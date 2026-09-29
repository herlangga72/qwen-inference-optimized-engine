// What the store's read path actually costs, without Python in the loop.
//
// llama_io_read_direct issues one pread of LLAMA_IO_BLOCK (4096) bytes per block
// (src/llama-io.cpp: next_block). This measures that pattern against larger reads on the
// same file, and against buffered reads, so the per syscall cost can be separated from
// the device's bandwidth.
//
// build:
//   gcc -O2 -o <out> scripts/research/ssdcache/block_read_probe.c
// run:
//   <out> [size_mib]

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define BLOCK 4096

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

static void * aligned_alloc_pages(size_t size) {
    void * p = NULL;
    if (posix_memalign(&p, BLOCK, size) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        exit(1);
    }
    memset(p, 0, size);
    return p;
}

static double cached_mib(void) {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) {
        return -1.0;
    }
    char line[256];
    double out = -1.0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "Cached:", 7) == 0) {
            out = strtod(line + 7, NULL) / 1024.0;
            break;
        }
    }
    fclose(f);
    return out;
}

static void drop_cache(const char * path) {
    int fd = open(path, O_RDONLY);
    if (fd >= 0) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
    }
}

static void write_blocks(const char * path, size_t size, const void * buf, size_t chunk) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) {
        fprintf(stderr, "O_DIRECT open for write failed: %s\n", strerror(errno));
        exit(1);
    }
    for (size_t off = 0; off < size; off += chunk) {
        ssize_t n = pwrite(fd, (const char *) buf, chunk, (off_t) off);
        if (n != (ssize_t) chunk) {
            fprintf(stderr, "pwrite at %zu returned %zd: %s\n", off, n, strerror(errno));
            exit(1);
        }
    }
    fsync(fd);
    close(fd);
}

// returns MiB/s, and sets *us_per_op
static double read_pattern(const char * path, size_t size, size_t chunk, int direct, double * us_per_op) {
    void * buf = aligned_alloc_pages(chunk);
    int flags = O_RDONLY | (direct ? O_DIRECT : 0);
    int fd = open(path, flags);
    if (fd < 0) {
        fprintf(stderr, "open for read failed: %s\n", strerror(errno));
        exit(1);
    }
    const double t0 = now_s();
    size_t ops = 0;
    for (size_t off = 0; off < size; off += chunk) {
        ssize_t n = pread(fd, buf, chunk, (off_t) off);
        if (n != (ssize_t) chunk) {
            fprintf(stderr, "pread at %zu returned %zd: %s\n", off, n, strerror(errno));
            exit(1);
        }
        ops++;
    }
    const double dt = now_s() - t0;
    close(fd);
    free(buf);
    *us_per_op = dt / (double) ops * 1e6;
    return (double) size / (1024.0 * 1024.0) / dt;
}

int main(int argc, char ** argv) {
    const size_t size_mib = argc > 1 ? (size_t) atoi(argv[1]) : 256;
    const char * path = argc > 2 ? argv[2] : "/home/herlanggays/.jcode/scratch/ssd-cache/disk/blocks.bin";
    const size_t size = size_mib * 1024 * 1024;

    printf("file %zu MiB, %s\n", size_mib, path);
    printf("Cached before %.0f MiB\n", cached_mib());

    void * fill = aligned_alloc_pages(1 << 20);
    for (size_t i = 0; i < (1 << 20); ++i) {
        ((unsigned char *) fill)[i] = (unsigned char) (i * 7 + 11);
    }

    double us;
    // write once at 1 MiB granularity so the file exists in a sane layout
    double t0 = now_s();
    write_blocks(path, size, fill, 1 << 20);
    double tw = now_s() - t0;
    printf("  O_DIRECT write, 1 MiB per pwrite   %6.3f s  %7.1f MiB/s   Cached %.0f MiB\n",
           tw, (double) size_mib / tw, cached_mib());

    printf("\n  read back, O_DIRECT, chunk swept\n");
    printf("  %10s %10s %12s %12s\n", "chunk", "MiB/s", "us/op", "ops");
    for (size_t chunk = BLOCK; chunk <= (1u << 20); chunk <<= 1) {
        drop_cache(path);
        double r = read_pattern(path, size, chunk, 1, &us);
        printf("  %9zuK %10.1f %12.2f %12zu\n", chunk >> 10, r, us, size / chunk);
    }

    printf("\n  read back, buffered (page cache warm)\n");
    printf("  %10s %10s %12s %12s\n", "chunk", "MiB/s", "us/op", "ops");
    for (size_t chunk = BLOCK; chunk <= (1u << 20); chunk <<= 1) {
        double r = read_pattern(path, size, chunk, 0, &us);
        printf("  %9zuK %10.1f %12.2f %12zu\n", chunk >> 10, r, us, size / chunk);
    }

    // the writer in llama_io_write_direct::put_block issues one pwrite of one block per call
    printf("\n  write, O_DIRECT, chunk swept (one pwrite per chunk)\n");
    printf("  %10s %10s %12s %12s\n", "chunk", "MiB/s", "us/op", "ops");
    for (size_t chunk = BLOCK; chunk <= (1u << 20); chunk <<= 1) {
        const double t = now_s();
        write_blocks(path, size, fill, chunk);
        const double dt = now_s() - t;
        printf("  %9zuK %10.1f %12.2f %12zu\n", chunk >> 10,
               (double) size_mib / dt, dt / (double) (size / chunk) * 1e6, size / chunk);
    }
    printf("  Cached after writes %.0f MiB\n", cached_mib());

    // the number the design needs: sequential 4 KB reads, as the store does today
    drop_cache(path);
    double rate4k = read_pattern(path, size, BLOCK, 1, &us);
    const double us4k = us;
    drop_cache(path);
    double rate1m = read_pattern(path, size, 1 << 20, 1, &us);

    printf("\n4 KB O_DIRECT   %7.1f MiB/s  %6.2f us per read\n", rate4k, us4k);
    printf("1 MiB O_DIRECT  %7.1f MiB/s  %6.2f us per read\n", rate1m, us);
    printf("ratio %.1fx\n", rate1m / rate4k);

    // restore cost for a cached prefix: 7400 B/token attention plus 62.8 MiB recurrent,
    // framed 1.6 percent by the store's block headers
    const double pp_us = 1e6 / 120.0;
    printf("\nrestore, 4 KB reads as the store does today (pp 120 t/s = %.0f us/token)\n", pp_us);
    printf("  %8s %10s %11s %11s %11s\n", "tokens", "on disk", "at 4 KB", "us/token", "pp/x");
    const int lens[] = { 4000, 13500, 64000, 262144 };
    for (int i = 0; i < 4; ++i) {
        const double bytes = (double) lens[i] * 7400.0 * 1.016 + 62.8 * 1024 * 1024;
        const double sec = bytes / (1024.0 * 1024.0) / rate4k;
        const double ut = sec / lens[i] * 1e6;
        printf("  %8d %9.1fM %10.1fms %11.2f %11.0f\n", lens[i], bytes / (1024 * 1024), sec * 1000, ut, pp_us / ut);
    }
    printf("\nrestore, 1 MiB reads (what the reader could do)\n");
    printf("  %8s %10s %11s %11s %11s\n", "tokens", "on disk", "at 1 MiB", "us/token", "pp/x");
    for (int i = 0; i < 4; ++i) {
        const double bytes = (double) lens[i] * 7400.0 * 1.016 + 62.8 * 1024 * 1024;
        const double sec = bytes / (1024.0 * 1024.0) / rate1m;
        const double ut = sec / lens[i] * 1e6;
        printf("  %8d %9.1fM %10.1fms %11.2f %11.0f\n", lens[i], bytes / (1024 * 1024), sec * 1000, ut, pp_us / ut);
    }

    free(fill);
    return 0;
}
