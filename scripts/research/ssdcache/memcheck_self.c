// Does this box compute and store the same answer twice?
//
// docs/research/26-prefill-vulkan-profile.md ends with "run a memory test, since
// non-deterministic arithmetic under load is also the signature of failing RAM", and
// docs/research/28-ubatch-determinism-results.md finds the same prompt missing to different
// values in one process. This is the cheap version of that test: arithmetic and memory
// integrity, repeated, with the results compared bit for bit.
//
// It does not replace a real memtest86 run. It is the part that can run under a normal user
// and next to a decoder.
//
// build: gcc -O2 -march=native -o memcheck_self memcheck_self.c -lm
// run:   ./memcheck_self [rounds]

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

// write a pattern, read it back, fold it into a checksum. two passes must agree, and the
// checksum must be the same in every round and every process
static uint64_t memory_round(uint8_t * buf, size_t size, uint32_t seed) {
    for (size_t i = 0; i < size; ++i) {
        buf[i] = (uint8_t) ((i * 2654435761u + seed) >> 13);
    }

    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < size; i += 8) {
        uint64_t v;
        memcpy(&v, buf + i, sizeof(v));
        h ^= v;
        h *= 1099511628211ull;
    }
    return h;
}

// a tight floating point reduction over a small hot array, twice. bitwise comparison, so a
// single differing bit in the last place is a failure
static uint64_t fma_round(size_t n, int iters, uint32_t seed) {
    float * a = malloc(n * sizeof(float));
    float * b = malloc(n * sizeof(float));
    if (!a || !b) {
        exit(1);
    }

    for (size_t i = 0; i < n; ++i) {
        a[i] = (float) ((i % 977) * 0.001953125f + 1.0f);
        b[i] = (float) ((i % 331) * 0.00390625f - 0.5f);
    }

    // a deterministic serial reduction with an fma chain, the shape a matmul uses
    double acc = 0.0;
    for (int it = 0; it < iters; ++it) {
        const uint32_t s = seed + (uint32_t) it;
        float x = (float) (s & 0xFFFF) / 65536.0f;
        for (size_t i = 0; i < n; ++i) {
            x = fmaf(x, a[i], b[i]);
            acc += (double) x;
        }
    }

    uint64_t h = 1469598103934665603ull;
    uint64_t bits;
    memcpy(&bits, &acc, sizeof(bits));
    h ^= bits;
    h *= 1099511628211ull;

    free(a);
    free(b);
    return h;
}

int main(int argc, char ** argv) {
    const int rounds = argc > 1 ? atoi(argv[1]) : 4;
    const size_t mib   = argc > 2 ? (size_t) atoi(argv[2]) : 512;

    printf("rounds %d, buffer %zu MiB\n", rounds, mib);

    uint8_t * buf = malloc(mib << 20);
    if (!buf) {
        printf("allocation failed\n");
        return 1;
    }

    uint64_t h_mem_first = 0;
    uint64_t h_fma_first = 0;
    int mem_diff = 0;
    int fma_diff = 0;

    for (int r = 0; r < rounds; ++r) {
        const double t0 = now_s();
        const uint64_t hm = memory_round(buf, mib << 20, 12345u);
        const uint64_t hf = fma_round(4096, 2000, 777u);
        const double dt = now_s() - t0;

        if (r == 0) {
            h_mem_first = hm;
            h_fma_first = hf;
        }
        const int md = hm != h_mem_first;
        const int fd = hf != h_fma_first;
        mem_diff += md;
        fma_diff += fd;

        printf("  round %d  memory %016llx%s  fma %016llx%s  %.2f s\n", r,
               (unsigned long long) hm, md ? "  DIFFERS" : "",
               (unsigned long long) hf, fd ? "  DIFFERS" : "", dt);
    }

    printf("\n  memory checksum stable: %s\n", mem_diff == 0 ? "yes" : "NO");
    printf("  fma checksum stable:    %s\n", fma_diff == 0 ? "yes" : "NO");
    printf("  memory %016llx\n  fma    %016llx\n",
           (unsigned long long) h_mem_first, (unsigned long long) h_fma_first);

    free(buf);
    return (mem_diff == 0 && fma_diff == 0) ? 0 : 1;
}
