// what does the per block crc32 cost, at the granularity the store uses
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <time.h>

uint32_t llama_io_crc32(const void * data, size_t size);

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

int main() {
    const size_t blk = 4096;
    const size_t n   = 6900;          // blocks in a 27 MiB state
    uint8_t * buf = new uint8_t[blk];
    for (size_t i = 0; i < blk; ++i) buf[i] = (uint8_t) (i * 31 + 7);

    // warm
    volatile uint32_t sink = 0;
    for (size_t i = 0; i < 100; ++i) sink ^= llama_io_crc32(buf + 12, blk - 12);

    double t = now_s();
    for (size_t i = 0; i < n; ++i) sink ^= llama_io_crc32(buf + 12, blk - 12);
    double dt = now_s() - t;
    printf("crc32 over %zu x %zu B: %.4f s, %.3f us per block, %.1f MiB/s\n",
           n, blk - 12, dt, dt / n * 1e6, (n * (blk - 12) / 1048576.0) / dt);
    printf("(sink %u)\n", (unsigned) sink);
    delete[] buf;
    return 0;
}
