// Sequential read bandwidth probe used by scripts/research/bench-stage.sh.
// Reads a 3 GiB zero-filled array and sums it, once per thread count.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int main(int argc, char ** argv) {
    const size_t n = 3UL << 30;
    const int nth = argc > 1 ? atoi(argv[1]) : 8;

    unsigned char * a = aligned_alloc(4096, n);
    if (!a) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    memset(a, 1, n);

    double best = 0.0;
    for (int r = 0; r < 3; r++) {
        const double t0 = omp_get_wtime();
        unsigned long long s = 0;
        #pragma omp parallel num_threads(nth) reduction(+:s)
        {
            unsigned long long ls = 0;
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n/8; i++) {
                ls += ((unsigned long long *) a)[i];
            }
            s += ls;
        }
        const double t = omp_get_wtime() - t0;
        const double gbs = n/t/1e9;
        if (gbs > best) {
            best = gbs;
        }
        printf("membw: threads=%d run=%d %.2f GB/s (chk=%llu)\n", nth, r, gbs, s);
    }
    printf("membw: threads=%d BEST %.2f GB/s\n", nth, best);
    free(a);
    return 0;
}
