// Does MALLOC_PERTURB_ actually reach the allocations ggml uses?
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int all_of(const unsigned char * p, size_t n, unsigned char v) {
    for (size_t i = 0; i < n; ++i) if (p[i] != v) return 0;
    return 1;
}

int main(int argc, char ** argv) {
    const size_t n = argc > 1 ? (size_t) atoi(argv[1]) : 1048576;
    const char * env = getenv("MALLOC_PERTURB_");
    const unsigned char want = env ? (unsigned char) (atoi(env) ^ 0xff) : 0;

    unsigned char * a = malloc(n);
    memset(a, 0xAB, 1); // touch it so the allocation is real, then look at the next byte
    printf("malloc %zu: supposed fill %02x, first bytes", n, want);
    for (int i = 0; i < 8; ++i) printf(" %02x", a[i]);
    printf("  uniform=%d\n", all_of(a, n, want));
    free(a);

    void * b = NULL;
    if (posix_memalign(&b, 64, n) == 0) {
        printf("memalign %zu: first bytes", n);
        for (int i = 0; i < 8; ++i) printf(" %02x", ((unsigned char *) b)[i]);
        printf("  uniform=%d\n", all_of(b, n, want));
        free(b);
    }
    return 0;
}
