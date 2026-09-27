#!/bin/bash
# Tokenizer microbenchmark. Builds a small harness against build-cpu and prints ms/repetition,
# throughput and an id checksum that is comparable between builds. Prefix with LLAMA_TOKENIZER_LEGACY=1
# to force the reference path, which is inert for the merge loop in the current tree. The checksum
# covers all repetitions, so compare runs made with the same repetition count.
#
# Usage: scripts/research/tok_bench.sh [reps]
set -e

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN_DIR="$REPO/build-cpu/bin"
VOCAB="${VOCAB:-$REPO/models/ggml-vocab-qwen35.gguf}"
REPS="${1:-20}"
TEXT="${TEXT:-}"
SRC="$(mktemp -d)"

trap 'rm -rf "$SRC"' EXIT

if [ -z "$TEXT" ]; then
    TEXT="$SRC/text.txt"
    cat "$REPO"/scripts/research/tok_corpus/*.txt > "$TEXT"
fi

cat > "$SRC/bench_tok.cpp" <<'EOF'
#include "llama.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <vocab.gguf> <text file> <reps>\n", argv[0]);
        return 1;
    }
    const int reps = atoi(argv[3]);

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (model == nullptr) {
        fprintf(stderr, "model load failed\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string text;
    {
        FILE * f = fopen(argv[2], "rb");
        if (f == nullptr) {
            fprintf(stderr, "cannot open %s\n", argv[2]);
            return 1;
        }
        char buf[65536];
        size_t got;
        while ((got = fread(buf, 1, sizeof(buf), f)) > 0) {
            text.append(buf, got);
        }
        fclose(f);
    }

    std::vector<llama_token> tokens(text.size() + 8);
    const auto t0 = std::chrono::steady_clock::now();

    size_t total = 0;
    uint64_t checksum = 1469598103934665603ull;
    for (int r = 0; r < reps; r++) {
        const int n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), false, true);
        if (n < 0) {
            fprintf(stderr, "tokenize failed, needs %d\n", -n);
            return 1;
        }
        total += (size_t) n;
        for (int i = 0; i < n; i++) {
            checksum ^= (uint64_t) tokens[i];
            checksum *= 1099511628211ull;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    printf("bytes %zu reps %d ms/rep %.3f MB/s %.2f tokens/rep %.0f checksum %016llx\n",
           text.size(), reps, 1000.0*sec/reps, (double) text.size()/1e6/(sec/reps),
           (double) total/reps, (unsigned long long) checksum);

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
EOF

g++ -O2 -std=c++17 -I "$REPO/include" "$SRC/bench_tok.cpp" \
    -L "$BIN_DIR" -lllama -Wl,-rpath,"$BIN_DIR" -o "$SRC/bench_tok"

"$SRC/bench_tok" "$VOCAB" "$TEXT" "$REPS"
