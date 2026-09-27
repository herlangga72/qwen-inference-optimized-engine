#!/usr/bin/env bash
# Differential tokenizer check: ids from the fast path must equal ids from the reference path.
# usage: scripts/research/tok_diff.sh [corpus_dir]
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${BIN:-$REPO/build-cpu/bin/llama-tokenize}"
VOCAB="${VOCAB:-$REPO/models/ggml-vocab-qwen35.gguf}"
CORPUS="${1:-$REPO/scripts/research/tok_corpus}"

fail=0
for f in "$CORPUS"/*.txt "$REPO"/src/*.cpp "$REPO"/common/*.cpp; do
    ref=$(LLAMA_TOKENIZER_LEGACY=1 "$BIN" -m "$VOCAB" -f "$f" --ids 2>/dev/null)
    new=$(LLAMA_TOKENIZER_LEGACY=0 "$BIN" -m "$VOCAB" -f "$f" --ids 2>/dev/null)
    if [ "$ref" != "$new" ]; then
        echo "DIFFERS: $f"
        fail=1
    fi
done

[ "$fail" = 0 ] && echo "all identical" || echo "divergences found"
exit "$fail"
