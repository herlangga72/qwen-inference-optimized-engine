#!/usr/bin/env bash
# Smoke test for the multimodal path (Qwen3.5-VL: text GGUF + qwen3vl mmproj).
#
# Usage:
#   MODEL=/path/to/qwen35-vl-text.gguf MMPROJ=/path/to/mmproj-qwen35-vl.gguf ./tools/mtmd/tests.sh
#
# Optional environment variables:
#   IMAGE   image to send      (default: test-1.jpeg next to this script)
#   PROMPT  prompt to use      (default: "Describe this image.")
#   N_PREDICT number of tokens (default: 128)

set -eux

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJ_ROOT="$SCRIPT_DIR/../.."

: "${MODEL:?set MODEL to the text GGUF (arch qwen35 or qwen35moe)}"
: "${MMPROJ:?set MMPROJ to the matching qwen3vl mmproj GGUF}"

PROMPT="${PROMPT:-Describe this image.}"
N_PREDICT="${N_PREDICT:-128}"
IMAGE="${IMAGE:-$SCRIPT_DIR/test-1.jpeg}"

cmake --build "$PROJ_ROOT/build" -j --target llama-mtmd-cli

"$PROJ_ROOT/build/bin/llama-mtmd-cli" \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --image "$IMAGE" \
    -p "$PROMPT" \
    --temp 0 -n "$N_PREDICT"
