#!/usr/bin/env bash
# Launch the Qwen3.6-35B-A3B Vulkan engine with MTP speculative decoding enabled.
#
# MTP is the model's own NextN head (blk.40.nextn.*), so no separate draft model is needed: the head is
# already in the GGUF and the --spec-type draft-mtp path loads it instead of ignoring it.
#
# Depth 3 is the measured optimum on this machine (-c 3000, fixed prompt, interleaved runs):
#
#   no spec   21.35 t/s      depth 1   24.05      depth 2   24.95
#   depth 3   27.3 - 28.2    depth 4   26.1       depth 6   22.8      depth 8   14.5
#
# so generation runs 24 to 32 percent faster than without it and prefill is unchanged. Depth 2 would cost
# about 9 percent against depth 3, and depth 8 falls below the no-spec baseline because the draft cost
# exceeds what verification saves. See docs/research/26 for the measurements.
#
# Overridable by environment: MODEL, CTX, PORT, HOST, NPARALLEL, EXTRA.
#   MODEL=/path/to/model.gguf CTX=8192 PORT=8080 ./scripts/run-qwen-mtp.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build-vk/bin/llama-server"

MODEL="${MODEL:-/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf}"
CTX="${CTX:-4096}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
NPARALLEL="${NPARALLEL:-1}"
EXTRA="${EXTRA:-}"

if [ ! -x "$BIN" ]; then
    echo "$0: missing $BIN" >&2
    echo "$0: build it with: cmake --build $ROOT/build-vk -j8 --target llama-server" >&2
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "$0: model not found: $MODEL" >&2
    exit 1
fi

echo "$0: model $MODEL"
echo "$0: mtp on, draft depth 3, ctx $CTX, http://$HOST:$PORT"

set -x
exec "$BIN" \
    -m "$MODEL" \
    -ngl 99 \
    -fa on \
    -c "$CTX" \
    -np "$NPARALLEL" \
    --spec-type draft-mtp \
    --spec-draft-n-max 3 \
    --host "$HOST" \
    --port "$PORT" \
    $EXTRA
