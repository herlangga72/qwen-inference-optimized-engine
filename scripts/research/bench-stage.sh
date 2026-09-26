#!/usr/bin/env bash
# Baseline harness for the qwen-only fork research.
#
# Records, for one model: DRAM read ceiling, page cache residency, and a llama-bench
# matrix over threads and load modes. Output goes to benches/<tag>/<tag>.md.
#
# usage: scripts/research/bench-stage.sh MODEL.gguf [TAG]
#
# env:
#   BUILD=build-vk            which build tree to use (default build-cpu)
#   BENCH_ARGS="-ngl 99"      extra llama-bench args, for the GPU path

set -u

MODEL="${1:?usage: bench-stage.sh MODEL.gguf [TAG]}"
TAG="${2:-$(uname -n | tr ' ' '-')}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${BUILD:-$REPO/build-cpu}"
case "$BUILD" in
    /*) ;;
    *) BUILD="$REPO/$BUILD" ;;
esac
BENCH_ARGS="${BENCH_ARGS:-}"
BIN="$BUILD/bin"
OUT="$REPO/benches/$TAG"
MD="$OUT/$TAG.md"
BENCH="$BIN/llama-bench"

[ -x "$BENCH" ] || { echo "no llama-bench in $BIN" >&2; exit 1; }

mkdir -p "$OUT"
exec > >(tee "$MD") 2>&1

echo "## System info"
echo
echo '```bash'
uname -a
echo "$(nproc) threads, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(awk '/MemTotal/{printf "%.1f GiB", $2/1048576}' /proc/meminfo) RAM"
echo '```'
echo

echo "## Memory ceiling"
echo
echo '```'
gcc -O2 -march=native -fopenmp "$REPO/scripts/research/membw.c" -o "$OUT/membw" || exit 1
for t in 8 16; do "$OUT/membw" "$t"; done
echo '```'
echo

echo "## Model"
echo
echo '```'
ls -l "$MODEL"
python3 "$REPO/scripts/research/pagecache.py" "$MODEL"
echo '```'
echo

echo "## llama-bench"
echo

echo '#### prompt processing, 8 threads'
echo
"$BENCH" -m "$MODEL" -r 3 -p 256 -n 0 -t 8 $BENCH_ARGS
echo

echo '#### token generation against thread count'
echo
for t in 4 8 16; do
    "$BENCH" -m "$MODEL" -r 3 -p 0 -n 128 -t "$t" $BENCH_ARGS
done
echo

echo '#### token generation against load mode, 8 threads'
echo
for lm in mmap none mlock; do
    "$BENCH" -m "$MODEL" -r 3 -p 0 -n 128 -t 8 -lm "$lm" $BENCH_ARGS
done
echo
