#!/usr/bin/env bash
# Run a memory-heavy child serially with a watchdog. Aborts the child if MemAvailable
# drops below the floor, because the box has livelocked under GTT over-allocation.
# See docs/research/freeze-notes-2026-09-27.md
FLOOR_MIB="${FLOOR_MIB:-2000}"
LOG="${LOG:-/dev/null}"
"$@" >"$LOG" 2>&1 &
CHILD=$!
MIN=999999999
while kill -0 "$CHILD" 2>/dev/null; do
  AV=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
  [ "$AV" -lt "$MIN" ] && MIN=$AV
  if [ "$AV" -lt "$FLOOR_MIB" ]; then
    echo "WATCHDOG: MemAvailable ${AV} MiB below floor ${FLOOR_MIB} MiB, killing child"
    kill -9 "$CHILD" 2>/dev/null
    wait "$CHILD" 2>/dev/null
    echo "WATCHDOG_MIN_MIB=$MIN"
    exit 99
  fi
  sleep 0.5
done
wait "$CHILD"
RC=$?
echo "MIN_MEMAVAILABLE_MIB=$MIN"
exit $RC
