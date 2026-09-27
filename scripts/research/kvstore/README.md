# Session KV store harness

Layer 1 of the session KV store design: the two file format, the append only status log,
the checkpoint root, and the recovery algorithm. Pure Python, no model, no backend, no
build step.

Spec: `docs/superpowers/specs/2026-09-27-session-kv-store-design.md`
Plan: `docs/superpowers/plans/2026-09-27-session-kv-store.md` (Task 1, Task 7 = E0)
Results: `docs/research/14-session-kv-store-format-results.md` (layer 1 format)
         `docs/research/15-e0-park-unpark-results.md` (existing park and unpark)
         `docs/research/16-disk-tier-results.md` (disk tier, buffered versus O_DIRECT)
         `docs/research/17-device-to-disk-path-results.md` (session size law, device transfer)
         `docs/research/18-direct-io-framing-results.md` (O_DIRECT alignment, block framing)
         `docs/research/19-block-granular-crash-matrix-results.md` (current matrix, block granular)
         `docs/research/20-direct-io-sink-results.md` (the engine sink, cross checked against Python)
         `docs/research/21-park-unpark-disk-results.md` (park and unpark a real sequence to disk)

## Run

```sh
python3 scripts/research/kvstore/scenarios.py      # crash matrix, expect pass
python3 scripts/research/kvstore/mutations.py      # rule falsification, expect pass
python3 scripts/research/kvstore/e0_park_unpark.py # existing park and unpark, ~80 s, needs a model
python3 scripts/research/kvstore/disk_bench.py     # disk tier, ~2 s, writes 512 MiB to scratch
python3 scripts/research/kvstore/direct_io_probe.py # O_DIRECT alignment and framing, ~0.5 s
```

The first two exit non zero on failure and need nothing but Python. The last two drive real
hardware and print tables; see the results docs for how to read them.

## Files

| file | role |
| --- | --- |
| `sim.py` | the format: framing, crc32, slot pages, root encoding, recovery, and a `Disk` that can lose or tear writes |
| `scenarios.py` | record plans, the crash matrix, and the controls |
| `mutations.py` | removes one rule at a time and re-runs the matrix, to show each rule is load bearing |
| `e0_park_unpark.py` | measures the park and unpark path that already ships, with a cache-disabled control and a capacity case |
| `disk_bench.py` | measures buffered versus `O_DIRECT` disk I/O and the page cache growth, which is the residency policy check |
| `direct_io_probe.py` | measures `O_DIRECT` alignment, write amplification, block framing, and RSS under a reused buffer |
| `direct_sink_check.cpp` | C++ check for the engine sink: byte exact readback, RSS, and two damage controls. Build with the command in `docs/research/20-direct-io-sink-results.md` |
| `cross_check.py` | reads a file written by the C++ sink with the Python reader, so two implementations of the format have to agree |
| `park_disk_check.cpp` | parks a real sequence to disk and unparks it, requiring an identical greedy continuation and a flat RSS. Build with the command in `docs/research/21-park-unpark-disk-results.md` |

## How the crash matrix works

`Disk` models a byte addressed file whose atomic unit is a 512 byte block, because that is
what `O_DIRECT` allows on this filesystem. One shared `Budget` covers both files, so a crash
stops everything.

For each plan, the matrix sweeps `limit` in `0..n_ledger_entries` over four tear cases: no
tear, and a torn prefix of 4, 32 or 256 bytes. The ledger has one entry per block flush, not
per record, because that is the real write granularity. After each crash it recovers and
asserts two things:

1. recovery returns the state at the newest committed epoch, or at the epoch before it when
   the crashing write was torn and fully landed, and never anything else
2. every rule in the spec holds: block framing, generations, page crc, monotonic positions,
   and for hybrid models the recurrent blob epoch matching the attention epoch

The expected states come from a second, independent application of the same logical
operations, so the log decoding path and the model can disagree.

## Plans

- `safe`: prefill, decode, checkpoint, a second session, eviction, prefix and middle range
  compaction, defrag with legal slot reuse, log compaction, more decode. Passes 170 crash
  points, returning all 12 committed states across the matrix.
- `unsafe_reuse`: one checkpoint removed, so a move reuses a slot the newest durable
  checkpoint still references. Must be detected. 6 of 56 runs caught.
- `stale_rs`: a bind whose recurrent blob is from an earlier epoch. Must be detected. 1 of
  46 runs caught, which is the full completion run.

## Controls

| control | expected |
| --- | --- |
| corrupt one byte of a live slot page | refuse |
| corrupt the newest root page | refuse |
| corrupt a superseded root page | accept, log compaction made it unreachable |

## What this does not cover

Shared cells and cross-session prefix sharing (refcounts), real filesystem and `fsync`
semantics, tiering and capacity policy, and performance. Those are layers 2 and 3, and the
spec defers the first of them.
