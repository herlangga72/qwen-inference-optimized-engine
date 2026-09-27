# Block granular crash matrix: results

Date: 2026-09-27
Tree: `qwen-only-backends`
Harness: `scripts/research/kvstore/sim.py`, `scenarios.py`, `mutations.py`
Spec: `docs/superpowers/specs/2026-09-27-session-kv-store-design.md`
Supersedes: `docs/research/14-session-kv-store-format-results.md`, which ran the same matrix
at record granularity, before the atomic unit became a block.

**Verdict: the block framed format survives a crash at every write boundary. The matrix now
models what the file can actually do under `O_DIRECT`, one atomic write per 512 byte block,
and in the process it found a real ordering bug that would have lost every session in the
store.**

## What changed since the record level matrix

| | record level (doc 14) | block level (here) |
| --- | --- | --- |
| atomic unit | one record | one 512 byte block |
| status layout | bare records | `[u32 used][u32 crc32(payload area)][records][padding]` |
| ledger entry | per record | per block flush |
| durable commit | the commit record landed | the block holding the commit landed |
| crash points per plan | 170 | 88 |

Crash coverage is coarser for the same plan, because there are fewer writes: 21 ledger
entries instead of 84 records. That is the real granularity, not a weaker test.

## Result

| plan | runs | ledger entries | states recovered | reclaimed base | failures |
| --- | --- | --- | --- | --- | --- |
| safe | 88 | 21 | 7 | 15 | 0 |
| unsafe_reuse (control) | 44 | 10 | 3 | 11 | 7, all detected |
| stale_rs (control) | 28 | 6 | 2 | 0 | 2, both detected |

Controls, each an assertion:

| control | expected | observed |
| --- | --- | --- |
| corrupt one byte of a live slot page | refuse | refused |
| corrupt the newest root page | refuse | refused |
| corrupt a superseded root page | accept, unreachable | not read |
| zero the header of the newest live block | lose the tail, not read through it | fell back to epoch 9 |

## Finding 1: the base advance raced the block flush

The first run failed 8 of 44 crash points on the safe plan, all in the log compaction epoch.
Recovery returned an empty state.

`set_base` is an independent in-place write. The block holding the checkpoint and its commit
is buffered until it fills or the epoch ends. So the base could be advanced to point at a
block that had not been written yet, and every byte behind it was then unreachable:

| step | state |
| --- | --- |
| base points at block 4 | block 4 not yet flushed |
| crash before block 4 is written | base says start at block 4 |
| recovery reads block 4 | zeros, `used == 0`, stop |
| result | **empty state, every session lost** |

The fix is one line of ordering: flush the block before advancing the base. It is now step 4
of the log compaction sequence in the spec, with the reason written down, because it is
invisible in the code and the failure is total.

This is exactly what "the ledger has one entry per block flush" was for, and it is the
strongest argument so far for keeping the matrix rather than trusting the argument.

## Finding 2: a torn write can fully land

One crash point kept failing after that fix. The torn prefix was half a block, 256 bytes, and
the block's payload area was 168 bytes. The prefix therefore covered the entire payload, the
header crc matched, and the block was valid. Recovery correctly accepted it; my expectation
had wrongly assumed a torn write is always lost.

Both behaviours are correct and now both are modelled:

- a torn prefix that cuts into the payload leaves the block invalid and it is lost whole
- a torn prefix that covers the payload leaves a complete, valid block

The second case is indistinguishable from a full write and does not need to be distinguished,
because every byte the block claims is present. The matrix now allows either outcome at a
torn boundary and sweeps the prefix length over 4, 32 and 256 bytes, so both are exercised.

Nothing observable sits between them: there is no partial block state.

## Rules that are load bearing, re-checked

Each row is a mutation, then the full matrix re-run.

| mutation | safe plan failures |
| --- | --- |
| defer all page writes to the end | 75 of 88 |
| record scan that ignores the block framing | 65 of 88 |
| apply stray records outside an epoch | 15 of 88 |
| blind the page crc check | 0, the generation check still catches it |

The 65 is the new one and the important one: with the naive record scan the block header stops
mattering, and the format breaks almost everywhere. This is the same failure the direct I/O
probe measured at 93 of 2000 records, now reproduced inside the crash matrix.

## What this does not cover

- Shared cells and cross session prefix sharing. Still deferred by the spec.
- Real device and filesystem behaviour. `O_DIRECT` alignment and framing were measured
  directly in `docs/research/18-direct-io-framing-results.md`; the crash model itself is
  simulated.
- Multi-file consistency, and any crash that spans the two files in a way a single lost
  write cannot express.
- Compaction under pressure, where slot reuse needs free space. The plan only reuses a slot
  after a checkpoint, which is the legal case.

## Reproduce

```sh
python3 scripts/research/kvstore/scenarios.py   # expect exit 0
python3 scripts/research/kvstore/mutations.py   # expect exit 0
```
