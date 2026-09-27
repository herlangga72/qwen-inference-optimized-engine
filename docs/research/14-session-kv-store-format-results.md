# Session KV store (layer 1): format results

Date: 2026-09-27
Machine: Ryzen 7 6800H. No GPU and no model needed, the format is pure bytes.
Tree: `qwen-only-backends`
Spec: `docs/superpowers/specs/2026-09-27-session-kv-store-design.md`
Plan: `docs/superpowers/plans/2026-09-27-session-kv-store.md` (Task 1)
Harness: `scripts/research/kvstore/sim.py`, `scripts/research/kvstore/scenarios.py`

Superseded: the format moved to block granularity under `O_DIRECT`, so the current matrix and
its results are in `docs/research/19-block-granular-crash-matrix-results.md`. This document is
kept for the record-level reasoning and the findings that still hold.

**Verdict: the two file format survives a crash at every write boundary. Across 170
crash points per plan, recovery always returns exactly the newest committed state, never a
partial one. The rules that make that true were identified by mutation, not assumed: four
of them are individually load bearing, and one of them (slot reuse) gives the design a
constraint that was not obvious from the spec text.**

## What was built

`sim.py` implements `kv.status` (append only, framed with length plus crc32), `kv.pages`
(header plus fixed size slots), the checkpoint root, and the recovery algorithm. `Disk`
models a byte addressed file where each atomic write, up to 4096 bytes, either lands
whole, lands a prefix (torn), or does not happen. One shared `Budget` covers both files, so
one crash stops everything, which is what a real power loss does.

`scenarios.py` builds record plans and, in parallel, the expected states from a second,
independent application of the same logical operations. The comparison is between the log
decoding path and the logical model, so the two can disagree.

The matrix sweeps, for each plan, `limit` in `0..n_records` times `tear` in
`{false, true}`. That is every crash boundary and every torn boundary.

## Result

| plan | runs | records | states recovered | runs using a reclaimed base | failures |
| --- | --- | --- | --- | --- | --- |
| safe | 170 | 84 | 12 | 31 | 0 |
| unsafe_reuse (control) | 56 | 27 | 3 | 9 | 6, all detected |
| stale_rs (control) | 46 | 22 | 2 | 0 | 1, detected |

`states` is the number of distinct committed epochs recovery returned across the matrix. The
safe plan returns all 12, so the matrix is not passing by always landing on the same
trivially easy state.

Additional controls, each an assertion rather than a measurement:

| control | expected | observed |
| --- | --- | --- |
| corrupt one byte of a live slot page | refuse | refused, page crc or generation mismatch |
| corrupt the newest root page | refuse | refused, checkpoint root crc mismatch |
| corrupt a superseded root page | accept, it is unreachable | not read |

The last one is what proves log compaction actually took effect: after the base advance, the
older root is never consulted, so damage there is invisible.

## Rules that are load bearing

Each row is a mutation applied to the harness, then the full matrix re-run. A mutation that
does not change the outcome is a rule that was not being tested.

| mutation | safe plan failures | conclusion |
| --- | --- | --- |
| defer all page writes to the end (log durable before data) | 138 / 170 | page before record is required |
| apply stray records outside an epoch instead of refusing | 31 / 170 | the epoch bracket is a visibility rule, not bookkeeping |
| accept records with a bad crc (no torn tail rule) | 11 / 170 | framing plus crc is required |
| blind the structural page check | 0 / 170 | the generation comparison alone still catches it, so there are two independent catches |
| reuse a slot the newest durable checkpoint still references | 6 / 56 in the control | the reuse rule is required, see below |

The safe plan recovered 12 distinct states. Under the page-last mutation that collapsed to
2, because recovery could no longer trust any page it had not yet written.

Run `scripts/research/kvstore/mutations.py` to reproduce the first four rows.

## Findings that change the design

### 1. Slot reuse has a precondition, and it is not in the spec yet

The spec said compaction writes new slots under new generations. What the harness shows is
that reusing a slot is only legal when that slot is not referenced by the newest durable
checkpoint. Otherwise a crash between the page write and the record that names the new
generation leaves the checkpoint's root pointing at a page whose generation has already
changed, and recovery refuses.

That is the whole difference between the `safe` and `unsafe_reuse` plans: one checkpoint.
After the checkpoint, slots 0 to 5 are no longer in any root, so slot 0 can be reused at
epoch 8. Without that checkpoint, the same move corrupts the recoverable state.

Consequence: log structured garbage collection here needs free space. You cannot compact in
place in one pass unless you accept that the recoverable point moves back. Compact a
bounded number of slots per checkpoint, and only into slots the newest durable checkpoint
does not name.

### 2. A torn base advance is safe, but it removes the reclaim

Advancing the status base to the last checkpoint is a single small write, and if it is torn
recovery falls back to the file header and replays from the beginning, which is idempotent
because the checkpoint is still in the log. Correct, but it means the older region is read
again. So physically truncating the status file is only safe once the base advance is
durable. Logical reclaim is free, physical reclaim is not.

Measured both directions on the same damage, one byte flipped in the superseded root:

| base header | outcome |
| --- | --- |
| intact | accepted, the older root is never read |
| invalid, as a torn write would leave it | refused, `checkpoint root crc mismatch` |

So the dead region is dead only while the base advance holds. That is the ordering
constraint on any future physical truncation work.

### 3. The recurrent state cannot be compacted, only rewritten

Middle range compaction on a hybrid sequence, removing a middle range, cannot be expressed in
the gated delta net state at all: it is a running summary, not per token data. The harness
models this as a required rewrite of the blob in the same epoch as the `seq_rm`, and
detects the stale case. One detection in 46 runs, which is the whole point: without the
same epoch rule the failure is silent, because no crc anywhere disagrees.

So `seq_rm` on a hybrid sequence must either be rejected or must roll the recurrent blob
back to a saved snapshot, and that rollback has to be in the same committed epoch as the
attention change.

## What this does not cover

- Shared cells. Cross-session prefix sharing means refcounts and pages referenced by
  several sequences. Deliberately deferred in the spec, and not exercised here.
- Real device semantics. Writes are atomic up to 4096 bytes with an optional torn prefix.
  Real filesystems, `fsync` barriers and discard are not modelled.
- Capacity and eviction policy, the tier model, and the write ordering against actual KV
  bytes. Those are layers 2 and 3.
- Performance. Nothing here says a park and unpark is fast enough.

## Reproduce

```sh
python3 scripts/research/kvstore/scenarios.py
```
