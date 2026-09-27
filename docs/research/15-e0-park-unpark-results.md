# E0: existing park and unpark path (results)

Date: 2026-09-27
Machine: Ryzen 7 6800H, CPU backend, `build-cpu`.
Tree: `qwen-only-backends`
Model: `/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf`
Harness: `scripts/research/kvstore/e0_park_unpark.py`
Logs: `/home/herlanggays/.jcode/scratch/kvstore/e0-*.log`

**Verdict: park and unpark already works, through `server_prompt_cache` plus `--cache-ram`
plus `--cache-idle-slots`, and the round trip is nearly free: a returning session costs
about the same as one that was never evicted. Its limit is arena capacity. A restore needs
free KV equal to the whole session footprint while the state it replaces still occupies the
arena, so with a small context the restore cannot fit and the request falls back to a full
re-prefill. See the correction immediately below before relying on that.**

The first version of this document said the feature does not work. That was wrong, and one
extra case falsified it. The record of how is in "Two explanations" below, because the
distinction is the whole finding.

**Correction, added later: the capacity conclusion below rests on a single pair of runs.**
Each number here was measured once or twice. Repeating the same path later turned up a separate
instability on the *file* path (`docs/research/22-state-serialization-not-reproducible.md`),
which does not touch this host buffer path: two host serializations of the same state are always
identical. So the capacity explanation is not contradicted by that bug. It is still supported
by one 2048 run against one 8192 run, and should be re-measured with repeats.

## Setup

Three requests per case, all with `cache_prompt: true`:

| request | slot | prompt | purpose |
| --- | --- | --- | --- |
| A | 0 | P1, 1339 tokens | fill the slot |
| B | 1 | P2, 1274 tokens | a new task, which parks idle slots |
| C | 0 | P1 again | the unpark |

Each case is run with the cache on (`--cache-ram 512 --cache-idle-slots`) and off
(`--cache-ram 0`). Repeat runs are stable.

## Result

Request C, 1339 tokens. `A/C` is the speedup against that case's own first request.

| configuration | cache | C prompt eval | A/C | restore attempts | restore failures |
| --- | --- | --- | --- | --- | --- |
| separate KV per slot, 8192 ctx | on | 34 ms | 90x | 0 | 0 |
| separate KV per slot, 8192 ctx | off | 37 ms | 85x | 0 | 0 |
| unified, 2048 ctx, slot pinned | on | 2968 ms | 1.0x | 0 | 0 |
| unified, 2048 ctx, slot pinned | off | 3056 ms | 1.0x | 0 | 0 |
| unified, 2048 ctx, slot auto | on | 3091 ms | 1.0x | 3 | 1 |
| unified, 2048 ctx, slot auto | off | 3171 ms | 1.0x | 0 | 0 |
| unified, 8192 ctx, slot auto | on | **32 ms** | **99x** | 3 | 0 |
| unified, 8192 ctx, slot auto | off | 36 ms | 91x | 0 | 0 |

Read the last two rows together, because they are fast for different reasons:

- cache off, 8192: nothing was evicted, so slot 0 still held P1's KV and C reused it
  directly. `restore attempts = 0`.
- cache on, 8192: idle slots were parked and cleared, so slot 0's KV was gone and C
  restored it from host RAM. `restore attempts = 3, failures = 0`.

So the park and restore round trip costs 32 ms against 36 ms for never evicting. It works,
and it is essentially free.

## Why the 2048 case fails

Same path, same exact match, different capacity:

```
load:    - prompt with length    1339, lcp =    1339, f_keep = 1.000, f_sim = 1.000
load:  - found better prompt with f_keep = 1.000, f_sim = 1.000
load: failed to restore state with size 36687860
```

`36687860` bytes is 34.99 MiB, exactly the size that was saved for 1339 tokens. The lookup
is perfect and the restore call fails.

The arithmetic: at `-c 2048`, unified KV gives all slots one shared arena of 2048 cells. When
C is restored, the other slot still holds P2 at 1274 cells, so only 774 are free and the
restore needs 1339. `prompt_save` writes C's outgoing state to the cache but does not release
its cells, and the restore happens before the idle slot pass runs.

So the constraint is a total over the shared arena: the sum of all resident sessions plus
the arriving one must fit. It is not about the destination slot. That distinction was tested,
see below, and it is the reason the obvious fix does not work.

## The obvious fix, tested and rejected

The natural reading of the capacity arithmetic is that the destination slot should be
released before the restore, since the outgoing state is already in the cache by then. That
was implemented: `ret->prompt_clear()` between `prompt_save` and `prompt_load` in
`tools/server/server-context.cpp:1647`. Re-run, the 2048 case failed identically:

| | restore failures | C prompt eval |
| --- | --- | --- |
| before, no clear | 1 | 3091 ms |
| after, clear before load | 1 | 3055 ms |

No effect. That rules out the destination slot as the blocker and points at the other
resident session, which owns 1274 of the shared 2048 cells and is only released by the idle
slot pass, which runs after the restore attempt. The change was reverted; it was not a fix.

So on a shared arena the real limit is: the sum of all resident sessions plus the arriving
one must fit. Two long sessions cannot be resident at once at all, and no amount of clearing
the destination changes that. Freeing the other session before the restore is a scheduling
change, not a one-liner, and it is not attempted here.

## Two explanations, and how they were told apart

After the 2048 result there were two readings: the restore path is broken, or the restore
does not fit. They predict the same thing at 2048 and opposite things at 8192. Raising the
context to 8192 raised the free space and the restore succeeded, so the second reading
holds. The first version of this document recorded the first reading, from the 2048 data
alone.

Two things make the failure easy to misread:

1. **Pinning the slot skips the restore entirely.** With `id_slot` set, the slot is chosen
   by id and the code path that calls `prompt_load` never runs. The logs show
   `selected slot by id (0)` and `restore attempts = 0`. Only automatic slot selection
   exercises the restore, and only then does the capacity limit show.
2. **`tokens_cached` is not a cache hit.** In every case, including the ones that
   re-prefilled everything, the response reported `tokens_cached = 1339`. It reports the
   length of the matched prefix, not work avoided. Wall time and `prompt_n` are the only
   reliable signals, and any future work should be validated with those.

## State size

Measured from the save path:

| session length | serialized state |
| --- | --- |
| 1274 tokens | 34.225 MiB |
| 1339 tokens | 34.988 MiB |

Those two points separate into a fixed term and a slope: **19.3 MiB per session plus 12.0 KB
per token**. The fixed term is the gated delta net recurrent state, which is a per session
running summary and not per token data. An earlier version of this document reported the two
points as "27.4 KB per token", which conflated the fixed term with the slope. The separate
law and the 35B equivalent are in `docs/research/17-device-to-disk-path-results.md`.

The practical difference matters at short contexts, where the fixed term dominates: a 200
token session parks at about 22 MiB, not at 2.4 MiB.

## What this means

- The goal is closer than it looked. Session parking exists, is keyed by token prefix, is
  LRU bounded in MiB, and costs about 4 ms over not evicting at all. It stores to host RAM
  though, which the residency policy now forbids for KV, so it is a reference implementation
  and a source of numbers, not a solution.
- The gap is capacity and eviction, not the mechanism. Three concrete gaps:
  1. a restore must fit in the shared arena alongside every other resident session, so two
     long sessions cannot both be resident, and the destination slot is not the constraint;
  2. there is no disk tier, so with `--cache-ram 0` there is no persistence at all;
  3. parking and restoring are per session whole-state operations, not page granular, so a
     restore needs the whole session footprint free at once.
- Cheapest next step is no longer the `update_cache` clear. That was tested and rejected
  above. The remaining options are to free the other resident session before a restore, or to
  accept the shared-arena limit and make the disk tier the answer, which is the path the spec
  now takes.
- Layers 2 and 3 are needed for the parts capacity cannot fix: many sessions at once,
  surviving a restart, and never holding KV in host RAM.

## What this does not cover

- Freeing the other resident session before a restore, which is the fix the rejected test
  points to. Not attempted.
- Whether the constraint is total free cells or contiguous free cells. The 2048 numbers are
  consistent with both, since 774 free is less than 1339 either way.
- Non-CPU backends, restarts, and multi-process behaviour. The Vulkan figures for the same
  path are in `docs/research/17-device-to-disk-path-results.md`.

## Reproduce

```sh
python3 scripts/research/kvstore/e0_park_unpark.py
```

About 80 seconds. Logs in `/home/herlanggays/.jcode/scratch/kvstore/`.
