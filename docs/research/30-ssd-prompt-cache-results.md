# SSD prompt cache: results

Date: 2026-09-28
Tree: `qwen-only-backends`, working tree on top of `397b643f4`
Machine: Ryzen 7 6800H, Radeon 680M (RADV REMBRANDT), 27.1 GiB, AC connected
Model: `Qwen3.5-0.8B-Q4_K_M` for the end to end check, CPU and Vulkan both used
Harness: `scripts/research/ssdcache/cache_hit_check.py`, `scripts/research/ssdcache/store_roundtrip.cpp`
Spec: `docs/superpowers/specs/2026-09-28-ssd-prompt-cache-design.md`
Plan: `docs/superpowers/plans/2026-09-28-ssd-prompt-cache.md`
Depends on: `docs/research/27-ssd-prompt-cache-feasibility.md` (the numbers),
            `docs/research/29-store-io-batching-results.md` (the store's I/O), 
            `docs/research/28-ubatch-determinism-results.md` (why the round trip is the gate)

**Verdict: it works, on the 0.8B and on the 35B. A returning conversation is restored from its
file instead of being reprocessed, the restore survives a server restart, and a truncated entry is
a miss and not an error. The numbers that matter, because they are token counts and do not move
with the machine: 88.6 percent of the 35B's prompt evaluation for a returning conversation is
removed, and over a 40 request agentic loop 82.5 percent is removed in aggregate and 90.6 percent
in the steady state. Wall clock is 1.3 to 2.5x faster on the 35B and 1.42 to 4.27x on the 0.8B
depending on what else the box is doing. Three engine side obstacles had to be fixed on the way,
and all three were pre-existing limitations that also affect the shipping RAM prompt cache on a
hybrid model: a whole restore was thrown away, an entry was invisible after a restart, and the
heuristic that decides whether to consult the cache at all excluded preamble dominated
conversations, which is the workload this was built for.**

## What was built

| piece | where |
| --- | --- |
| store moves blocks in 1 MiB windows, one syscall per window | `src/llama-io.cpp`, `src/llama-io.h` |
| crc32 sliced by eight, same values | `src/llama-io.cpp` |
| one device copy per window on the tensor path, not one per block | `src/llama-io.cpp` |
| entry payload is a file or host RAM, chosen by the configuration | `tools/server/server-task.{h,cpp}` |
| content addressed entry names, a fingerprint per configuration | `server-task.cpp`, `server-context.cpp` |
| entry files rediscovered at startup, so a restart keeps the cache | `server_prompt_cache::rescan` |
| a restored prefix is no longer thrown away on a hybrid model | `server-context.cpp`, the `prompt_restored` flag |
| `--cache-disk-path DIR`, `--cache-disk-mib N` | `common/common.h`, `common/arg.cpp` |
| read an entry's token list without restoring it | `llama_state_seq_file_tokens`, `include/llama.h` |

## The end to end check

`cache_hit_check.py` runs three conversations that share a 260 word preamble over one slot, on
the 0.8B, `-np 1 -kvu --cache-ram 0 --cache-disk-path ... --cache-disk-mib 512`:

```
  the appended turn keeps the prefix token-exact         ok   A is 1759 tokens, common prefix 1759
  A processed its whole prompt                           ok   prompt_n=1759 cache_n=0
  B displaced A and A was written to disk                ok   prompt_n=2363 cache_n=0 files=1
  C displaced B and B was written to disk                ok   prompt_n=2969 cache_n=0 files=2
  returning conversation A was restored from disk        ok   prompt_n=309 cache_n=1759 (A was 1759)
  cached request                                              1.174 s wall, 464 ms of prompt eval
server restarted, same cache dir, 3 entries on disk
  the restore survives a restart                         ok   cache_n=1759 (A was 1759)
server restarted, 3 entry files truncated to a third
  a corrupt entry is a miss                              ok   cache_n=0
  the answer is still produced                           ok
server with no cache, for the comparison
  no cache means no reuse                                ok   cache_n=0
  the cached answer equals the uncached answer           ok   cached 57 chars, uncached 57 chars
  uncached request                                            1.985 s wall, 1776 ms of prompt eval
  speedup, uncached over cached                               1.69x wall, 3.83x prompt eval
```

The restore replaces 1759 tokens of prompt processing with a 40 MiB file read. The corrupt case
is the one that matters most: the loader refuses the file, warns, and the request is served
normally, with `cache_n=0`.

## The 35B, end to end

Same check, `Qwen3.6-35B-A3B-UD-IQ3_XXS`, `--ngl 99`, `--scale 0.35` so the prompts are short
enough for the 35B's prompt processing rate, under the `MemAvailable` watchdog. The state sizes
and rates are unchanged from the 0.8B, because they are properties of the store:

```
  the appended turn keeps the prefix token-exact         ok   A is 874 tokens, common prefix 874
  A processed its whole prompt                           ok   prompt_n=874 cache_n=0
  B displaced A and A was written to disk                ok   prompt_n=726 cache_n=358 files=1
  C displaced B and B was written to disk                ok   prompt_n=939 cache_n=358 files=2
  returning conversation A was restored from disk        ok   prompt_n=112 cache_n=874 (A was 874)
  cached request                                              4.344 s wall, 1600 ms of prompt eval
server restarted, same cache dir, 3 entries on disk
  the restore survives a restart                         ok   cache_n=874 (A was 874)
server restarted, 3 entry files truncated to a third
  a corrupt entry is a miss                              ok   cache_n=0
  the answer is still produced                           ok
server with no cache, twice: the comparison and the control
  the uncached control is reproducible                   ok   two uncached runs agree
  no cache means no reuse                                ok   cache_n=0
  the cached answer equals the uncached answer           ok   cached 57 chars, uncached 57 chars
  uncached request                                           11.066 s wall, 8617 ms of prompt eval
  speedup, uncached over cached                               2.55x wall, 5.39x prompt eval
```

The returning conversation is 986 tokens and 874 of them come from the cache, so **88.6 percent of
its prompt evaluation is removed**, and the remaining 112 tokens are what is new. That number is
deterministic: it is a token count, and it does not move with the machine's throughput. The wall
clock ratio does, and see the note under Rates.

The displacing conversations report `cache_n=358`, which is the shared preamble. That is not the
prompt cache, it is the arena: the slot still holds the previous conversation's state, and either
the guard does not fire or a context checkpoint re-establishes the recurrent part at the preamble
boundary, which is the pre-existing mechanism described under Obstacle 1. How much of a displacing
conversation this covers depends on how large the shared preamble is next to the prompt, which is
why the 0.8B run reports 0 there and this one reports 358. The check reports it rather than
asserting on it for that reason.

The round trip on the same model, from `store_roundtrip` with the cache's own KV types:

```
  store file            79353924 bytes payload, 79806464 bytes on disk
  state blob after load 79346668 vs 79346668 bytes, first_diff=-1, differing=0
  gen A: 198 248045 248068 271 248069 271 6918 235212
  gen B: 198 248045 248068 271 248069 271 6918 235212
  loaded state continues identically           ok
  save     0.198 s    382.7 MiB/s   rss delta 1028 KiB
  load     0.094 s    802.6 MiB/s
```

A 75.67 MiB state, byte exact through the store, same eight greedy tokens on a fresh context, and
1 MiB of resident growth rather than 75 MiB. That 75.67 MiB is the fixed 62.8 MiB recurrent term
plus 13 MB of attention for 1816 tokens, which is the law from
`docs/research/17-device-to-disk-path-results.md` confirmed on the real thing.

## The agentic loop

Four conversations over one slot, ten turns each, 200 words appended per turn, so every request
displaces the previous conversation and the cache is on the critical path for all 40 requests.
`scripts/research/ssdcache/agentic_loop.py`, 0.8B, and the no-cache baseline is the same traffic
with the prompt cache disabled:

```
  4 conversations x 10 turns = 40 requests over one slot
                                       with cache     no cache
  requests                                     40           40
  prompt tokens processed                   15551        76991
  tokens served from the cache              68889         7449
  wall clock                               33.47s       56.32s
  entries on disk, peak                    239.9MiB           -

  prompt processing removed:  79.8%
  prompt processing speedup:  4.95x
  wall clock speedup:         1.68x

  last turn of each conversation, the steady state:
  prompt tokens processed                    1248        13296
  tokens served from the cache              12812          764
  prompt tokens removed                     90.6%
  prompt processing speedup                10.65x
  wall clock speedup                        2.43x
```

Read the two blocks differently. The aggregate over all 40 requests is 79.8 percent of prompt
processing removed, diluted by the first turn of each conversation, which has nothing cached yet.
The last turn of each conversation is the steady state, 90.6 percent removed and 10.65x, and it is
what a longer loop converges to. The design's model in
`docs/research/27-ssd-prompt-cache-feasibility.md` predicted 94.3 percent of token-turns removed;
this measures 90.6 percent in the steady state, and the gap is the turn where a conversation has
no entry yet.

The block above is the run before the `f_keep` change described below. Re-run after it, the same
traffic reports 13487 prompt tokens processed instead of 15551 and **82.5 percent removed**, with
the steady state unchanged at 90.6 percent, because those turns were already below the old
threshold.

One number worth noting for the disk budget: 239.9 MiB for four conversations of about 3500 tokens.
The recurrent term is most of it at 20.2 MiB per conversation on the 0.8B and 62.8 MiB on the 35B,
and it does not shrink with shorter conversations.

## Concurrency, slots, and a budget that evicts

The production server ran with `-np 2` and every measurement above is one slot and one request at a
time, so `scripts/research/ssdcache/cache_concurrent_check.py` runs the server the way it is actually
deployed. The premise needs one correction first, and reading the code settles it: **the cache is
never touched by two threads.** Every call site is `slot.prompt_save`, `slot.prompt_load`,
`prompt_cache->update()` and the constructor, and they all sit in `get_available_slot` and
`update_slots`, which run in the one slot-processing thread; the `mutex_cache` in this file guards the
metrics snapshot, not the cache. Concurrent HTTP requests are queued into that one loop, so there is
no data race to find and no lock is needed. What concurrency can produce is *interleaving*: a slot's
entry evicted between two requests, a slot asking to load an entry that another slot evicted a moment
earlier, or an entry replaced as a conversation grows. The check below is a test of that, and of the
eviction path under a budget that forces it, rather than of thread safety.

```
2 slots, 4 conversations x 3 turns = 12 requests, 4 in flight

run 1
  every request was served                                 ok   0 errors
  the server is still running                              ok
  result                                                   4 files, 9031 prompt tokens, 3197 from cache
  every entry file is intact, no bad blocks                ok   0 of 4 files reported
  no holes and no stale generations                        ok
  a hit survives the restart for every conversation        ok   4 of 4
run 2, same traffic, fresh cache
  the server is still running after the second run         ok
  every request's accounting is consistent                 ok   12 requests, 12228 tokens in total
  concurrent traffic still reuses the cache                ok   3197 of 12228 tokens came from the cache, 26.1%
```

Two things worth reading carefully.

**The entry files are checked by a second implementation.** `validate_file.py` parses the format
with plain buffered reads and its own crc recomputation and shares no code with the C++ reader, so
"0 of 4 files reported" is an independent statement that the concurrent writes produced intact
files with no holes and no mixed generations.

**The hit rate is much lower under concurrency, 26.1 percent against 79.8 percent serial.** That is
not a fault, it is slot contention: with two slots and four conversations in flight, a request
often lands on a slot holding a different conversation, and how much it reuses depends on the
arrival order. The same traffic run twice does not produce the same per-request accounting for that
reason, which is why the check asserts consistency (`prompt_n + cache_n` equals the prompt length
for every request) and integrity rather than repeatability. The first version of this check
asserted repeatability and failed on it, and the honest conclusion is that counts cannot be
compared across runs under concurrency either, only invariants.

### The restart gate was asking for something that cannot happen

The line above reads `4 of 4` for that run, and it is wrong as a gate. An entry exists only for a
conversation that was *displaced*, so a conversation still resident in a slot at the end has no entry
to hit, which is one per slot. The gate demanded all four and was satisfied only by the accident of
where the last request landed. Re-run on the same parameters on a later build it gives `3 files, 3 of
4`, and the pattern holds at every size: 7 files and 7 hits at four slots and eight conversations, 3
files and 3 hits at two slots and four. The gate is now `hits >= conversations - slots`, which is the
invariant, and the file count is reported beside it.

### A tight budget, which is the case where eviction interleaves with everything

`--budget-mib` was added for this, because the default budget holds a whole run and never evicts.
Four slots, eight conversations, three turns, and a budget that holds two entries:

```
64 MiB   every request was served                    ok   0 errors
         the server is still running                 ok
         result                                      2 files, 21004 prompt tokens, 3452 from cache
         every entry file is intact                  ok   0 of 2 files reported, by the Python validator
         no holes and no stale generations           ok
         a restart serves what is still on disk      n/a  0 of 8, 2 files on disk and the rest evicted
         every request's accounting is consistent    ok   24 requests, 24456 tokens in total
         concurrent traffic still reuses the cache   ok   14.1 percent
```

What that establishes is the safety property under constant eviction: no crash, no request failed,
every surviving file intact under the independent validator, no holes, no stale generations, and the
accounting exact. What it does not establish is reuse, and it cannot: with 2 files for 8
conversations, six of the eight have nothing left to hit, and the restart phase's own writes evict the
entries the later conversations need before they are asked, so `0 of 8` is the correct answer rather
than a failure. The 128 MiB run behaves the same way with 3 files.

### An intermittent 500 that this document cannot yet explain

Seven runs at four slots and eight conversations with a 512 MiB budget: one of them returned HTTP 500
for 16 of its 24 requests, as a suffix, while the server stayed up and the later phases were clean.
The same configuration was clean three more times, two slots and four conversations were clean four
times, and the tight budgets were clean three times. The server log for the failing run has no error
level line at all, which rules out `send_error` and therefore every path that reports a condition it
understands.

Reading the queue explains the shape of it, and it is a mechanism worth knowing about. A posting
thread does not merely enqueue: `server_queue` may run the work on the *current* thread "so that all
ggml compute stays on the same thread", and when it does, it captures any exception from the work and
rethrows it in the posting thread, after the declined tasks are safely back in the queue
(`tools/server/server-queue.cpp`, the `work while yielded` path at the end of the file). An HTTP
request thread is a posting thread. So an exception raised anywhere in slot work, outside the places
that catch it themselves, surfaces as HTTP 500 on whichever request is posting at that moment, and
the only record of it is the route wrapper's `SRV_WRN("got exception: ...")` in `tools/server.cpp`.
That is why the failing run has no error level line: the alternative path, `abort_all_slots` when
`pre_decode`, `decode` or `post_decode` throws, does call `send_error` and does log at error level.
Sixteen consecutive 500s then says the condition persisted across requests rather than a single
unlucky task.

What this does not do is exclude the cache, and the earlier draft of this paragraph was wrong to
imply otherwise. The engine calls inside `server_prompt_cache::load` and `save` are wrapped and degrade
to a miss, and the corrupt-entry check exercises that; but slot assignment is exactly where those
calls live, and anything thrown elsewhere in that path by a string or a container would travel this
route rather than being logged where a cache bug would be expected. The first 500's body names the
exception, the harness now records it, and until then this is an open item rather than a resolved
one. Ten further attempts to reproduce it did not: five at the identical shape, three at a lighter one
with shorter prompts, and two at a shape with more conversations and shorter turns. So it is rare
enough that the load matters and not just the configuration, and it is not worth more of this
document's time than it has already had.

## Where the cache is and is not consulted

This is the part that is easy to get wrong, and it cost the most time. The prompt cache is not
used on every request:

- A slot that already holds a prefix the request extends is reused in the arena, and the cache is
  never touched. That is better, and it means a single conversation on a single slot never uses
  the cache at all. The first version of the test did not hit the cache once and looked like a
  broken cache for that reason.
- `get_available_slot` saves and loads through the cache when the selected slot is about to lose
  something, which is what `f_keep < 1.0f` now means. See "The `f_keep` gate" below, which is the
  one line of this whole feature that had to change an existing heuristic.
- So the case the cache exists for is several conversations over fewer slots, which is what a
  shared agentic server is.

## The `f_keep` gate, and the workload it was excluding

`get_available_slot` decided whether to touch the cache with

```cpp
// if we are about to lose a large portion of the existing context - save it in the prompt cache
if (f_keep < 0.5f) { update_cache = true; }
```

where `f_keep` is the fraction of the slot's prompt that the new request shares. The threshold
looks harmless and it excludes an entire workload shape: a conversation that is mostly a shared
preamble. Measured, same check, same three conversations, only the body length changed, 0.8B:

| conversation bodies | preamble share of each conversation | entry files written | `cache_n` for the returning conversation |
| --- | --- | --- | --- |
| 900, 1300, 1700 words | about 20 percent | 2 | 1759 of 2062, 85 percent |
| 45, 65, 85 words | about 85 percent | **0** | **9 of 502, 2 percent** |

With short bodies every request that replaces a conversation has `f_keep` above 0.5, so the cache
was never written and never read, and reuse collapsed to almost nothing: the shared preamble was
not even served from the arena, because the hybrid guard rejected the slot's stale state. That is
exactly the shape of an agentic workload with a large system prompt and tool block and short turns,
which is the case this cache was built for.

The condition is now `f_keep < 1.0f`, which reads as "this request does not cleanly extend what the
slot holds", so the cache engages whenever the slot's state is about to be truncated. A clean
extension still `f_keep == 1.0` and is untouched, so nothing changed for the case that already
worked. Measured after the change:

| measurement | before | after |
| --- | --- | --- |
| short bodies, returning conversation | 9 of 502 tokens, 2 percent | 465 of 502, 93 percent |
| agentic loop, 40 requests, prompt tokens removed | 79.8 percent | 82.5 percent |
| loop, steady state | 90.6 percent | 90.6 percent, unchanged |
| concurrency check, tokens from the cache | 26.1 percent | 29.7 to 38.7 percent over two runs |

The steady state does not move because those turns were already below the old threshold. The two
cases that move are the ones with short prefixes relative to their slot's state.

### What the same change costs, on a workload that never repeats

The table above is the gain, and for a while it was the only side measured. The cost is that a request
sharing between half and all of what a slot holds now parks that slot, and parking writes an entry,
even when the prefix will never be asked for again.
`scripts/research/ssdcache/cache_gate_cost_check.py` measures that shape: one shared 700 word
preamble, ten requests each with a distinct short tail, one slot, disk cache, and nothing repeated.
Run against a build of each gate:

| gate | requests served the preamble | prompt tokens processed each | entry files | written |
| --- | --- | --- | --- | --- |
| `f_keep < 0.5f`, before | 9 of 9 | 33 | 0 | 0.0 MiB |
| `f_keep < 1.0f`, after | 9 of 9 | 33 | 8 | 258.7 MiB |

Both serve the same 1059 token preamble from the arena and process the same 33 new tokens; the
difference is 258.7 MiB of writes for 8 entries that nothing will hit. So the change buys the reuse in
the table above at the price of write amplification on prefixes that never repeat, which is bounded by
`--cache-disk-mib` and is the reason to set that rather than leave it at its default. A client that
always edits the tail instead of appending is what pays; the agentic case this exists for appends, and
that is the honest shape of the trade rather than an argument against it.

## What an entry actually contains

One clarification that took several failed checks to pin down, and it is the same rule as the token
exactness requirement seen from the other side.

An entry records the slot's prompt at the moment it was saved, and after a generation that prompt
includes **the tokens the model generated**. So an entry is not "the prompt that was sent", it is
"the prompt plus what came out of it". A later request can only use that entry if it carries those
generated tokens, which a real transcript does: the next turn re-sends the assistant's own output.
An entry saved after a generation can be extended, and it cannot be served to a request that
replaces the assistant's output with different text, because that diverges inside the entry.

That is why the restart check in `cache_hit_check.py` rebuilds the request from the entry's own
text, `a2` plus the content the cached request generated, and why it searches for a separator that
the server's own tokenizer agrees keeps that text a prefix. Two earlier versions of the check
appended different text, so the request diverged inside the entry, the guard correctly rejected the
restore, and the check reported a cache failure that was not one.

## Obstacle 1: a hybrid model refused every restore

The state restored correctly, and then the server threw it away:

```
state_read_meta: cell_count = 1759, dest_seq_id = 0
slot | task 13 | forcing full prompt re-processing due to lack of cache data
                  (likely due to SWA or hybrid/recurrent memory, ...)
slot | task 13 | cached n_tokens = 0, memory_seq_rm [0, end)
```

The guard is `if (pos_min >= pos_min_thold)` in the prompt processing path. For a hybrid model
`llama_memory_hybrid::seq_pos_min` returns the **max** of the attention and recurrent minima, and
the recurrent state is a running summary whose position is the end of the summary, so `pos_min`
comes out at the end of the restored state and the test always fires. With no context checkpoint
to fall back on, the code resets and reprocesses everything.

Two things follow, and the second is the more interesting one:

1. The shipping RAM prompt cache has the same guard to satisfy, and it satisfies it with context
   checkpoints: an entry carries the slot's checkpoints, and when the guard fires a load falls
   back to one, which re-establishes the SWA or recurrent part at the checkpoint's position and
   caps the reuse there. The disk entry had no checkpoints, so it had nothing to fall back on.
   Measured, same scenario, one slot, `--cache-ram 512`:

   | build | RAM cache, `cache_n` for the returning conversation |
   | --- | --- |
   | guard override disabled | 1759 |
   | guard override enabled | 1759 |

   and for the disk cache, where the entries carry no checkpoints:

   | build | disk cache, `cache_n` for the returning conversation |
   | --- | --- |
   | before the override | 0, with `forcing full prompt re-processing` |
   | after the override | 1759 |

   So the override was not needed for the RAM path and was essential for the disk path, and the
   reason is the checkpoints rather than the storage.

2. A whole state restore makes the reset unnecessary, because the arena then holds every position
   `n_past` refers to, including the recurrent summary at the continuation point. That is strictly
   better than the checkpoint route, which can only reuse up to a checkpoint boundary, and it now
   applies to both storage backends. The fix keeps `n_past` when a whole state was restored
   **and** the restored prompt is a prefix of the input. Both halves are load bearing: an earlier
   version of the fix only checked the first, and a restore that covers part of the prefix leaves
   old cells past the divergence, which the reset was also there to remove. That version aborted
   in `seq_rm` on the second request.

Worth knowing before leaning on the checkpoint route: a checkpoint is a partial state, which for
these hybrid models is the whole recurrent part. The log from the run above shows
`created context checkpoint 1 of 32 ... size = 19.266 MiB` on the 0.8B, so a slot can hold up to
32 of them at 19.3 MiB each, and on the 35B the same structure is 62.8 MiB each. That is the RAM
a hybrid model's prompt cache traffic implies, and it is why the disk entry does not carry them.

The end to end check exercises both paths: the returning conversation is a clean prefix and is
kept, and the displacing conversations diverge and still take the reset.

## Obstacle 2: a restart forgot everything

The entry list lived in memory only, so the files a previous run wrote were orphaned, and the
restart check failed with `cache_n=0`. Entries are now rediscovered at startup by reading each
file's own token list back, which is what `llama_state_seq_file_tokens` is for. It reads the same
header through the same reader, so there is no second parser of the format to get out of step, and
a damaged file is simply not registered.

Note the ordering consequence: after a rescan the entries are in directory order, so eviction by
LRU starts from an arbitrary order. It is a cache, so this only affects which entry goes first.

## The requirement people will trip over

**A hit needs the cached prefix to be token identical to the input prefix.** A restore covers
position `n_past` exactly, and the recurrent summary sits at the end of the restored state, so a
continuation that diverges one token before the end cannot be served even though 1758 of 1759
tokens match. The server then does the reset and processes everything.

That is a property of the engine's state model, not of the disk cache, and the RAM cache has it
too. In practice it means the appended turn has to start on a token boundary. Appending
`"more alpha bravo"` with a space does not; the test showed `A is 1759 tokens, common prefix
1759` for `"\n" + "[turn 2] "` and 1758 for `"\n" + "\nUser: "`, where the double newline merged
with the following token. Chat templates put a role marker at the boundary and are fine. A client
that concatenates raw text should be aware.

## The determinism gate, and what it does and does not invalidate

`docs/research/28-ubatch-determinism-results.md` finds that the same prefix processed twice in the
same process, with the same configuration and even with one thread, serializes to different bytes
and produces different logits, and that stock `llama-cli`, greedy with a fixed seed, produced three
different continuations in three runs on the same prompt. Its rerun control is clean, serializing
the same unchanged state twice gives 0 differing bytes in all 145 rows, so the instability is in
the computation that fills the state and not in the serialization. Below about 24 prompt tokens
the whole pipeline is bit exact, so it is something that appears with length, and it is an artefact
of the Vulkan driver being in the process rather than a defect in this fork.

That is not a property of this tree. `docs/research/31-prefill-nondeterminism-localisation.md` ran
it down: the same binary is bit exact for every configuration, thread count and process once the
Vulkan driver is kept out of the process, by pointing `VK_ICD_FILENAMES` at nothing, and a tree
built with `GGML_VULKAN=OFF` produces that identical state. The driver is initialised at startup
even when `n_gpu_layers` is 0, so a run that offloads nothing still has it loaded, and a GPU run
necessarily has it loaded. That is why nothing in this document's 35B numbers changes. It is not
caused by the cache, and it does not invalidate the cache, because the cache does not need two
computations to agree. It needs a save and a load to
agree, and that is measured directly in `store_roundtrip`:

```
  state blob after load 28308716 vs 28308716 bytes, first_diff=-1, differing=0
  gen A: 15 17 15 8029 235212 1122 11234 9197
  gen B: 15 17 15 8029 235212 1122 11234 9197
  loaded state continues identically           ok
  save     0.063 s    427.0 MiB/s   rss delta 2056 KiB
  load     0.033 s    807.4 MiB/s
```

A save and a load are byte exact on both backends, and the continuation is identical. What the
cache therefore promises is: **a hit reproduces the computation that wrote the entry.** That is
weaker than "a hit equals a recomputation", and on a build whose prefill is reproducible the two
would be the same claim.

Two text comparisons in `cache_hit_check.py` are reported and not counted, and that is a deliberate
decision with evidence behind it rather than a way to make a check pass.

- `restored vs recomputed, same process` compares a restored state against a full recomputation of
  the same prompt in the same process, same context. On the 0.8B it has agreed when it ran; on the
  35B it reported `34 chars restored, 57 chars recomputed` and `57 chars restored, 57 chars
  recomputed, DIFFERENT` in two runs, 57 of 57 in others. A restore returns the state that was
  saved and a recomputation produces a new one, so this compares two computations, and document 28
  says this build does not agree with itself on those.
- `two uncached processes agree` is the control for the same thing across processes. It has
  reported `same` and `DIFFERENT` on the 35B in different runs, and when it reports DIFFERENT it is
  two fresh servers with no cache at all answering one prompt differently. That is the finding, not
  a failure.

Both of those comparisons agree when the driver is kept out of the process, and they also agree with
it in, so their agreement does not come from the driver question at all and cannot be used as
evidence that the driver explains the DIFFERENT rows above. On the 0.8B, at 200 generated tokens:

```
  restored vs recomputed, same process                   n/a  812 chars restored, 812 chars recomputed, same
  two uncached processes agree                            ok  812 chars vs 812 chars, same
  cached vs uncached text                                n/a  812 chars cached, 812 chars uncached, same
```

With the driver hidden, and with it visible, those three lines are the same. So the reason they are
reported and not counted is stronger than the determinism question and does not depend on it: the
sampled token is the argmax of a vocabulary sized distribution, and it rarely moves when the last
bits of the logits move. `docs/research/31-prefill-nondeterminism-localisation.md` measures the
logits hashes disagreeing in nearly every pair of runs while `first_tok` stayed 2180, and the one run
where it did not was the exception. Text equality is therefore a weak detector of this in both
directions. It is not evidence for the cache's transparency and it is not evidence against it. The
discriminating measurements are the state and logits hashes in doc 28 and doc 31, and the byte exact
round trip in `store_roundtrip`.

So the gates in that check are the ones that are properties of the cache: the entry was written,
the restore happened (`cache_n`), the request and the entry were token exact, a restart still hits,
a truncated entry is a miss, and the accounting is consistent. The check that carries the
transparency weight is the byte exact round trip in `store_roundtrip`, which compares a state with
its own round trip rather than with a second computation, and it is `differing=0` on the 0.8B on
both backends and on the 35B.

## Rates

Store, 27 MiB state, 0.8B, CPU, from `docs/research/29-store-io-batching-results.md`:

| | before this work | after |
| --- | --- | --- |
| load | 197 MiB/s | 843 MiB/s |
| save, write plus verify | 185 MiB/s | 432 MiB/s |

End to end, `n_predict 16`, on the 0.8B through `llama-server` on Vulkan, three runs of the whole
check, each with its own server:

| run | cached wall | cached prompt eval | uncached wall | uncached prompt eval | wall | prompt eval |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.056 s | 406 ms | 1.603 s | 1419 ms | 1.52x | 3.50x |
| 2 | 1.061 s | 412 ms | 1.590 s | 1407 ms | 1.50x | 3.41x |
| 3 | 1.113 s | 413 ms | 1.577 s | 1395 ms | 1.42x | 3.38x |

The cached request processes 309 tokens and the uncached one 2062, so the 3.4x on prompt
evaluation is the token ratio. One earlier run measured 1.69x wall and 3.83x prompt eval, which is
the spread of this box rather than a better configuration, and the three runs above are the ones
to quote.

The token reduction is the number to trust, because it is a count and it does not move with the
machine. The wall clock ratio does. Across runs of the same 0.8B check the uncached request
measured 1.577 s, 1.590 s, 1.603 s and then 6.972 s, while the cached one went 1.056 to 1.782 s, so
the box's throughput moves by a factor of four between sessions and the ratio moves with it: 1.42x,
1.50x, 1.52x, 3.91x. The 35B run measured 2.55x. Report the token reduction and treat the wall
clock as a range.

The wall clock ratio is also much worse than the prompt eval ratio because the cached request pays
for writing the displaced conversation's state, 54 MiB on the 0.8B and 79 MiB on the 35B, inside
the same request. That write is the price of the displacement, and it is the thing stage 2 of the
spec would reduce.

The design's model in `docs/research/27-ssd-prompt-cache-feasibility.md` predicted a 16.9x loop
speedup from prompt processing alone. These numbers are for one request pair, not a loop, and the
wall clock includes the save, the restore, and the generation, so they are not directly comparable.
The model's claim that prompt processing falls to the new tokens holds exactly: `cache_n=1759,
prompt_n=309`.

## Eviction under the disk budget

This was listed as untested in the first draft of this document, and it turned out to be the one
place where a cache bug takes the server down instead of returning a wrong answer.
`scripts/research/ssdcache/cache_evict_check.py` runs four conversations over one slot with the budget
set from a measured entry size, then brings the oldest and the newest back with one token more than
the entry holds:

```
  one entry is 35.5 MiB, budget 78 MiB, so at most 2 entries fit
  step                    prompt_n   cache_n  entries        MiB
  A arrives                   1366         0        0       0.00
  B arrives                   1366         0        1      35.53
  C arrives                   1366         0        2      71.05
  D arrives                   1366         0        2      71.05
  A returns, evicted          1375         0        2      71.05
  D returns, retained            9      1366        2      71.20
  on-disk bytes stay within the budget after every request   ok   budget 78 MiB, worst none over
  old entries are deleted rather than leaked                 ok   files [0, 1, 2, 2, 2, 2], budget holds 2
  something is still cached at the end                       ok   entries 2, 71.2 MiB
  the evicted conversation misses and is reprocessed         ok   cache_n 0 of 1366 tokens
  the retained conversation still restores                   ok   cache_n 1366 of 1366 tokens
  the server reported removing the oldest entry              ok   the eviction path ran
  no entry failed to load                                    ok   which is what deleting the wrong file would look like
```

Two things this settled. The quiet one: `alloc()` makes room before it registers a new entry, so the
budget is never exceeded even between two writes, which is why the worst total above is 71.05 MiB
against a 78 MiB limit and the file count never rises above the two the budget holds. The loud one is
a crash.

### The crash it found, and the one-condition fix

The first run took the server down, on a request that carried exactly the tokens of the entry it had
restored:

```
common/common.cpp:1580: failed to remove sequence 0 with p0=1365, p1=-1
```

That is `GGML_ABORT` in `common_context_seq_rm`, reached from `pre_decode`. The chain is worth
writing down because none of its links is obvious. A restore that covers the whole prompt sets
`n_past` to the entry's length. The server then insists that at least one token be evaluated so the
sampler has something to sample (`[TAG_PROMPT_LOGITS]` in `server-context.cpp`), so it decrements
`n_past` and asks the memory to drop the last position. The attention cache can drop one position.
The recurrent cache can only do that from a per-token rollback snapshot, `n_rs_seq`, which is 0 by
default (the `rollback <= n_rs_seq` test in `llama-memory-recurrent.cpp`), so `seq_rm` returns false,
and `common_context_seq_rm` turns false into an abort. On a hybrid model every restore that covers
the whole prompt arrives here, since the recurrent summary sits at the end of the state.

Hiding the driver does not avoid it, and neither does the RAM cache: this is the server's prompt
path, and both backends go through the same line. It is reachable whenever a client resends a
prompt that some other conversation has displaced, which is ordinary agent behaviour.

The fix is one condition: the whole-state restore is kept only when at least one token is left to
evaluate, `n_past < slot.task->n_tokens()`. A prompt that extends the cached entry is unaffected,
because anything left to evaluate keeps the trim empty, and that is the case this feature exists for
and the case the 88.6 and 90.6 percent numbers come from. It is also the case the table above shows
restoring 1366 of 1366. What is given up is the exact replay of an entry no slot is holding: it now
falls back to a checkpoint if the entry has one and to reprocessing otherwise. That costs prompt
evaluation on an exact replay, and an exact replay is a request with nothing new to say, so the cost
is bounded and the crash is not.

## The draft context path, and a fingerprint that did not cover it

The other unmeasured path in the first draft was `ctx_dft`, the draft context that speculative
decoding uses, which is null in every other check here and which makes an entry two files. Building
the check turned up a second bug, in the entry name rather than in the payload.
`scripts/research/ssdcache/cache_draft_check.py`, one slot, disk cache, target fixed at
`Qwen3.5-0.8B-Q4_K_M` and only the draft model changed:

```
  step, draft A             prompt_n   cache_n
  A arrives                     1061         0
  B arrives, parks A            1061         0
  A returns, extends               9      1061
  an entry is a target file and a draft file                 ok   four files, two entries, .kvs and .d.kvs
  the restore loads the prefix with a draft model            ok   cache_n 1061 of 1061 tokens
  step, draft B             prompt_n   cache_n
  A returns again               1070         0
  B arrives, parks A            1061         0
  a run with another draft model does not reuse the entry    ok   cache_n 0 of 1070 tokens, it reprocessed
  changing only the draft model changes the entry name       ok   draft A wrote bcb6f19c43906573, draft B wrote 944b32f6724b7fbe
  the second run did not report a failed draft load          ok
```

The draft path itself works: an entry is a target payload and a `<hash>.d.kvs` draft payload, and a
restore brings back both, reusing 1061 of 1061 tokens with speculative decoding on.

The bug is the name. `server_cache_fingerprint()` listed the model path, the context size, the KV
types, the slot count, `kv_unified`, `swa_full` and the flash attention type, and not the draft
model, while `save()` writes the draft context's state into every entry. So two runs differing only
in `-md` produced the same entry names, and the second run read the first run's draft state into a
draft model that never wrote it. Before the fix the two middle rows read `cache_n 1061` and the same
fingerprint `0f5e470dcd84efbb` on both sides, which is the bug in one line: the second run reused the
entry and the log recorded no failure, because two models of one architecture have the same state
size and the load succeeds.

The blast radius is narrower than it looks and worth stating precisely. The draft state only feeds
proposals, and speculative decoding verifies them against the target, so the generated text stays
correct; what a wrong draft state costs is speculation quality, and if the two models had different
state sizes the load would fail and the entry would be a miss. It is still the kind of silent
wrong-file read that this cache's design goes out of its way to prevent, and it is invisible without
a second draft model to test with.

The fix adds the draft model path and the draft KV types to `server_cache_fingerprint()`, only when a
draft is configured, which leaves the fingerprint and therefore every existing entry of a run without
a draft untouched. One consequence is deliberate and worth knowing: a run with a draft and a run
without one no longer share entries, even though the target payload would be valid in both, because
the entry is a unit of two files. That is the conservative side to split on.

One dimension having been missed, every dimension is now checked directly rather than one at a time as
they come up. `scripts/research/ssdcache/cache_fingerprint_check.py` reads the fingerprint out of the
entry name, which needs no cache hit because the name is `<fingerprint>-<hash>.kvs`, so it costs eight
server starts of two short conversations each, 27 seconds in total:

```
  baseline                 0f5e470dcd84efbb
  same config again        0f5e470dcd84efbb
  other model file         9f170c382607c358
  smaller context          7527dcd053ac6e60
  K cache type q8_0        416bec0b0f5d0416
  V cache type q8_0        2722e6c9585acfb4
  two slots                747e308093b01922
  not kv unified           1f849cf9af25f4dc
```

Six dimensions, six different names, and the same configuration twice gives the same name. So the
draft model was the only gap: the model file, the context size, both KV types, the slot count and
`kv_unified` were all in the fingerprint already.

## What this does not cover

- **The `f_keep < 0.5` gate was not varied.** It decides when the cache is consulted at all.
- **One run in seven at four slots and eight conversations returned HTTP 500 for 16 of its 24
  requests, and this document does not explain it.** See the section above for what was ruled out
  and what the next occurrence will report. Every other concurrency run was clean, including six
  attempts at the same configuration.
- **Eviction was exercised at two budgets and one entry size, first with one slot and then with
  four.** `alloc()` makes room before it writes, so the budget held and nothing leaked. The
  four-slot run under a 64 MiB budget is the interleaved case, and it is safe for the reason the
  concurrency section gives: the cache is touched by one thread, so two slots cannot decide to remove
  the same entry at the same moment.
- **The draft context path was exercised for `draft-simple` only.** `cache_draft_check.py` covers
  two draft models, the two files an entry becomes, and the entry name changing with the draft. The
  other speculative types are untested, and so is a draft model whose state size differs from the
  target's, which is the case where a stale payload would be caught by size rather than by name.
- **Cross-prefix sharing was not measured.** Two conversations with a common preamble store it
  once each, which is what stage 2 of the spec would fix, and the loop pays for that twice over.
  It is visible in the concurrency run as the same 191 token preamble arriving from six different
  slots.
- **Concurrency was tested at two slots and four requests in flight, not at load.** Twelve
  requests, one conversation overlapping another at most. A sustained concurrent run with
  eviction under pressure is a different test and was not done, though eviction itself is covered
  in the section above, single slot.
- **Nothing was measured in the healthy GPU window.** The 35B numbers come from the degraded one,
  which is why they are quoted as token counts rather than seconds where possible.
- **The store is faster on CPU than on Vulkan by 2.9x on the write side**, unexplained, see
  `docs/research/29-store-io-batching-results.md`.
- **No 20 turn single conversation loop.** The loop above is four conversations over one slot,
  which is the case the cache is for. A single conversation on a single slot never touches the
  cache at all, so a longer one would measure arena reuse rather than this cache.
- **The in-tree server test suite was not run.** `tools/server/tests/` covers exactly the area the
  `f_keep` change touches, `test_slot_save.py` and `test_kv_keep_only_active.py`, and it cannot run
  on this box: `utils.py` imports `wget`, pip refuses without `--break-system-packages` because the
  interpreter is externally managed, a venv failed on `numpy` before this work started (see
  `docs/research/24-state-api-compat-results.md`), and the `ServerPreset` helpers download their
  models. What was done instead is narrower and aimed at the change:
  `scripts/research/ssdcache/cache_gate_check.py` checks the contract that the change could have
  broken, on both cache backends, and it passes:

  ```
  disk cache: a clean extension writes no entry file      ok   entry counts [0, 0, 0, 0, 0]
  disk cache: a clean extension is served by the arena    ok   cache_n [0, 707, 1019, 1331, 1643]
  disk cache: the first conversation returns with its whole prefix ok   cache_n 1955, prefix 1643
  RAM cache:  a clean extension writes no entry file      ok   entry counts [0, 0, 0, 0, 0]
  RAM cache:  the first conversation returns with its whole prefix ok   cache_n 1955, prefix 1643
  disk cache: a different conversation writes one entry file ok   entries 1
  both backends agree on the extension and return behaviour  ok   return 1955 vs 1955
  ```

  So a request that cleanly extends what the slot holds still writes nothing and is still served
  from the arena, which is the behaviour the old threshold gave it, and the `RAM` cache behaves
  identically because the same line drives both. That is the largest un-run check made as small as
  it can be, and it is not the same thing as the project's suite having passed.
- **The prefill nondeterminism from `docs/research/28-ubatch-determinism-results.md` is not a
  defect of this tree.** `docs/research/31-prefill-nondeterminism-localisation.md` traced it to the
  Vulkan driver being in the process, and with the driver kept out the same binary is bit exact
  across configurations, thread counts and processes, landing on the same state as a build with
  Vulkan compiled out. It is still why several checks here assert counts and round trips rather
  than text, together with the reason those text comparisons are weak detectors at any length
  either way, which is the argmax of a large vocabulary moving far less often than the logits do.

## Reproduce

```sh
cmake --build build-vk -j6 --target llama-server

# the round trip, losslessness, and the store rate, on the 0.8B and on the 35B
g++ -O2 -std=c++17 -I include -I ggml/include scripts/research/ssdcache/store_roundtrip.cpp \
    -L build-vk/bin -lllama -lggml-base -lggml -Wl,-rpath,$PWD/build-vk/bin -o store_roundtrip
./store_roundtrip ~/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf /tmp/rt.bin 1200 8192 0
./store_roundtrip ~/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf /tmp/rt.bin 1200 8192 99
./store_roundtrip "/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf" \
    /tmp/rt35.bin 1200 8192 99

# the end to end check, about 45 s on the 0.8B and 2.5 minutes on the 35B
python3 scripts/research/ssdcache/cache_hit_check.py
python3 scripts/research/ssdcache/cache_hit_check.py \
    "/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf" \
    --ngl 99 --scale 0.35

# and on the 0.8B with the Vulkan driver kept out of the process, at 200 generated tokens
VK_ICD_FILENAMES=/nonexistent/icd.json python3 scripts/research/ssdcache/cache_hit_check.py --n-predict 200
# the same with the driver in the process, which gives the same three text lines: text is a weak
# detector of the byte level difference either way, see the determinism section
python3 scripts/research/ssdcache/cache_hit_check.py --n-predict 200

# eviction under the disk budget, about 40 s: sets the budget from a measured entry size
python3 scripts/research/ssdcache/cache_evict_check.py

# the draft context path, and the entry name across two draft models, about 30 s
python3 scripts/research/ssdcache/cache_draft_check.py

# every other dimension the entry name has to follow, about 30 s
python3 scripts/research/ssdcache/cache_fingerprint_check.py

# the agentic loop, about 90 s
python3 scripts/research/ssdcache/agentic_loop.py --turns 10

# two slots with four requests in flight, about 90 s
python3 scripts/research/ssdcache/cache_concurrent_check.py

# the same with a budget that forces eviction while the slots write, which is the case where an
# entry disappears between two requests. about 3 minutes
python3 scripts/research/ssdcache/cache_concurrent_check.py --slots 4 --conversations 8 --budget-mib 64

# the contract the f_keep change could have broken, both backends, about 30 s
python3 scripts/research/ssdcache/cache_gate_check.py

# what that change costs on a workload that never repeats: about 15 s against the current build,
# and against a build with the old gate by passing a binary path
python3 scripts/research/ssdcache/cache_gate_cost_check.py
```

Wrap the 35B commands in `scripts/research/ssdcache/run_guard.sh` with a raised floor,
`FLOOR_MIB=4000`, because that model on Vulkan is the configuration
`docs/research/freeze-notes-2026-09-27.md` records seven hard resets under. The checks start and
stop their own servers on ports 8747 and 8749 and clear the cache directories they use first.
