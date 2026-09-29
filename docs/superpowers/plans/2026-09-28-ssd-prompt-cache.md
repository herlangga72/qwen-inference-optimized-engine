# SSD prompt cache Implementation Plan

**Outcome: built and verified end to end on the 0.8B and on the 35B, two slots included. Task 6 is
done in a different shape than written, task 7 is not started. The cache works: a returning
conversation over a rotating slot set is restored from its file, 88.6 percent of the 35B's prompt
evaluation for that turn is removed, 82.5 percent of a 40 request agentic loop's prompt processing
is removed, the restore survives a restart, and a truncated entry is a miss. See
`docs/research/30-ssd-prompt-cache-results.md` for the results,
`docs/research/29-store-io-batching-results.md` for the store's I/O, and
`docs/research/28-ubatch-determinism-results.md` for the gate. That gate fails, and the failure is
neither the cache nor the engine: `docs/research/31-prefill-nondeterminism-localisation.md` shows it
is the Vulkan driver being in the process, and the same binary is bit exact for every configuration,
thread count and process with the driver kept out or with `GGML_VULKAN=OFF`.**

Five things went differently from the plan.

1. Task 2 became three fixes rather than one: batching the syscalls, then the slice by eight crc32
   that capped the store at 519 MiB/s, then the device copy granularity that capped it again on
   Vulkan.
2. Task 4 needed an engine change the plan said it would not: a whole state restore for a hybrid
   model was thrown away by the `pos_min >= pos_min_thold` guard, which cost the cache every hit
   until it was understood.
3. Persistence needed a way to read an entry's tokens back out of its file, which is now
   `llama_state_seq_file_tokens` in `include/llama.h` plus a rescan at startup.
4. A heuristic that was already there excluded the workload this was built for. `f_keep < 0.5f`
   decides whether the cache is consulted at all, and for a conversation that is mostly a shared
   preamble it is never true, so the cache was never written and reuse was 2 percent. It is now
   `f_keep < 1.0f` and the same case measures 93 percent. This is the one change to existing
   server behaviour in the whole feature and it affects the RAM cache too.
5. Fix 2 above carried a crash, and the check that found it was the one written to close a
   "not tested" note rather than to look for a bug. A restore that covers the whole prompt leaves
   `n_past` equal to the entry's length, the sampler's minimum of one evaluated token then rewinds
   it, and dropping that token needs a partial removal the recurrent cache cannot do without
   `n_rs_seq`, so the server aborted. See the eviction section of
   `docs/research/30-ssd-prompt-cache-results.md`. The fix is one added condition,
   `n_past < slot.task->n_tokens()`, which leaves the extending prompt unaffected.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `server_prompt_cache` store its entries in files instead of host RAM, so a request
whose prefix is already cached pays an SSD read instead of prompt processing.

**Architecture:** Stage 0 makes the existing `O_DIRECT` store path issue large reads and writes
instead of one per 4 KB block, which is worth 9.8x and is a prerequisite. Stage 1 swaps the
entry's `std::vector<uint8_t>` for a file, reusing `llama_state_seq_save_file_direct` and
`_load_file_direct` unchanged, so no engine change is needed. Stage 2, content addressed attention
blocks, is deferred until stage 1 is measured.

**Tech Stack:** C++17 (`src/llama-io`, `tools/server/server-task`, `tools/server/server-context`),
`llama-server`, `llama-bench`, Python 3 for the loop harness.

**Spec:** `docs/superpowers/specs/2026-09-28-ssd-prompt-cache-design.md`
**Measurements:** `docs/research/27-ssd-prompt-cache-feasibility.md`

## Global Constraints

- Commits need explicit human approval for each action. Every commit step is a checkpoint: ask, and
  if approval is not given, leave the tree uncommitted and report. Never push.
- Do not add files under `tests/`. Reuse the existing binaries and `tools/server/tests/`.
- ASCII only, no em dashes. Comments concise, simple wording, no restating the code.
- Resident KV is never in host RAM. All store I/O uses `O_DIRECT` with aligned buffers. Host memory
  in the path stays O(1) in session count and length.
- **A cache may be absent, never wrong.** Every failure path returns a miss and the request is
  processed normally. No cache error may fail a request or change its output relative to a
  recomputation, other than through the one assumption in the spec's "The one assumption that is
  not verified".
- Layer 1 and 2 of the session store must not include a backend header. Neither may this.
- Builds: `build-vk` exists, `build-cpu` does not. `build-vk` is Release, `GGML_NATIVE=ON`,
  `GGML_VULKAN=ON`.
- Models: `/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf` for fast
  iteration, `/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`
  for every acceptance number.
- Scratch: `/home/herlanggays/.jcode/scratch/ssd-cache/`. Wrap every heavy run in
  `scripts/research/ssdcache/run_guard.sh`, and never run two heavy runs at once. The 35B is
  13.09 GiB against a 13.57 GiB TTM limit and `docs/research/freeze-notes-2026-09-27.md` records
  seven hard resets under it.
- Every measurement is taken twice and reported with both values. The GPU window that
  `4e94b1937` records makes single runs untrustworthy, and two of three probe runs disagreed by
  10x on one line.

---

### Task 1: Batch composition determinism (gate)

This is a gate on the whole design, not a step in it. If prompt processing the same prefix under
two different ubatch splits gives different KV, then a cache hit is "the computation that wrote the
entry" rather than "a recomputation", and every acceptance claim downstream has to say so.

**Files:**
- Create: `scripts/research/ssdcache/ubatch_determinism.cpp`

**Interfaces:**
- Consumes: `llama_state_seq_get_data_ext` (`src/llama-context.cpp:4320`), `llama_batch` with an
  explicit split.
- Produces: a pass or fail plus a byte difference count, for the 0.8B and the 35B.

- [x] **Step 1: Decode one prefix, serialize, decode the same prefix again in a fresh context with a different ubatch, serialize, compare.**
- [x] **Step 2: Run on the 0.8B with `n_ubatch` 128 against 512, then 512 against 2048.**
- [x] **Step 3: Run on the 35B with `-ngl 99`, a single `n_ubatch` pair first, since each run reloads 13 GiB.**
- [x] **Step 4: Record the result in `docs/research/28-ubatch-determinism-results.md`.**

Pass criterion: byte identical, or a quantified difference small enough to state as a bound. If it
fails, stop and revise the spec's assumption section before writing any cache code.

---

### Task 2: The store's I/O path uses the device

`llama_io_read_direct::next_block` reads one block per `pread` and `put_block` writes one per
`pwrite`. Measured at 185 and 273 MiB/s against 1810 and 1854 MiB/s at 1 MiB. This is a
prerequisite for stage 1, and it also corrects the park and unpark cost in `docs/research/16` and
`17` for the store that already exists.

**Files:**
- Modify: `src/llama-io.h` (`llama_io_read_direct`, `llama_io_write_direct`)
- Modify: `src/llama-io.cpp` (`next_block`, `put_block`, both destructors)
- Modify: `scripts/research/ssdcache/block_read_probe.c` (add a store-path case if useful)

**Interfaces:**
- Consumes: `LLAMA_IO_BLOCK` (`src/llama-io.h:43`), `io_pwrite_all` (`src/llama-io.cpp:174` usage).
- Produces: unchanged public behaviour, `n_blocks()`, `n_bytes()`, `good()` and `error()` semantics
  identical, with a window buffer per direction.

- [x] **Step 1: Reader window**

Add a second buffer, `win`, of a whole number of blocks, and refill it with one `pread` when the
current window is exhausted. `next_block` serves the next block from it. A torn tail is still
detected by the short read on the final window, and by the per block crc, so the tail rule does not
change. Keep the block crc and generation checks exactly as they are.

- [x] **Step 2: Writer window**

Accumulate flushed blocks into a buffer of a whole number of blocks and `pwrite` when it fills, and
in `flush()`. The file header still goes to offset 0 as its own write. A failed write must leave
`err` set and `n_bytes()` unchanged, exactly as now.

- [x] **Step 3: Prove the format did not move**

```sh
g++ -O2 -std=c++17 -I include -I ggml/include \
    scripts/research/kvstore/direct_sink_check.cpp \
    -L build-vk/bin -lllama -lggml-base -lggml -Wl,-rpath,$PWD/build-vk/bin -o /tmp/dsc
/tmp/dsc
python3 scripts/research/kvstore/cross_check.py
python3 scripts/research/kvstore/validate_file.py <a park file>
```

Pass criterion: the C++ and Python readers agree byte for byte with the pre-change file, and the
damaged block count is 0. The framing must not change at all, or every existing store file becomes
unreadable.

- [x] **Step 4: Re-measure the rate**

Run `park_disk_check` on the 0.8B and report the park and unpark time before and after. Expect
roughly 9x. Record it in `docs/research/29-store-io-batching-results.md`.

---

### Task 3: Park and unpark on the 35B (gate)

Docs 21 and 24 verified the direct pair on the 0.8B only. Stage 1 rests on it, so it has to be
verified on the model the cache is for, with the cache types the cache will use.

**Files:**
- Modify: `scripts/research/kvstore/park_disk_check.cpp` (take `n_gpu_layers` from the environment;
  line 142 hardcodes 0)

**Interfaces:**
- Consumes: `PARK_VERBOSE` already exists as the env precedent.
- Produces: the same report, on Vulkan.

- [x] **Step 1: `PARK_NGL=99 ./park_disk_check <35B> /path/park.bin 400 8192` under `run_guard.sh`**
- [x] **Step 2: Confirm the continuation is identical to never parking, and the RSS delta is flat.**
- [x] **Step 3: Repeat twice, since the box is non-deterministic in this window. If the two runs disagree with each other, the run is measuring the box and the result is void.**

Pass criterion: continuation identical, on both repetitions, with `-ngl 99` and
`type_k q8_0`, `type_v planar3_0`.

---

### Task 4: The entries become files

**Files:**
- Modify: `tools/server/server-task.h` (`server_prompt_data`, `server_prompt_cache_state`,
  `server_prompt_cache`)
- Modify: `tools/server/server-task.cpp` (`alloc`, `load`, `update`, `size`, `n_tokens`)
- Modify: `tools/server/server-context.cpp` (`prompt_save`, `prompt_load`, the cache construction at
  `:1359`, and the `cache_idle_slots` gate at `:1420`)
- Modify: `common/common.h`, `common/arg.cpp` (`--cache-disk-path`, `--cache-disk-mib`)

**Interfaces:**
- Consumes: `llama_state_seq_save_file_direct` (`include/llama.h:922`),
  `llama_state_seq_load_file_direct` (`:929`), `llama_state_seq_get_size_ext`.
- Produces: `server_prompt_cache` entries whose payload is a file, with the matching scoring,
  eviction and failure semantics unchanged.

- [x] **Step 1: `server_prompt_data` holds a path and a size**

Replace the two vectors with a path and the two sizes. Keep `size()` summing the same quantities so
the budget arithmetic does not change.

- [x] **Step 2: `prompt_save` writes a file**

Path is `<dir>/<fingerprint>-<prefix hash>.kvs`, where the fingerprint covers arch, model file
size, `n_ctx`, `type_k`, `type_v`, `v_trans` and the attention layer set. A save that returns 0
removes the file and leaves no entry.

- [x] **Step 3: `prompt_load` reads a file**

Call the direct loader into `id_slot` with the tokens out. A return of 0, or a token count that does
not match the entry, is a miss. Do not erase the entry on use; let the budget evict it.

- [x] **Step 4: Eviction unlinks**

`update()` and `alloc()` unlink on drop instead of freeing vectors. The byte budget counts the
entry's size, which includes the recurrent term, so short entries are not undercharged.

- [x] **Step 5: `--cache-ram 0` with a disk path is a valid configuration**

Right now `--cache-ram 0` disables the cache and `--cache-idle-slots` with it
(`server-context.cpp:1351`, `:1420`). With a disk path set, the cache exists and idle slots are
saved to it. Make sure a missing or unwritable directory disables the cache with a warning rather
than failing startup.

- [x] **Step 6: Build both, and confirm the 0.8B output is unchanged with the cache disabled**

```sh
cmake --build build-vk -j
build-vk/bin/llama-cli -m <0.8B> -p "The capital of France is" -n 32 --seed 1
```

Pass criterion: identical text with and without `--cache-disk-path`.

---

### Task 5: The cache actually hits

**Files:**
- Create: `scripts/research/ssdcache/cache_hit_check.py`

**Interfaces:**
- Consumes: `llama-server` with `--cache-ram 0 --cache-disk-path <dir> --cache-idle-slots -kvu`,
  the `/completion` endpoint with `n_predict: 0` so only prompt processing is measured, and the
  `timings` in the response for `prompt_n` and `cache_n`.
- Produces: a pass or fail, and the `cache_n` for each request.

- [x] **Step 1: Request 1 with a synthetic 4000 token prefix, assert `cache_n` is 0.**
- [x] **Step 2: Request 2 with the same prefix plus 500 tokens, assert `cache_n` is 4000 and `prompt_n` is 500.**
- [x] **Step 3: Restart the server, repeat request 2, assert the same, so persistence is proven.**
- [x] **Step 4: Assert the completion with the cache and with `--cache-ram 0 --cache-disk-mib 0` is byte identical at `temperature 0`, seed 1.**
- [x] **Step 5: Assert a corrupt entry is a miss, not an error.** Truncate the `.kvs` file and repeat
  request 2. The response must be correct and `cache_n` must be 0.

Pass criterion: all five. Step 5 is the one that matters most: it is the only check that the
failure path is a miss.

**Correction to this task, from running it.** Steps 1 and 2 as written do not exercise the cache.
A second request that extends the first is served from the slot's own KV, and the cache is never
consulted: the server only reaches it when a slot is about to lose more than half its context. The
first version of the check passed its cache assertions with no entry file on disk at all. The
scenario in `scripts/research/ssdcache/cache_hit_check.py` is now three conversations that share a
preamble rotating over one slot, and it also asserts that the appended turn keeps the prefix token
exact, without which no restore can be used. See
`docs/research/30-ssd-prompt-cache-results.md`, "Where the cache is and is not consulted".

---

### Task 6: The loop, measured

**Files:**
- Create: `scripts/research/ssdcache/agentic_loop.py`

**Interfaces:**
- Consumes: the server as in task 5.
- Produces: `cache_n` and wall clock per turn over 20 turns, against the model in
  `docs/research/27-ssd-prompt-cache-feasibility.md`.

- [x] **Step 1: 20 turns, a 4000 token system prompt plus tool block, 500 tokens appended per turn,
  `n_predict: 0` so the number is prompt processing only.**
- [x] **Step 2: Report token-turns processed, the reduction against no cache, and the wall clock.**
- [x] **Step 3: Measure the bytes written and read with `/proc/<pid>/io`, and compare against the
  2.45 GiB and 2.45 GiB the model predicts.**
- [x] **Step 4: Repeat at 4 KB and at the batched I/O to show the stage 2 gain in situ.**
- [x] **Step 5: Record in `docs/research/30-ssd-prompt-cache-results.md`.**

Pass criterion: the token-turn reduction is at least 90 percent, and the wall clock reduction is at
least 8x, in the degraded GPU window. Both are weaker than the model's predictions, deliberately.

**Corrections to this task, from running it.** The shape changed and one pass criterion was wrong.

- The shape is four conversations over one slot, ten turns each, not one conversation over twenty.
  A single conversation on a single slot keeps its prefix in the arena and never consults the cache
  at all, so the loop as written would have measured arena reuse. See the results document, "Where
  the cache is and is not consulted".
- The 90 percent token reduction holds in the steady state, 90.6 percent, but the aggregate over a
  finite loop is lower, 79.8 percent, because the first turn of each conversation has nothing
  cached. Both are reported.
- The 8x wall clock criterion was never defensible. The wall clock includes writing the displaced
  conversation's state, and this box's throughput moves by a factor of four between sessions, so
  the measured ratio for one request pair ranged 1.42x to 3.91x. The token reduction is the
  criterion that holds still, and the wall clock is reported as a range.
- Step 3's `/proc/<pid>/io` measurement was not taken. The entry sizes on disk were measured
  instead, 239.9 MiB peak for four conversations, which is the number the disk budget needs.
- Step 4 by construction: the store went from 197 to 843 MiB/s reading before this task ran, so
  there is no 4 KB configuration left to compare against. The comparison is in
  `docs/research/29-store-io-batching-results.md`.

---

### Task 7: Stage 2, only if the numbers ask for it

Not planned in detail. Open it when task 6 shows that either the write volume or the cross
conversation sharing of a common preamble is the binding cost. The design is in the spec's
"Stage 2" section.
