# SSD prompt cache: design

Date: 2026-09-28
Status: stage 0 and stage 1 built and verified on the 0.8B. Stage 2 is design only.
Tree: `qwen-only-backends`
Measurements: `docs/research/27-ssd-prompt-cache-feasibility.md`
Results: `docs/research/30-ssd-prompt-cache-results.md`
Depends on: `docs/superpowers/specs/2026-09-27-session-kv-store-design.md` (the store format)

## What the implementation changed about this design

Three corrections, all from building and running it. Read the rest of the document with these in
mind, since two of them contradict what it says below.

1. **"Stage 1 needs no engine change" is false.** A whole state restore for a hybrid model was
   thrown away by the `pos_min >= pos_min_thold` guard in the prompt processing path, for the
   reason given in the results document: `llama_memory_hybrid::seq_pos_min` returns the maximum of
   the attention and recurrent minima, and the recurrent summary's position is the end of the
   restored state. A server change was needed, and the shipping RAM prompt cache turns out to
   satisfy the same guard through its context checkpoints rather than through the storage.
2. **Persistence needs an engine addition this document did not foresee.** A restarted server had
   no way to learn what was on disk, because the entry list is in memory only and the tokens live
   inside the files. `llama_state_seq_file_tokens` reads them back through the store's own reader,
   and `server_prompt_cache::rescan` registers what it finds.
3. **Stage 0 is three fixes, not one.** Batching the syscalls is what this document describes.
   After that the byte at a time crc32 capped the store at 519 MiB/s and one device copy per block
   capped it again on Vulkan. Both are in `docs/research/29-store-io-batching-results.md`.

One thing this document got right and worth keeping visible, because it is the requirement users
will trip over: a hit needs the cached prefix to be **token identical** to the input prefix. A
continuation that diverges anywhere inside the restored prefix cannot be served, and the server
falls back to reprocessing. See the results document for the measurement.

## Goal

Stop reprocessing a prefix the server has already processed. A client that resends its
transcript every turn, which is what an agentic loop does, should pay prompt processing only for
the tokens that are new, at any distance in the past, after a slot change, after an idle gap, and
after a server restart.

The cache is the server's existing prompt cache, `server_prompt_cache`. The change is where its
bytes live: a file instead of a `std::vector<uint8_t>`.

## What the numbers say

Per cached token, from `docs/research/27-ssd-prompt-cache-feasibility.md`:

| | per token |
| --- | --- |
| prompt processing, 120 t/s on Qwen3.6-35B-A3B | 8333 us |
| restore from SSD, batched reads | 4.25 us |
| restore from SSD, as the store reads today | 39.0 us |

Over a 20 turn loop with a 4000 token prefix growing 500 a turn: 175000 token-turns to 10000,
1458 s of prompt processing to 86 s, and the disk traffic is 0.16 percent of the time saved.

So the answer to "is this free" is: the I/O is 0.2 percent of what it replaces, and the parts
that are not free are the suffix, which is unavoidable, and the write volume, which is a wear
question rather than a time question.

## Non-goals

- Changing prefill, attention, or the memory layouts. The cache moves bytes that already exist.
- Serving more concurrent sessions than the arena holds. That is the session store's problem and
  it needs the placement policy from its layer 2, which is not built.
- Eviction policy sophistication. LRU over entries is enough to start.

## What exists, and the exact seam

| piece | where |
| --- | --- |
| the cache, host RAM, `std::vector<uint8_t>` per entry | `server_prompt_cache`, `tools/server/server-task.h:588-660` (`server_prompt_data` at `:588`, the cache at `:612`), `tools/server/server-task.cpp:1689-1880` |
| longest common prefix match, scored `f_keep` and `f_sim`, entry consumed on load | `server_prompt_cache::load`, `server-task.cpp:1793` |
| entries contained in a new prompt dropped on alloc | `server_prompt_cache::alloc`, `server-task.cpp:1738-1748` |
| the slot calls save then load on every assignment | `server-context.cpp:1647-1656` |
| a failed load clears the slot | `server-context.cpp:1649-1651` |
| the same, at idle | `server-context.cpp:2441` |
| purge of idle slots to the cache | `try_clear_idle_slots`, `server-context.cpp:1667` |
| save and load a whole sequence state to a file, framed, `O_DIRECT`, verified, retried | `llama_state_seq_save_file_direct` / `_load_file_direct`, `src/llama-context.cpp:3303` and `:3336`, public at `include/llama.h:922` and `:929` |
| the file carries its own token list, so matching needs no side table | header in `state_seq_save_file_direct_once`, `src/llama-context.cpp:3268` |
| KV serialization is already range based | `state_write_data(io, cell_ranges_t)`, `src/llama-kv-cache.h:352` |

The whole state, attention KV and the gated delta net recurrent state, goes into one blob with
`LLAMA_STATE_SEQ_FLAGS_NONE`, and `state_seq_load_file_direct` restores with flags 0, so a
restored entry carries both halves at the same position. That is the property the session store
spec calls out as mandatory for a hybrid model, and the existing pair already has it.

### The lifecycle is already the one a prefix cache wants

Read `server_prompt_cache::load` and `::alloc` together and the shape is: match by longest common
prefix, prefer an entry that keeps more of the slot, drop entries that a new longer prompt
contains, consume an entry when it is used. For a conversation that only grows, that collapses to
one live entry per conversation with the longest prefix, which is exactly the cache-entry model in
the loop table above. The disk tier does not need a new lifecycle. It needs the entry to be a file
and the eviction to unlink.

Two details have to change for the disk case:

- `load` erases the entry it used (`server-task.cpp:1864`) and clears its vectors. On disk the
  file is worth keeping, because the same prefix will be asked for again by a sibling request.
  Keep the entry and let the byte budget evict it.
- `alloc` clears the vectors of an obsolete entry. On disk that is an unlink.

### One structural consequence of the current config

`--cache-ram 0` disables the prompt cache outright (`server-context.cpp:1351`), and
`--cache-idle-slots` is force-disabled without it (`server-context.cpp:1420-1422`). With unified
KV, an idle slot is purged to the cache and its KV leaves the arena
(`[TAG_IDLE_SLOT_CLEAR]`). So in the agentic case, where a client is silent for a while between
turns, the steady state per turn is a full save and a full load through the cache, and the arena
holds nothing between turns.

That is the configuration the design has to be fast in, and it is the one the numbers above
assume. It is not the only possible one: reviving the slot's KV instead of purging it would avoid
most of the traffic, but only while the arena can hold every live conversation, which on this box
at 13.09 GiB of model is not many. Section "Residency" below keeps the option open without
requiring it.

## Stage 0: make the store's I/O path use the device

This is a prerequisite, not an optimisation. `llama_io_read_direct::next_block`
(`src/llama-io.cpp:367`) issues one `pread` of `LLAMA_IO_BLOCK` bytes per block, and
`llama_io_write_direct::put_block` (`src/llama-io.cpp:174`) one `pwrite` per block. Measured,
that path runs at 185 MiB/s reads and 273 MiB/s writes, against 1810 and 1854 MiB/s for the same
file in 1 MiB chunks. It is latency bound at about 21 us per 4 KB request with one request in
flight.

The change is to hold a larger aligned buffer and serve blocks out of it:

- reader: fetch, say, 1 MiB per `pread` into a second buffer, and hand `next_block` a window
  inside it. The framing is self-describing, so nothing above it changes. A block that straddles
  the end of the window is the only edge case, and a window size that is a multiple of the block
  size removes it.
- writer: accumulate blocks into a 1 MiB buffer and `pwrite` when it fills, plus on `flush()`.

Cost: one extra buffer of a few MiB, which violates nothing, because the policy is O(1) host
memory and not zero host memory. Verification: the existing checks re-run, `park_disk_check` for
the continuation, `validate_file.py` for the layout, and the probe above for the rate.

This also corrects the park and unpark cost in `docs/research/16` and `17` for the existing store,
which is the same code path.

## Stage 1: the entries become files

The smallest change that gets the win, and it needs no engine change at all.

- `server_prompt_data` (`tools/server/server-task.h:588`) becomes a path and a size instead of two
  `std::vector<uint8_t>`.
- save: `prompt_save` calls `llama_state_seq_save_file_direct(ctx_tgt, path, id, tokens, n)`.
  The direct pair already retries up to three times, verifies every block on read back, and
  returns 0 rather than reporting success over a damaged file.
- load: `prompt_load` calls `llama_state_seq_load_file_direct(ctx_tgt, path, id_slot, ...)`.
  A return of 0 is a miss.
- the token list stays in RAM per entry, because the match needs it on every request and reading
  it back from disk to compare would cost a seek per entry. 4 bytes per token, so 100 entries at
  16k tokens is 6.4 MB. The text is dropped.
- eviction: `update()` unlinks instead of popping, and the budget is bytes on disk.
- the path is content addressed: `<dir>/<fingerprint>-<hash of the token prefix>.kvs`, so a
  concurrent rewrite of the same prefix is idempotent and a different prefix never collides. The
  fingerprint covers the model, `n_ctx`, `type_k`, `type_v`, `v_trans`, and the attention layer
  set, so a file written under one configuration can never be loaded under another.

New flags: `--cache-disk-path <dir>` and `--cache-disk-mib <n>`. `--cache-ram 0` plus a disk path
is the intended pairing, with `--cache-idle-slots` then meaningful again rather than disabled.

What it gets: everything in the loop table except the write volume. What it does not get:
attention blocks are stored once per conversation, so two conversations sharing a 30k token
system prompt and tool set store that preamble twice.

### Failure handling, which is the whole safety argument

A cache is allowed to be absent. It is not allowed to be wrong.

- a save that returns 0 leaves no entry, or an entry that later fails to load
- a load that returns 0 is a miss and the request is processed normally
- a load that returns a wrong length is a miss
- a load that returns something structurally valid but stale cannot happen, because every block
  carries the generation of the save that wrote it and a block from a different save is rejected
  on read
- any error from the cache is not surfaced to the client and never fails a request

There is no path in which a request fails because of the cache, and no path in which a request
gets a wrong answer from it that the engine would not also produce on its own. The second half of
that sentence is the one assumption in this document, and it is named below.

## Stage 2: attention blocks become content addressed

Worth doing after stage 1 is measured, for two reasons that are different from each other.

**Sharing.** Attention KV for a token block is a function of the prefix and nothing else, so two
conversations with the same preamble share every block of it. Key with a chain,
`key(b_i) = H(key(b_{i-1}), tokens of b_i)`, which is the standard construction: a request walks
the chain for its own blocks and takes the longest prefix that resolves.

**Write volume.** Stage 1 writes the whole state per turn. At 13500 tokens that is 101.5 MiB of
attention plus 62.8 MiB of recurrent, so 2.45 GiB over the loop. Appending only new attention
blocks and one recurrent blob per turn is 1.30 GiB. The recurrent blob is the floor and block
granularity cannot go below it.

The shape:

    kv.pages    append only, slots of one (layer, block) column, padded to a whole block
    kv.index    append only records, PUT(block_key, layer, slot) and ENTRY(prefix_key, n_blocks,
                rs_gen), plus CHECKPOINT
    <entry>.rs  the recurrent blob for one entry, fixed size, rewritten whole per turn

The crash story is simpler than the session store's, and the reason is worth stating: these pages
are immutable. There is no free, no move, no in place reuse, and no context compaction to
tombstone, so the recovery argument reduces to "a torn or lost page write means the page is
absent or rejected", which is the generation check the store already has. The two file store's
hard parts, the slot reuse precondition and the base advance ordering, exist because sessions are
mutable and do not apply here.

What stage 2 needs that stage 1 does not:

- a public API to save and restore a *range* of cells. `llama_kv_cache::state_write_data` already
  takes `cell_ranges_t` (`src/llama-kv-cache.h:352`) and `state_read_data` already takes a
  destination `slot_info` (`:356`), so the range machinery exists, but it is internal. This is an
  engine change and therefore goes through the layer 1 and 2 rules in the session store spec.
- a decision about where the index lives. A manifest per entry, named by its prefix key, is
  enough, and it keeps the file count at three.
- eviction with references: an entry names blocks, blocks outlive entries only while referenced.

## The recurrent state, which is the term that does not compress out

On Qwen3.6-35B-A3B the gated delta net state is 62.8 MiB per conversation, fixed, and independent
of context length. Consequences, all of them already in the numbers table:

- it is the dominant term for short prefixes: at 4000 tokens it is 62.8 of 91.5 MiB, 69 percent
- it is why there is one entry per conversation rather than a blob per block boundary. A snapshot
  per 256 token block would be 245 KB per token, 33x the attention slope
- it is why a per token restore cost falls with prefix length instead of staying flat
- it is why stage 2 only removes 47 percent of the write volume rather than most of it

It is also not derivable from a subset of tokens, so it cannot be split or recomputed cheaply. One
blob per entry, written whole, is the design. The only reduction available is precision, and
quantizing it changes the continuation of a restored prefix relative to a re-processed one, which
is the property the whole design rests on. Leave it at f32.

## The one assumption that is not verified

**That prompt processing the same prefix twice produces the same KV.** Attention is per token
projections, so it does. The gated delta net is a scan over a ubatch, and its floating point
reduction order can depend on where the ubatch boundaries fall, which depends on how many slots
are active and what else is in the batch. If it does, then a prefix restored from a cache entry
written under one batch composition will continue slightly differently from the same prefix
processed under another, and the "identical continuation" property that docs 21 and 24 verified on
the 0.8B becomes a same-configuration property rather than an unconditional one.

This is not introduced by the cache. A server that reprocesses the same prompt under different
batch compositions already has whatever this is. But the cache makes it observable by pinning a
prefix to the composition that happened to write the entry, and the design should not claim a
property it has not measured.

The test is cheap and should precede any acceptance claim: process the same prefix twice in the
same process with different `n_ubatch`, serialize both, and compare. If the bytes match, the
assumption holds and the cache is transparent. If they differ, the difference is bounded and
should be quantified, and the design becomes "a cache hit reproduces the computation that wrote
it" rather than "a cache hit equals a recomputation".

## Residency

The session store spec's rule is that resident KV is never in host RAM. This design inherits it:
`O_DIRECT` for all store I/O, one reused staging buffer per direction, host cost O(1) in session
count and length.

The rule is what makes the disk tier the *only* home rather than a spill, and it has one
consequence worth naming: it rules out the cheapest possible version of this cache, which is to
let the page cache hold the entries. Warm buffered reads measure 8.5 to 12.7 GiB/s against 1810
MiB/s for `O_DIRECT`, and background writeback can slow the 4 KB `O_DIRECT` read to 16 MiB/s
while leaving large reads at 1380. So the page cache version would be faster and simpler and it
would put every cached entry in RAM, which is the thing the user is trying to stop paying for.

Two later refinements, both deferred:

- Do not purge an idle slot whose KV fits, and park it only when another request needs the slot.
  This removes the per turn round trip entirely for a workload with fewer live conversations than
  arena slots, and it is what the session store's layer 2 is for.
- Reuse an entry that is already in the arena instead of reading it back from disk. The current
  code cannot, because the load path always goes through the cache, and the check "is this
  sequence's state already the one I want" is a comparison the server does not currently make.

## Budget

Defaults to start: `--cache-disk-mib 8192`. At 7400 B per token plus 62.8 MiB per entry that is

| conversation length | entries |
| --- | --- |
| 4000 tokens | 89 |
| 16000 tokens | 46 |
| 64000 tokens | 15 |

Evict by LRU over entries, with the byte count including the recurrent term, since that is what
makes short entries cost more than their token count suggests.

## Verification

Ordered, each one a gate for the next.

1. **The batch composition test above**, on the 35B with `type_k q8_0` and `type_v planar3_0`.
   Nothing else is worth measuring before this.
2. **Reboot and re-check the gate.** `4e94b1937` records that the GPU path became 2.2x slower and
   non-deterministic in this window, and the standing advice is a reboot followed by
   `llama-perplexity -f prose.txt -ngl 99 -fa on -c 2048 --chunks 2` returning exactly 6.0495 and
   pp512 returning about 255. Until it does, any "identical output" comparison is measuring the
   box. *Superseded for this work the same day:
   `docs/research/31-prefill-nondeterminism-localisation.md` shows the prefill nondeterminism is the
   Vulkan driver being in the process rather than the box or the engine, and the gate passes for
   every configuration, thread count and process with the driver kept out
   (`VK_ICD_FILENAMES=/nonexistent/icd.json`) and on a build with `GGML_VULKAN=OFF`. The reboot is
   still the right thing to do for the box's other symptoms, `4e94b1937`'s slowdown among them, but
   it is no longer a gate for this.*
3. **Stage 0**, then re-run the probe and `park_disk_check` on the 35B, extending the harness to
   take `n_gpu_layers` instead of hardcoding `0` (`park_disk_check.cpp:142`). The 35B has never
   been through the park and unpark check, only the 0.8B.
4. **Stage 1, two requests.** Request one fills the cache with a 4000 token prefix; request two
   sends the same prefix plus 500 tokens with a fresh server and the cache warm; assert
   `n_prompt_cached` is 4000 and the completion matches a run with the disk cache disabled.
   Restart the server between them for the persistence half.
5. **Stage 1, the loop.** A 20 turn harness against `llama-server`, measuring `n_prompt_cached`
   and wall clock, reproducing the 175000 to 10000 token-turn reduction and checking that the
   disk traffic matches the 2.45 GiB the model predicts.
6. **Stage 2** only after stage 1's numbers are in hand, and only if the write volume or the
   cross conversation sharing turns out to matter for the real workload.

## Open questions

1. Block size for stage 2. 256 tokens gives 1.9 MiB per block across the 10 attention layers,
   which is a reasonable write unit. It has to be a fixed function of position, so that the same
   tokens always give the same boundaries, whatever the request looks like.
2. Whether the entry should be keyed on the token prefix alone or also on the rendered template
   version. Keying on tokens is what the engine can see; a template change with the same tokens
   would silently reuse the old rendering. A template hash in the fingerprint is the cheap
   answer.
3. What happens to the draft or MTP state. `server_prompt_data` has a `drft` vector and
   `common_prompt_checkpoint` has a `data_spec`, so the live configuration may carry a second
   context. Either it is cached the same way, one file per context, or speculative decoding is
   disabled for cache hits. Not decided.
4. Whether the idle slot purge should stay on by default once the disk tier exists, given that it
   forces a full round trip per turn and the arena could often hold the state for free.
