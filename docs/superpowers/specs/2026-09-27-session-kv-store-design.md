# Session KV store: design

Date: 2026-09-27
Status: layer 1 partly built, layers 2 and 3 are design only
Tree: `qwen-only-backends`
Layers: 1 (format, this document), 2 (placement), 3 (backend acceleration)

## Status: what exists, and how to read this document

Read this first, because the rest of the document is a design and parts of it are not code yet.

Built and verified:

| piece | note |
| --- | --- |
| `llama_io_write_direct` / `llama_io_read_direct` | block framed store I/O through `O_DIRECT`, one reused 4 KB buffer, so host memory stays O(1) in the state |
| `llama_state_seq_save_file_direct` / `llama_state_seq_load_file_direct` | the park and unpark pair, verified against a never-parked continuation on a fresh context into an empty sequence |
| per-save generation, verify, rewrite | see "Silent write loss" below. A save either writes an intact file or fails |
| fault-injection harness | `scripts/research/kvstore/`, a Python model of the format with 88 crash points, plus the C++ and Python cross checks |

Designed in this document but not built:

- the two-file store, `kv.pages` and `kv.status`, with its record log, checkpoint, recovery and
  compaction. The framing and the direct sink it would sit on are built; the two files are not.
- the residency arithmetic of layer 2, and the fast paths of layer 3.

Falsified since this document was written, and corrected in place below:

- the claim that the store is a strict superset of `LLAMA_STATE_SEQ` v3, so that
  `--slot-save-path` and the prompt cache keep working. See "Compatibility".
- the 512 byte block size, which loses blocks on this volume. See the `kv.status` section.

## Goal

Let a server hold more sessions than fit in device memory, by parking a session's KV
state outside the compute arena and restoring it on demand. Residency must be a policy,
not a property of the allocation, and the durable form must be portable across machines.

Hard constraint: resident KV is never in host RAM. The only homes for KV are device memory
and disk. Host RAM is for the model and for transient staging, nothing else. See "Residency
policy" below for what that rules out and what it costs.

## Non-goals

- Cross-session prefix sharing. Deferred, see "Open questions".
- Changing attention math. The arena layouts and kernels stay as they are.
- Fitting any particular GPU. Layer 1 and 2 must not reference a backend.

## Why the current shape cannot do it

`llama_kv_cache` allocates one dense tensor for the whole context
(`src/llama-kv-cache.cpp:233`):

    ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream)

`kv_size` is the total cell budget for all sequences. So residency is all-or-nothing per
buffer, and the only way to release a sequence is to serialize it out through
`state_seq_get_data_ext` and `seq_rm` it. There is no per-session allocation unit to
evict.

## What already exists

| piece | where |
| --- | --- |
| per-sequence serialize/restore | `llama_state_seq_get_data_ext` / `set_data_ext` (`src/llama-context.cpp:3062-3100`) |
| byte sinks and sources | `llama_io_write_host`, `llama_io_read_host`, `llama_io_write_file`, `llama_io_read_file`, `llama_io_write_device`, `llama_io_read_device` (`src/llama-context.cpp:2537-2854`) |
| size-only pass | `llama_io_write_dummy` (`src/llama-context.cpp:2537`) |
| host-RAM session cache | `server_prompt_cache` (`tools/server/server-task.h:597-635`), `--cache-ram`, `--cache-idle-slots` |
| file form, host buffered | `LLAMA_STATE_SEQ_MAGIC` / `_VERSION` 4 (`include/llama.h:48-49`), `state_seq_save_file` / `state_seq_load_file` |
| file form, direct | `llama_state_seq_save_file_direct` / `_load_file_direct`, `llama_io_write_direct` / `read_direct` (`src/llama-io.{h,cpp}`) |
| 3 bit V cache type | `GGML_TYPE_PLANAR3_0` (`ggml/include/ggml.h:433`) |
| on-device form | `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` (`include/llama.h:928`) |
| cell metadata | `llama_kv_cells` (`src/llama-kv-cells.h`) |
| recurrent state | `llama-memory-recurrent`, `llama-memory-hybrid` |

The important one is the `llama_io_*` set. The format is already split from the medium, so
the store can be built on the existing serializers without touching a backend.

## Layout

Three layers, bottom up. Only layer 3 knows what a device is.

    layer 3  acceleration      per-backend probes and fast paths
    layer 2  placement         tiers from ggml_backend_dev_* plus a cost model
    layer 1  format            pages, status log, checkpoint, recovery

Layer 1 is pure bytes and is testable with no model and no GPU. This document specifies it.

## Layer 1: two files

### `kv.status`

Append-only. Written in 4096 byte blocks, because `O_DIRECT` requires both offset and length
aligned on this filesystem, and because 4096 is the device's own block size. The second part is
not a preference. The first version of this spec used 512 byte blocks, which forces the
filesystem to read and modify a whole 4 KB device block per 512 byte write, and on this volume
that loses blocks: about ten out of every 112590, on every write. Alignment is satisfied by any
multiple of 512, so 4096 costs nothing and removes the loss. See
`docs/research/23-write-path-loses-blocks-results.md`.

    block  := [u32 used][u32 crc32(payload area)][u32 gen][records][zero padding]
    record := [u32 len][u32 crc32(payload)][payload]
    payload := [u64 epoch][u32 op][op args...]

`used` is the payload byte count in the block. `len` is the payload length of a record. `gen`
is the generation of the save that wrote the block, see below. Records are batched into blocks
and the buffer is reused; one `O_DIRECT` write per record costs 93x write amplification, so
batching is required.

The block header is not optional. Without it the padding is indistinguishable from a torn
tail, and a reader stops at the first block boundary: measured at 93 records recovered out of
2000. With it, readback is byte exact and a torn block is rejected as a unit, losing only the
records inside that block. Numbers in `docs/research/18-direct-io-framing-results.md`.

#### Silent write loss, and why the header carries a generation

A write can be lost on the way to the device while `pwrite` still reports success. What comes
back is not zeroes, it is whatever occupied that block before, and for a store file that is
usually an earlier save of the same file, laid out the same way. Those stale bytes are a
structurally valid frame with a matching crc.

The crc alone therefore cannot detect this, and this is the failure that matters: with crc
alone the reader accepts the stale block, the restore reports success, and the session comes
back holding part of an older state with no error anywhere. It was observed exactly that way, as
a park whose restore produced a different continuation from a run that never parked, with every
block passing crc.

The header therefore carries the generation of the save that wrote the block, and the file
header carries the same value. A block belonging to a different save is rejected on read even
though it is internally consistent. Zero is never a valid generation.

Two rules follow, and both are required:

1. After a save, read the file back and check every block: `used` in range, crc matching, and
the generation equal to this save's. One reused block buffer, so the check stays O(1) in host
memory and never holds the payload resident. If any block is damaged, rewrite the whole file,
up to three attempts.
2. If the file cannot be written intact, the save fails and returns 0. It never reports success
over a file it has not verified.

`fsync` is not used. It does not prevent the loss, and on this volume it makes it worse: with
`fsync` after the writes, no save completed inside three attempts.

A reader that sees a short read, a bad block crc, a generation from another save, a length that
overruns its block, or a bad record `crc32` stops there and treats everything after it as
absent. That is the whole torn-tail rule. The atomic unit is a block, so the durability point is
a block boundary, not a record boundary.

One consequence of tearing at block granularity, established by the matrix: a torn write whose
prefix happens to cover the whole payload area leaves a block that is complete and valid. It
cannot be told apart from a full write, and it does not need to be, because every byte it claims
is present. The only observable outcomes are "the block landed" and "the block is absent", never
a partial block.

Ops:

| op | args | meaning |
| --- | --- | --- |
| `ALLOC` | slot, gen, layer, block, n_cells | slot now holds block `block` of `layer` |
| `FREE` | slot, gen | slot is dead as of `gen` |
| `MOVE` | src_slot, src_gen, dst_slot, dst_gen | compacted copy, used by defrag |
| `SEQ_RM` | seq, p0, p1 | context compaction, a tombstone |
| `SEQ_BIND` | seq, slot, gen | attach a parked session to arena slots |
| `RS_PUT` | seq, slot, gen | recurrent state blob for `seq` (hybrid models) |
| `BEGIN` / `COMMIT` | epoch | brackets a multi-record atomic unit |
| `CHECKPOINT` | epoch, root_page, root_crc, model_fp | new visible root |

Rules:

1. A record is invisible until a `COMMIT` for its epoch has been validated. An
   unterminated `BEGIN` bracket at the tail is dropped with the rest of the tail.
2. `MOVE` and `FREE` always carry the generation they apply to. A record applying to a
   generation other than the one currently resolved for that slot is ignored, not an
   error. A stale reference is structurally unreadable.
3. `CHECKPOINT` is the only op that changes the visible set.

### `kv.pages`

Header, then fixed-size slots:

    header := [magic][version][model_fp][page_tokens][n_layer]
              [n_embd_k_gqa][n_embd_v_gqa][v_trans][n_swa][dtype_k][dtype_v]
              [layout_k][layout_v][endian][header_crc]

    slot   := [slot_id][gen][crc32][K block][V block][padding]

One slot is one `(layer, block)` pair, `page_tokens` cells wide. The slot size is padded to a
multiple of the 4096 byte store block, so a slot write is a whole number of aligned blocks. A
logical attention block is therefore a column of `n_layer` slots, and the block table maps
block index to slots. Layout fields carry the planar3 rotation seed when that type is in use,
so a store cannot be read back with a different rotation.

Gap. The slot carries a crc, but no generation, and the file is not read back after a write.
So a stale slot that is internally consistent cannot be told from a fresh one, which is the
failure the status file now handles and this one does not. It matters for the same reason: a
lost slot write would restore a page of an older session instead of failing.

The root page is stored in `kv.pages` like any other page and is referenced by the
checkpoint record. That keeps the checkpoint record small and the file count at two.

### Recovery

    read tail of kv.status backwards for the newest CHECKPOINT whose record crc is valid
      and whose root page exists in kv.pages with a matching crc
    if none: fail closed, cold start
    load root page: slot -> generation map, per-seq bindings, recurrent blobs
    replay delta records forward from the checkpoint
      apply only inside a COMMITted epoch
      stop at the first bad block
    resolve each live slot, then verify every referenced (slot, gen) against kv.pages crc
    run the semantic check

Semantic check, independent of the log:

- for each sequence, positions are strictly increasing and inside `[0, n_ctx)`
- the reconstructed cell count per sequence equals the count the session expects
- every shared cell (`llama_kv_cells::seq_count > 1`) still has all its referrers present
- for hybrid models, each bound sequence has exactly one recurrent blob, and the blob's
  epoch equals the attention epoch for that sequence

Any failure steps back one checkpoint. Two independent mechanisms, structural and
semantic, both fail closed.

The hybrid rule is not optional. Gated delta net state is a running summary with no
per-token pages. If attention is parked at token `t` and the recurrent blob is at `t' != t`,
output is silently wrong and no crc anywhere fires.

### Compaction

Two things are called compaction. They are different problems.

| | what | how |
| --- | --- | --- |
| context compaction | dropping or shifting a sequence's tokens (`seq_rm`, `seq_div`, SWA evict) | append a tombstone, no special handling |
| log compaction | reclaiming space in `kv.status` and `kv.pages` | write a new root, append one `CHECKPOINT` |

Log compaction is safe under append-only because the switch is a single record:

1. write the new compacted slots under new generations
2. write a new root page covering only live slots
3. append the `CHECKPOINT` naming that root
4. flush the block holding that checkpoint and its commit
5. advance the status base to the block that starts that epoch
6. everything before it in `kv.status`, and every slot not in the new root, is dead

Step 4 is not optional and it is easy to miss. The base advance is an independent in-place
write, while the block holding the checkpoint is buffered until it fills or the epoch ends.
If the base moves first, it points at a block that has not been written, the reader finds an
empty block there, and recovery returns nothing at all. That failure was found by the crash
matrix, not by reading the code: it cost every session in the store.

A crash before step 3 leaves the previous checkpoint valid and the new pages unreferenced.
That is shadow paging. No reader ever observes a half-compacted state, and nothing has to
be checked for the answer to be "no". The verification pass exists as a second net, not as
the argument.

One precondition, established by the crash matrix rather than by reasoning: a slot may only
be reused when it is not referenced by the newest durable checkpoint. Otherwise a crash
between the page write and the record naming the new generation leaves that checkpoint's
root pointing at a page whose generation already changed. So compaction output goes into
slots the newest durable checkpoint does not name, which means reclaim needs free space and
cannot always be done in a single pass.

If the base advance is torn, recovery falls back to the file header and replays from the
beginning, which is idempotent because the checkpoint is still in the log. So physical
truncation of `kv.status` is only safe once the base advance is durable.

Eviction is safe for the same reason: a sequence is parked at a quiescent point, with no
in-flight compute for it and speculative state already rolled back
(`llm_arch_supports_rs_rollback`). Single writer, one checkpoint at park time.

### Compatibility with `LLAMA_STATE_SEQ` v3

Tested, and the claim that stood here was wrong. It said the store is a strict superset of v3,
so that `server_prompt_cache` and `--slot-save-path` keep working against store files. They do
not. Measured in both directions through the real public API:

| save | load | result |
| --- | --- | --- |
| shipping `llama_state_seq_save_file` | shipping loader | ok |
| direct | direct loader | ok, continuation identical to a run that never parked |
| shipping | direct loader | returns 0, not loaded |
| direct | shipping loader | returns 0, not loaded |

The payload stream is the same v3 shape in both cases. The container is not: a store file is
block framed and carries a generation, a v3 file is neither, so neither loader reads the
other's file. `LLAMA_STATE_SEQ_VERSION` is now 4, and store files are not readable by a version
3 reader.

Two ways to close this, and the choice is still open:

- Detect the format on read. Block 0 of a store file is its own header and a v3 file starts
  with its own magic, so either loader could sniff and dispatch. This is the direction worth
  having, because it decides whether a user's existing saved slots survive.
- Or drop the claim, and say plainly that migrating an existing `--slot-save-path` file is a
  conversion step.

Numbers in `docs/research/24-state-api-compat-results.md`.

## Residency policy

KV lives in exactly two places: device memory, and disk. It is never resident in host RAM.

| | resident in | note |
| --- | --- | --- |
| device tier | device local memory | the compute arena, sized for concurrent sessions |
| disk tier | the store files | the only durable home |
| host RAM | transients only | one bounded staging buffer, reused, never grown with context |

The distinction that makes this enforceable is resident versus transient. A transfer needs
somewhere to land on the host side, because Vulkan has no direct device to file path. That
buffer is one page, reused for every transfer, so host cost is O(1) in session count and
session length. Anything that grows with context length on the host is a violation.

Two system behaviours break the rule silently, so both must be handled, not assumed away:

1. **The page cache.** A buffered write to a file lands in the page cache, which is system
   RAM. Measured: a 512 MiB buffered write grew `Cached` by 514 MiB. So all store I/O must
   use `O_DIRECT` with aligned buffers. Measured on this box, that is also the faster path,
   so the rule costs nothing (see `docs/research/16-disk-tier-results.md`). It does impose one
   requirement: the block size has to be the device's own 4 KB, because smaller aligned writes
   lose blocks. See the generation section under layer 1.
2. **Swap.** This box has 54 GiB of zram swap, which is compressed RAM. Under memory
   pressure any resident page, including a staging buffer, can end up there. Keeping host
   use at O(1) is what prevents it.

Config consequence: `--cache-ram 0`. The shipping prompt cache is a host RAM store, so it
is off by policy. Note the boundary: with `--cache-ram 0` there is no persistence at all
today, so the disk tier is not an optimisation over the current state, it is the feature.

Session size is a fixed term plus a slope, not a per token rate:

    bytes(session) = rs_bytes + tokens * kv_bytes_per_token

Measured: 19.3 MiB plus 12.0 KB per token on Qwen3.5-0.8B, and 62.8 MiB plus 20.5 KB per
token on Qwen3.6-35B-A3B. The fixed term is the gated delta net recurrent state, and context
compaction cannot shrink it. Two consequences for the policy: parking has a floor, and the
number of parked sessions depends on their length, not only on the disk size. Numbers in
`docs/research/17-device-to-disk-path-results.md`.

### Capacity assumption

**Ten sessions, each holding the full context of the loaded model.** For the model this store is
being built around, Qwen3.6-35B-A3B, the GGUF metadata gives:

    layers 41, full attention 10 (one every 4), state layers 31
    kv heads 2, key_length 256, value_length 256, context_length 262144

with K cached q8_0 and V cached planar3_0:

    K q8_0       5440 B/token   10 x 2 x 256 dims x 1.0625 B
    V planar3_0  1960 B/token   10 x 2 x 256 dims x 0.3828 B
    slope        7400 B/token   = 7.23 KiB/token

    per session  62.8 MiB + 262144 x 7400 B = 1.868 GiB
    ten sessions                             = 18.68 GiB

The slope is 2.77x smaller than the f16 figure measured above. Predicted f16 from the same
metadata is 20480 B/token, or 20.0 KiB/token, against the measured 20.5, so the layer and head
counts are the right ones to compute from.

Reproduce with `scripts/research/kvstore/kv_capacity.py <model.gguf>`, which reads only the GGUF
header and needs no model load.

Two things this assumption does not settle, both flagged rather than assumed:

- Which context. 262144 is the model's own context length. If the intent is ten sessions at the
  context the server was launched with, `-c 2048`, the same arithmetic gives 77.3 MiB per session
  and 0.75 GiB for ten, which is 25x smaller. The assumption above takes the model's own context.
- Disk. The box has 35 GB free on `/`, so 18.68 GiB of parked sessions is over half of it before any
  other model. This is the first constraint that will bite, not RAM, since parked KV never sits in
  RAM by policy.

## Layer 2: placement

A tier is a tuple, never a memory name:

    tier = { buft, capacity, read_bw, write_bw }

Every field comes from a backend-neutral query: `ggml_backend_dev_buffer_type`,
`ggml_backend_dev_memory`, and `ggml_backend_dev_supports_buft`
(`ggml/include/ggml-backend.h:148-190`). `ggml_backend_dev_host_buffer_type` exists and is
deliberately not used for KV, by the policy above.

Residency is then arithmetic, not hardware detection:

    keep(t)    = bytes_touched / read_bw[t]
    unpark(t)  = bytes * (write_bw_disk + read_bw[t])

The policy keeps a session in the device tier while its expected remaining work costs more
than parking and restoring it. Moving the system to a different box means supplying that
box's numbers. No logic changes.

## Layer 3: acceleration

Per-backend, optional, and never assumed by layers 1 and 2. For KV the interesting rungs
have narrowed to the direct device to file path and its alignment requirements. The host
pointer import work (`docs/research/11-uma-zero-copy-findings.md`) applies to model weights,
not to KV, because KV is not resident on the host.

## Verification

A fault-injection harness, before any engine change:

- model the format and the reference session state in isolation
- run a scenario: prefill, decode, park, unpark, evict, defrag, checkpoint
- crash at every write boundary from 0 to the total number of writes
- recover, then assert every rule above and that the recovered state equals the committed
  state at the crash point

Pass criterion: for all crash points, recovery yields either the state at the last
`COMMIT` or the state at the previous checkpoint, and never anything else. This is the
only way to be sure about compaction, and it needs no GPU.

## Open questions

1. Page size. 16 tokens is the conventional default; SWA sizes in the tree are powers of
   two, so it may want to track `n_swa`. Needs measurement.
2. Prefix sharing. If sessions share a prefix, pages become shared and refcounts plus
   cross-session epochs are needed. Deferred until a use case exists.
3. Defrag trigger. The engine currently relies on cells being contiguous for batched
   decode; the store must expose a `MOVE` path that keeps an arena compaction atomic.
4. Whether the arena should be sized by active concurrency `K` rather than `n_seq_max`,
   and what `K` the server wants by default.
