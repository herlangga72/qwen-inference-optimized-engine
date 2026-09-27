# Session KV store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a server park and restore per-session KV state so more sessions can be held than fit in device memory, with a format that is provably correct under crash and compaction, and that is portable to other machines.

**Architecture:** Three layers. Layer 1 is a pure-bytes format (two files, append-only status log, checkpoint root) and carries all the correctness argument. Layer 2 is a placement policy built only from `ggml_backend_dev_*` queries, so it never names a device. Layer 3 is per-backend acceleration (host pointer import, async copies) and is optional. Layer 1 and 2 must build and pass with no GPU present.

**Tech Stack:** C++17 (`llama-kv-cache`, `llama-context`, `llama-io`), Python 3 for the protocol harness, CMake, the fork's existing test binaries.

**Spec:** `docs/superpowers/specs/2026-09-27-session-kv-store-design.md`

## Global Constraints

- Commits need explicit human approval for each action (`AGENTS.md`). Every commit step is a checkpoint: ask, and if approval is not given, leave the tree uncommitted and report. Never push.
- Do not add files under `tests/`. Reuse the existing binaries.
- ASCII only, no em dashes. Comments concise, simple wording, no restating the code.
- The format is a strict superset of `LLAMA_STATE_SEQ` v3 (`include/llama.h:48-49`). Today's `server_prompt_cache` and `--slot-save-path` must keep working unchanged throughout.
- Resident KV is never in host RAM. Device memory and disk only. Run `--cache-ram 0`, and all store I/O uses `O_DIRECT` with aligned buffers. The only host memory in the path is one reused staging page, O(1) in session count and length. See `docs/research/16-disk-tier-results.md` for why `O_DIRECT` is a correctness requirement and not just an optimisation.
- Layer 1 and layer 2 must not include a backend header. If a change needs `ggml-vulkan.h` or a device name, it is layer 3.
- Builds: `build-cpu` and `build-vk`, Release, `GGML_NATIVE=ON`.
- Models: `/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf` (default), `/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` (scale check).
- Scratch dir: `/home/herlanggays/.jcode/scratch/kvstore/`
- Gate: the harness must pass at every crash point before any engine code is written.

---

### Task 1: Protocol harness with fault injection

**Files:**
- Create: `scripts/research/kvstore/sim.py`
- Create: `scripts/research/kvstore/scenarios.py`
- Create: `scripts/research/kvstore/README.md`

**Interfaces:**
- Consumes: nothing. Pure Python, no repo imports.
- Produces: `sim.Disk` (a byte-addressed simulated file with a crash injection point), `sim.Budget`, `sim.Writer`, `sim.recover(pages, status) -> (state, epoch, checkpoint)`, `scenarios.run_matrix(plan, label) -> report`.

- [x] **Step 1: Simulated disk with crash injection**

A byte-addressed file object whose atomic writes draw from a shared `Budget`. Below the simulated 4096 byte sector it can lose the whole write, land a prefix (torn), or skip it, so torn tails are covered and not only whole-record loss.

- [x] **Step 2: Implement the format**

`kv.status` framing with `len`, `crc32`, payload; the ops from the spec; `kv.pages` header and slots; `CHECKPOINT`; recovery per the spec's algorithm.

- [x] **Step 3: Reference model**

An in-memory expected state: per sequence, the ordered cell list, plus per-sequence recurrent blobs. Independent of the store so the comparison is a real check.

- [x] **Step 4: Scenarios**

`prefill`, `decode`, `park`, `unpark`, `seq_rm`, `defrag_move`, `checkpoint`, `log_compaction`, and a mixed sequence of all of them.

- [x] **Step 5: Crash matrix**

For every scenario, for every crash point `k` in `0..n_writes`, crash, recover, and assert:
- recovery returns the state at the last `COMMIT` or at the previous checkpoint, nothing else
- every live `(slot, gen)` resolves with a matching page crc
- per-sequence positions are strictly increasing and in range
- per-sequence recurrent blob epoch equals the attention epoch
- the semantic check fails closed when a page crc is corrupted deliberately

- [x] **Step 6: Run and record**

Pass criterion: zero invariant violations across the whole matrix. Record the matrix size and the result in `docs/research/14-session-kv-store-format-results.md`.

---

### Task 2: Superset compatibility

**Files:**
- Modify: `scripts/research/kvstore/sim.py` (add a v3 codec)
- Create: `scripts/research/kvstore/v3_compat.py`

**Interfaces:**
- Consumes: `LLAMA_STATE_SEQ` v3 layout: `n_tokens`, `tokens[]`, per-cell `(pos, n_seq_id, ext, seq_ids)`, `v_trans`, `n_layer`, per-layer K rows, per-layer V rows.
- Produces: a proof that a v3 blob is exactly one block of one sequence, byte for byte.

- [ ] **Step 1: Encode a v3 blob**
- [ ] **Step 2: Decode it through the store and re-encode**
- [ ] **Step 3: Assert byte equality with the original v3 blob**

Pass criterion: round-trip is byte identical for single-layer, multi-layer, `ext` present and absent, and multi-seq cells.

---

### Task 3: Layer 2 tier model (no behaviour change)

**Files:**
- Modify: `src/llama-kv-cache.h` / `.cpp` (new internal struct only)

**Interfaces:**
- Consumes: `ggml_backend_dev_buffer_type`, `ggml_backend_dev_host_buffer_type`, `ggml_backend_dev_memory`, `ggml_backend_dev_supports_buft`, `ggml_backend_dev_get_props` (`ggml/include/ggml-backend.h:148-190`).
- Produces: an internal `kv_tier` list and a `kv_placement` policy object, unused by default.

- [ ] **Step 1: Enumerate tiers per device at context creation**
- [ ] **Step 2: Log the tiers at debug level**
- [ ] **Step 3: Add the cost comparison as a pure function**
- [ ] **Step 4: Assert the default path is byte-identical to pre-change**

Pass criterion: generated text identical to pre-change for 0.8B on CPU and Vulkan0, and the tier log shows the real capacities from `ggml_backend_dev_memory`.

---

### Task 4: Walking skeleton, park and unpark one session to disk

**Files:**
- Modify: `src/llama-io.cpp` / `src/llama-io.h` (an `O_DIRECT` file sink, aligned)
- Modify: `src/llama-kv-cache.cpp` (`state_write` / `state_read` paths)
- Modify: `src/llama-context.cpp` (expose park/unpark)
- Modify: `common/arg.cpp` (a store path flag, off by default)

**Interfaces:**
- Consumes: `llama_io_write_file` / `llama_io_read_file` (`src/llama-context.cpp:2672-2720`) as the shape to extend, plus `llama_state_seq_get_data_ext` / `set_data_ext`.
- Produces: park one sequence to a store file and restore it, with one reused aligned staging buffer and no host residency.

- [x] **Step 1: An `O_DIRECT` file sink with a reused aligned staging buffer**

Done: `llama_io_write_direct` / `llama_io_read_direct` in `src/llama-io.{h,cpp}`, with
`llama_io_crc32`. Framed mode for the status log, plain mode for the pages file. Verified by
`scripts/research/kvstore/direct_sink_check.cpp` and cross checked against the Python reader by
`cross_check.py`; see `docs/research/20-direct-io-sink-results.md`. The cross check found a real
bug: the sink split a logical write across a block boundary, which made the record unreadable.
A logical write now never straddles.
- [x] **Step 2: Park a sequence to disk, then unpark it**
- [x] **Step 3: Assert generation is identical to no park, and assert host RSS does not grow with context length**

Done: `llama_state_seq_save_file_direct` / `llama_state_seq_load_file_direct`, public and
mirroring the existing pair, but streaming the state device to file so it never lands in host
memory whole. Checked by `scripts/research/kvstore/park_disk_check.cpp`: the greedy
continuation after an unpark is identical to never parking, and the RSS delta is 0 KiB while
the state is 26.4 MiB. See `docs/research/21-park-unpark-disk-results.md`. Still open: no
policy drives the parking, one sequence only, and the unpark still needs the whole state to fit
in the arena.

The RSS assertion is the policy check, and it is the one that fails silently otherwise. Pass criterion: byte-identical generated text after unpark vs no park on 0.8B and 35B, and RSS growth bounded by the staging buffer plus page slop, not by session size.

---

### Task 5: Device to disk path, and the disk budget

**Files:**
- Create: `docs/research/17-device-to-disk-path-results.md`
- Create: `scripts/research/kvstore/device_to_disk.py`

**Interfaces:**
- Consumes: `ggml_backend_tensor_get` for the device to staging copy, `O_DIRECT` for the file.
- Produces: the end to end park and unpark cost, including the copy this tree has not yet measured.

- [x] **Step 1: Measure the device to staging copy for a KV shaped tensor**
- [x] **Step 2: Measure end to end park and unpark, and convert to a per session cost**
- [x] **Step 3: Measure the 35B session size per token, so the disk budget stops being extrapolated**
- [x] **Step 4: State the disk budget from the real numbers, given 25.7 GB free**

Pass criterion met for steps 1, 3 and 4 by direct measurement. Step 2 is a sum of measured
components, 2.7 ms device plus 48 ms disk, not one end to end run, and is labelled as such in
the results. The result that changed the design: session size is a fixed term plus a slope,
so parking has a floor set by the recurrent state.
