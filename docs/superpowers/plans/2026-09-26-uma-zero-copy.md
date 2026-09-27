# UMA zero-copy weights (F part 1) Implementation Plan

**Outcome: dropped.** Task 1 was run and its premise fails on this driver: RADV refuses to import a
file backed host pointer through `VK_EXT_external_memory_host`, at every size, while an anonymous
pointer on the same device imports fine. Per the plan's own step 6, part 1 was dropped and the
enabling changes reverted. Findings in `docs/research/11-uma-zero-copy-findings.md`.
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Back the model weights with the GGUF's own pages on this UMA iGPU, removing the 13.33 s copy-based load and the second resident copy, without changing inference results.

**Architecture:** The import path already exists on both sides: `ggml_vk_buffer_from_host_ptr` (`ggml-vulkan-buffers.cpp:759`) imports host memory when the device has `external_memory_host` and the pointer and size are aligned, and the loader already builds device buffers from mapped ranges when a device advertises `buffer_from_host_ptr` (`src/llama-model.cpp:1446-1466`, the path Metal uses). Three things stop it here: the Vulkan device advertises `buffer_from_host_ptr = false`, the loader clears `use_mmap` for any device reporting `mmap_support = false` (true for iGPUs), and both the range alignment and the 4.29 GB `maxStorageBufferRange` are unmet. Tasks 1 and 2 fix the first three; Task 3 handles ranges larger than one allocation.

**Tech Stack:** C++17, `ggml-vulkan` (`ggml-vulkan.cpp`, `ggml-vulkan-buffers.cpp`), the loader (`src/llama-model.cpp`, `src/llama-model-loader.cpp`), CMake, `llama-completion`, `llama-bench`, `llama-gguf-split`.

**Spec:** `docs/superpowers/specs/2026-09-26-uma-zero-copy-prefill-design.md` (part 1)

## Global Constraints

- Commits need explicit human approval for each action (`AGENTS.md`). Every commit step is a checkpoint: ask, and if approval is not given, leave the tree uncommitted and report. Never push.
- ASCII only, no em dashes. Comments concise.
- Model: `/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`, 13.1 GiB.
- Scratch dir: `/home/herlanggays/.jcode/scratch/zerocopy/`. Disk has about 25 GB free, so a split model for Task 1 needs the original deleted or the split kept small; check free space before writing.
- Device facts: `minImportedHostPointerAlignment = 4096`, `maxStorageBufferRange = 4294967295`, `maxMemoryAllocationSize = 0xfffffffc`, memory types 3 and 4 are `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`.
- Correctness requirement: byte identical generation against the current build. The weights are the same bytes; only who owns the pages changes.
- Throughput is not expected to change. The gates are load time, resident set, and behaviour under memory pressure.

---

### Task 1: Enable the import path and prove it with a split model

The 13.1 GiB model exceeds `maxStorageBufferRange`, so a single-file import cannot work until Task 3. This task enables everything else and proves the whole premise on a model split into shards below 4 GB.

**Files:**
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (device props at line 15219)
- Modify: `src/llama-model.cpp` (the AUTO resolve of `use_mmap`, around line 1111)
- Modify: `src/llama-model-loader.cpp` (align the mapped range before binding, and fall back instead of throwing)

**Interfaces:**
- Consumes: `device->external_memory_host` and `device->min_imported_host_pointer_alignment` (already populated by the Vulkan backend).
- Produces: a device that advertises `buffer_from_host_ptr` on capable hardware, and a loader that keeps mmap on and can fall back.

- [ ] **Step 1: Advertise the capability**

In `ggml-vulkan.cpp`, in the device props:

```cpp
        /* .buffer_from_host_ptr  = */ ctx->device->external_memory_host,
```

- [ ] **Step 2: Do not clear mmap for an import-capable device**

In `src/llama-model.cpp`, the loop that clears `use_mmap` when a device lacks `mmap_support` should keep mmap when the same device can bind host pointers, because that is precisely the zero-copy case:

```cpp
    if (ml.use_mmap && params.load_mode == LLAMA_LOAD_MODE_AUTO) {
        for (const auto & dev : devices) {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev.dev, &props);
            if (!props.caps.mmap_support && !props.caps.buffer_from_host_ptr) {
                ml.use_mmap = false;
                break;
            }
        }
    }
```

- [ ] **Step 3: Align the range and fall back instead of throwing**

`ggml_backend_vk_device_buffer_from_host_ptr` rejects unaligned pointers and sizes, and `get_mapping_range` produces tensor offsets, which are 32-byte aligned at best (`src/llama-model-loader.cpp:1449-1464`). Two changes in the loader's binding block (`src/llama-model.cpp:1446-1466`):

```cpp
                // the import needs a page aligned pointer and size
                const size_t align = 4096;
                size_t first_a = first & ~(align - 1);
                size_t last_a  = (last + align - 1) & ~(align - 1);
                if (last_a > ml.mappings.at(idx)->size()) {
                    last_a = ml.mappings.at(idx)->size();
                }
```

and replace the throw on a null buffer with a fall back to the ordinary allocation path for that buft, because a driver may refuse the import for its own reasons and that must not be fatal:

```cpp
                ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, (char *) addr + first_a, last_a - first_a, max_size);
                if (buf == nullptr) {
                    LLAMA_LOG_WARN("%s: host pointer buffer unavailable for %s, falling back to a copy\n", __func__, ggml_backend_buft_name(buft));
                    bufs.clear();
                    break; // leave the block and use the normal allocation below
                }
```

Adjust the surrounding control flow so the fallback actually reaches the ordinary allocation, rather than leaving the loop with an empty buffer list.

- [ ] **Step 4: Build and check the flags**

```bash
cd /home/herlanggays/RISET/llama.cpp
cmake --build build-vk -j8 2>&1 | tail -3
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
./build-vk/bin/llama-completion -m "$M" -p "hi" -n 1 -c 64 -ngl 99 -v 2>&1 | grep -iE "load_mode|host pointer|falling back|buffer size" | head -8
```

Expected: `load_mode = mmap` now, and either host pointer buffers are used or the warning explains the fallback. If the single 13.1 GiB file fails to import (expected, range limit), the warning appears and the run still completes: that is the fallback working.

- [ ] **Step 5: Split the model so each shard fits one range**

```bash
cd /home/herlanggays/RISET/llama.cpp
D=/home/herlanggays/.jcode/scratch/zerocopy
mkdir -p "$D"
df -h / | tail -1
./build-cpu/bin/llama-gguf-split --split-max-size 3800M -m "/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf" "$D/qwen-split"
ls -la "$D" | head -8
```

Expected: four or five shards, each below 3.8 GB, and enough free space (if not, skip to Task 3 and do the chunking first, then come back). Note the shard sizes for Task 3's chunk sizing.

- [ ] **Step 6: Prove the premise**

```bash
cd /home/herlanggays/RISET/llama.cpp
D=/home/herlanggays/.jcode/scratch/zerocopy
S=$(ls "$D"/qwen-split-00001-of-*.gguf)
time ./build-vk/bin/llama-completion -m "$S" -p "hi" -n 1 -c 64 -ngl 99 -v 2>&1 | grep -iE "load_mode|host pointer|falling back|KV buffer size" | head -6
```

Expected: `load_mode = mmap`, no fallback warning, and a load time near 2 s against the measured 13.33 s. Measure the resident set with the same technique used before (read `/proc/<pid>/smaps_rollup` while the process loads) and compare against the copy path. If the import is refused for the shards too, stop and report: the premise fails on this driver and part 1 is dropped.

- [ ] **Step 7: Confirm output is unchanged**

```bash
cd /home/herlanggays/RISET/llama.cpp
D=/home/herlanggays/.jcode/scratch/zerocopy
S=$(ls "$D"/qwen-split-00001-of-*.gguf)
./build-vk/bin/llama-completion -m "$S" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 2048 -ngl 99 > "$D/out-zerocopy.txt" 2>&1
diff <(grep -v "^0\." "$D/out-zerocopy.txt") <(grep -v "^0\." /home/herlanggays/.jcode/scratch/rs-view/gen-35b-vk.txt) | head -5
```

Expected: identical text to the recorded copy-path baseline (the same prompt, seed 1, temp 0, 32 tokens). A difference means the tensors are not backed by the same bytes, which is the one way this change can corrupt results.

- [ ] **Step 8: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add ggml/src/ggml-vulkan/ggml-vulkan.cpp src/llama-model.cpp
git commit -m "vulkan : bind mapped weights as host pointer buffers on UMA devices"
```

---

### Task 2: Ranges larger than one allocation

Needed for the single-file model, and for any model whose tensor range exceeds `maxStorageBufferRange` (4.29 GB).

**Files:**
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp` (`ggml_vk_buffer_from_host_ptr` at line 759)
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (`ggml_backend_vk_device_buffer_from_host_ptr` at line 15981)
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan-common.h` (the `vk_buffer` shape, if it must hold several allocations)

**Interfaces:**
- Consumes: Task 1's enabling changes.
- Produces: a host-pointer buffer that spans several Vulkan allocations, so a 13.1 GiB file binds in one call.

- [ ] **Step 1: Decide the shape with a measurement, not a guess**

Check how a tensor's buffer is resolved to a `vk_buffer` in the compute path (find the function that maps `tensor->buffer` to the buffer used for descriptor binding). If that resolution goes through `ggml_backend_buffer_get_context(buffer)` and a single `bufctx->buffer`, the change is to make that struct hold a list of allocations with base offsets and have the resolution return the piece containing the tensor's offset, translating the offset accordingly. Record the number of call sites in the commit message or the results doc, because it is the cost estimate for a reviewer.

- [ ] **Step 2: Implement the split**

In `ggml_vk_buffer_from_host_ptr`, when `size > device->max_buffer_size`, create several imported allocations, each aligned start and length, at most `max_buffer_size` bytes, covering the requested range. Do not change the alignment preconditions: the whole range is 4096-aligned, and each chunk boundary is chosen as a multiple of 4096, so every piece is aligned.

- [ ] **Step 3: Build and run the single-file model**

```bash
cd /home/herlanggays/RISET/llama.cpp
cmake --build build-vk -j8 2>&1 | tail -3
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
./build-vk/bin/llama-completion -m "$M" -p "hi" -n 1 -c 64 -ngl 99 -v 2>&1 | grep -iE "load_mode|host pointer|falling back" | head -6
```

Expected: `load_mode = mmap`, no fallback, load near 2 s. If the fallback appears, the chunking is wrong; the warning names the buffer type.

- [ ] **Step 4: Prove the fallback still works**

Force the failure deliberately, for example by temporarily lowering the chunk limit in the import so a tensor straddles a boundary, and confirm the run completes through the copy path with a warning rather than crashing. Revert the temporary change afterwards and confirm the tree is clean.

- [ ] **Step 5: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp ggml/src/ggml-vulkan/ggml-vulkan.cpp ggml/src/ggml-vulkan/ggml-vulkan-common.h
git commit -m "vulkan : split host pointer imports across allocations"
```

---

### Task 3: Stop double-caching on the paths that still copy

**Files:**
- Modify: `src/llama-model-loader.cpp` (`load_all_data`, the non-mmap branch near line 1680)

**Interfaces:**
- Consumes: nothing from earlier tasks; this one stands alone.
- Produces: a copy-based load that does not hold the file in page cache as well as the copy.

- [ ] **Step 1: Drop the pages after copying each tensor**

In the branch that reads a tensor into a buffer, after the read completes:

```cpp
                // the copy is owned by the backend now, do not keep the file pages too
                posix_fadvise(file->file_id(), weight->offs, n_size, POSIX_FADV_DONTNEED);
```

Guard it for POSIX and keep it inside the existing `use_mmap == false` path. The `llama_file` exposes the descriptor; if it does not, the call site can open the file separately or the file class gains an accessor, whichever fits the surrounding code.

- [ ] **Step 2: Build and measure on the CPU build, which always copies**

```bash
cd /home/herlanggays/RISET/llama.cpp
cmake --build build-cpu -j8 2>&1 | tail -3
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
./build-cpu/bin/llama-completion -m "$M" -p "hi" -n 1 -c 64 -lm none -v 2>&1 | grep -iE "load_mode|buffer size" | head -4
```

Expected: the load still completes, and the resident set after load is about the model size rather than about twice it. Measure with `/proc/<pid>/smaps_rollup` during the load, as in Task 1.

- [ ] **Step 3: Confirm decode speed is not harmed**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
./build-cpu/bin/llama-bench -m "$M" -r 2 -p 0 -n 64 -t 8 -lm none 2>&1 | grep tg64
```

Expected: within noise of the 15.71 t/s recorded for `-lm none` in `docs/research/00-baseline-stage1.md`. Dropping pages must not slow decode, because the tensors are already in backend memory.

- [ ] **Step 4: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add src/llama-model-loader.cpp
git commit -m "loader : drop file pages after copying a tensor"
```

---

### Task 4: Gates and record

**Files:**
- Create: `docs/research/12-zero-copy-results.md`
- Modify: `QWEN_ONLY.md` (verification log)

**Interfaces:**
- Consumes: Tasks 1 to 3.
- Produces: the measured verdict.

- [ ] **Step 1: Load time and memory**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/zerocopy
mkdir -p "$D"
# read the model into page cache first so the comparison is warm
cat "$M" > /dev/null 2>/dev/null || true
for i in 1 2; do
  /usr/bin/time -v ./build-vk/bin/llama-bench -m "$M" -p 0 -n 1 -r 1 -t 8 -ngl 99 2>&1 | grep -E "wall clock|Maximum resident"
done
```

Expected: load time at or below 3 s and peak resident set at or below 16 GB, against the 13.33 s and about 26 GB recorded in the spec. If `/usr/bin/time` is unavailable on this box, read `/proc/<pid>/smaps_rollup` from a background run instead, as in Task 1.

- [ ] **Step 2: Correctness and throughput**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
./build-vk/bin/llama-bench -m "$M" -r 3 -p 256 -n 128 -t 8 -ngl 99 2>&1 | grep -E "pp256|tg128"
```

Expected: pp256 near 202 t/s and tg128 near 22.5 t/s, within noise. A material regression means the imported memory is slower than an ordinary allocation for the GPU to read, which would be a driver-level finding worth recording.

- [ ] **Step 3: Decode under memory pressure**

Fill page cache with unrelated data so the model does not sit resident, then measure decode with the import and with the copy path, and compare against the 15.77 to 16.06 t/s spread measured in `docs/research/00-baseline-stage1.md`. Record both numbers.

- [ ] **Step 4: Write the results and update the log**

Create `docs/research/12-zero-copy-results.md` with the load times, resident sets, the driver's `VK_EXT_memory_budget` reading and the loader's own accounting, the byte identical check, and the throughput and pressure numbers. State plainly whether the gates passed. Add the corresponding entry to the verification log in `QWEN_ONLY.md`.

- [ ] **Step 5: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add docs/research/12-zero-copy-results.md QWEN_ONLY.md
git commit -m "docs: record the zero-copy weight results"
```

---

## Self-Review

**Spec coverage:** the spec's part 1 requirements map to Task 1 (enable and prove), Task 2 (range limit and the fallback), Task 3 (the page-cache adjunct), Task 4 (the gates, including the memory-pressure run and the recording of both memory accountings). The spec's risk about mapping lifetime is addressed by not unmapping imported ranges: Task 1 keeps `use_mmap` on, and the loader's `unmap_weight` is only reached for tensors it copied.

**Placeholder scan:** no TBD. Task 2 step 1 requires counting the resolution call sites before editing, which is a measurement with a stated output, not a deferred design decision.

**Type consistency:** `buffer_from_host_ptr`, `external_memory_host`, `min_imported_host_pointer_alignment`, `max_buffer_size` and `get_mapping_range` are the names already in the tree at the cited lines, and Task 1 uses them without inventing new ones.

**Known gap, stated rather than hidden:** Task 2's exact shape depends on how many places resolve a tensor's buffer to a `vk_buffer`, which is counted in its first step. If the count is large, the honest alternative is the split-model workaround that Task 1 proves, plus a note that a single-file model needs the plumbing; the plan says so rather than pretending the change is small.
