# UMA zero-copy weights (F part 1): findings

Date: 2026-09-27
Machine: Ryzen 7 6800H, Radeon 680M (RADV REMBRANDT), 27.1 GiB, Mesa/amdvlk as packaged.
Tree: `qwen-only-backends` at 4f5831ba2 plus the plan E change.
Plan: `docs/superpowers/plans/2026-09-26-uma-zero-copy.md` (Task 1).
Spec: `docs/superpowers/specs/2026-09-26-uma-zero-copy-prefill-design.md`.

**Verdict: the premise fails on this driver. RADV refuses to import a file backed host pointer, at
every size, while importing an anonymous pointer on the same device succeeds. Per the plan's Task 1
step 6, part 1 is dropped and the enabling changes were reverted.**

## What the design needs and what the driver does

The plan binds the GGUF's own mapped pages to the Vulkan device with `VK_EXT_external_memory_host`, so
the weights exist once in RAM and the GPU reads those pages instead of a private copy.

Four changes were made to enable it, and all four worked as intended mechanically:

| change | file | outcome |
| --- | --- | --- |
| advertise `buffer_from_host_ptr` when the device has `external_memory_host` | `ggml-vulkan.cpp` | loader enters the import path |
| keep `load_mode = mmap` when the device can bind host pointers | `llama-model.cpp` | `load_mode = mmap` confirmed for the iGPU |
| align the imported range to the device's alignment | `llama-model.cpp` | needed, see below |
| fall back to a copy instead of throwing when the import is refused | `llama-model.cpp` | needed, the import is always refused here |

Then the import is refused, and the run falls back to the copy path, which is why those first two
changes are a regression on this box (they add the mmap and the failed import attempt to a load that
used to be a plain copy) and were reverted. The diagnostics added while chasing it are kept, they cost
nothing and they explain a fallback that otherwise happens silently.

## Evidence chain

1. The plan's alignment snippet is wrong in a way that is invisible. It clamps the range end to the
   mapping size after rounding up, and the clamped value is no longer a multiple of the alignment, so
   `ggml_vk_buffer_from_host_ptr` refuses the range in its own size check and returns nullptr with no
   message. With the clamp re-derived to keep the alignment the import gets as far as the driver.

2. The driver then returns `VK_ERROR_INVALID_EXTERNAL_HANDLE` from `vkAllocateMemory`, for every size
   from 4 KiB to 3.47 GiB, on a shard small enough to fit one allocation (3.47 GiB against the
   device's 4.00 GiB `maxStorageBufferRange`). Size is not the problem.

3. A fresh anonymous 1 MiB buffer, imported through the same code path, on the same device, with the
   same memory type, **succeeds**. So the extension is present and functional here. What RADV rejects
   is a pointer that belongs to a file mapping, which is exactly the pointer the design needs.

4. Asking for a memory type that includes `DEVICE_LOCAL` finds no candidate at all: the memory types
   the driver reports for this host pointer and the buffer's requirements do not intersect with the
   device local types.

5. Two defects made this expensive to find and are worth keeping fixed:
   - the import branch only tries the first rung of the memory property ladder
     (`*req_flags_list.begin()`), so the caller's fallback rungs never apply to an import;
   - the allocation failure was swallowed by an empty catch block, so the driver's own reason
     (`ErrorInvalidExternalHandle`) was discarded and the code reported "No suitable memory type
     found" instead. Both now log.

## Measured

| path | load time | output |
| --- | --- | --- |
| copy path, `load_mode = none`, single 13.1 GiB file | 22.9 to 26.5 s | baseline |
| import enabled, single file (over the range limit) | 26.8 s | identical |
| import enabled, 3.5 GiB shard (within the range limit) | 26.8 to 63.3 s | identical |

All three produce byte identical text for the same prompt, seed and token count against the recorded
baseline, so nothing about correctness was at risk; the import path is simply slower here because it
mmaps and then copies anyway.

## Options left

- Drop part 1, which is what the plan says to do and what was done.
- A different design that the same evidence supports: load the weights once into an anonymous buffer
  and import *that*. RADV accepts anonymous host pointers, so this removes the second resident copy
  without needing file backed imports. It does not remove the load time, so the spec's 13.33 s to 2 s
  target would not be met by it. This is not in the plan and needs a decision before it is built.
- Part 1 could be revisited on a driver that imports file backed pointers. Nothing in this tree would
  need to change except re-enabling the two reverted lines.

## Scratch

`/home/herlanggays/.jcode/scratch/zerocopy/` holds the logs and the probe script. The four model
shards the task needed (`qwen-split-*.gguf`, 13.1 GiB) were deleted afterwards to give the disk its
headroom back; `llama-gguf-split --split-max-size 3800M` reproduces them from the model.
