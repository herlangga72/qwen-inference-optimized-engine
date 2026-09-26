# Unified-memory iGPU: zero-copy weights first, then the prefill profile

Date: 2026-09-26
Status: approved design, not yet implemented
Scope: part 1, the weight memory path on the Vulkan backend for UMA devices; part 2, a prefill profile and the fused kernel it justifies. Ordered 1 then 2, each with its own plan.
Device: AMD Radeon 680M (RADV REMBRANDT), Mesa 26.2.2, Vulkan 1.4.354, integrated
Related: `docs/research/02-gpu-vulkan-recheck.md` (load modes), `docs/research/04-stage5-diagnosis.md` (op costs)

## Context

This machine has one memory. The Vulkan device reports `PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`,
`VK_EXT_external_memory_host`, and memory types 3 and 4 with `DEVICE_LOCAL | HOST_VISIBLE |
HOST_COHERENT`. There is no VRAM and no bus, so every "GPU write to RAM" is a DRAM write that the CPU
can read back directly.

Two heaps exist: 9.33 GiB without the device-local flag, and 18.67 GiB with it but a budget of
18.07 GiB. The whole model is 13.1 GiB, and today it is copied into the device-local heap.

What that costs, measured:

| measurement | value |
| --- | --- |
| `-ngl 99` load time, page cache warm | 13.33 s |
| CPU build mmap load, same conditions | 1.94 s |
| page cache residency for the file after other work | 48 to 57% |
| resident set implied by a copy-based load | about 26 GB on a 27 GB machine |

The cause is a policy: `ggml-vulkan` reports `mmap_support = !is_integrated_gpu`, so on an iGPU the
loader clears `use_mmap` and reads every tensor into a Vulkan buffer. The device can instead be given
the file's own pages.

Prefill is a separate problem. `pp256` at `-ngl 99` is 202 t/s: 1.27 s for a 256-token pass, which
reads essentially all 13.8 GB of weights, so 10.9 GB/s against a 68 GB/s ceiling, and about
614 GFLOP in that time, 476 GFLOPS against roughly 3.4 TFLOPS of peak. Prefill runs at 16% of
bandwidth and 14% of FLOPs, so neither is the limit, and any kernel work has to be aimed at what
actually holds it.

Relevant capability facts, from vulkaninfo:

- `minImportedHostPointerAlignment = 0x1000` (4096), and `mmap` bases are page aligned.
- `bufferImageGranularity = 1`, `minStorageBufferOffsetAlignment = 4`.
- `maxStorageBufferRange = 4294967295` (4.29 GB), so a 13.1 GiB import needs several allocations.
- shared memory 64 KB per workgroup, 1024 invocations, subgroup size 64 by default with
  `subgroupSizeControl` and a minimum of 32, 4x8-bit packed dot product accelerated, fp16 and 8/16-bit
  storage, float atomics, and no cooperative matrix support.

## Part 1: zero-copy weights

**Requirement.** On a device that reports the import capability and a host-visible device-local
memory type, the model tensors must be backed by the file's own pages, with no copy, no second
resident copy, and no unmapping of ranges that back live tensors.

**Design.**

- The loader asks the device for an import-capable host buffer type. `ggml-vulkan` gains such a
  buffer type: it imports a range of the model mapping with `vkImportMemoryHostPointerEXT` and binds
  it as a storage buffer. Weight tensors are read-only, so no coherent writeback concerns apply.
- Import granularity: the mapping base is 4096-aligned by construction, and the file is 13.1 GiB, so
  the loader splits it into allocations below `maxStorageBufferRange`, **choosing each boundary so it
  falls between tensors**. That removes straddling entirely for this model, where the largest single
  tensor is 417 MB. A tensor larger than one chunk still falls back to a copy, and that path is
  reachable only in theory here, so the plan must test it with a forced small chunk size.
- Mapping lifetime: today the loader unmaps ranges whose tensors were copied. With imports, the
  ranges backing imported buffers must stay mapped for the model's lifetime. The loader's unmap and
  `mmap` bookkeeping must know which ranges are imported.
- Loader path: with an import-capable device, `use_mmap` must stay on and the tensors must be bound
  rather than copied. This is the same code path that today clears `use_mmap` for iGPUs.
- Adjunct, independent and cheap: on paths that still copy, drop the file's page cache behind each
  tensor as it is copied, so a copy-based load stops holding two copies. Applies to the CPU build and
  to non-import-capable devices as well.

**What this buys, and what it does not.** Load time should fall from 13.33 s toward the disk-read
floor near 2 s, and the resident set should halve from about 26 GB to about 14 GB, because the model
stops existing twice. Decode and prefill throughput are expected to be unchanged, since the data was
already in DRAM; the benefit under memory pressure is that the file no longer competes with a
duplicate of itself, so residency should stop falling into the 50s.

What happens to the 18.07 GiB device-local budget is measured rather than assumed. Imported host
memory is not an ordinary device allocation, so whether RADV also reduces its reported budget usage
by the 13 GB depends on the driver, and the loader's own accounting and the driver's
`VK_EXT_memory_budget` numbers are both recorded after the change. If the budget is freed, long
contexts gain headroom; if it is not, the memory is still de-duplicated and the load time still
collapses, which are the two things this part is for.

**Gates.** Load time at or below 3 s on a warm page cache; resident set at or below 16 GB after load;
the loader's own report of device-allocated weight memory down by about 13 GB, with the driver's
`VK_EXT_memory_budget` reading recorded alongside it; byte identical generation against the
current build on both backends; decode and prefill within noise of the current numbers; and a decode
run after deliberately filling page cache with other data, where the no-duplicate arrangement should
now hold up better than the measured 48 to 57% residency case.

## Part 2: prefill, profiled and targeted

**The profile.** The prefill graph was exported at a 512-token batch and replayed per op on Vulkan0.
Per-op time, and what each becomes across the 40 layers, against the 2.53 s a 512-token pass takes
at 202 t/s:

| op | per call | across layers | share |
| --- | --- | --- | --- |
| FLASH_ATTN_EXT, 10 full-attention layers | 26.9 ms | 269 ms | 10.6% |
| attn_qkv projection, 30 delta layers | 6.30 ms | 189 ms | 7.5% |
| MoE gate projection, 30 delta layers | 6.07 ms | 182 ms | 7.2% |
| GATED_DELTA_NET, 30 layers | 4.32 ms | 130 ms | 5.1% |
| CONCAT of the conv input, 30 layers | 4.16 ms | 125 ms | 4.9% |
| ssm_out projection, 30 layers | 3.30 ms | 99 ms | 3.9% |
| attn_gate, 30 layers | 3.12 ms | 94 ms | 3.7% |
| MoE down projection, 30 layers | 2.56 ms | 77 ms | 3.0% |
| attention q projection, 10 layers | 5.16 ms | 52 ms | 2.1% |
| MoE weighted-sum multiply, 30 layers | 1.51 ms | 45 ms | 1.8% |

Two readings of the profile change what part 2 is:

- The MoE expert path is about 20% of the pass once the gate projection is counted twice for gate and
  up, but at this batch size each expert already receives 16 rows and the kernels run at 41% of peak,
  1.4 TFLOPS. It is a real grouped GEMM, not per-pass overhead, so the earlier premise that fusing
  would win big is weaker than assumed.
- Attention is the worst-utilized thing on the device. One full-attention layer at 512 tokens is
  4.3 GFLOP in 26.9 ms, 160 GFLOPS, about 4.7% of the GPU's FP32 peak, and the KV it touches is only
  about 1 MB, so it is not byte-starved either. With its projections it is 21% of the pass sitting at
  a twentieth of what the hardware can do.

Two caveats recorded rather than hidden. The `result_output` row in the raw profile reads 190 ms
because the export reserves the prefill graph with every token as an output; a real prefill computes
one logits row, which the same profile shows at 8 ms, so that row is an artifact of the instrument.
And the MoE gate and up projections are identical in shape and type, so one profiled row represents
both.

**Workstream A, attention, first.** Two parts:

- **Flash attention.** Sweep the configuration the kernel family already offers on the real shapes:
  scalar versus dot2 accumulation, split-k on and off, workgroup shape, and subgroup size, which the
  device defaults to 64 while the shaders assume 32, with `subgroupSizeControl` available to pin
  either. Fix the selection rule that currently reaches the slow configuration. Target: halve the
  269 ms, about 5% of the pass.
- **Attention projections.** `attn_qkv`, `attn_q` and `attn_output`, 268 ms together across their
  layers, are plain quantized GEMMs at this batch size. Same tuning question, no new kernel.

**Workstream B, MoE expert fusion, second.** One dispatch per layer takes the layer's rows grouped by
expert, keeps an expert's weights in the 64 KB of LDS across its 16 rows, applies SiLU between gate
and up, and writes the down projection once, replacing three projections that each re-read and
re-dispatch. The bar: the current kernels already run at 41% of peak, so the fused version is kept
only if it clearly beats the measured 513 ms MoE segment; otherwise it is dropped rather than kept
for elegance.

**Small ops, noted not planned.** A bare `CONCAT` costs 125 ms across the layers and `RMS_NORM`
about 1 ms per call. They are worth folding into whichever workstream touches those layers and not
worth their own plan.

**Gate.** `pp256` and `pp512` on Vulkan0 against the current 202 t/s, measured in the same session,
with the profile as the justification for each change. A workstream that measures no better than the
baseline is recorded and reverted, as stage 5 did for the MoE expert kernels.

## Risks

- **Import support in practice.** RADV advertises `VK_EXT_external_memory_host`, but the combination
  of a file-backed mapping, coherence, and this driver is unproven here. This is the first gate, and
  a failure stops part 1 with the finding recorded rather than falling back silently to copies.
- **Chunk boundaries.** The no-straddle property depends on choosing boundaries between tensors. The
  plan must compute them from the tensor table and assert that no tensor crosses one, with the copy
  fallback exercised by a forced tiny chunk size.
- **Mapping lifetime and unmapping.** The loader currently unmaps ranges it has copied. Getting this
  wrong would hand the GPU freed pages, which fails loudly at best and silently at worst. The plan
  must show which ranges are imported and must test unload and reload.
- **Shared memory pressure.** With the model out of the device heap, the KV cache and compute buffers
  have more room, which changes how far a long context gets before allocation fails. That is a gain,
  but it moves the failure point rather than removing it, and the result must be recorded.
- **Prefill may have no kernel left to win.** The profile may show the current shape is already near
  the iGPU's structural ceiling for dequant-and-dot. That is a legitimate, useful result.

## Out of scope

- Discrete-GPU staging and PCIe transfer paths. This spec is written for UMA and says so.
- The third option from the approach list, fixing the `use_mmap` policy for CPU-resident weights
  under partial offload, which measured 11% on `-ngl 0`. It is worth its own small change; it is not
  this spec.
- Requantizing any weight, which the user ruled out.
- The already-planned sub-projects: the delta-net output projection (A), the recurrent state view
  (B phase 1), MTP speculation (C), and the planar3 KV cache (E).
- Other backends.
