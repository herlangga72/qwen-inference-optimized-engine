# Stage 1 and 2 recheck on Vulkan (Radeon 680M)

Same machine and model as the CPU baseline. Build: `build-vk`, `GGML_VULKAN=ON`, Release,
`GGML_NATIVE=ON` (identical CPU flags to `build-cpu`).

```
ggml_vulkan: 0 = AMD Radeon 680M (RADV REMBRANDT) (radv) | uma: 1 | fp16: dot2 | bf16: 0 |
              fp4: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
Vulkan0: AMD Radeon 680M (RADV REMBRANDT) (28672 MiB, 27851 MiB free)
```

## Device caps that shape stage 1

`ggml/src/ggml-vulkan/ggml-vulkan.cpp`:

- `ggml_backend_vk_device_get_type` returns `GGML_BACKEND_DEVICE_TYPE_IGPU` for this device
  (`is_integrated_gpu` is derived from `VkPhysicalDeviceType::eIntegratedGpu`).
- `caps.mmap_support = !is_integrated_gpu` -> **false** here.
- `caps.host_buffer = true`, `caps.buffer_from_host_ptr = false`.
- The reported device memory is the whole system RAM, because it is UMA.

Consequence: `llama_model_base::load_tensors` walks the selected devices and clears `use_mmap`
for any device without mmap support, so an integrated GPU in the device list forces the AUTO load
mode to `none`. Verified in the verbose load log:

```
load_tensors: loading model tensors, this can take a while... (load_mode = none)
load_tensors: layer   0 assigned to device Vulkan0, is_swa = 0
... (all 40 trunk layers)
```

## Load time and memory

Page cache fully warm before each measurement, `-p 0 -n 1 -r 1 -t 8`, two runs each
(seconds, includes load plus one generated token):

| config | run 1 | run 2 |
| --- | --- | --- |
| CPU build, `-lm mmap` | 1.99 | 1.94 |
| CPU build, `-lm none` | 49.22 | 16.13 |
| Vulkan build, `-ngl 99` (AUTO -> none) | 14.13 | 13.33 |
| Vulkan build, `-ngl 99 -lm mmap` | 19.20 | 26.29 |

So the copy based load costs ~12 s more than mmap on this 13 GiB model, the GPU path is forced into
it, and forcing mmap back on the GPU path makes loading worse, not better. Decode throughput is
unaffected by the choice.

## Throughput

Reproducible baseline via the harness, `BUILD=build-vk BENCH_ARGS="-ngl 99"
scripts/research/bench-stage.sh MODEL advan-6800h-vulkan`, results in
`benches/advan-6800h-vulkan/advan-6800h-vulkan.md`:

| test | t/s |
| --- | --- |
| pp256, 8 threads | 202.41 +- 2.99 |
| tg128, 4 threads | 22.67 +- 0.12 |
| tg128, 8 threads | 22.46 +- 0.10 |
| tg128, 16 threads | 22.48 +- 0.06 |
| tg128, 8 threads, `-lm mmap` | 21.62 +- 0.98 |
| tg128, 8 threads, `-lm none` (default here) | 22.40 +- 0.13 |
| tg128, 8 threads, `-lm mlock` | 22.37 +- 0.05 |

Ad hoc sweep from the same session, for the configurations the harness does not cover:

| config | pp256 | tg128 |
| --- | --- | --- |
| CPU build, CPU compute | 61.27 | 15.26 to 15.88 |
| Vulkan build, `-ngl 0` | 79.24 | 14.04 |
| Vulkan build, `-ngl 20` | - | 15.92 |
| Vulkan build, `-ngl 99`, t8 | 198.93 | 21.76 |
| Vulkan build, `-ngl 99`, t16 | - | 21.70 |
| Vulkan build, `-ngl 99`, graph reuse off | - | 19.69 |

- Full offload is 1.4x the best CPU decode and 2.5x the same build's CPU-only prefill.
- Partial offload (20 of 40 layers) gives nothing, because the CPU layers become the bottleneck.
- Thread count is irrelevant with the GPU doing the work, and so is the load mode.
- `-ngl 0` on the Vulkan build is 11% slower than the CPU only build. With a Vulkan device in the
  list, `make_cpu_buft_list` puts the Vulkan host buffer type ahead of the CPU buffer type, so
  CPU-computed weights are allocated as Vulkan host visible memory. Worth keeping in mind, not a
  blocker for the GPU path.

## Stage 1 conclusion on the GPU: it is at the memory wall

Decode at `-ngl 99` is 22.46 t/s in the harness run, 21.76 t/s in the ad hoc run. At ~2.08 GB of
weights per token that is 46.7 and 45.2 GB/s against a ceiling measured at 46.81 GB/s in the same
session. That is 97% to 100% of the wall, so there is nothing left in layout, kernels or placement
for single stream decode on the GPU.

Cross-check with a different model, Qwen3.5-0.8B Q4_K_M (497 MiB, ~0.5 GB per token):

| config | tg256 | implied bandwidth |
| --- | --- | --- |
| CPU build, CPU compute | 69.38 | 34.7 GB/s |
| Vulkan build, `-ngl 99` | 87.51 | 43.8 GB/s |

Both backends reproduce their wall on a second model, so these are hardware limits and not
artifacts of this model's quant mix: CPU saturates at ~33 to 35 GB/s, the 680M at ~44 to 45 GB/s.

## Stage 2 recheck: graph reuse matters more on the GPU

| setting | tg128 | ms/token |
| --- | --- | --- |
| reuse on (default) | 21.76 | 45.96 |
| `LLAMA_GRAPH_REUSE_DISABLE=1` | 19.69 | 50.79 |

Build plus alloc plus reset costs 4.8 ms per token on the GPU path, 9.5% of the step, against
1.8 ms / 2.8% on the CPU. Stage 2 itself still has no headroom (reuse is already on), but the
constraint it imposes is stronger: on the GPU, any change that breaks `can_reuse` costs 9.5%.

## The opening for the MoE work

`llama-batched-bench`, all layers on the GPU, `-c 16384 -npp 512 -ntg 32`, 8 threads:

| B | S_TG t/s | S_PP t/s |
| --- | --- | --- |
| 1 | 21.19 | 241.33 |
| 2 | 30.35 | 241.46 |
| 4 | 37.40 | 241.71 |
| 8 | 39.47 | 242.36 |
| 16 | 34.76 | 241.34 |

Aggregate decode peaks at B=8 with 1.86x, then falls off. But the bytes say it should do better.
At B=8 a step reads ~1.72 GB dense plus ~2.55 GB of experts (about 57 distinct of 256 per layer),
so ~0.53 GB per token. At the 46.5 GB/s wall that is ~87 t/s, and prefill already proves the GPU
can aggregate 241 t/s. Measured is 39.47 t/s, that is 21 GB/s of 46.5, only 45% of the wall.

So the batched expert path is where the engine leaves half the bandwidth unused. That is stage 5
territory (MoE FFN: routing, `mul_mat_id` dispatch, grouped per expert GEMM at 1 to 2 rows per
expert) and it is the single largest gap found so far.

## Note on MTP

The loader reports the MTP block as unused and skips it:

```
W model has unused tensor blk.40.ffn_down_exps.weight (size = 115343360 bytes) -- ignoring
W model has unused tensor blk.40.nextn.eh_proj.weight (size = 8912896 bytes) -- ignoring
```

consistent with `load_mtp = false` by default. The amortization path is on disk and switched off.
