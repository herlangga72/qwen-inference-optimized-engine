# Stage 1 evidence: load and memory layout

Machine: AMD Ryzen 7 6800H (8C/16T), 27 GiB RAM, Radeon 680M iGPU, f2fs root, 16 MB L3 per CCX.
Model: `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`, 13.09 GiB, 35.51 B params, 41 blocks (40 trunk + 1 MTP),
256 experts, top-8, n_embd 2048, n_ff_exp 512, vocab 248320.
Build: `build-cpu`, release, GGML_NATIVE=ON, commit 04e10983e.

## Baseline

| test | result |
| --- | --- |
| pp256, 8 threads | 77.69 t/s |
| tg64, 8 threads | 15.03 t/s |
| tg128, 8 threads | 15.72 t/s |
| tg128, 16 threads | 11.28 t/s |
| load time, page cache warm | ~1.95 s (measured alone, see below; the first table row above was a whole llama-bench run) |
| model load mode resolved | mmap (MAP_SHARED, r--s, single 13096 MB region) |
| page-cache residency (mincore) | 100% of 3434879 pages |
| measured DRAM read ceiling (3 GiB array, 8T) | 46.88 GB/s (46.54 in the harness rerun) |
| measured DRAM read ceiling (3 GiB array, 16T) | 44.29 GB/s (44.04 in the harness rerun) |

Load mode comparison (tg128, 8 threads, -r 3):

| load mode | t/s |
| --- | --- |
| mmap | 15.72 +- 0.12 |
| none (copy into buffers) | 14.85 +- 0.78 |
| mlock | 15.06 +- 0.10 |

## Byte accounting for one decode token

| part | bytes/token | share |
| --- | --- | --- |
| dense attention + delta-net + shared experts + routers (40 layers) | ~1.72 GB | 83% |
| routed experts (8 of 256, 40 layers) | ~363 MB | 17% |
| total | ~2.08 GB | 100% |

Biggest single tensors (not per expert):

| tensor | type | shape | bytes |
| --- | --- | --- | --- |
| output.weight | Q6_K | 2048 x 248320 | 417 MB |
| token_embd.weight | Q6_K | 2048 x 248320 | 417 MB (gather, ~0 traffic) |
| blk.N.attn_qkv.weight | Q6_K | 2048 x 8192 | 13.76 MB x 30 |
| blk.N.ffn_down_exps.weight | IQ3_S | 512 x 2048 x 256 | 115.34 MB x 39 |
| blk.N.ffn_gate_exps.weight | IQ2_S | 2048 x 512 x 256 | 85.98 MB x 39 |
| blk.N.ffn_up_exps.weight | IQ2_S | 2048 x 512 x 256 | 85.98 MB x 39 |

Achieved weight bandwidth: 15.72 t/s x 2.08 GB = 32.7 GB/s = 70% of the 46.88 GB/s ceiling.
Ceiling at batch 1 with these weights: 46.88 / 2.08 = 22.5 t/s.

## Prefill is compute bound

pp256 = 77.69 t/s means 256 tokens in 3.29 s. Prefill touches essentially all 13.8 GB of weights,
so effective weight traffic is ~4.2 GB/s, 9% of the ceiling. Same window is ~600 GFLOP, ~187 GFLOPS,
about 35% of the AVX2 FMA peak of this CPU (8 cores x 2 FMA x 8 lanes x 2 x 4.2 GHz ~ 537 GFLOPS).

## Kernel level evidence (test-backend-ops perf, CPU, n_mats=128 n_used=8 m=768 k=2048)

n=1 (one token per batch):

| type_a | us/run | GFLOPS |
| --- | --- | --- |
| q4_K | 60.16 | 418 |
| q6_K | 69.08 | 364 |
| iq4_xs | 80.67 | 312 |
| q8_0 | 76.22 | 330 |
| iq2_xs | 126.45 | 199 |

n=4: q4_K 329.86, q6_K 176.32, iq2_xs 200.37, iq4_xs 264.70 GFLOPS.

Grid-LUT i-quants are roughly 2x less efficient per FLOP than k-quants in the GEMV regime. The expert
tensors of this model are exactly those types (IQ2_S gate/up, IQ3_S down).

## Layout findings

- Weights are a single MAP_SHARED read-only mapping of the GGUF; zero copy, fully resident.
- `select_weight_buft` prefers CPU_REPACK, but the loader demotes to the plain CPU buffer when
  `use_mmap` is set ("avoid using a host buffer when using mmap", llama-model-loader.cpp ~1279).
  The verbose log reports 733 tensors moved off the CPU_REPACK preference, and no
  "repack tensor ... with q6_K_8x8" line is emitted, so no weight is repacked in this configuration.
  Measured cost of that: ~0 at batch 1 (see load mode table above).
- IQ2_S / IQ3_S have no repack instance (repack.cpp covers Q1_0, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0,
  IQ4_NL, MXFP4), so expert tensors use vec_dot kernels at batch 1.
- The tiled k-quant path (tiled.cpp) covers IQ2_S and IQ3_S but only engages at rows >= 8, so prefill
  uses it and decode does not.
- `nextn_predict_layers = 1`: the MTP block (blk.40) is present on disk, `load_mtp` defaults to false,
  so the amortization path is available but unused.

## Relevant knobs that already exist

`--load-mode {auto,none,mmap,mlock,mmap+mlock,dio}`, `-mp`, `--no-host`, `-t/-tb`,
`GGML_CPU_TILED_MM`, `GGML_CPU_TILED_MM_FORCE`, `--override-tensor`.

## Addendum: the harness run and page-cache residency

Repeat of the matrix with `scripts/research/bench-stage.sh` (results in `benches/advan-6800h/`):
pp256 61.27 t/s, tg128 12.97 / 15.26 / 11.29 t/s at 4 / 8 / 16 threads, load modes
mmap 15.88, none 15.71, mlock 15.71. DRAM ceiling re-measured at 46.54 GB/s (8T), 44.04 (16T).

Residency is not stable on this machine. After the `-lm none` run (13 GiB of anonymous weights)
the model was only 48 to 57% resident, i.e. the 27 GiB box cannot always hold the model plus the
rest of the desktop. That does not cost throughput, measured directly:

| state | resident | tg128 |
| --- | --- | --- |
| mixed (48.4%) | 6.35 GiB | 15.89 t/s |
| fully populated after a full read | 13.10 GiB | 16.06 t/s |
| all model pages dropped (posix_fadvise DONTNEED) | low | 15.77 t/s |

So the engine is not disk-bound. The always-hot dense set (~1.7 GB) stays in DRAM, and the rotating
expert pages read from disk overlap with the DRAM traffic instead of serializing with it.

## What actually accounts for the 63 ms/token

| component | ms/token |
| --- | --- |
| weight streaming of the always-hot dense set, 1.72 GB at 46.5 GB/s | ~37 |
| weight streaming of the selected experts, 0.36 GB at 46.5 GB/s | ~8 |
| per-token compute, ~2.4 GFLOP at the measured 187 GFLOPS prefill rate | ~13 |
| perfect overlap bound (max of the two, not the sum) | ~46 ms -> 21.7 t/s |

Measured is 63 ms, so the gap to the bound is roughly 1.4x. That gap is kernel inefficiency and
imperfect overlap, not bytes, not layout, not disk. The bound is optimistic on purpose: compute and
bandwidth contend for the same 8 threads, so no real kernel reaches max() of the two.

## See also

`docs/research/02-gpu-vulkan-recheck.md` rechecks this stage on the Vulkan backend. On the GPU
nothing in this stage matters either, but the GPU path is 1.4x faster on decode and reaches the
DRAM wall, so it is the configuration to build on.
