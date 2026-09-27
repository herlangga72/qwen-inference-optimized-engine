# Delta-net output projection: results

Date: 2026-09-27
Machine: Ryzen 7 6800H, Radeon 680M (RADV REMBRANDT), 27.1 GiB.
Change: `final_output` is declared `[value_dim, n_seq_tokens*n_seqs]` instead of
`[value_dim, n_seq_tokens, n_seqs]` at the three delta-net output projections:
`src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`, `src/models/qwen4exp.cpp`.

Baselines and method: `docs/research/05-ssm-out-fix-baselines.md`.

## Correctness

The two reshapes are views of the same bytes with the same element order, so the change is expected to
be bit identical. Checks that were run:

| check | result |
| --- | --- |
| `test-llama-archs -a qwen35 -s 1` | pass, CPU 0.00e+00, Vulkan0 8.54e-08 NMSE |
| `test-llama-archs -a qwen35moe -s 1` | pass, CPU 0.00e+00, Vulkan0 8.72e-08 NMSE |
| `test-llama-archs -a qwen4exp -s 1` | pass, CPU 0.00e+00, Vulkan0 1.23e-07 NMSE |
| `test-backend-ops test --test-file` on the exported 35B ops, Vulkan0 | both `linear_attn_out` shapes OK against the CPU reference |

The `n_seqs > 1` path is not covered by `test-llama-archs`, which decodes a single sequence; for
`n_seqs = 1` the two reshapes are the same tensor, so that test cannot tell them apart. The coverage
for the batched shape is the exported op replay, which is the real 35B graph at `-np 8`.

## Op level, 35B qwen35moe, np=8, `-c 4096`

Before and after shapes alternated in one session, three rounds each, us/run:

| case | before | after | change |
| --- | --- | --- | --- |
| Vulkan0, tg, 1 token x 8 seqs, `[2048,1,8,1]` -> `[2048,8,1,1]` | 823.28, 823.38, 1035.04 | 354.11, 379.98, 380.51 | 2.2x faster |
| Vulkan0, pp, 64 tokens x 8 seqs, `[2048,64,8,1]` -> `[2048,512,1,1]` | 3278.26, 3280.13, 3299.06 | 3278.85, 3278.95, 3281.78 | unchanged |
| CPU, tg | 310.30, 491.92 | 502.61, 508.96 | about 1.3x slower |
| CPU, pp | 12658.26, 13204.75 | 12168.44, 12213.06 | about 5% faster |

## Op level, 0.8B qwen35, np=8

| case | before | after | change |
| --- | --- | --- | --- |
| Vulkan0, tg | 101.35, 149.53, 257.76 | 240.68, 331.03, 537.68 | noise, no clear direction |
| Vulkan0, pp | 792.63, 794.32 | 788.56, 788.67, 791.96 | unchanged |
| CPU, tg | 105.67, 149.53 | 115.24, 141.77, 185.08 | unchanged |
| CPU, pp | 2877.11, 2933.59 | 3067.63, 3232.78, 3604.20 | 10 to 20% slower |

## Verdict

The premise holds only where the weight does not fit in cache, and the gain is smaller than predicted.

- Vulkan, the target backend: the np=8 decode op drops from 823 us to 380 us, 2.2x. The spec predicted
  110 us, so the prediction did not hold; the residual is the single GEMM at `M = 8` per sequence
  group, which is far from peak, not bandwidth.
- The np=8 prefill shape is unchanged on Vulkan. At 64 tokens per sequence the weight reads were
  already amortized, so removing them buys nothing.
- CPU gets slightly worse on the np=8 decode shape and slightly better on the np=8 prefill shape. The
  CPU weight (6912 kB) is read from L3 by the batched path, so the batched path has no bandwidth to
  save, and the CPU's small-`M` GEMM is the slower kernel of the two. The 0.8B pp case is the worst
  reading in the set (up to 23% slower) and is inside the noise band of these micro-benchmarks.
- Single stream is untouched by construction, `n_seqs = 1` produces the identical tensor.

Per the plan, the change stays: correctness holds and single stream is unaffected. The aggregate B=8
throughput prediction (39.47 to about 43.5 t/s) was not measured, because the instruction for this
round was to test the affected part only and not to run the engine.

Two follow-ups fall out of this:

- The `M = 8, K = 4096, N = 2048` GEMM at 380 us is 134 MFLOP, about 0.35 TFLOP/s, roughly 10% of this
  part's fp32 peak. That is the remaining cost, and it belongs with the prefill attention and MoE work
  (F part 2), not here.
- If the CPU batched decode path matters, the reshape has to be conditional on the backend, because the
  CPU loses on the same shape the GPU wins.
