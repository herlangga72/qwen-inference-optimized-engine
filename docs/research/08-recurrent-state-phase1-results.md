# Recurrent state view (phase 1): results

Date: 2026-09-27
Machine: Ryzen 7 6800H, Radeon 680M (RADV REMBRANDT), 27.1 GiB, Vulkan0 + CPU.
Tree: `qwen-only-backends` at 4f5831ba2 plus the change below, both build trees rebuilt from it.
Plan: `docs/superpowers/plans/2026-09-26-recurrent-state-view-phase1.md`.
Baselines and method: `docs/research/07-recurrent-state-phase1-baselines.md`.

## The change

`build_rs` gathers the active recurrent state rows into a graph buffer, every delta-net layer every
token. When the active rows are the identity mapping `[head, head + n_rs)` the gather is a copy of
rows onto themselves, and so is the extra-state write-back at the end of `build_rs`. The change
detects that case and returns a view of the cache instead.

| file | change |
| --- | --- |
| `src/llama-memory-recurrent.h` | declare `llama_memory_recurrent_context::state_rows_are_contiguous()` |
| `src/llama-memory-recurrent.cpp` | implement it: `n_rs_seq == 0` and `cells[head + i].src0 == head + i` for `i < n_rs` |
| `src/llama-graph.h` | `llm_graph_input_rs::rows_contiguous`, and a trailing `bool state_contiguous` on the first `build_rs` overload |
| `src/llama-graph.cpp` | set the flag in `build_rs_inp_impl`, check it in `can_reuse` of `llm_graph_input_rs` and of the three hybrid inputs, return `ggml_view_2d` instead of the gather when it holds |

50 insertions, 4 deletions across the four files.

One departure from the plan. With the view path taken, `s_copy` is no longer an operand of any node
in the graph, so the allocator does not give it a buffer and the `GGML_ASSERT` in `set_input`
aborts. The plan's Task 2 step 3 does not cover this. The fix is the guard the tree already uses for
conditionally allocated inputs (`if (s_copy && s_copy->buffer)`), in the four `set_input`
implementations that read it. Without it `test-llama-archs` aborts on the first decode.

## Correctness

Every run below is in `capture.sh`, which serializes the steps and watches memory; logs are in
`/home/herlanggays/.jcode/scratch/rs-view/`.

| check | result |
| --- | --- |
| CPU build, Vulkan build | 0 errors |
| `ctest -L main`, build-cpu and build-vk | 34 / 34 both |
| `test-llama-archs -a qwen35 / qwen35moe / qwen4exp -s 1` | all pass, NMSE 0.00e+00 |

Output identity, digest of the generated text before against after the change:

| case | before | after |
| --- | --- | --- |
| 0.8B, single sequence, CPU | `a599102cce02ed30` | `a599102cce02ed30` |
| 0.8B, single sequence, Vulkan0 | `a599102cce02ed30` | `a599102cce02ed30` |
| 35B, single sequence, Vulkan0 | `dfbe0e02dec9d892` | `dfbe0e02dec9d892` |
| 0.8B, passkey, CPU | `ae5f3046d0bfef0e` | `ae5f3046d0bfef0e` |
| 0.8B, passkey, Vulkan0 | `ae5f3046d0bfef0e` | `ae5f3046d0bfef0e` |
| 0.8B, 2 sequences in one batch, CPU | `031fec546e66` | `031fec546e66` |
| 0.8B, 2 sequences in one batch, Vulkan0 | `031fec546e66` | `031fec546e66` |
| 0.8B, 4 sequences in one batch, CPU | `166ad7dab0f5` | `166ad7dab0f5` |
| 0.8B, 4 sequences in one batch, Vulkan0 | `166ad7dab0f5` | `166ad7dab0f5` |

The single sequence digest is `grep -v "^0\." <file> | sha256sum`, the multi-sequence digest is
`python3 norm-msgen.py <file> | sha256sum`. The multi-sequence runs are `llama-batched -np N -kvu`,
which is the case the predicate keys on. `llama-passkey` cannot cover it and runs one sequence only;
see the baselines doc.

Slow path proof. With the predicate forced to `return false` and both trees rebuilt, the gather path
reproduces every pre-change baseline:

| case | baseline | forced gather path |
| --- | --- | --- |
| 0.8B single sequence, CPU and Vulkan0 | `a599102cce02ed30` | `a599102cce02ed30` |
| 0.8B passkey, CPU and Vulkan0 | `ae5f3046d0bfef0e` | `ae5f3046d0bfef0e` |
| 0.8B 2 sequences, CPU and Vulkan0 | `031fec546e66` | `031fec546e66` |
| 0.8B 4 sequences, CPU and Vulkan0 | `166ad7dab0f5` | `166ad7dab0f5` |

So the two paths return the same numbers for every case that was measured, and the change is safe to
land with the predicate on.

Graph reuse across a mapping change. The dangerous direction is a graph built while the mapping was
the identity being reused for a batch where it is not: the view would read another sequence's rows
and fail silently. `llama-server -np 2` with three staggered requests (slot 0, then slot 1 joins,
then slot 0 again) changes the mapping under a graph the previous request built. All three replies
are identical between this build and the same build with the predicate forced false
(`server-fast.txt` against `server-slow.txt`):

| request | reply |
| --- | --- |
| slot 0 | ` the capital of the country. The capital of France is the capital of the` |
| slot 1 | ` Tokyo. The capital of Japan is Tokyo. The capital of Japan is` |
| slot 0 again | ` the capital of the country. The capital of France is the capital of the` |

`test-recurrent-state-rollback` covers the rollback case in the suite but runs with `n_rs_seq = 8`,
so it always takes the gather path; the server probe is what covers the reuse transition.

## Performance, 35B qwen35moe IQ3_S, Vulkan0, `-ngl 99`

`llama-bench -r 3 -p 0 -n 128 -t 8`, three builds of the same tree:

| build | tg128 t/s |
| --- | --- |
| before the change | 22.33 +/- 0.21 |
| change in, predicate forced false | 22.70 +/- 0.03 |
| change in, predicate on | 24.21 +/- 0.01 |

The middle row is the control: same binary as the last row with only the predicate body changed, so
the 6.6% between them is the change itself. The spec predicted about +7%, and its ablation B
measured 24.01.

`llama-batched-bench -ngl 99 -t 8 -c 4096 -npp 128 -ntg 32 -npl 1,2,4,8`:

| B | S_TG before | S_TG after | change |
| --- | --- | --- | --- |
| 1 | 21.46 | 22.72 | +5.9% |
| 2 | 32.54 | 34.98 | +7.5% |
| 4 | 40.51 | 45.91 | +13.3% |
| 8 | 44.30 | 49.11 | +10.9% |

| B | S_PP before | S_PP after | change |
| --- | --- | --- | --- |
| 1 | 143.23 | 144.70 | +1.0% |
| 2 | 190.05 | 188.60 | -0.8% |
| 4 | 235.95 | 233.02 | -1.2% |
| 8 | 239.60 | 241.35 | +0.7% |

Decode improves, prefill is unchanged, which is what the spec expects: the copies are per decode
step and at 128 tokens per sequence per prefill they are amortized.

The single-sequence generation baseline moves with the throughput: 21.36 t/s eval before, 23.78
after, on a 12 token run.

## Against the spec's expectations

- Single stream +7%: held, +6.6% on the controlled comparison, +8.4% against the pristine baseline.
- Batch 8 aggregate +41%: did not hold, +10.9%. The spec's ablation started from 31.61 and read
  44.52 after; this tree's baseline already reads 44.30 before the change, as recorded in the
  baselines doc. The prediction was made on a tree without the delta-net output projection change,
  and that change moves the same batched decode path. The two do not overlap completely, which is
  why phase 1 still adds about 11% on top.
- Prefill unchanged: held.
- Risk, a stale predicate reading another sequence's state: not observed. The multi-sequence runs
  (2 and 4 sequences, both backends) are byte identical to the gather path.
