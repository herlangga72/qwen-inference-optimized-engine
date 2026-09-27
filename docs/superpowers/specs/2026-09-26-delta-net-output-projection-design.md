# Delta-net output projection: one weight read per step instead of one per sequence

Date: 2026-09-26
Status: implemented and measured
Outcome: built at the three delta net output projections, declaring `final_output` as
`[value_dim, n_seq_tokens*n_seqs]` instead of 3D. Vulkan token generation with 1 token and 8
sequences improved 2.2x and Vulkan prefill was unchanged, while CPU token generation fell about
1.3x and CPU prefill rose about 5%. That is a deliberate CPU-for-Vulkan trade, and it is the wrong
trade on a CPU-only box. Results in `docs/research/06-ssm-out-fix-results.md`.
Scope: `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`, `src/models/qwen4exp.cpp`

## Context

This fork runs Qwen3.5 and up on CPU and Vulkan. Qwen3.5 layers alternate between gated delta net
(linear attention) and full attention. The delta-net output projection multiplies a per layer weight
with the gated delta net output:

- `blk.N.ssm_out.weight`, shape `[value_dim, n_embd]` = `[4096, 2048]`, 6.9 MB at Q6_K
- the activation from `build_norm_gated`, shape `[head_v_dim, num_v_heads, n_seq_tokens, n_seqs]`

The graph reshapes that activation to 3D before the matmul:

```cpp
ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);
```

`ggml_mul_mat` treats dims 2 and 3 of `src1` as a batch dimension. With `src1` declared as
`[..., n_seq_tokens, n_seqs]`, the kernel walks the entire weight once per batch index, so the 6.9 MB
weight is read `n_seqs` times per step instead of once.

## Evidence

Per op timing from the real model graph (`test-export-graph-ops -np 1` and `-np 8`, then
`test-backend-ops perf --test-file`):

| backend | `linear_attn_out`, 1 token | 8 tokens | ratio |
| --- | --- | --- | --- |
| Vulkan | 109.57 us | 823.85 us | 7.52x |
| CPU | 39.35 us | 279.14 us | 7.09x |

Control in the same graph: `attn_output` of the full attention layers, which takes a 2D activation,
scales 1.09x over the same 8x token increase. A weight bound op should scale about 1.0x, so 7.5x is
`n_seqs` weight reads, not extra work.

End to end that is about 21 ms of a 202.7 ms step at batch 8 on the Vulkan backend (30 delta-net
layers x 0.7 ms), which is about 10% of aggregate token generation throughput. Single stream
(`n_seqs == 1`) is unaffected. Full measurements: `docs/research/04-stage5-diagnosis.md`.

## Change

Replace the 3D reshape that immediately precedes the `ssm_out` matmul with a 2D reshape of the same
element count, at three sites:

| file | line | current | new |
| --- | --- | --- | --- |
| `src/models/qwen35.cpp` | 457 | `ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs)` | `ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs)` |
| `src/models/qwen35moe.cpp` | 481 | same | same |
| `src/models/qwen4exp.cpp` | 977 | same | same |

Nothing else changes. The trailing `ggml_reshape_2d(cur, n_embd, n_seq_tokens*n_seqs)` stays; with a
2D input `cur` is already that shape and the reshape is a free no-op.

`qwen4exp` shares this code path exactly, including the `build_norm_gated` helper (its only
difference is a sigmoid gate instead of silu), so it takes the same one line change.

## Why it is correct

`attn_out_norm` is produced by `build_norm_gated`, which returns `ggml_mul(normalized, gate)`.
`ggml_mul` allocates a fresh contiguous tensor whose `ne` is the broadcast of its inputs, that is
`[head_v_dim, num_v_heads, n_seq_tokens, n_seqs]` with standard contiguous strides
(`nb1 = ne0*4`, `nb2 = ne0*ne1*4`, `nb3 = ne0*ne1*ne2*4`).

The existing `ggml_reshape_3d` to `[head_v_dim*num_v_heads, n_seq_tokens, n_seqs]` already assumes
dims 0 and 1 are adjacent and that dim 2 then dim 3 follow in memory order. Flattening dims 2 and 3
into one dimension of size `n_seq_tokens*n_seqs` is the same memory in the same order; only the view
metadata differs. Both reshapes are views, so neither copies data.

Downstream, the matmul result is already consumed as 2D by
`ggml_reshape_2d(cur, n_embd, n_seq_tokens*n_seqs)`, so no consumer depends on the 3D view.

## Expected outcome

| measurement | before | expected after |
| --- | --- | --- |
| Vulkan `linear_attn_out`, 8 tokens | 823.85 us | about 110 us |
| CPU `linear_attn_out`, 8 tokens | 279.14 us | about 40 to 60 us |
| Vulkan aggregate TG, B=8 | 39.47 t/s | about 43.5 t/s |
| Vulkan aggregate TG, B=1 | 21.19 t/s | unchanged |
| Vulkan tg128, `-ngl 99` | 22.46 t/s | unchanged |

The CPU figure is smaller in absolute terms because that weight is largely L3 resident there
(6.9 MB in a 16 MB L3), so the repeated reads are cheap; the Vulkan reads hit DRAM.

These are predictions, not results. The verification below is what decides.

## Verification

1. `test-llama-archs -a qwen35 -a qwen35moe -a qwen4exp`, which builds a synthetic model, runs the
   graph and compares against a reference forward pass. Run with `-s 1` for a reproducible seed.
2. Byte identical generation against the pre-change build, both backends: 35B MoE
   (`llama-cli -n 32`, fixed seed) and 0.8B dense. Capture those outputs from the current build
   before making the change, into files under the scratch directory, then diff after the change.
3. Op level: re-export the batch 8 graph and re-run `test-backend-ops perf --test-file`, checking
   `linear_attn_out` drops and that no other op regresses.
4. End to end: `llama-batched-bench` at 1, 2, 4, 8, 16 sequences on both builds, plus `llama-bench`
   tg128 and pp256 to confirm single stream is unchanged.
5. `ctest -L main` to catch collateral damage.

## Risks and fallback

- If `attn_out_norm` is not contiguous in the assumed order, `ggml_reshape_2d` asserts at graph build
  time. That is a loud, immediate failure, not a silent one. Fallback: keep the 3D reshape and insert
  an explicit `ggml_cont_2d` before the matmul, which materialises the layout at the cost of copying
  the activations (about 128 KB per layer at batch 8).
- `build_lora_mm` also handles the LoRA adapter path; the change touches only `src1`'s shape, so it is
  unaffected. Verify with the existing archs test rather than assuming.
- `qwen4exp` has QSA and PLE paths of its own. The `ssm_out` site is shared with the other two models,
  but if the archs test for `qwen4exp` fails, drop that file from this change and file it separately
  rather than guessing.

## Out of scope

- Reducing the recurrent state gather in `build_rs` (`src/llama-graph.cpp:2551`), about 4% of single
  stream, and correctness sensitive.
- Non weight work: flash attention, `GATED_DELTA_NET`, norms, elementwise ops, softmax, argmax, 10 to
  14 ms of a 47.2 ms single stream step, unlocalized.
- MoE expert kernels. Measured byte optimal at batch, no headroom.
- Amortization work: MTP speculative decoding, batching policy.
- Dense requantization (Q6_K to Q4_K), which is a quality decision with no engine code.

## Follow-ups

Separate specs, in the order the measurements suggest: the recurrent state gather, then the
non weight work localization, then amortization. Batching policy and MTP both multiply tokens per
byte read, which is the only way to make the 17% expert share of per token bytes pay off.
