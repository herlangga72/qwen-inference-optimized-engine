# Delta-net output projection: pre-change baselines

Date: 2026-09-27
Machine: Ryzen 7 6800H, Radeon 680M (RADV REMBRANDT), 27.1 GiB, Vulkan0 + CPU.
Tree: `qwen-only-backends` before the reshape change (commit f32e022f9).
Plan: `docs/superpowers/plans/2026-09-26-delta-net-output-projection.md`.

## Method

The op is measured on the real graph but without running the engine, as instructed: the graph is
exported once and each op replayed by the backend's own perf harness.

```sh
cd /home/herlanggays/.jcode/scratch/ssm-out-fix/before
build-vk/bin/test-export-graph-ops -m <model> -np 8 -c 4096      # writes tests.txt
build-vk/bin/test-backend-ops perf --test-file tests.txt -b Vulkan0 -p 'name=linear_attn_out'
build-vk/bin/test-backend-ops perf --test-file tests.txt -b CPU     -p 'name=linear_attn_out'
```

`test-export-graph-ops` builds the model graph and writes the ops it needs; `perf` times them and
never decodes. The `-p name=linear_attn_out` filter keeps the run to a few seconds.

Caveat: these are ten-microsecond ops on an integrated GPU, so single readings drift with clocks. All
numbers below were taken with the before/after shapes alternated in the same session, and repeats are
listed.

## Shapes

`linear_attn_out` is `MUL_MAT(ssm_out, final_output)` with `ssm_out` declared `[n_embd, value_dim]` in
the GGUF (so `ne=[value_dim, n_embd]` for the graph operand).

| model | arch | weight | op at np=8, `-c 4096` |
| --- | --- | --- | --- |
| Qwen3.5-0.8B-Q4_K_M | qwen35 | q5_K `[2048,1024]`, 1408 kB | tg shape `ne=[1024,1,8,1]`, pp shape `ne=[1024,64,8,1]` |
| Qwen3.6-35B-A3B-UD-IQ3_XXS | qwen35moe | q6_K `[4096,2048]`, 6912 kB | tg shape `ne=[2048,1,8,1]`, pp shape `ne=[2048,64,8,1]` |

In the np=8 tg graph each sequence holds one token, so `ne2 = n_seq_tokens = 1` and the declared 3D
shape puts the 8 sequences in the batch dimension. In the np=8 pp graph each sequence holds 64 tokens,
so the batch is 8 and every sequence reads the whole weight.

## Measured baselines

35B, qwen35moe (the shape the plan cares about):

| backend | tg shape `[2048,1,8,1]` | pp shape `[2048,64,8,1]` |
| --- | --- | --- |
| Vulkan0 | 1020.04, 823.38, 823.28 us | 3299.06, 3280.13, 3278.26 us |
| CPU | 342.39, 491.92, 310.30 us | 13439.39, 13204.75, 12658.26 us |

0.8B, qwen35:

| backend | tg shape `[1024,1,8,1]` | pp shape `[1024,64,8,1]` |
| --- | --- | --- |
| Vulkan0 | 101.35, 149.53, 257.76 us | 792.63, 794.32 us |
| CPU | 105.67, 149.53 us | 2933.59, 2877.11 us |

The 35B Vulkan tg figure reproduces the reference in `docs/research/04-stage5-diagnosis.md` (823.85 us)
to within 0.1%, on the same method. The 0.8B weight (1408 kB) fits in the 16 MB L3, so its tg shape is
cache resident and much less sensitive to the extra weight reads.

## What has to change

`reshape_3d(a, value_dim, n_seq_tokens, n_seqs)` and `reshape_2d(a, value_dim, n_seq_tokens*n_seqs)`
are the same view of the same bytes; only the declared batch dimension differs. For `n_seqs = 1` the
two are literally the same tensor (`ne = [value_dim, n_seq_tokens, 1, 1]`), so single-stream prefill
and single-stream decode are untouched by construction. The change only has an effect where
`n_seqs > 1`.
