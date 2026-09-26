# Stage 5 diagnosis: where batched decode loses, and a correction to the bandwidth ceiling

## Correction: 46.9 GB/s is the CPU core rate, not the machine's

The stage 1 ceiling probe (`scripts/research/membw.c`, an 8 thread sum loop over 3 GiB) reports
46.5 to 46.9 GB/s. That is what this CPU's cores can pull, not what the memory can deliver. The
Vulkan backend demonstrably reads faster, from `test-backend-ops perf -b Vulkan0 -o MUL_MAT`:

| case | weight bytes | us/run | GB/s |
| --- | --- | --- | --- |
| MUL_MAT f32 m=4096 k=14336 n=1 | 235 MB | 3440.36 | 68.3 |
| MUL_MAT f32 m=4096 k=14336 n=8 | 235 MB | 3528.36 | 66.6 |
| MUL_MAT f16 m=4096 k=14336 n=1 | 117 MB | 1722.73 | 68.2 |
| MUL_MAT q8_0 m=4096 k=14336 n=4 | 62 MB | 964.24 | 64.7 |

So the machine's sustained read rate is about 68 GB/s (DDR5-4800 dual channel theoretical peak is
76.8 GB/s, so 89%). Corrections to earlier docs:

- `02-gpu-vulkan-recheck.md` said the GPU decode path is "at the memory wall, 97 to 100% of the
  wall". It is at 46.7 GB/s, which is 69% of the achievable 68 GB/s. The claim was overstated.
- `03-stage3-stage4.md` "at 97 to 100% of DRAM wall" for the GPU and "at 70% of DRAM wall" for the
  CPU became: GPU 69% of the machine rate, CPU 33 GB/s against a 46.9 GB/s CPU core rate (71%).
- The CPU conclusions are unaffected in substance, they were always measured against the CPU rate.

Both single-stream paths still leave roughly a third of the achievable read rate on the table, and
batched decode leaves much more.

## Method

`test-export-graph-ops` dumps the real model's graph ops (131 unique ops over 3787 nodes) into a
file that `test-backend-ops perf --test-file` can replay per op on a chosen backend. Run twice, with
`-np 1` and `-np 8`, it gives the batch 1 and batch 8 decode graphs with the real shapes and quant
types. Per op time then shows which ops amortize over a batch and which do not.

Note: ops identical in shape and type collapse into one entry (the gate and up expert projections of
a layer are byte identical, so one entry represents both). The tool also reports per run bytes as
the full source tensor size, which is meaningless for `MUL_MAT_ID`, so expert bandwidth has to be
recomputed by hand from the real expert bytes.

## Per op time, batch 1 against batch 8, Vulkan, real model

Sorted by extra microseconds at batch 8. `x` is the time ratio, `n=8` ideal for a weight bound op is
about 1.0, and for `MUL_MAT_ID` it is the expert union growth (about 7x for 8 tokens).

| extra us | bs1 us | bs8 us | x | op | tensor | src |
| --- | --- | --- | --- | --- | --- | --- |
| 3763 | 6140.0 | 9903.4 | 1.61 | MUL_MAT | result_output (lm_head) | q6_K |
| 900 | 85.8 | 985.9 | 11.49 | MUL_MAT_ID | ffn_moe_down | iq3_s |
| 714 | 109.6 | 823.9 | 7.52 | MUL_MAT | linear_attn_out (ssm_out) | q6_K |
| 493 | 74.8 | 567.5 | 7.59 | MUL_MAT_ID | ffn_moe_down | iq4_xs |
| 483 | 68.3 | 551.3 | 8.07 | GET_ROWS | node_23 (recurrent state) | f32 |
| 480 | 66.5 | 546.1 | 8.22 | CPY | cache_s_l0 (recurrent state) | f32 |
| 348 | 60.0 | 407.9 | 6.80 | MUL_MAT_ID | ffn_moe_gate | iq3_s |
| 281 | 213.8 | 494.8 | 2.31 | MUL_MAT | node_13 (attn_qkv) | q6_K |
| 269 | 47.5 | 316.3 | 6.66 | MUL_MAT_ID | ffn_moe_gate | iq2_s |
| 127 | 18.4 | 145.1 | 7.91 | CPY | conv_state_update | f32 |
| 91 | 275.4 | 366.3 | 1.33 | MUL_MAT | Qcur_full | q8_0 |
| 62 | 112.2 | 174.4 | 1.55 | MUL_MAT | z (attn_gate) | q6_K |
| 53 | 17.5 | 71.0 | 4.05 | MUL_MAT | ffn_shexp | q6_K |
| 37 | 24.7 | 61.3 | 2.48 | MUL_MAT | shared_expert_gate (router) | f32 |

Also measured: `FLASH_ATTN_EXT` 22.56 us at 1 token against 167.20 us at 8, which is expected since
attention has no weight to amortize; `GATED_DELTA_NET` 71.5 us for 129 rows against 586.9 us for
1032 rows, also correctly linear; `SSM_CONV` 4.68 against 23.97; `GET_ROWS` on the conv state 3.89
against 21.55.

## What this says

Three distinct behaviours:

1. **True amortization.** `result_output` (lm_head, 417 MB) 1.61x, `Qcur_full` 1.33x, `attn_output`
   1.09x, `z` 1.55x, `ffn_gate` (shared expert) 1.61x. These are the big weight matmuls and they
   behave correctly: one weight read per step.

2. **Byte bound growth, not a bug.** The four `MUL_MAT_ID` expert ops grow 6.7x to 11.5x for 8x the
   tokens. That is close to the expert union growth (8 tokens x 8 experts of 256 gives about 57
   distinct experts per layer, 7x the bytes of one token's 8 experts). The MoE kernels are
   therefore doing the right amount of work. **The MoE expert path is not the batched bottleneck,
   and optimizing its kernels cannot recover much, because the bytes are real bytes.**

3. **No amortization at all, in the recurrent path.** `linear_attn_out` (ssm_out, 6.9 MB per layer)
   grows 7.52x for 8x tokens, which means the 6.9 MB weight is read once per sequence instead of
   once per step: about 55 MB per layer per step instead of 6.9 MB. Its destination is
   `[2048, 1, 8, 1]`, a per sequence layout, and the same pattern shows on the recurrent state ops.
   At batch 8 this costs about 21 ms of a 202.7 ms step (10%).

4. **Recurrent state traffic is large even single stream.** Per delta-net layer per token:
   `SCALE cache_s_l0` 67.29 us, `CPY cache_s_l0` 66.45 us, `GET_ROWS node_23` 68.33 us, 202 us
   total. The state is 524288 f32 elements, 2 MB per sequence. Times 30 delta-net layers that is
   6.06 ms of a 47.2 ms single stream step, **12.8% of decode**, before any batching.

## Revised budget, single stream, GPU, all layers offloaded (47.2 ms step)

| part | ms | share |
| --- | --- | --- |
| attention and delta-net weights (qkv, gate, ssm_out, attn_output) | ~13 | 28% |
| MoE expert projections (gate, up, down, 40 layers) | 7.2 | 15% |
| lm_head | 6.1 | 13% |
| recurrent state ops (SCALE, CPY, GET_ROWS) | 6.1 | 13% |
| shared experts and routers | ~2.8 | 6% |
| norms, elementwise, softmax, argmax | ~4 | 8% |
| flash attention and delta-net compute | ~5 | 11% |

Total weight traffic is 2.08 GB per token, so the 47.2 ms step equals 44 GB/s, 65% of the 68 GB/s
the machine can stream.

## Candidate work, with measured stakes

| # | target | measured stake | where |
| --- | --- | --- | --- |
| 1 | per sequence layout of the delta-net output projection | up to 10% of a batch 8 step, none single stream | delta-net graph builder reshape |
| 2 | recurrent state gather (`GET_ROWS` of the 2 MB per sequence state row) | about 4% of single stream, only when the row mapping is the identity | llama-graph build_rs, hybrid memory |
| 3 | MoE expert kernels | ~0, bytes are real; scaling matches the expert union | not worth it |
| 4 | non weight work: attention and delta-net compute, norms, elementwise, softmax | 10 to 14 ms of the 47.2 ms step, unlocalized | broad |

## Refinements after reading the code

Two claims in the first pass were too generous and are corrected here.

**The state ops are less avoidable than they look.** `build_rs` (src/llama-graph.cpp:2551) does three
things per layer: a conditional zeroing of one state row (`ggml_scale_inplace` on a zero sized view
when no slot is being cleared, so this is free in steady state), a gather of the active rows
(`get_state_rows`, the `GET_ROWS`), and a write back of the non active rows. The write back of the
updated state is inherent to a 2 MB recurrent state, and the gather is only removable when the active
sequence rows are already the identity mapping. So the avoidable part is roughly the single 2 MB read
per layer per token, about 2 ms of 47.2 ms, 4%, not the 12.8% first estimated.

**The single stream gap is not mostly a gap.** Individual weight bound ops run at 55 to 63 GB/s
against the 68 GB/s machine rate, which is 81 to 93%, close to what a dequantising kernel can do.
The step as a whole moves 2.08 GB in 47.2 ms, 44 GB/s, because between 10 and 14 ms of the step is
non weight work: flash attention, `GATED_DELTA_NET`, the state traffic, norms, elementwise ops,
softmax and argmax. Adding those to the 33 to 38 ms of weight streaming accounts for the whole step.
So the realistic ceiling for this model on this GPU is low 50s GB/s, not 68, and the remaining fat is
in the non weight work.

Control: a dense only 0.8B Q4_K_M model on the same GPU reaches 87.51 t/s over 0.5 GB per token,
which is 43.8 GB/s, essentially the same figure. About 44 GB/s is what this Vulkan backend sustains
for quantised GEMV and GEMM on this iGPU, independent of model and node count (the 0.8B model has
far fewer nodes per step, so per node dispatch cost is not the explanation).

## Artifacts

- `scripts/research/mmid_bw.py` turns test-backend-ops perf output into per op bandwidth.
- Export and replay: `test-export-graph-ops -m MODEL -np N` then
  `test-backend-ops perf --test-file tests.txt -b Vulkan0`.
- Raw captures: `vk_realops.txt` (`-np 1`), `vk_np8.txt` (`-np 8`), `vk_mm.txt`, `vk_mmid.txt`.
