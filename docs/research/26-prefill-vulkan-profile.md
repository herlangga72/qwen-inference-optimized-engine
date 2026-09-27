# Prompt processing on Vulkan: where the time goes

Date: 2026-09-27
Tree: `qwen-only-backends`, commit `742cf16d9` plus the llama-bench type map change
Machine: Ryzen 7 6800H, Radeon 680M (RADV REMBRANDT), 27.1 GiB
Model: `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` (qwen35moe, 35B-A3B, IQ3_S 3.4375 bpw, 13.09 GiB)
Backend: Vulkan, `-ngl 99`, `-fa on`
Instrument: `GGML_VK_PERF_LOGGER=1`

**Verdict: prompt processing runs at 230 t/s and is flat across prompt length. 47 percent of it is the
MoE expert matmul, which runs at about a third of the efficiency the dense matmuls reach on the same
GPU and whose cost barely depends on batch size. Flash attention, which the prefill plan puts first, is
1.2 percent.**

## Throughput

| test | ubatch 512 | ubatch 1024 | ubatch 2048 |
| --- | --- | --- | --- |
| pp512 | 230.27 ± 0.69 | 231.67 ± 1.48 | 231.98 ± 1.43 |
| pp2048 | 228.97 ± 0.72 | 169.92 ± 0.32 | 209.43 ± 0.01 |

Flat at about 230 t/s. Prompt length does not change it, and a larger micro batch makes it worse, not
better. Whatever limits this, it is not fixed cost that a bigger batch would amortise.

## Per-op breakdown, pp512

Shares are of a 2.22 s pass. The last column is the backend's own reported rate.

| op family | calls | total ms | share | rate |
| --- | --- | --- | --- | --- |
| `MUL_MAT_ID` (MoE experts) | 120 | 1049 | 47% | 878 to 1494 GFLOPS/s |
| `MUL_MAT` dense q6_K and q8_0 | 176 | 518 | 26% | 2430 to 3372 GFLOPS/s |
| `GATED_DELTA_NET` | 30 | 163 | 8% | 5.4 ms per call |
| `CONCAT` | 30 | 161 | 8% | 5.4 ms per call |
| `MUL`, `MULTI_ADD`, `GLU` | 200 | 103 | 5% | elementwise |
| `RMS_NORM` family | 181 | 92 | 4.6% | 570 us for a 128 wide norm |
| `FLASH_ATTN_EXT` | 10 | 23 | 1.2% | 1844 GFLOPS/s |
| `ROPE`, `CPY`, `CONT`, `ADD` | 110 | 15 | 0.7% | |

## The MoE cost does not scale with tokens

Same shape, `iq2_s m=512 n=8 k=2048 n_expert=256`, per call:

| batch | us per call | GFLOPS/s |
| --- | --- | --- |
| 128 | 8622 | 249 |
| 512 | 9781 | 878 |

Four times the tokens costs 13 percent more time. So the expert matmul is not limited by its arithmetic
and not by reading the expert weights: it is dominated by something that costs the same whether the
batch is 128 or 512. The shape is 256 experts with 8 used per token, so the natural suspect is work
done per expert slot rather than per token, most of it on empty slots at these batch sizes.

That also explains why ubatch 512 beats 128 but 1024 does not help: a pass of 512 amortises the fixed
part across more tokens, and beyond that something else in the pass gets worse.

## The tokenizer is not the problem, and it is not close

Measured with `scripts/research/tok_bench.sh`'s harness, compiled against `build-vk`, on the 250 KB
corpus in `scripts/research/tok_corpus/`, 20 repetitions:

| path | ms/rep | MB/s | tokens/s | checksum |
| --- | --- | --- | --- | --- |
| current merge table | 93.457 | 2.68 | 703208 | `ba424148b2873e23` |
| `LLAMA_TOKENIZER_LEGACY=1` | 99.766 | 2.51 | 658742 | `ba424148b2873e23` |

So the merge table work bought about 7 percent, byte identical. The plan's target for it was 10M
tokens/s from a 342k baseline, and this is 0.70M tokens/s, so that target is not met by a factor of 14.

It does not matter for prompt processing. Tokenising a 512 token prompt is about 2 KB of text, which at
2.68 MB/s is 0.75 ms against 2200 ms of prefill: **0.03 percent**. Even a 100k token prompt, 400 KB of
text, is 150 ms against 435 s. The tokenizer can be 14x off its target and it would still not show up in
prompt latency. If it is to be optimised, it should be for its own sake, not for this.

## What to implement, in order of measured payoff

1. **`MUL_MAT_ID` expert iteration.** 47 percent of the pass, at 878 GFLOPS/s for `iq2_s` against 2636
   for dense `q6_K` on the same hardware, and insensitive to batch size. If the fixed per-expert-slot
   cost is removed, the threefold gap to the dense rate is the headroom, worth roughly 700 ms of 2220,
   so about 1.5x on prompt processing.
2. **`CONCAT`.** 8 percent for 30 calls at 5.4 ms each. A concatenation should not cost as much as a
   delta-net layer. Cheap to look at, and a pure data movement win if the shape or the path is wrong.
3. **`GATED_DELTA_NET`.** 8 percent, 5.4 ms per call across 30 layers.
4. **The `RMS_NORM` family.** 4.6 percent, with individual calls at 570 us for a 128 wide norm over 512
   tokens, which is a poor rate for elementwise work and worth a look for the same reason as `CONCAT`.
5. **Flash attention configuration.** 1.2 percent. This is the prefill plan's first task and the measured
   smallest lever. Re-ordering that plan is the cheapest change available here.

The global constraint from the prefill plan holds: no changes to model weights or their quantization, so
the MoE gap has to be closed in the kernel, not by moving the experts to a cheaper type.

---

## Follow-up: both of my first two hypotheses were wrong

Date: 2026-09-27, same session

**Correction to the section above.** It attributed the MoE cost to per-expert-slot work and predicted
about 1.5x on prompt processing from fixing it. That inference came from comparing GFLOPS/s across
*different quantization formats*, which is not a valid comparison: `iq2_s` does far fewer FLOPs per byte
than `q6_K`, so a lower FLOP rate is expected and says nothing about efficiency. Checked against the
fast path and the byte accounting, both hypotheses fail.

### The kernel is already on the fast path

`MUL_MAT_ID` has no shader of its own: it is `mul_mm.comp` with `MUL_MAT_ID` defined, and it has two ways
to find the rows belonging to an expert, a precomputed packed table and a per workgroup scan.

The precomputed one is enabled: `hoist_row_ids` requires `n_as <= 1024`, both `nei0` and `nei1` to fit 16
bits, and the table to fit one binding, and our shape is 256 experts, 8 used per token, 512 tokens.

Shared memory is not the limit either. For `iq2_s` the LUT is 8192 bytes, the largest of any type here,
and against a 65536 byte limit the three `mmqid` warp tiles come to 12560, 16928 and 25664 bytes, so
`mul_mat_id_s`, `_m` and `_l` are all still supported and no fallback to the vector path happens.

### What the op actually costs

`MUL_MAT_ID iq2_s m=512 n=8 k=2048 n_expert=256 batch=512`, 9781 us per call:

| quantity | value |
| --- | --- |
| expert weights read, 256 experts at 2.0625 bpw | 69.2 MB |
| arithmetic, 512 tokens x 8 experts | 8.59 GFLOP |
| arithmetic intensity | 124 FLOP/byte |
| achieved bandwidth | 7.08 GB/s of a 68 GB/s ceiling |
| achieved rate | 878 GFLOPS/s against 2900 for dense q6_K |
| time if it were at the memory ceiling | 1018 us |

So the op is not at the memory ceiling, not at the dense matmul's compute rate, and not limited by
expert-slot bookkeeping either. At 124 FLOP/byte it should be compute bound, and it is: the bound is in
the `iq2_s` dequantize path, whose grid lookup is why that type reserves 8192 bytes of shared memory
when `iq3_s` needs 2048 and `iq4_xs` needs 64. That is deep kernel work in the dequant, and the type is
fixed by the no-requantization constraint.

The batch insensitivity is explained without any per-slot theory: at both batch 128 and batch 512 nearly
every expert is touched, so the weight bytes are the same, and the extra tokens add little on top of a
dequant bound.

### `CONCAT` is the better first target

It is the delta-net convolution input, `delta-net-base.cpp:472`, `ggml_concat(conv_states, qkv_mixed, 0)`,
once per state layer, 30 calls a pass.

| quantity | value |
| --- | --- |
| elements, inner 4096 x (kernel 4 + 1) x 512 tokens | 10.5 M |
| bytes moved, read plus write | 83.9 MB |
| measured | 15.6 GB/s, 23% of the ceiling |
| at the ceiling | 1234 us against the measured 5380 |
| headroom | about 4.4x, worth roughly 120 ms of a 2220 ms pass |

A concatenation is a copy. It should run near the ceiling, and it is running at a quarter of it. That is
the cheapest real win in this profile, and unlike the MoE line it does not depend on the model's
types.

### State of the work

No code has been changed by this investigation. Two plausible fixes were tested as hypotheses first and
both were disproved by measurement, so nothing was worth committing. The next concrete step is the
`CONCAT` path, not the MoE kernel.

---

## Follow-up 2: the concat experiment, and an instrument caveat

Date: 2026-09-27, same session

### The third hypothesis, also disproved

The concat shader assigns with a select:

```glsl
    data_d[get_doffset() + dst_idx] = D_TYPE(is_src0 ? data_a[get_aoffset() + src0_idx]
                                                     : data_b[get_boffset() + src1_idx]);
```

Both loads are written unconditionally, and for the source that is not wanted the index underflows, so
the suspicion was two loads per element, one of them wild. The generator already has a branch variant,
`OPTIMIZATION_ERROR_WORKAROUND`, used for `get_rows` and the f16 copies, and for a dim 0 concat the split
falls on a workgroup boundary so that branch would be uniform rather than divergent. It was enabled for
`concat_i32` and measured:

| variant | CONCAT per call |
| --- | --- |
| select, first measurement | 5380 us |
| branch, three runs | 5367, 5294, 5313 us |

The change was reverted. The double load is not the cost.

### The instrument may not be attributing these two ops independently

`CONCAT` and `GATED_DELTA_NET` both come out at 30 calls and both at about 5.38 ms per call, in the same
runs:

| op | calls | us per call |
| --- | --- | --- |
| CONCAT | 30 | 5367 |
| GATED_DELTA_NET | 30 | 5378 |

They sit next to each other in the graph, once per state layer, and the perf logger takes timestamps
around dispatches in one command buffer. Two different ops agreeing to within 0.2 percent is more likely
to mean the window includes a shared dependency or barrier than to mean two separate ops cost the same.

So the previous section's claim that `CONCAT` runs at 23 percent of the ceiling with 4.4x headroom is
**not verified**. It rests on an attribution that has not been tested. Testing it is cheap: time a
variant where the concat is removed entirely, or instrument the two ops separately with a barrier between
them, and see whether the 5.38 ms follows the concat or stays with whatever it is waiting for.

Flash attention, as a cross-check on the instrument: `-fa 0` measures pp512 at 232.69 against 230.27 with
`-fa 1`, so the 1.2 percent the profile attributes to it is consistent with turning it off entirely.

### Where this leaves the prefill work

Three hypotheses have now been tested and all three are disproved: expert slot bookkeeping, batch
amortisation, and the concat double load. The one remaining number with headroom is the MoE expert
matmul at 7 GB/s and 124 FLOP per byte, whose bound looks like the `iq2_s` dequantize path. The honest
next step is not to optimise anything yet but to test the instrument on the two ops it may be
mis-attributing, because every remaining target except the MoE line depends on those numbers.

---

## Follow-up 3: how the dequantize actually works, and why `iq2_s` is the slow one

Date: 2026-09-27, same session

There are two matmul paths for quantized weights on Vulkan, and which one a type gets decides its speed.

**The int dot path.** `mul_mmq.comp` dequantizes the weights into an int8 domain and uses integer dot
products against the activation quantized to q8_1. On this GPU that is the fast path, because RDNA2 has
`dot2`, and it is why the dense `q6_K` and `q8_0` matmuls reach 2430 to 3372 GFLOPS/s.

**The fp16 path.** `mul_mm.comp` dequantizes the weights to fp16 into shared memory through `store_a`,
then runs an fp16 matrix multiply reading them back.

The types each path supports:

| path | types |
| --- | --- |
| int dot, `mul_mmq_funcs.glsl` | Q2_0, Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, **IQ3_S**, **IQ4_XS**, MXFP4 |
| fp16, `mul_mm_funcs.glsl` | F32, F16, BF16, IQ1_S, IQ1_M, **IQ2_XXS**, **IQ2_XS**, **IQ2_S**, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4, NVFP4 |

**`IQ2_S` is only in the second list.** It has no int dot kernel, so it is dequantized to fp16 and run as
an fp16 matmul, on a GPU whose integer dot product rate is what the dense matmuls are exploiting.

That lines up with the profile, where the three expert types in this model split exactly as the two
lists do:

| expert tensor | calls | total ms | rate | path |
| --- | --- | --- | --- | --- |
| `iq2_s` m=512 k=2048 | 78 | 763 | 878 GFLOPS/s | fp16 |
| `iq3_s` m=2048 k=512 | 39 | 269 | 1230 GFLOPS/s | int dot |
| `iq4_xs` m=2048 k=512 | 3 | 18 | 1453 GFLOPS/s | int dot |

The two that take the int dot path run at 1.4 to 1.7x the rate of the one that does not, and the one
that does not is the largest single item in the whole prefill pass.

### What the fp16 path does per 8 weights of `iq2_s`

From `mul_mm_funcs.glsl`, per thread and per 8 k-values: four integer divides and shifts to derive the
block and group indices, one shared memory lookup into `iq2s_grid[1024]`, where each entry is a `uvec2`
so every access is 8 bytes wide and irregular, two `unpack8` calls, eight conditional negations driven by
the sign byte, and four shared memory stores of `FLOAT_TYPEV2`. Then the matmul reads all of that back
out of shared memory and does fp16 FMAs.

Against that, the int dot path keeps the weights in a packed int8 form and dots them against a q8_1
activation, with no fp16 round trip at all. That is the whole difference.

### The change, its size, and its expected value

Adding an int dot kernel for `IQ2_S` means a new branch in `mul_mmq_funcs.glsl` that maps the 2 bit codes
through `iq2s_grid` into the int8 domain, applies the group scale `d * 0.25 * (0.5 + scale)` and the sign
byte in integer arithmetic, and packs them for the dot product, plus the shared memory budget for the
8192 byte LUT in `ggml_vk_matmul_int_shmem_support`, plus the generator and host entries. It is a new
kernel variant, not a tweak, and the sign and scale handling in the integer domain is exactly where a
mistake would be silent, so it needs the CPU and Vulkan outputs compared, as the planar3 work did.

Expected value, from the table above: if `iq2_s` reached the 1450 GFLOPS/s that `iq4_xs` gets on the int
path, its 763 ms would become about 460 ms, so a pass of 2220 ms would become about 1920 ms, and prompt
processing would go from 230 to about 265 t/s. That is roughly 15 percent, which is the largest single
win available in this pass, and it is the only one left that the measurements support.

### What was not done

No code was written for this yet. The dequantize question has a specific answer that is structural, not a
micro-optimisation: the type is missing from the fast path's type list. Before adding the kernel it is
worth confirming the attribution on a cheap shape, since three earlier hypotheses in this document were
wrong, and the `CONCAT` numbering above is already flagged as suspect.

---

## Baseline before the `iq2_s` int dot path

Date: 2026-09-27, same session. Commit `7939ef6b0`. Taken in one session, back to back.

| measurement | value |
| --- | --- |
| CPU pp512, `-ngl 0 -t 8` | 97.38 t/s |
| Vulkan pp512, `-ngl 99 -fa on`, current fp16 path | 231.52 ± 0.42 t/s |
| `MUL_MAT_ID iq2_s` per call | 9561 to 9794 us, 877 to 898 GFLOPS/s |
| `MUL_MAT_ID iq2_s` per pass | 78 calls, 746 to 764 ms |

Correctness reference, a 180 word prompt and 8 greedy tokens at seed 1: CPU and Vulkan wrote 938 bytes
and `diff` reports them identical. That file pair is the check the new kernel has to pass, and it is kept
at `/home/herlanggays/.jcode/scratch/iq2mmq/`.

Note on how those runs end: `llama-completion` writes its output and then does not exit, so both runs
report `rc=124` from the timeout despite having produced complete, identical output. That is the tool, not
the run, and it cost a false alarm here.

## The int dot path for `iq2_s`: design, and the one thing that is not like `IQ3_S`

`IQ3_S` is the template, because it also uses a grid lookup, a sign byte per group of eight, and a four bit
scale per 32 values. Three functions need an `IQ2_S` arm: `block_a_to_shmem`, `block_a_to_registers`, and
`mmq_dot_product`. The grid is already compatible:

- `iq2s_grid` is 1024 entries of `uvec2`, 8 bytes, 8 values per entry as magnitudes. The largest magnitude
  in the table is 43, so the values fit int8 and the sign can be applied with the same branchless
  `(v ^ m) - m` the `IQ3_S` arm uses. No precision is lost packing them.

The obstacle is the scale. `IQ2_S` carries **two scales inside one 32 value group**:

```c
        db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
        db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
```

`db[0]` applies to the first two of the four eight-value sub-groups and `db[1]` to the last two, so the 32
packed int8 values carry two different multipliers. The `IQ3_S` template has one `d` per group and one dot
product per group, so it cannot be copied as is: the arm needs either a dot product split in two halves
with a scale each, or a change to how the block scale is carried. That is where a mistake would be silent,
because both paths would still produce plausible text.

State: **nothing implemented.** The baseline is taken, the semantics are pinned down against
`dequantize_row_iq2_s` in `ggml/src/ggml-quants.c`, and the remaining work is the three functions, the 8192
byte LUT in the shared memory budget of `ggml_vk_matmul_int_shmem_support`, the generator entries, and the
host type list. That is a kernel addition of a few hundred lines with a verification step, and it should be
done in one sitting rather than half.

---

## Does moving `iq2_s` to int8 degrade the values?

Date: 2026-09-27, same session

**The weights do not degrade at all.** The dequantized weight is `grid_value * scale`, and the int8 path
stores `grid_value` as int8 and keeps `scale` outside the integer domain. The largest grid magnitude in
`iq2s_grid` is 43, so every weight fits int8 exactly, with no rounding, and the sign is applied exactly by
the branchless `(v ^ m) - m`. The accumulation is `dotPacked4x8EXT` into int32, and 32 terms of at most 43
by 127 is about 175 thousand, nowhere near overflow. So the integer side is exact, and the fp16 path it
replaces is not more precise about the weights either: it also represents these small integers exactly.

**The activation is what changes.** The int path quantizes the activation to `block_q8_1`, which is 32 int8
values plus an fp16 scale per block, against one fp16 scale and a sum term. The fp16 path leaves the
activation in fp16. That is the whole trade.

The important context is that this trade is already made everywhere else in this model. The dense attention
projections run `q6_K` on the int path, and of the three expert tensors, `iq3_s` and `iq4_xs` are already
on it. So the int path is not a new precision decision for this model, it is the existing one, and
`iq3_s` in particular is the other half of the same MoE:

| expert tensor | calls | path today |
| --- | --- | --- |
| `iq2_s` | 78 | fp16 |
| `iq3_s` | 39 | int8 |
| `iq4_xs` | 3 | int8 |

And the existing evidence on what the int path costs: with `iq3_s` and `iq4_xs` already on it, the model's
greedy output is byte identical to the CPU reference. Adding `iq2_s` to the same path does not introduce a
new kind of approximation, it removes the one type that was not using it.

Order of magnitude for the activation change: `q8_1` picks its scale as half the block maximum, so a
single element is off by at most about 1 part in 254 of the block peak, and the error across the 32 terms
of a dot product partly cancels. Against a model whose expert weights are already 2 bit, that is a second
order term rather than the dominant one.

### Baseline for the degradation number

| measurement | value |
| --- | --- |
| PPL, 24 chunks of 512, technical prose, fp16 path | 5.9148 +/- 0.19103 |
| greedy output vs CPU, 180 word prompt, 8 tokens | byte identical, 938 bytes |

The perplexity uncertainty is 3.2 percent, which is larger than the effect being looked for. A final
estimate PPL comparison at this size cannot resolve a sub one percent change. The two instruments that
can:

- The greedy output equality, which is exact and deterministic, and which any real error in the sign or
  scale handling would break on the first token.
- A paired comparison, running both builds over the same chunks and comparing chunk by chunk, so the
  corpus variance cancels instead of dominating.

So the answer to the question asked is: the values are not degraded, the activation is quantized the same
way the rest of the model already quantizes it, and the measurement that settles it is the exact output
comparison rather than the perplexity number.

---

## The int path is real on this device, and `iq2_s` is genuinely the only type missing from it

Date: 2026-09-27, same session

Checked against the built artifact rather than the source lists, using the generated variant header at
`build-vk/ggml/src/ggml-vulkan/ggml-vulkan-shaders.hpp`, which declares every shader variant this build
produced. For the MoE op:

| variant | present |
| --- | --- |
| `matmul_id_iq3_s_q8_1` | yes |
| `matmul_id_iq4_xs_q8_1` | yes |
| `matmul_id_q2_0_q8_1`, `matmul_id_mxfp4_q8_1` | yes |
| **`matmul_id_iq2_s_q8_1`** | **no, only `matmul_id_iq2_s_f16` and its dot2 forms** |

That is the same split the profile shows in speed, and it confirms the plan: the int dot family is
generated and selected on this device for exactly the types named earlier, and `iq2_s` is absent from it.
Mid-investigation I briefly thought the int path might be dead on a `dot2` device, because the generator
gates one list on `!dot2`; the header disproves that for this build, since the `_q8_1` variants are there
for the other types.

## The remaining edit list

Nothing was changed in the tree. The work is these six edits plus a build and a verification:

1. `mul_mmq_shmem_types.glsl`: an `IQ2_S` arm with `QUANT_R_MMQ 1` and a cache of
   `int32_t qs[8]` plus `FLOAT_TYPEV2 dm`, since this type needs two scales where `IQ3_S` needs one.
2. `mul_mmq_funcs.glsl`: `block_a_to_shmem`, `block_a_to_registers` and `mmq_dot_product` for `IQ2_S`,
   modelled on the `IQ3_S` arm. The dot splits into two int32 sums, `qs[0..3]` scaled by `dm.x` and
   `qs[4..7]` by `dm.y`, which is the `db[l/2]` split in `dequantize_row_iq2_s`.
3. `ggml-vulkan.cpp`, `ggml_vk_matmul_int_shmem_support`: a `block_a_size` case for `IQ2_S`, and the
   LUT accounted in `total_size`. The grid is copied into shared memory by `init_iq_shmem`, so it is
   8192 bytes of the 64 KB budget.
4. `ggml-vulkan.cpp`: `IQ2_S` added to the two `sg_create_mmq` lists, the MoE one on `tc_mmqid_int_k`
   beside `iq3_s`, and the dense one on `tc_mmq_int_k`.
5. `vulkan-shaders-gen.cpp`: `iq2_s` added to the `_q8_1` generation list so the variant exists.
6. Build, then verify against the 938 byte CPU reference, then bench pp512 against 231.52.

Step 2 is the one that must not be rushed, because the two-scale split is silent when wrong.
