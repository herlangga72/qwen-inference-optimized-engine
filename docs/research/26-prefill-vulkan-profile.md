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

---

## The int dot path was implemented, verified, and is not faster

Date: 2026-09-27, same session

Implemented all six edits, built clean, and ran the two gates.

**Correctness: passed.** The MoE output with `iq2_s` on the int dot path is byte identical to the CPU
reference, 938 bytes, same prompt and seed. That means the grid lookup, the sign handling, the two-scale
split and the packing are all right, including the part predicted to fail silently.

**Speed: nothing.** Same session, same command, two repetitions before and after:

| measurement | before | after |
| --- | --- | --- |
| pp512 | 231.52 +/- 0.42 t/s | 231.29 +/- 0.33 t/s |
| `MUL_MAT_ID iq2_s` per call | 9561 to 9794 us | 9413 to 9662 us |
| `MUL_MAT_ID iq2_s` rate | 877 to 898 GFLOPS/s | 889 to 912 GFLOPS/s |

The op itself moved about 1.5 percent, which is inside the variation between runs, and the pass moved not
at all. The projected 15 percent is not there.

**So the change was reverted**, by the rule the prefill plan sets out: a change that does not beat the
in-situ baseline is recorded and reverted. The tree is byte identical to before it (`git diff` against
`6c636f124` is empty) and the built variant is gone from the artifact. The code is preserved in history if
it is ever wanted:

- `6a4e23e5c` the implementation
- `fc2bb21cd` the revert

### Why the hypothesis was wrong

The projection came from comparing rates across types: `iq2_s` at 878 GFLOPS/s against `iq3_s` at 1230 and
`iq4_xs` at 1453, with the two fast types present in the int dot type list and `iq2_s` absent. The
correlation was real and the conclusion drawn from it was wrong, because the difference between those types
is not the dot domain. It is how much work the dequantize does per value:

| type | per value dequantize work |
| --- | --- |
| `iq4_xs` | one nibble and a 16 byte LUT |
| `iq3_s` | one grid entry per 4 values, one sign bit per value, one scale per 32 |
| `iq2_s` | one grid entry per 8 values, one sign bit per value, **two** scales per 32 |

Moving `iq2_s` to the int dot path does not remove any of that work, because `block_a_to_shmem` still has
to run the grid lookup, extract the eight sign bits and apply the two scales to build the int8 block. It
only makes the multiply that follows cheaper, and the multiply was not the bottleneck. The cost is in the
dequantize, in both domains.

That is the fourth hypothesis in this document to be disproved by measurement, after expert slot
bookkeeping, batch amortisation, and the concat double load. The pattern is consistent and worth stating:
in every case the candidate was plausible from the code and wrong when measured, and in this one the
mistake was instrumented by a cross-type rate comparison, which is exactly the comparison this document had
already warned about two sections earlier.

---

## The MoE op wastes its column tile, and that is the measured wall

Date: 2026-09-27, same session

This is the first explanation in this document that is confirmed by a prediction it made in advance rather
than fitted afterwards.

### The measurement

Same command, only the batch changes, `ub 2048`:

| batch | `MUL_MAT_ID iq2_s` rate | per call | pp throughput |
| --- | --- | --- | --- |
| 512 | 863 to 900 GFLOPS/s | 9.5 to 10.0 ms | 229.88 t/s |
| 2048 | **1637 to 1639 GFLOPS/s** | 21.0 ms | 225.07 t/s |

Four times the tokens costs 2.15 times the time, and the rate nearly doubles. Since the useful work is four
times larger and the time is not, the kernel was doing work that did not scale with tokens, and most of it
was not useful.

### Why

The tile selector for `mul_mat_id` picks by the narrow dimension:

```c
        if (m <= 32 || n <= 32) return 0;   // the small tile
```

and the small tile for the int path is 32 columns wide. Our MoE has one column per token routed to that
expert, which at batch 512 with 8 experts over 256 experts is about 16, and at batch 2048 about 64. So:

| batch | useful columns per expert | column tiles of 32 dispatched | useful fraction |
| --- | --- | --- | --- |
| 512 | ~16 | 1 | 50 percent |
| 2048 | ~64 | 2 | 100 percent |

At batch 512 every expert's matmul dispatches a full 32 wide tile and half of it multiplies padding. At
batch 2048 the tiles are full, and the rate doubles. The prediction is arithmetic: 2x the useful fraction,
2x the rate, and the measured 900 to 1638 is 1.82x.

The same table explains the batch insensitivity that misled an earlier section of this document: going from
batch 128 to 512 keeps the tile count at one per expert, so the time barely moves even though the tokens
grow fourfold.

### What this says about lowering the dispatch

Dispatching less padding is exactly the right lever, and the two ways to try it are not equivalent:

- **A larger batch does not work.** It fills the tiles, and the op does go twice as fast, but the pass does
  not benefit: pp2048 with `ub 2048` measures 225.07 t/s against 229.88 for a 512 prompt. The other ops in
  the pass get worse over a larger micro batch, which was already measured in the throughput table earlier.
  A short prompt cannot be given more columns anyway.
- **A narrower column tile does work in principle, and the size of the prize is now measured rather than
  guessed.** If the small tile were 16 columns instead of 32, a batch of 512 would dispatch half the padding
  and the op should reach the rate it already reaches at batch 2048. That is 1.82x on its 760 ms, worth
  about 345 ms of a 2220 ms pass, or **pp512 from 230 to roughly 272 t/s, about 18 percent**.

That is the same plumbing the `iq2_s` int path needed and it is now familiar: a new narrow tile entry in
`mul_mmq_shmem_types` terms, the generator emitting the variant for that tile size, both host pipeline
lists, and the tile selector choosing it when `n <= 16`. Unlike the dequant work, this one attacks the
padding itself, which is where the measurements say the time goes.

### What is not yet tested

The narrow tile is a prediction too. It has one advantage over the previous four candidates: it is derived
from a measurement of the same kernel at a fuller tile, so the rate it would reach is observed rather than
modelled. The thing to test first, cheaply, is whether a 16 wide variant changes anything at all, before
building the full selector wiring.

### A caveat on the label, and a consistency check that holds up

The profiler prints `n` as `node->ne[1]`, which for `MUL_MAT_ID` is `n_expert_used` and is 8 in both runs. It
is not the column count. Reading it as the column count is what made an earlier draft of this document
incoherent, and it is the same class of mistake as the others listed above: the label was measured but its
meaning was assumed. The columns the kernel actually tiles over are the routed pairs, `n_expert_used *
batch`, divided among the experts.

Read correctly, the two runs give a check that the numbers agree on. Dispatched work means the workgroup
tiles, including the padding:

| batch | routed pairs | per expert | padded to | dispatched work | dispatched rate |
| --- | --- | --- | --- | --- | --- |
| 512 | 4096 | ~16 | 32 | 17.2 GFLOP | 1773 GFLOPS/s |
| 2048 | 16384 | ~64 | 64 | 34.4 GFLOP | 1638 GFLOPS/s |

The dispatched rate is the same at both batch sizes to within 8 percent. The hardware does the same work per
second either way, and the only thing that changed is how much of the tile was padding. That is a coincidence
the padding story predicts and no other story does, and it also removes the alternative reading in which the
batch 512 case is simply a badly scheduled kernel.

### Two snags found while preparing the change

Neither is fatal, but both mean this is not the one line edit it looked like:

- **`iq2_s` is not in the `mmqid_int_k` family.** That family is `Q2_K` through `Q6_K` plus `IQ3_S`, so the
  MoE op for this model uses the generic `tc_mmqid` configuration, whose small tile is also `BN` 32. The
  narrow tile has to be added there, and the two families are selected by different type lists.
- **The `mmqid` family shares `s_mmq_wg_denoms` with the dense family.** The workgroup count is computed by
  dividing the problem by these, so the N denom has to equal the N tile, and they cannot be edited in place
  without disturbing dense matmuls. The narrow tile needs its own denom triple.

Both are the same shape of work the `iq2_s` integer path needed, which is a new pipeline entry plus both
host lists, rather than a new subsystem.

---

## Two experiments on the MoE op, and the one that looked like a 1.7x win

Date: 2026-09-28, same session

Both experiments were run against the same baseline, pp512 231.52, with the op reporting 862 to 900
GFLOPS per second at 9.5 ms per call. One was a null result and one was a bug that presented as a large
speedup. The second is the more useful of the two.

### Experiment one, a narrow matmul_id column tile, null

The small `matmul_id` tile is 32 columns wide. On this device `BLOCK_SIZE` and `WARP` are both 32, so one
workgroup is one warp covering a single 32 by 32 tile, and the padding argument above said half of it was
wasted. BN and WN were both halved to 16 in the subgroup branch, with their own `wg_denoms`, which the
selector reaches without any change because it returns entry 0 whenever n is 32 or less.

Result: 878 and 899 GFLOPS per second against 862 and 900 baseline, pp512 230.99 against 231.52. A null
result, less than the run to run noise. The half empty tile was not the cost.

### Experiment two, shrinking the x grid, which broke correctness

The fit on the two batch sizes gives a clean two parameter model, T = L + W over r:

| quantity | value |
| --- | --- |
| marginal rate r | 2244 GFLOPS/s, near the dense 2430 to 3372 |
| fixed overhead L | 5.67 ms per call |
| over 78 calls | 442 ms, 20 percent of a 2220 ms pass |

The grid is `{ m, nei1, n_as }`, which for this op is `{ 512, 8, 256 }`, 1048576 workgroups. The shader
derives `ir = WorkGroupID.x % blocks_m` and `ik = WorkGroupID.x / blocks_m` with `blocks_m` 16, so passing
`m` looked like dispatching 32 workgroups per M tile when one would do. The x grid was changed to
`CEIL_DIV(m, wg_denoms[0])`, which is 16.

The measurement was spectacular: 1605 and 1558 us per call against 9542 and 9780, 5351 and 5513 GFLOPS per
second, **pp512 231.52 to 388.87, a 68 percent gain**. The overhead model predicted about 24 percent, so the
result was better than predicted, and the reported rate of 5.5 TFLOPS per second sat at 82 percent of the
680M's 6.7 TFLOPS per second fp16 peak. Every number was consistent and pleasing.

Perplexity said otherwise. On the same input, same flags, baseline build against changed build:

| build | perplexity |
| --- | --- |
| baseline | 6.0495 +/- 0.36037 |
| x grid reduced | **90105.6877** |

### What was actually wrong

`ik` is not dead for the MoE path. It indexes the split k slices, and the quant shader accumulates
`start_k = ik * p.k_split` over `end_k = min(p.K, (ik + 1) * p.k_split)`, with the partial results summed by
a reduction afterwards. The call site even says so: `prealloc_split_k_need_sync` is set on the path out of
`ggml_vk_matmul_id`. Dropping `ik` above zero therefore dropped most of the k range, which is why the op got
faster and why the answer was wrong.

The source reading that produced the mistake was of `mul_mm.comp`, whose `MUL_MAT_ID` branch does set
`start_k = 0` and `end_k = p.K`, and whose output offset for `ik` is guarded by `#ifndef MUL_MAT_ID`. The
quantized path for this model does not use that branch. It uses `mul_mmq.comp`, where the k split is live.

### The lesson, which is about instruments again

`GGML_VK_PERF_LOGGER` reports a rate computed from the *nominal* shape of the operation, not from the work
actually done. A change that silently computes less work therefore reports a large speedup, and the reported
rate can even land just under the hardware peak while doing so, which is exactly what a plausible
optimization looks like. The denominator moved, and the numerator was assumed.

This is the ninth time in this work that a measuring instrument agreed with me for the wrong reason. The
check that caught it was the only one in the set that measures the *output* rather than the time: a
deterministic perplexity value, compared between two binaries built from the same tree, same input file,
same flags.

Practical instruments established here, all cheap:

- `llama-perplexity -f prose.txt -ngl 99 -fa on -c 2048 --chunks 2` gives a deterministic single number.
  Baseline for this model and file is 6.0495. Use it as an A/B on every grid or kernel change before
  believing any speedup.
- `llama-cli` needs `-st` or it runs the conversation until the context fills; one runaway wrote 83 MB
  before it was killed.
- CPU against Vulkan byte equality is *not* available at `-c 2048` on this model: the baseline build alone
  differs between backends there, 1803 bytes against 2050, so that comparison proves nothing and only
  looked like it did.

### State after these experiments

The x grid change is reverted. Revert verified: perplexity 6.0495, exactly the baseline value, and pp512
231.87. The narrow tile change is also reverted, as a null result. The tree is clean, and both the padding
story and the overhead model are documented here as wrong or incomplete: the cost is in the split k work and
the grid that indexes it, which is the next thing to read, in `mul_mmq.comp`.

---

## The x grid cannot be reduced, and two instruments failed their own checks

Date: 2026-09-28, same session

### The measurement

The x grid divisor was made a runtime knob (`GGML_VK_MMID_XDIV`, 1 reproduces upstream) so that one
build could sweep it, gated on perplexity, which had already returned 6.0495 twice to five significant
figures on independent runs:

| divisor | x grid | perplexity |
| --- | --- | --- |
| 1 | 512 | 6.0495 |
| 2 | 256 | 81555.6552 |
| 4 | 128 | 97551.4202 |
| 8 | 64 | 102641.0678 |
| 16 | 32 | 94190.0921 |
| 32 | 16 | 90105.6877 |

Halving the grid already destroys the output. So `x` is not padded, and the 1048576 workgroups are not
waste: the grid is `blocks_m` times the number of k slices, which is 16 times 32 for K 2048, over 256
experts. The y dimension, 8, does exit early, on `data_expert_count[expert_idx]`, at the top of `main`.

That means the MoE path splits k into 32 slices of 64 while the two shaders this document already quoted set
`start_k = 0` and `end_k = p.K` for `MUL_MAT_ID`, and the MoE push constant has no `k_split` field at all.
The contradiction is unresolved. A debug print placed in `ggml_vk_matmul_id` never fired, and the string was
not found in `libggml.so`, which turned out to be the wrong library to search because the Vulkan backend
builds as its own. The leading hypothesis is a split k variant with an atomic or reduction store, and it is
recorded as a hypothesis rather than a finding, because the instrument designed to settle it did not work
and was not pursued to the end.

What is established is the conclusion that matters for the optimization: **the x grid is load bearing and
cannot be shrunk.** The earlier 68 percent result was computing one thirty second of the k range.

### Two instruments that failed

The `llama-cli` output is **not deterministic in this build**. Two runs, same binary, same flags, same
prompt, same seed produced 1551 and 1543 bytes with different checksums. Everything this session concluded
from `llama-cli` comparisons is therefore retracted:

- the divisor sweep verdicts, which called every divisor broken, were noise and would have said the same
  about a correct change;
- the 3 byte difference between the baseline and the changed build (1803 against 1806) was noise, not the
  corruption it was read as;
- the claim that CPU and Vulkan disagree at `-c 2048` (1803 against 2050) was noise as well. The earlier
  session's 938 byte equality may still hold, but it was not reproduced here and should not be relied on
  without a determinism check.

Perplexity passed the determinism check that `llama-cli` failed, on the same machine, same session. It is
the only instrument in this work so far that has measured the output and could be shown to be repeatable,
and it is what caught the corrupted k range. It should be the primary gate, with `llama-cli` used only for
eyeballing.

The lesson count is now ten, and this pair sharpens it. It is not enough for an instrument to be plausible
and to agree with the hypothesis. It has to be shown repeatable on this machine before its verdicts count,
and any check that greps a library, a log, or a file must first be shown to be looking at the right thing.

---

## The x grid is load bearing, measured by scaling rather than by output

Date: 2026-09-28, same session

Perplexity said the x grid cannot be shrunk, and this says why. Same binary, same workload, only the
divisor changes, times are per call as reported by the perf logger:

| divisor | x | per call | rate |
| --- | --- | --- | --- |
| 1 | 512 | 9723.40 us | 883 GFLOPS/s |
| 2 | 256 | 5885.45 us | 1459 GFLOPS/s |
| 4 | 128 | 3052.56 us | 2813 GFLOPS/s |
| 8 | 64 | 1650.77 us | 5202 GFLOPS/s |
| 16 | 32 | 1671.32 us | 5138 GFLOPS/s |
| 32 | 16 | 1699.73 us | 5052 GFLOPS/s |

The top four points fit **T = 0.5 ms + 18 us per x unit**, which predicts the x 64 point at 1.65 ms exactly,
and below that the time stops falling. A linear term with a small constant means these workgroups do real
work in proportion to their number. They are not exiting early, and their cost cannot be removed by making
the grid smaller, only correctness removed.

This also corrects the accounting two sections above. That section said only 4096 of 1048576 workgroups do
work, a useful fraction of 0.39 percent. That number came from treating the grid as M tiles alone, which is
the x 16 configuration, which is the broken one. Counting it properly, 16 M tiles times 32 x units times 256
experts is 131072 useful workgroups, 12.5 percent, and it is the y dimension of 8 that exits early on
`data_expert_count[expert_idx]`.

### What this closes off

The x dimension of the MoE grid carries real per-unit work and is required in full. The 68 percent result was
that work, removed. There is no padding to reclaim there, so the earlier targets in this document, the
padding of the column tile and the over dispatch of the x grid, are both now falsified by measurement.

What remains unshown is the mechanism, because the sources read here say the MoE path does not split k while
the behaviour says the x axis carries 32 units of real work per M tile. Two candidates are a split k with an
atomic store, which would make the reduction traffic proportional to the slice count, and something in the
row id path that the x index partitions. The probe built to distinguish them never printed, and the
instrument meant to name the shader was checked against the wrong library.

### A constraint worth stating plainly

If the mechanism does turn out to be a split k with a host chosen slice count, then changing that count
changes the order in which partial sums are added, and therefore changes the last bits of the result. That
would preserve correctness in the usual sense and break this project's byte equality standard, which has
been the acceptance test throughout. Any work on the slice count should be treated as a change to that
standard, not as a free optimization, and decided explicitly rather than as a side effect.

---

## Correction: the narrow tile experiment measured a tile the op does not use

Date: 2026-09-28, same session

The earlier sections of this document conclude that the padding story is falsified by the narrow tile
experiment. That conclusion is wrong and is retracted here.

A probe placed in `ggml_vk_matmul_id`, printed during prefill where the op actually runs in batch mode,
names the pipeline and its denoms:

```
matmul_id_subgroup_iq2_s_f32_f16acc_aligned_1 m=512 n=8 k=2048 nei0=8 nei1=512 n_as=256 denom0=64 denom1=64
matmul_id_subgroup_iq3_s_q8_1_1              m=2048 n=8 k=512  nei0=8 nei1=512 n_as=256 denom0=64 denom1=64
```

`denom0` is 64, and the small entry of `tc_mmqid` carries `s_mmq_wg_denoms`, which is 32. So the op runs on
the medium tile. The narrow tile experiment changed `s_warptile_mmqid`, the small entry, which this op never
selects. It measured nothing about the column tile, and the padding hypothesis is therefore **untested**, not
falsified. The lesson is the same one that has been recurring: the experiment was not shown to be connected to
the thing it claimed to test.

The corrected result from the timing sweep stands: the x grid carries real, proportional work and cannot be
shrunk. The narrow tile has to be applied to the medium tile to be a real test, which is a different edit,
`m_warptile_mmqid`, along with `m_mmq_wg_denoms` for the N denom.

## Correction: ids is [n_expert_used, n_tokens], so the y grid is 512

The probe also settles the grid geometry, which this document had wrong twice. `nei0` is 8 and `nei1` is
512, and those are `ids->ne[0]` and `ids->ne[1]`, so the ids tensor is `[n_expert_used, n_tokens]`. The y
grid is therefore `nei1`, which is the token count, 512, and not 8. The shader exits those workgroups at
`if (ic * BN >= _ne1) return;` where `_ne1` is the number of columns that expert actually received, about 16
at this batch size, so the y over dispatch is very large in nominal terms and cheap per workgroup.

Earlier sections of this document describe the y grid as 8 and derive the over dispatch from it. Those
numbers are wrong and should be read as superseded by this section. What survives is the direction: this op
dispatches a grid far larger than its useful work in both the x and y axes, and the useful columns per expert
are far fewer than the tile width, but which of those is worth attacking is now open again rather than
settled.

---

## The MoE x axis has no split k behind it in the host code

Date: 2026-09-28, same session

The leading hypothesis for why the x grid is load bearing was a split k with a partial buffer or an atomic
accumulation. The host code does not support that:

- `ggml_vk_matmul`, the dense path, computes `k_split = ROUNDUP_POW2(CEIL_DIV(k, split_k), 256)`, dispatches
  `x = CEIL_DIV(m, wg_denoms[0]) * wg_denoms[0] * split_k`, passes a `split_k_buffer`, and reduces afterwards
  with `pipeline_matmul_split_k_reduce` over `m * n * batch`.
- `ggml_vk_matmul_id` takes no `split_k`, has no `split_k_buffer` parameter, and dispatches exactly
  `{ m, nei1, n_as }`.

So the MoE path has no host driven split k, and no reduction pass, and yet with the medium tile selected,
`blocks_m` is 8 and the required x of 512 means 64 separate contributions per M tile, every one of which
changes the output when removed. The mechanism is **unexplained**, and the earlier sections of this document
that assert a split k should be read as disproved rather than as background.

There is one host detail that would make the arithmetic much less strange and that is worth one grep before
any further experiment: whether `ggml_vk_dispatch_pipeline` divides the counts it is given by the pipeline
`wg_denoms`, or uses them directly as workgroup counts.

| reading | grid | consequence |
| --- | --- | --- |
| counts divided by denoms | 8 x 8 x 256, 16384 workgroups | the op is normal compute, 2048 useful, y over dispatched 8x |
| counts used directly | 512 x 512 x 256, 67 million workgroups | the op is almost entirely launch overhead, 0.003 percent useful |

These predict completely different next moves, and the second one would mean the 9.5 ms is mostly dispatch
at roughly 7 billion workgroups per second, which is at the edge of plausible for 12 CUs. The measurement in
hand, that time is linear in the x count down to 64 and then flat, is consistent with the first reading and
not obviously with the second, but it is not conclusive either way.

### Where this leaves the MoE work

Established by measurement: the x grid cannot be shrunk; the y grid is the token count rather than 8; the op
runs on the medium tile, 64 columns wide, while an expert receives about 16 columns of an actual batch. Not
established: why the x axis carries 64 load bearing contributions with no split k in the host, and whether
the grid counts are divided before dispatch.

The untested hypothesis that still has the best motivation is the column tile. It is 64 wide against about
16 useful columns, a factor of four, and the earlier attempt to test it changed the small tile, which this op
never selects. A real test is `m_warptile_mmqid`, BN 64 to 32 with WM 32 to 16 so that the warp grid stays
consistent at four warps along M and one along N, plus a dedicated denom triple because the medium entry
shares `m_mmq_wg_denoms` with the dense family. It must be gated on perplexity, since the perf logger
reports its rate from the nominal shape and the last grid experiment looked like a 68 percent win while
computing one thirty-second of the work.

---

## The medium column tile, narrowed at warp granularity, is a real win

Date: 2026-09-28, same session

This is the first change in this document that is correct and faster.

### The change

The medium entry of `tc_mmqid` in the subgroup branch, which is the tile this op actually selects, is
narrowed in the column direction, with its own denom triple because the medium entry shares the dense
family's:

- BN 64 to 32
- WN 32 to 16
- WM left alone at 32

### The measurements

| quantity | baseline | with the change |
| --- | --- | --- |
| perplexity on prose.txt | 6.0495 +/- 0.36037 | 6.0495 +/- 0.36037, three separate runs |
| `MUL_MAT_ID iq2_s` per call | 9542 / 9780 us | 7230 / 7042 us |
| 78 calls per pass | 762 ms | 551 ms |
| reported op rate | 878 / 900 GFLOPS/s | 1187 / 1219 GFLOPS/s |
| pp512 | 231.52 / 231.87 | 253.81, then 255.10 +/- 0.55 |
| reported op rate against the 6.7 TFLOPS/s fp16 peak | well under | under |

So about 211 ms of a 2220 ms pass, and 10 percent on pp512.

Perplexity is expected to be bit identical rather than merely close: the change moves which warp computes
which column, and does not change the order in which k is accumulated for any given output element. That is
consistent with it matching to four decimal places on three runs.

### Why WN is the lever and not BN

The shader already refuses to compute padding warps:

```
required_warp_c = (_ne1 - ic * BN + WN - 1) / WN;
if (warp_c < required_warp_c) { ... compute ... }
```

So a 64 wide tile holding 16 useful columns does not spend four times the work, it spends the work of one
WN wide warp block. WN was 32 against about 16 useful columns, a factor of two, and that factor is what
halving it recovered. BN has to follow WN because the warp grid requires BN to equal WN times the number of
warps along N.

### Why the three earlier attempts did not find this

- The first narrow tile experiment edited the **small** entry, `s_warptile_mmqid`, with denoms 32. The probe
  later showed this op selects the medium entry with denoms 64, so that experiment measured an unused tile.
- The second set WM to 16 while WN was 16, giving
  `WNITER = (WM * WN) / (WARP * TM * TN * WMITER) = 256 / 512 = 0`, so the N loop never executed at all.
- The third left WM alone but set WN 32 alongside BN 32, breaking `BN = WN * warps along N`.

The second and third both reported rates between 26 and 104 TFLOPS/s against a peak of about 6.7. **A
reported rate above the device peak means the change is skipping work, and it is cheaper to check than a
perplexity run.** That single check would have caught both immediately, and it is now the first thing to
look at for any tile change.

### What bounds further narrowing

`WNITER >= 1` requires `WM * WN >= WARP * TM * TN * WMITER`, which with tm_m 4 and tn_m 2 is 512. WM is
fixed at 32 by the warp grid, so WN 16 is the floor: WN 8 gives 256 and breaks the loop. Getting below that
needs TM or TN to change, which are device properties rather than tile ones.

### Not measured

pp2048 at `ub 512` has no baseline in this document, so whether the gain holds on longer prompts is
unconfirmed. The changed build measures 176.85 +/- 0.50 there, which is below pp512 for this workload
because prefill attention is quadratic in the sequence, and comparing it to nothing would be the same
mistake this document keeps recording.

---

## pp512 is solid, pp2048 is interference limited on this machine

Date: 2026-09-28, same session

The pp2048 figure recorded as missing above was chased, and the honest outcome is that it cannot be
established in this session, while pp512 can.

### pp512, five independent measurements per side

| build | pp512 | pool |
| --- | --- | --- |
| baseline | 231.52, 231.87, 232.37, 232.98 | 232.2 |
| with the change | 253.81, 255.10, 255.16, 255.22 | 254.8 |

Per run spreads were 0.3 to 1.0. That is plus 9.5 to 10.1 percent, and it has been reproducible all
session.

### pp2048 at ub 512, and why no number is claimed

| build | measurement |
| --- | --- |
| baseline | 196.28 +/- 47.11 (r=2), 161.70 +/- 0.54 (r=4) |
| with the change | 176.85 +/- 0.50 (r=2), 177.32 +/- 0.38 (r=2), 195.81 +/- 37.47 (r=4), 189.30 +/- 30.50 (r=6), 214.06 +/- 40.92 (r=6) |

The pattern is that both builds intermittently enter a fast mode that lifts the mean to about 196 to 214
with spreads of 30 to 47, while the uncontaminated samples sit at 161.70 and 177. The two tight samples
agree with the pp512 result in both direction and size, which is what the mechanics predict: at ub 512 each
micro batch has the same shape and the same tokens per expert as pp512, so the same relative gain is
expected. But that is an expectation supported by two samples, not a measurement, and it is recorded as
such.

No regression at pp2048 is claimed or observed. What is claimed is that the machine has an intermittent
mode that affects the longer runs, and that a contaminated mean is not rescued by more repetitions.

### A benchmarking rule for this box

On this machine a run whose spread is more than a few percent is not a measurement. r=2 was not enough for
pp2048, and both of the wide spreads here came from runs that looked perfectly ordinary at the time. Prefer
pp512 for acceptance numbers, use r=4 or more for anything longer, and treat a wide spread as a reason to
re-run rather than as a result to average.

---

## Which matmul_id entry the MoE ops select, and why the others do not matter here

Date: 2026-09-28, same session

Two lines settle this. The tile selector is called with `n = nei1` for `mul_mat_id`:

```
7407:    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, nei1, aligned, true);
```

and the probe showed `nei1` is 512, the token count, not 8. So the selector sees `(m, n)` of `(512, 512)` for
the gate and up projections and `(2048, 512)` for the down projection.

Our device takes the non coopmat branch, whose logic is:

```
if (m <= 32 || n <= 32) return 0;
if (configs.size() == 2) return 1;
if (m <= 64 || n <= 64) return 1;
return last;
```

With `(512, 512)` the first two tests fail, so if the vector held three entries the selector would return the
last one, the large tile with BM 128. The probe instead reported the selected pipeline with `denom0` 64,
which is a BM of 64, the entry that was edited, at index 1. The reading that fits both facts is that this
key's config vector holds **two** entries rather than three, so `configs.size() == 2` returns 1 directly for
any `n` above 32. This is an inference from two measurements rather than a direct measurement of the vector
size, and is recorded as such.

Two consequences for this model:

- The large `matmul_id` entry is unreachable for these keys, so narrowing it would change nothing, and the
  small entry is reached only when `nei1` is 32 or less, that is a prompt of at most 32 tokens. That closes
  the question of whether the other two entries want the same treatment: for this model they are not used.
- The dense family's tiles are a different matter and are not padding limited: there `n` is the token count,
  so a 64 or 128 wide column tile is fully used. The warp granularity trick that worked here does not have a
  target there.

Where that leaves the profile: the MoE op is now 551 ms of a pass that is still around 2000 ms, dense matmul
is about 26 percent, GATED_DELTA_NET and CONCAT about 8 percent each, and flash attention 1.2 percent. The
MoE lever is exhausted at WN 16, which is the floor the WNITER condition allows.

---

## Fresh profile, and the same narrowing on the k quant family is inside the noise floor

Date: 2026-09-28, same session

### The pass as it now stands

With the base family win in place, one profiled pass, ranked by total time, with the counts as reported:

| share | calls | op |
| --- | --- | --- |
| 27.5% | 156 | `MUL_MAT_ID iq2_s` m=512 n=8 k=2048, the gate and up projections, already narrowed |
| 12.5% | 74 | `MUL_MAT_ID iq3_s` m=2048 n=8 k=512, the down projection |
| 11.9% | 78 | `MUL_MAT q6_K` m=8192 n=512 k=2048 |
| 8.4% | 60 | `CONCAT` |
| 7.9% | 60 | `GATED_DELTA_NET` |
| 6.2% | 78 | `MUL_MAT q6_K` m=2048 n=512 k=4096 |
| 4.3% | 60 | `MUL_MAT q6_K` m=4096 n=512 k=2048 |
| 2.1% | 192 | `MUL_MAT q6_K` m=512 n=512 k=2048 |
| 1.1% | 20 | `FLASH_ATTN_EXT` |

Dense `q6_K` is about 24.5 percent across those four shapes, and it has no column padding to remove because
there n is the token count and the tiles are full. CONCAT at 8.4 percent for 60 memcpy shaped calls is a
candidate on its own merits.

A correction to the arithmetic used earlier in this document: the gate and up projections are one op each per
layer, so `MUL_MAT_ID iq2_s` is called 156 times per pass, not 78. The 78 figure came from reading a single
profiled line. The wall clock remains the measure of record, since the per op times are GPU timestamps that
do not necessarily sum to the pass.

### The down projection, narrowed, and reverted

The down projection runs on the `mmqid_int_k` family, which the first change did not touch, so the same
narrowing was applied to its medium entry: BN 64 to 32, WN 32 to 16, WM untouched, with its own denoms. The
three conditions were checked first: with BM 64, WM 32, WMITER 1, TM 2, TN 2 and WARP 32, `BN = WN * 2` is
32, `WNITER = (32 * 16) / (32 * 2 * 2 * 1) = 4` is at least 1, and the thread grid is
`(32 / 1 / 2) * (16 / 4 / 2) = 16 * 2 = 32 = WARP`.

| quantity | with the base fix only | with both |
| --- | --- | --- |
| perplexity | 6.0495 +/- 0.36037 | 6.0495 +/- 0.36037, correct |
| `iq3_s` down projection per call | 6858 us | 6709 us, 2 percent |
| pp512 | 254.8, a pool of five | 255.89 +/- 0.53 |

Two percent on a 12.5 percent share is about 0.25 percent of the pass, and the pp512 difference sits inside
the per run spread of 0.5 to 0.7. It was reverted, per the standing rule that a change which does not clearly
beat the in situ baseline is recorded and reverted rather than carried on the strength of an expectation. It
is not a regression; it is unprovable with the instruments available here, and these numbers are the record
for anyone who revisits it with a better one.

What it does establish is a bound on the technique: it pays where the op is column work bound. The gate and
up projections, m 512 and k 2048, gained 1.35 times. The down projection, m 2048 and k 512, gains about 2
percent, so its time is not in the column work, and the most likely reason is that both ops stream the same
weight bytes and the down projection is closer to streaming bound, where narrowing a column tile changes
nothing.

---

## CONCAT runs at 5.9 GB/s against a measured 44.5 GB/s ceiling

Date: 2026-09-28, same session

CONCAT is 8.4 percent of the pass, 30 calls at about 5.7 ms each. That is 171 ms for what looks like a copy,
so both the volume and the ceiling were measured rather than estimated.

### The volume

A probe in `ggml_vk_concat` printed the shapes, first call and identical thereafter:

```
[concat] dim=0 src0=3x8192x1 src1=512x8192x1 dst=515x8192x1 moved=16.09 MiB type=f32
```

That is the GDN conv state, 3 previous positions by 8192 channels, concatenated with the current qkv, 512 by
8192. Per call the traffic is 16.88 MB read plus 16.88 MB written, so 33.8 MB. At 5.7 ms per call that is
**5.9 GB/s**.

### The ceiling

`scripts/research/membw.c` rebuilt with `-fopenmp`, 8 threads, three runs: 39.34, 41.79, 44.53 GB/s, best
**44.53 GB/s**. So the concat moves data at **13 percent** of what this machine does on a plain copy. Per
pass the 30 calls are 171 ms, and at half the ceiling they would be about 45 ms.

Across 30 layers that is 1.01 GB of traffic per pass just for these concatenations. At the machine ceiling it
is 23 ms of unavoidable work, against 171 ms now.

### Why the kernel is slow

`concat.comp` is a generic elementwise kernel that copies **one 4 byte element per thread** and does this per
element:

- three integer divisions by runtime divisors, `idx / (p.ne22*p.ne21*p.ne20)` and two more, to recover the
  four indices
- a private array `o[4]` indexed by the runtime `dim`, which typically forces local memory
- a ternary load that can issue a load from both sources
- no vectorization at all, so every read and write is 4 bytes wide

### Two ways to fix it, neither attempted here

1. **A dim 0 specialization.** Map a workgroup to one output row of 515 elements, read the 512 element src1
   body with float4 loads, and write the row coalesced, with the 3 element prefix handled scalar. The catch is
   alignment: dst rows begin at `i1 * 515` elements, so a 16 byte vector store is not aligned for every row,
   which is presumably why the generic kernel exists in the first place. Writing scalar but coalesced may
   still be several times faster than the current per element index math.
2. **Remove the concat altogether.** Keep the conv state as a rolling buffer and have the conv read across the
   wrap boundary, as newer recurrent implementations do. That removes 33.8 MB per layer per pass instead of
   making it cheaper, and it is the larger change.

Both are recorded as work rather than as results.

---

## Correction: the concat rate depends on the prompt length

Date: 2026-09-28, same session

The previous section calls the concat a bandwidth failure at 5.9 GB/s against a 44.5 GB/s ceiling. That is
measured at pp512 only. Changing nothing but the prompt length:

| prompt | calls | per call | volume | achieved rate |
| --- | --- | --- | --- | --- |
| pp128 | 30 | 399.4 us | 8.5 MB | **21.3 GB/s**, 48 percent of the ceiling |
| pp512 | 30 | 5851.5 us | 33.8 MB | **5.9 GB/s**, 13 percent |

Four times the volume costs 14.7 times the time, so the achieved rate falls by 3.6 times as the volume grows.
A kernel bound by its own instruction cost, which is what the per element divisions and the missing
vectorization would imply, would hold its rate as the volume grows. So at least part of the pp512 figure is
not the kernel: it is contention with the memory traffic of the surrounding graph, buffer placement, or a
throttling effect, and this measurement does not separate them.

The 7.5 times gap quoted above is therefore wrong as a kernel property. The defensible statements are that
the kernel reaches 48 percent of the ceiling at pp128, that the pp512 cost is 171 ms per pass, and that the
difference between the two is unexplained.

This matters for the fix. A rewrite could plausibly recover up to 3.6 times, turning 171 ms into about 48 ms
per pass, but only if the pp512 case can be brought to the pp128 rate, and this measurement cannot show that.
The cheap next step, before writing a specialization, is to find out which of the two it is, for example by
timing the concat when the rest of the graph is not saturating memory.

## hoist_row_ids is on, so there is no ids scan to remove

The condition in `ggml_vk_mul_mat_id_q_f16` is `n_as <= 1024`, `nei0 <= 0xffff`, `nei1 <= 0xffff`, and
`(2 * n_as + 1 + nei0 * nei1) * 4` within `maxStorageBufferRange`. For this model that is 256, 8, 512 and
18436 bytes, so all four hold and the row ids are precomputed once per pass by `count_experts`. Each
workgroup then reads the 16 entries it needs rather than scanning the whole ids tensor.

So the ids scan noted earlier as a possible cost does not happen. The remaining item on that list is the y
over dispatch, 16 workgroups to do 1 or 2 of work, which exits at the top of `main` on `data_expert_count`
and is bounded by the dispatch rate: about 0.2 ms per call, or 34 ms per pass. The host cannot reduce that
without reading the expert counts back, which would cost a synchronization.

---

## The machine drifted 14 percent, and the memory clock explains it

Date: 2026-09-28, same session

Between roughly 01:17 and 01:38, with the tree unchanged at commit 20c5397dd and `git status` clean, pp512 was
measured first at 253.81, 255.10, 255.16 and 255.22, and then at 219.02 and 219.49 +/- 0.97. The second pair
is reproducible, so it is not noise.

The core is not the cause. `temp1_input` reads 53000, which is 53 C, and `pp_dpm_sclk` reports `2: 2200Mhz *`
with the asterisk on the 2200 entry, so the core is at its maximum clock and cool.

The memory clock is the cause:

```
pp_dpm_mclk
0: 1000Mhz *
1: 2400Mhz
```

The fabric is running at 1000 MHz with 2400 MHz available. Every memory bound kernel, which in this pass is
the concat and every matmul streaming weights, slows in proportion, which is consistent with a uniform
slowdown rather than an op specific one. Contributing to the state: 3.3 GiB of the 70 GiB swap in use,
`baloo_file` holding 4 GB and VS Code running, and `kswapd0` active. The model is 13 GiB, so page cache
pressure can also cause re-reads.

### Why the recorded A/B still stands

The 9.5 to 10.1 percent result compares measurements taken inside the same window: baseline 232.37 and 232.98
at 01:11 to 01:14, the change 253.81 to 255.89 at 01:08 to 01:17, and the earliest baseline pair 231.52 and
231.87 from before the work began, which agrees with the 232 seen later. Both sides are inside a single
stable window, and the difference is ten percent against a per run spread under one.

What this does mean is that absolute numbers from this box are only comparable within a session window. Every
result in this document was obtained in matched pairs for that reason, and this is the second independent
confirmation that it was necessary. The check to run before trusting any benchmark here is `pp_dpm_mclk`: if
it does not show 2400 MHz selected, memory bound results are not comparable to the recorded ones.

### The micro batch sweep, measured in the drifted state

| micro batch | pp512 |
| --- | --- |
| 512 | 219.02, 219.49 |
| 256 | 196.07 |
| 128 | 155.67 |
| 64 | 95.03 +/- 32.79 |

Smaller micro batch is monotonically worse, even though it makes the concat 3.7 times cheaper per token,
because the matmul calls lose more than the concat gains. Consistent with the earlier finding that a larger
micro batch is not a lever; this confirms it from the other direction. The absolute values are from the
drifted state and should not be compared with the recorded 254.8.

---

## Correction: the drift is the core clock, and the check must be taken under load

Date: 2026-09-28, same session

The previous section blames the memory clock. That was read at idle and describes nothing about a run, which
is the same mistake as trusting an instrument without checking what it covers. Sampled *during* a run that
measured 219.82 +/- 1.14:

| t | memory clock | core clock |
| --- | --- | --- |
| 4s | 2400 MHz | 533 MHz |
| 8s | 1000 MHz | 533 MHz |
| 12s | 2400 MHz | 533 MHz |
| 16s | 2400 MHz | 1504 MHz |
| 20s | 2400 MHz | 533 MHz |
| 24s | 2400 MHz | 1971 MHz |
| 28s | 2400 MHz | 1773 MHz |
| 32s | 2400 MHz | 1773 MHz |

So the fabric runs at its full 2400 MHz, apart from a single transient dip to 1000, and the **core clock** is
what moves: 533 to 1971 against a 2200 maximum. The 1000 MHz memory figure seen earlier was simply the idle
state, which is what the hardware selects when nothing is running.

The drift between the two windows is therefore in the core clock, and the likely cause is the APU power
policy responding to load the GPU does not control, with VS Code, `baloo_file` and a 3.3 GiB swap in use at
the same time. It is recorded as correlated rather than as established, because a single sampling series
cannot separate a power policy from background load.

### The check to use

`pp_dpm_sclk` and `pp_dpm_mclk` both have to be sampled **while a run is in flight**. Reading them at idle
reports the idle state and will mislead, as it did here. The practical rule is unchanged and is the reason
none of the recorded results are affected: interleave A and B so that drift cancels, as done for the MoE
change, where the baseline and the change were measured inside the same window and the difference was ten
percent against a per run spread under one.

---

## The micro batch conclusion survives an interleaved test

Date: 2026-09-28, same session

The sweep above ran sequentially inside a drifting window, so its ordering could have been drift rather than
the micro batch. Re-run interleaved, one round of each per pair:

| round | ub 512 | ub 64 |
| --- | --- | --- |
| 1 | 218.46 | 75.66 |
| 2 | 219.20 | 118.72 |

ub 512 wins both rounds by 1.8 to 2.9 times, far outside any spread, so a smaller micro batch being worse is
confirmed rather than an artifact of drift. The matmuls lose more than the concat gains; the concat becomes
3.7 times cheaper per token at ub 128 and it is not enough.

What is not reliable is the absolute ub 64 value: 75.66 against 118.72, a 57 percent difference between two
runs of the same configuration. Short, dispatch dominated runs are the most exposed to a core clock that
wanders between 533 and 1971 MHz. Any future measurement of a small effect on this box should be interleaved,
and the ub 64 figure should not be quoted as a throughput.

This is the second comparison in this session that interleaving has rescued, after the tile change, and the
third time a claim had to be corrected because a reading was taken without asking what it covered.

---

## The change is in HEAD, and a window can be 20 percent slower than another

Date: 2026-09-28, re-measured at 03:23 after a gap

This is a check that the accepted change is really present, and a measurement of how much the machine moves
between windows.

Presence, established directly rather than inferred: `m_warptile_mmqid_narrow` occurs five times in the
working tree and five times in HEAD, the assignments are `[2] = 32` for BN and `[5] = 16` for WN, the commit
that added them is 2c1448af4, and `build-vk/bin/libggml.so` is timestamped after the source file it was built
from. So the running binary contains the change.

Throughput in this window, with everything else identical:

| quantity | accepted window | this window |
| --- | --- | --- |
| pp512 | 253.81 to 255.22 | **231.25 +/- 1.81** |
| `MUL_MAT_ID iq2_s` per call | 7230 / 7042 us | **8839 / 8586 us** |

The baseline in the accepted window was 232.37 and 232.98, so this window runs at about the speed of the
baseline window while the code is the changed code. The op is 20 percent slower here than in the accepted
window.

The cause is that this box is a working desktop, not a test rig. At the time of this measurement `docker`,
VS Code and `jcode` were each between 10 and 15 percent CPU, the load average was 1.18, and earlier
`baloo_file` was holding 4 GB with 3.3 GiB of swap in use. On an APU those share memory bandwidth and the
power budget with the GPU, and the core clock wanders between 533 and 1971 MHz against a 2200 maximum.

### What this does and does not affect

The win is established by two things that are not exposed to this: the interleaved comparison, where the
baseline was rebuilt and measured against the change inside the same window, 232.37 and 232.98 against 253.81
to 255.22 with spreads under 1, and the within run op timings, 9542 and 9780 microseconds per call falling to
7230 and 7042. Both compare two builds under the same conditions rather than one build against a remembered
number.

What it does affect is any temptation to compare absolute numbers across windows. The 254.8 belongs to its
window and the 231.25 now is not evidence against it, just as the earlier 219.49 was not. It also means the
remaining work, the concat and the dense matmuls, needs interleaved rounds to resolve a few percent, and
that a busy window cannot resolve one at all.

---

## The concat specialization plan, with the two unknowns resolved by reading

Date: 2026-09-28, same session

The plan below is recorded rather than executed, because it needs a build and verify cycle the remaining session
budget cannot cover, and starting it would leave an unverified kernel in the tree. Two things that would
otherwise have been guessed are now read:

1. **The shader generator does not discover `.comp` files.** Each variant is registered by hand, for example
   `vulkan-shaders-gen.cpp` lines 961 to 964, which map `concat_i8`, `concat_i16`, `concat_i32` and
   `concat_i64` all onto `concat.comp` with different `A_TYPE`, `B_TYPE` and `D_TYPE` defines. So a new
   kernel needs its own file plus exactly one registration line.
2. **The concat works in units rather than bytes.** `ggml_vk_concat_unit_size` returns 1, 2, 4 or 8 from the
   type size, and the host switches on it at `ggml-vulkan.cpp` 8657 to choose `pipeline_concat_i8` through
   `pipeline_concat_i64`. Our concat is f32, so its unit size is 4 and only the i32 case needs a
   specialization; the other three keep the generic kernel.

### The plan, in three pieces

1. A new `concat_dim0.comp`, registered once as `concat_dim0_i32` with `D_TYPE` `uint`. One workgroup per
   output row: the row index from `WorkGroupID.x`, the position within the row from `LocalInvocationID.x`,
   the three element prefix handled by the first three threads, and no divisions and no runtime indexed
   private array anywhere. It reuses the existing `vk_op_concat_push_constants`, so the host side stays small.
2. A `pipeline_concat_dim0` created next to the other concat pipelines, with
   `sizeof(vk_op_concat_push_constants)` and the same three bindings.
3. A direct dispatch for the specialized case. `ggml_vk_concat` currently goes through
   `ggml_vk_op_f32`, which derives its grid from the element count, while this kernel wants one workgroup per
   row. So call `ggml_vk_dispatch_pipeline` directly with a grid of `{ dst->ne[1], 1, 1 }`, selected when
   `op_params[0]` is 0, the unit size is 4, and `ne[2]` and `ne[3]` are 1.

### Gate

Perplexity 6.0495, which clock drift does not affect, and the op going well below the recorded 5851 us per
call at ub 512. The op level figure is resolvable even in the busy window, because a factor of 2 to 3 is far
larger than the 15 to 20 percent window spread, unlike the few percent effect at the pass level.

---

## The concat is memory bound, not instruction bound: a specialization bought only 16 percent

Date: 2026-09-28, same session

The planned specialization was implemented, measured and reverted. It answered the question it was built to
answer, and the answer is no.

### What was built

A new `concat_dim0.comp`, registered once as `concat_dim0_i32`, with one workgroup per output row: the row
index from the workgroup id, the position within the row from the local id, one division per **workgroup**
rather than three per element, and no runtime indexed private array at all. It was wired with a pipeline
whose denoms are 1 so the grid is the row count, a new case in the element count switch to pass the row
count, and a `pipeline_override` from `ggml_vk_concat` selected when the concat is along dimension 0 in 32 bit
units with the trailing dimensions 1. An `GGML_VK_NO_CONCAT_DIM0` toggle forced the generic kernel so the two
could be compared with the same binary in the same window.

### The interleaved A/B

Same binary, same window, alternating:

| round | kernel | concat per call | pp512 |
| --- | --- | --- | --- |
| 1 | specialized | 5453.98 us | 217.75 |
| 1 | generic | 6025.81 us | 215.81 |
| 2 | specialized | 5120.33 us | 216.13 |
| 2 | generic | 6255.68 us | 215.43 |

Perplexity was 6.0495 +/- 0.36037 with the specialization in place, which is the expected result for a pure
copy and confirms correctness.

So removing all of the per element index arithmetic, which is three divisions by runtime values plus a
private array indexed by a runtime value, made the operation **1.16 times** faster and left pp512 inside the
noise at plus 0.3 to plus 0.9 percent. At 5454 microseconds for 33.8 MB the kernel is moving about 6.2 GB/s
against the 44.53 GB/s ceiling, barely better than the 5.9 GB/s it managed before.

### What that establishes

The kernel is **memory bound** at this volume, not instruction bound. The division and the private array were
not what cost the time, so the earlier reading of the kernel was wrong on that point even though the
measurements around it were right. Two consequences:

- The specialization is reverted, per the standing rule, since it does not clearly beat the in situ
  baseline. Revert verified: perplexity 6.0495, and the restored build measures 5920 us per call and pp512
  216.11, matching the generic arm of the A/B at 6026 and 6256.
- The remaining lever is not to make the copy cheaper but to **not do it**: the rolling conv state, which
  removes the 33.8 MB per layer per pass instead of moving it better. That is the direction the evidence now
  points at, and it is the larger change.

The same measurement also strengthens the earlier observation about volume: the same kernel reaches 21.9 GB/s
at 8.5 MB per call and 6.2 GB/s at 33.8 MB per call, and removing its instructions does not change that. The
effect is in the memory system at that size, not in the kernel's arithmetic.

---

## Plan for removing the concat: what the graph does and what the conv would need

Date: 2026-09-28, same session

Read rather than assumed, so the next attempt starts from the real structure.

### Where it is

`src/models/delta-net-base.cpp`, `build_conv_state`, line 472:

```
ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
```

with `conv_states` a reshaped view of the recurrent state buffer, `(conv_kernel_size - 1) x conv_channels
x n_seqs`, which is 3 x 8192 here, and `qkv_mixed` transposed to `(n_tokens, conv_channels, n_seqs)`. The
concatenation is `515 x 8192` for a 512 token batch.

### The concat serves two purposes, not one

1. It gives the separate conv op a contiguous sliding window, which is what the 33.8 MB buys.
2. It is also the **source of the state update**. `conv_state_last` is a `ggml_view_3d` of `conv_input`
   starting at `conv_input->ne[0] - conv_states->ne[0]`, that is the last 3 rows, and those are copied back
   into the state buffer by `ggml_cpy`. So the update reads the last three rows of `qkv_mixed` through the
   concatenated tensor.

Consequence: the update can be pointed at a view of `qkv_mixed` directly, which is a small graph change, but
that alone does not remove the concat while the conv op still takes a single tensor.

### The fused path does not help here

`cparams.fused_gdn_ch` defaults to true at `src/llama-context.cpp:208` and is resolved by a probe at line 538,
but the profile shows a separate `SSM_CONV_SILU` op at 60 calls, so the conv is not fused and
`build_conv_state` is on the path. The concat is structural rather than a fallback artifact.

### What a fix requires

A conv that reads from two sources: the three state rows and the new input rows, with the window shift
handled in the kernel instead of by materialising the joined tensor. Then `conv_input` disappears, the state
update reads a view of `qkv_mixed`, and the 33.8 MB per layer per pass stops being moved at all. At the
measured 44.53 GB/s ceiling that traffic is about 23 ms per pass; at the 6.2 GB/s the concat currently
achieves it is 171 ms.

### The risk areas, from the code

- `cparams.n_rs_seq == 0` and the else branch marked `[TAG_RECURRENT_ROLLBACK_SPLITS]`, which assumes the
  last `n_rs_seq + 1` tokens of a sequence are in the same ubatch. Both need the same treatment.
- The state update currently happens by copying from inside the concatenated tensor; changing its source
  changes the read-after-write relationship with the conv.
- `build_delta_net_fused` and `build_delta_net_chunking` are two paths through the same builder, and the
  conv input is built for both.

Gate as before: perplexity 6.0495, which is clock independent, the concat op disappearing from the profile,
and clocks sampled during the run with the two builds interleaved, since windows here differ by 20 percent.
