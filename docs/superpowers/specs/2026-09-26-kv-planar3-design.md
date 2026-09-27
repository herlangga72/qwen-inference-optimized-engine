# Asymmetric KV cache with a planar3 V type (RotorQuant), CPU first

Date: 2026-09-26
Status: implemented on CPU and Vulkan, Vulkan speed gate still open
Outcome: `GGML_TYPE_PLANAR3_0` is in the tree as type 43. The CPU path is built and verified against
the Rust reference implementation byte for byte, including the packing, the codebook, the Givens
table and the rotation direction through `ggml_mul_mat`. Results in
`docs/research/10-kv-planar3-results.md`. The Vulkan path is
`docs/superpowers/plans/2026-09-26-kv-planar3-vulkan.md`.
Scope: a new KV-only ggml type, its CPU reference and flash attention support, KV plumbing, and a quality gate. Vulkan is a second plan.
Reference implementation: `~/ternary-bonsai-inference/src/kvquant.rs` and its `notes/kv-rotorquant-plan.md`
Related here: `docs/research/03-stage3-stage4.md` (attention and the CPU flash attention regression), `docs/research/04-stage5-diagnosis.md` (per-op costs)

## Context

This fork keeps KV for the 10 full-attention layers only; the other 30 layers hold a 2 MB recurrent
state instead. The KV cache is f16 by default, written and read as `[head_dim, kv_size, stream]` with
2 KV heads and `head_dim = 256`.

Traffic per token is `(k_dim + v_dim) x depth x bytes`, and it becomes a real share of decode only at
depth:

| depth | KV per token at f16 | share of the 2.08 GB weight stream |
| --- | --- | --- |
| 2048 | 42 MB | 2% |
| 16384 | 336 MB | 16% |
| 32768 | 672 MB | 27% |

Asymmetric K/V types already work (`--cache-type-k` and `--cache-type-v` are independent), and the
accepted set today is f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1 (`common/arg.cpp:304`).
The fork also already rotates K and V before quantizing them, with a Hadamard rotation applied as a
dense `n_rot x n_rot` matrix whenever the type is quantized (`llama-kv-cache.cpp:1426-1460`), with
`LLAMA_ATTN_ROT_DISABLE` to turn it off.

The quantization scheme here is the one validated in the reference implementation: per-row norm
split, a fixed Givens rotation of adjacent coordinate pairs, and a Lloyd-Max codebook of `2^bits`
levels matched to a rotated unit-vector coordinate. Its recorded numbers at 3 bits on the reference
model: `planar3k` logit relative difference 3.13e-2 CPU and 3.06e-2 GPU, `planar4` 2.07e-2, `f16`
4.0e-3, with greedy output identical in every mode, and error monotone in bits.

## Design

**Type.** A new KV-only ggml type, `GGML_TYPE_PLANAR3_0`. It never appears in model weight files in
this plan, only in the KV cache, which keeps the blast radius small.

- Block: 256 coordinates, which is this model's `head_dim` for both K and V. One f16 norm per block.
- Codes: 8 coordinates per 24-bit little-endian group, 32 groups, 96 B. Block size 98 B,
  3.0625 bit/coord including the norm.
- This is deliberately not the upstream `block_planar3_0` layout (a 2-bit plane plus a 1-bit sign
  plane over 128 coordinates). The reference implementation measured that split at 0.75x and the
  grouped layout at about 3x unpack speed, and took the grouped one; the f16 norm is its option 4,
  a 2% row saving, adopted here.

**Quantizer, per row of 256 coordinates.**

1. `n = ||x||`, `xhat = x / n`, with `1/n` zeroed below `1e-12`.
2. Givens rotation of each adjacent pair `(2i, 2i+1)` by `(cos t, sin t)` from a table that is
   generated once from `splitmix64` with seed 42: `u = (r >> 11) * 2^-53`, `t = u * 2*pi` in double,
   then `cos` and `sin` cast to f32. The table is shared by every layer and head.
3. Nearest Lloyd-Max centroid of the 8-level codebook for `sigma = 1/sqrt(256) = 1/16`, then the
   index is packed.

Dequantization is the reverse: centroid lookup, inverse Givens (sin negated), multiply by the norm.

Both tables are computed on the host once per head dimension and uploaded as buffers, never
recomputed in a backend kernel. The Givens table in double-precision trigonometry then cast to f32
reproduces the Rust implementation bit for bit, and the codebook has exactly one source of truth.

**Asymmetry.** K stays `q8_0`, the existing well tested type, and only V becomes `planar3_0`. That
is the split the user asked for and it has a concrete benefit: K is used in the scores path, where
storing it rotated would complicate the kernel, while a quantized V only needs the attention output
inverse-rotated once per query per head.

**Rotation interaction.** `attn_rot_k` and `attn_rot_v` must not fire when the KV type is
`planar3_0`, because the type carries its own rotation. The condition goes where the flags are
computed; the Hadamard path is unchanged for every other quantized type.

**Integration points.**

- `ggml`: the enum entry, a `ggml_type_traits` entry (block size 256, 98 B per block, quantization
  flag, dequantize to f16/f32), and a reference quantize/dequantize used by the KV insert path and
  by verification.
- `common/arg.cpp`: add the type to `kv_cache_types` so `--cache-type-v planar3_0` parses.
- CPU flash attention: dequantize a planar3 V block into the accumulation and apply the inverse
  Givens rotation once per query per head. This is the one substantial new kernel in the first plan.
- KV state save and restore: rows are self-describing, so no extra state; still verified by a
  `llama_state_seq` round trip.

**Split, as approved.**

- **Plan 1, CPU, quality gated.** Type, reference quantize/dequantize, KV plumbing, CPU flash
  attention, and the quality measurement below. Speed is measured and reported but is not the gate,
  because CPU flash attention measured 44% slower than the non-flash path at 16k
  (`docs/research/03-stage3-stage4.md`), which can mask the byte saving there.
- **Plan 2, Vulkan, speed gated.** Add the type to the Vulkan flash attention path. At 16k the KV
  traffic falls from 336 MB to about 117 MB per token and at 32k by about 14%, and Vulkan flash
  attention is a 41% win rather than a loss, so this is where the premise is actually tested.

## Validation and the gate

Depth points 2048, 16384 and 32768, K at `q8_0` in every case, V at `f16` as the baseline and at
`planar3_0` as the candidate, same prompts, same seed, `--temp 0`. Measurable with tools in this tree
(`llama-perplexity`, `llama-completion`, `llama-bench`), so the gate does not depend on a metric the
fork cannot produce:

1. **Perplexity.** `llama-perplexity` on a fixed corpus built by concatenating `src/*.cpp`,
   `src/models/*.cpp`, `common/*.cpp` and `ggml/src/ggml-cpu/ops.cpp` in sorted order, truncated to
   512 KiB (about 130k tokens), at depths 2048 and 16384. Gate: relative increase at or below 2%
   against the f16-V baseline at both depths. Between 2% and 3% is a judgement call with the numbers
   on the table; above 3% the premise fails and the work stops before plan 2. The measured value is
   recorded either way, not just the verdict.
2. **Greedy continuation.** A 64-token `--temp 0` continuation at depth 2048, f16 V against planar3
   V. The texts must match for the first 64 tokens. Quantization does perturb logits, so a
   divergence is not automatically a failure; if one appears, record its position and the logit gap
   at that step, and treat a divergence inside the first 16 tokens as a failure.
3. **KV memory** as reported by the loader, expected to drop about 2.7x for the V half.
4. **CPU speed** at 16k and 32k, reported with the flash attention caveat above. Not a gate.
5. **State round trip**: `llama_state_seq` save and restore with a planar3 V cache, output identical
   to a run without the round trip.
6. **Rotation stacking**: a planar3 V cache with `LLAMA_ATTN_ROT_DISABLE` unset and set must produce
   the same output, proving the Hadamard path is not applied on top of the type's own rotation.

## Risks

- Quality at 3 bits on this model is unproven: the reference numbers come from a different model with
  a different KV geometry. The gate above is what decides, and the plan must record the measured
  delta rather than only pass or fail.
- `blck_size = 256` assumes `head_dim` divides 256. True here for both K and V. The type must refuse
  or fall back for other shapes rather than miscompute, and the plan must include that check.
- Two rotation mechanisms now exist in the tree. The condition that keeps them from stacking is the
  riskiest line of the change and gets an explicit test: a planar3 V cache must produce the same
  output whether or not the Hadamard path would otherwise have applied.
- The type must not silently become usable for weights; a guard in the quantizer or a comment plus a
  test that `llama-quantize` refuses it is enough for now.

## Out of scope

- Quantizing K to planar3 as well (the reference implementation's `planarNk` mode). K stays `q8_0`.
- The GQA-aware read restructure from the reference notes (reading each KV row once per KV head,
  their option 6). It is a larger traffic lever, but it is a kernel restructure across both backends
  and is not needed to validate the type.
- f32-norm variants, 2-bit and 4-bit planar widths, and deferred prefill quantization (their R5).
- Vulkan and any other backend in plan 1.
- Removing or changing the existing Hadamard rotation for other types.
