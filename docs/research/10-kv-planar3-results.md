# KV planar3 (CPU): results

Date: 2026-09-27
Machine: Ryzen 7 6800H, Radeon 680M (RADV REMBRANDT), 27.1 GiB, CPU backend.
Tree: `qwen-only-backends` at 4f5831ba2 plus the plan E change, `build-cpu` rebuilt from it.
Plan: `docs/superpowers/plans/2026-09-26-kv-planar3-cpu.md`.
Spec: `docs/superpowers/specs/2026-09-26-kv-planar3-design.md`.
Reference implementation: `~/ternary-bonsai-inference/src/kvquant.rs`.
Scratch: `/home/herlanggays/.jcode/scratch/kvplanar3/`.

## What was built

Task 1, the type: `GGML_TYPE_PLANAR3_0 = 43`, a 98 byte block (f16 norm plus 96 bytes for 256
coordinates, 3 bits each packed 8 to 3 bytes), a Givens table from `splitmix64` seed 42, a Lloyd-Max
codebook for `sigma = 1/sqrt(256)`, `quantize_row_planar3_0_ref`, `dequantize_row_planar3_0`,
`quantize_planar3_0`, the generic and CPU type traits, a CPU-facing `quantize_row_planar3_0` wrapper
and `ggml_vec_dot_planar3_0_q8_0`.

Task 2, the plumbing: `--cache-type-v planar3_0` accepted, the Hadamard path kept off the type,
`llama-quantize` refusing it as a weight type, and a head-size guard that aborts if a head is not one
block.

Task 3, the rotation: `dequantize_row_planar3_0` returns rotated-space values and the graph undoes
the rotation once per token, on the attention output. `ggml_planar3_0_gen_rot` exposes the same
Givens table so the quantizer and the graph cannot drift. `llama_kv_cache` builds the inverse matrix
when the V type is planar3 and hands it to the graph as an input, mirroring `build_input_v_rot`. The
plan's Step 3 offers either owner for the inverse; the graph was chosen for the reason the plan gives,
and the cost argument is real: the dequantizer runs per cached row, so the inverse there costs about
`n_kv * n_head * 128` rotations per layer per token, against `n_head * 256^2` flops once per token in
the graph.

## Verification

Task 1 against the reference, byte for byte. A Rust dump of the reference crate
(`refdump/`, a copy of `kvquant.rs`) and a C harness (`planar3_dump.cpp`) quantize the same fixed
256-float input: all 96 packed bytes are identical, the norm is the f16 rounding of the reference's
f32 (`9.155233383` against `9.156250000`), and the reconstruction agrees to 5e-7 relative. This pins
the Givens table, the codebook, the nearest-centroid rule, the rotation order and the packing, none
of which the plan's own build-and-run check would have caught.

Rotation direction and matmul convention, settled by measurement (`rot_check.cpp`):

| matrix passed to `ggml_mul_mat` | rel L2 against the original |
| --- | --- |
| forward | 1.496329 |
| inverse | 0.167626 |

and the two matrices are exact transposes (`1.1e-08` worst element difference). The inverse through
`ggml_mul_mat` recovers the vector at exactly the round-trip error the old in-dequantizer rotation
gave (`1.676258e-01`), so moving the rotation to the graph is numerically equivalent, not merely
close.

Rotation-aware checks. The type's value space is rotated, so `to_float` is not the quantizer's inverse
and the stock quantize test would report nonsense. `tests/test-quantize-fns.cpp` now undoes the
rotation before the round-trip comparison and rotates the reference operand of the dot product check,
using `ggml_planar3_0_gen_rot` so the test cannot drift from the engine either:

| check | value | limit |
| --- | --- | --- |
| planar3_0 absolute quantization error | 0.003909 | 0.0040 (3 bit bucket) |
| planar3_0 reference implementation error | 0.000000 | 0.0001 |
| planar3_0 dot product error | 0.002274 | 0.0400 |

The absolute error is bit-identical to the value measured before the inverse rotation moved to the
graph, which is the strongest available evidence that the two placements are the same computation.

| check | result |
| --- | --- |
| `ctest -L main`, build-cpu | 34 / 34 |
| `test-llama-archs` qwen35 / qwen35moe / qwen4exp | all pass |
| 0.8B generated text, CPU, digest before plan E and after | identical (`a599102cce02ed30`) |
| `--cache-type-v planar3_0`, 35B, `-c 2048`, 32 tokens | loads, decodes, coherent text |
| `attn_rot_v` for planar3_0 (`-v`) | 0 |
| `llama-quantize --output-tensor-type planar3_0` | refused: "is a KV cache type, not a weight type" |

## Quality

Perplexity on two corpora, because the choice of corpus swings the answer by a factor of two and the
plan fixes one of them.

### The plan's corpus (C/C++ sources, 512 kB), 2048 depth

`llama-perplexity -c 2048 --chunks 16 -t 8 --cache-type-k q8_0`, the plan's corpus recipe
(`src/*.cpp`, `src/models/*.cpp`, `common/*.cpp`, `ggml/src/ggml-cpu/ops.cpp`, first 512 kB):

| V type | PPL | vs f16 |
| --- | --- | --- |
| f16 | 1.7552 +/- 0.02149 | - |
| planar3_0 | 1.7820 +/- 0.02145 | +1.69% |

Per chunk, paired over the same 16 windows: mean difference +0.0304 +/- 0.0010, t = 29.6. The effect is
real and reproducible, and at this depth it is inside the plan's 2% target.

### The docs corpus (English prose, 73 kB), 2048 depth

Same model, same cache types, `-c 2048 --chunks 8`, on the fork's own research docs:

| V type | PPL | vs f16 |
| --- | --- | --- |
| f16 | 6.2170 +/- 0.17126 | - |
| planar3_0 | 6.4322 +/- 0.17408 | +3.46% |

Paired over the same 8 windows: mean difference +0.2356 +/- 0.0102, t = 23. On out-of-distribution
prose the same change costs twice as much and is past the plan's 3% failure line. The gate verdict
therefore depends on the corpus, and the plan only fixes the code corpus. Worth recording: the model
is a coding model, so the code corpus is where its PPL is lowest (1.76 against 6.22) and where a
fixed relative perturbation is hardest to produce.

### Depth 512, a ladder across V types

`-c 512 --chunks 4` on the docs corpus, to see whether a 3 bit cache lands where it should:

| V type | bits/coord | PPL |
| --- | --- | --- |
| f16 | 16 | 9.3213 +/- 0.796 |
| q8_0 | 8 | 9.2191 +/- 0.784 |
| q4_0 | 4 | 9.3112 +/- 0.797 |
| planar3_0 | 3 | 9.4706 +/- 0.786 |

planar3_0 is the largest of the four, which is the expected ordering for the lowest bit width.

### Depth 16384

Same corpus, `-c 16384 --chunks 3` (three windows, about 1.1 hours for both runs):

| V type | PPL | vs f16 |
| --- | --- | --- |
| f16 | 1.7703 +/- 0.01737 | - |
| planar3_0 | 1.8010 +/- 0.01749 | +1.73% |

Paired over the three windows: mean difference +0.0263 +/- 0.0022, t = 11.8.

### Verdict on the gate

| depth | relative increase | target |
| --- | --- | --- |
| 2048 | +1.69% | at or below 2% |
| 16384 | +1.73% | at or below 2% |

Both depths are inside the plan's target on the corpus the plan specifies, so the gate passes and the
Vulkan plan is not dropped. Two caveats belong with that verdict. The cost is corpus dependent: on
out-of-distribution prose the same change is +3.46% at 2048, so a deployment that cares about text
other than code should re-measure. And the effect is systematic rather than noise, so it will not
wash out with more data.

### Rotation stacking and memory

`LLAMA_ATTN_ROT_DISABLE=1` leaves the generated text identical (first four generated lines), so the
planar3 rotation and the Hadamard path are not stacked. The KV cache reports its own sizes with `-v`,
at `-c 2048`, 10 layers, 2048 cells:

| V type | K half | V half | total |
| --- | --- | --- | --- |
| f16 | 10.62 MiB | 20.00 MiB | 30.62 MiB |
| planar3_0 | 10.62 MiB | 3.83 MiB | 14.45 MiB |

The V half is 5.22x smaller, exactly the block ratio (512 bytes against 98 bytes per 256
coordinates), and the whole cache is 2.12x smaller. The plan's step 4 says about 2.7x for the V half;
the measured number is 5.2x.

### Greedy divergence criterion

Task 4 step 3 asks for 64 greedy tokens identical to the f16 run and fails the gate on a divergence
inside the first 16. Every lossy V cache tested diverges from f16 within the first generated token,
including `q8_0`, which is +0.0% in quality:

| V type | first generated tokens against f16 |
| --- | --- |
| q8_0 | diverges |
| q4_0 | diverges |
| planar3_0 | diverges |

So the criterion as written cannot discriminate: it fails q8_0, a type nobody would reject. Either
the criterion needs a control (for example identical against q8_0 rather than f16) or it should be
dropped in favour of the perplexity gate. The divergent planar3_0 continuation is coherent
("Here's a thinking process: ..."), not garbage, which is the outcome the plan said to expect from a
quantization perturbation rather than from a rotation bug.

### Memory

The V half of the cache is 98 bytes per 256 coordinates, against 512 for f16, and the cache reports a
measured 5.22x reduction. The plan's step 4 says 2.7x; the arithmetic and the measurement do not
support that number.

## What the plan missed

- Task 1's type needs a CPU type traits entry. Without one, the KV insert path reaches
  `ggml_compute_forward_set_rows` and calls a null `from_float`: segfault on the first decode.
  Added `quantize_row_planar3_0` as a wrapper over the ref, mirroring `quantize_row_q2_0`.
- A quantized type with `from_float` also needs `vec_dot`, or `test-quantize-fns` and
  `test-quantize-perf` call a null pointer and segfault. Wrote `ggml_vec_dot_planar3_0_q8_0_generic`
  in the CPU backend plus the per-architecture fallback mappings and the traits entry.
- `test-quantize-fns` classifies every type in a threshold bucket; planar3_0 is in the 3 bit bucket
  (the plan does not mention this file).
- Task 3 needs the table exposed to `src/`, which the plan does not say how to do. It lives behind
  `ggml_planar3_0_gen_rot` in the public header, implemented next to the table in `ggml-quants.c`,
  because `src/` cannot include ggml's internal headers and the plan forbids the two paths drifting.
- The plan's Task 2 step 4 expects the KV size and `attn_rot_v` in the log. Neither is printed at
  default verbosity in this build; `-v` prints `attn_rot_v`.

## Follow-ups

- `ggml_vec_dot_planar3_0_q8_0_generic` dequantizes each block into a stack array and dots in f32. It
  is correct but not vectorized; the reference's rotated-space dot is the model to follow.
- No throughput comparison against f16 V yet.
- The state round trip Task 4 asks for was not run: it refers to a `llama_state_seq` save and restore
  pair, and `llama-completion`'s `--prompt-cache`/`--save-session` flags are not that. The generic
  coverage is `test-save-load-state`, which passes, but with the default cache types.
- The 16384 depth is the remaining gate number.

## Verification on the real path

Perplexity is the quality gate, but it is not how a cache type gets used. Same binary,
`llama-completion -m <0.8B> -p "The capital of France is" -n 32 -s 1 --temp 0 -c 512 -no-cnv`:

| KV type | result | output |
| --- | --- | --- |
| f16 (run twice) | rc=0, both byte identical | `The capital of France is the capital of the country.` |
| `-ctv planar3_0 -ctk f16` | rc=0, no errors or asserts | `The capital of France is Paris.` |

The repeat of the f16 run is the control: greedy decoding at a fixed seed is reproducible here, so
the 25-character agreement followed by divergence between f16 and planar3 is the quantization, not
noise. Both outputs are coherent English, which is what a lossy V cache should look like at the top
of a distribution; the quality claim stays with the perplexity numbers above.

Two interface notes. This fork does not log a `KV self size` line at this verbosity, so the proof that
`-ctv planar3_0` is live is the output difference plus the V-size measurement above, not a log line.
And `llama-completion` blocks reading stdin: it needs `< /dev/null` in a script, or it hangs until
killed, which is how an earlier attempt at this check wasted ten minutes.
