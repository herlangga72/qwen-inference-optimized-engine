# KV planar3 V cache on Vulkan: results

Date: 2026-09-27
Tree: `qwen-only-backends`, commit `742cf16d9`
Machine: Ryzen 7 6800H, Radeon 680M (RADV REMBRANDT), 27.1 GiB
Model: `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` (qwen35moe, 35B-A3B, IQ3_S 3.4375 bpw, 13.09 GiB)
Plan: `docs/superpowers/plans/2026-09-26-kv-planar3-vulkan.md`
Spec: `docs/superpowers/specs/2026-09-26-kv-planar3-design.md`
Note: this document is numbered 25, not the 11 the plan asks for, because `11-uma-zero-copy-findings.md`
already exists.

## What was built

`planar3_0` now works as a V cache type on the Vulkan backend. Both halves were needed, and the plan
only described one of them.

**Read side**, as planned: a `planar3_0` branch in the flash attention uber `dequantize4()`, gated to V
only. Accepting it for K as well would fall into the K switch with no matching case, which returns
zeros silently, so the host predicate is split into `fa_kv_ok_k` and `fa_kv_ok_v`.

**Write side**, which the plan assumed away. The plan says "the insert-side rotation happens in the
host quantizer, and the read-side inverse rotation is the graph matmul", which is true on CPU and not
true here: V is computed in f32 on the device, so `set_rows` has to quantize into the cache on the GPU.
That needs the whole transform, not just a dequant branch:

- `copy_to_quant.comp` gains the host quantizer's exact arithmetic: L2 norm, per-pair Givens rotation,
  nearest of the 8 centroids, 8 three bit indices packed per group into 3 bytes little-endian.
- `planar3_tables.glsl` holds the two constant tables both shaders need.
- The shader generator emits the `set_rows_*_planar3_0_*` variants.
- Two host gates: `fa_kv_ok_v` above, and `GGML_TYPE_PLANAR3_0` in the `SET_ROWS` destination
  whitelist of `ggml_backend_vk_device_supports_op`, which is what rejected the type first.

The tables are literals because a shader cannot reproduce them: the centroids come from a 300
iteration Lloyd-Max solve in double precision, and the rotation angles from a 64 bit PRNG plus a
double precision cosine. `scripts/research/kvplanar3/planar3_codebook.py` mirrors both generators and
prints the tables, so they can be regenerated.

`llama-bench` also needed `planar3_0` added to its own name to type map, which is separate from the
list the server and completion tools use.

## Correctness

Same prompt, seed 1, temperature 0, `-c 128 -n 8`, `-ctk q8_0 -ctv planar3_0`:

| run | result |
| --- | --- |
| CPU, `-ngl 0` | 83 bytes |
| Vulkan, `-ngl 99` | 83 bytes, byte identical |

Identical text, which is the arithmetic agreement check: the shader's codebook and rotation match the
host's exactly, and the shader's quantizer produces the same blocks the host would.

## Speed

Vulkan, `-ngl 99`, `K q8_0`, `V planar3_0`, `-p 0 -n 32 -r 2`:

| depth | t/s |
| --- | --- |
| 2048 | 19.95 ± 0.10 |
| 8192 | 18.42 ± 0.08 |
| 16384 | 18.04 ± 0.14 |
| 32768 | 15.05 ± 0.03 |

The KV byte accounting behind those numbers, from the model metadata: K at `q8_0` is 5440 B/token and
V at `planar3_0` is 1960 B/token, against 10240 B/token for V at f16. At 32768 depth that is 178 MiB of
K and 61 MiB of V with planar3, against 320 MiB of V at f16, so about 259 MiB less to read.

**The f16 baseline was not taken, so no speedup is claimed here.** The run was started and stopped on
request before it produced a number. That is the one missing measurement, and it is the one the plan's
gate is written against, about 8 percent at 16k and 14 percent at 32k. Without it these numbers say the
type works, not that it is faster.

Worth knowing before someone runs that baseline: the f16 V cache is the risky configuration on this
box, not planar3. Its KV is 2.77x larger, and with the model at 13.09 GiB against a 13.57 GiB TTM limit
there is only about 0.48 GiB of GPU-visible memory left for KV and compute buffers. `docs/research/freeze-notes-2026-09-27.md`
records seven hard resets on this machine under exactly that model on Vulkan, so the baseline should be
run with a smaller context or with `--n-cpu-moe` rather than casually at 32768 depth.

## Also worth knowing

The CPU path with a `planar3_0` V cache is very slow: about a minute per token on the 0.8B model, where
f16 takes a fraction of a second. The agreement run above needed over seven minutes for 8 tokens. If the
type is kept, that is a second thing to fix, and it makes the CPU side of any A/B comparison expensive.
