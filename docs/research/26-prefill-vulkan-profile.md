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
