# Stage 3 evidence: scheduler and compute buffer allocation

Where this lives: `ggml_backend_sched` in `ggml/src/ggml-backend.cpp`, driven from
`llama_context::graph_reserve` / `process_ubatch` in `src/llama-context.cpp`. The scheduler cuts the
graph into per-backend segments ("splits") and inserts a copy at each boundary.

`llama_context` reports the shape of the reserved graph. `llama-bench -v`, `-p 256 -n 4`:

| build | flag | graph nodes | splits (pp bs=256 / tg bs=1) |
| --- | --- | --- | --- |
| build-cpu | `-ngl 0` | 3787 | 1 / 1 |
| build-vk | `-ngl 0` | 3787 | 844 / 61 |
| build-vk | `-ngl 99` | 3787 | 2 / 2 |

Readings:

- The pure CPU path is one split: no cross-backend copies at all per token.
- The fully offloaded GPU path is two splits, and the count does not change between a 256 token
  prefill and a single token decode, so it is one structural boundary, not something proportional
  to work. Decode on that path runs at 100% of the DRAM wall, so that boundary costs ~0.
  `llama_context::graph_get_cb` is what keeps it that low: when `full_offload` is set it pins the
  per layer `norm` and `l_last` tensors to the layer's own device, precisely to stop the scheduler
  from shuffling them across backends.
- The mixed configuration is pathological. With a Vulkan device present but zero layers offloaded,
  the graph is cut 61 times per decode step and 844 times for a 256 token prefill, i.e. 61
  cross-backend copies per token. That matches the measured 11% regression against the CPU only
  build (14.04 against 15.87 t/s, see `02-gpu-vulkan-recheck.md`).
- `sched_reserve: max_nodes = 23456` against 3787 nodes in the real graph: the compute buffer is
  sized for a graph six times larger than anything this model builds at these batch sizes.

## Per token cost of this stage

Bounded by the stage 2 experiment: `LLAMA_GRAPH_REUSE_DISABLE=1` costs 1.8 ms/token on CPU and
4.8 ms/token on GPU, and that delta covers build, scheduler reset and buffer allocation together.
With reuse on, the cost is already paid once, not per token.

## Headroom

Zero on the CPU path and on the fully offloaded GPU path. The only defect found is the mixed
configuration, which is cheap to detect: a device is in the scheduler's list but holds no tensor,
and the graph is then cut on every op that the device claims to support while its inputs live on
the CPU.

# Stage 4 evidence: attention

The model has 40 trunk layers: 30 gated delta net layers (linear attention, constant size state)
and 10 full attention layers (`full_attention_interval = 4`).

Context sweep, tg128, 8 threads:

| depth | GPU `-ngl 99` | CPU |
| --- | --- | --- |
| 0 | 22.43 | 16.01 |
| 4096 | 20.44 (-8.9%) | 14.02 (-12.4%) |
| 16384 | 19.06 (-15.0%) | 7.05 (-56.0%) |
| 32768 | 17.31 (-22.8%) | - |

Byte accounting at 16384: the KV cache for the 10 full attention layers is 10 x 2 kv heads x 512
dim x 16384 tokens x 2 bytes = 168 MB read per token, about 8% on top of the 2.08 GB of weights. The
GPU loss of 15% is in line with that. The CPU loss of 56% is not a bandwidth effect at all.

## The CPU loss is flash attention

tg64 at 16384, CPU, with and without flash attention:

| depth | `-fa 1` | `-fa 0` |
| --- | --- | --- |
| 0 | 16.41 | 16.52 |
| 4096 | 14.29 | 14.22 |
| 16384 | 7.24 | 12.91 |

FA is neutral up to 4k and then costs 44% at 16k on the CPU. The crossover is between 4k and 16k.
`-fa auto` turns FA on here (the verbose graph shows `FLASH_ATTN_EXT` nodes with a
`ne = {16640, 1, 1, 1}` f16 mask), which is why the plain context sweep collapsed.

On the GPU the sign is reversed, tg64 at 16384: `-fa 1` 18.88 against `-fa 0` 13.35, so FA buys 41%
there.

Hypothesis for the CPU side, not yet tested: without FA the two attention products are ordinary
matmuls and go through the well optimised CPU GEMM kernels, while with FA the fused kernel has to
be hand written and does not reach the same rate for head_dim 256 with 16 query heads and 2 KV
heads. At long context the matmul work dominates, so the fused kernel loses despite doing less
memory traffic.

## Headroom

GPU: the attention share matches its byte cost, so ~0.
CPU long context: 1.8x, measured, and reproducible from the command line with `-fa 0`. This is the
largest single loss found outside the batched MoE path.

# Stage summary so far

| stage | backend | measured loss | status |
| --- | --- | --- | --- |
| 1 load and layout | CPU | 0 | at 70% of DRAM wall, layout independent |
| 1 load and layout | GPU | 0 | at 97 to 100% of DRAM wall |
| 2 tokenize and graph build | both | 2.8% / 9.5% ceiling | already amortized by graph reuse |
| 3 scheduler and allocation | both | ~0 | only defect is the mixed config |
| 4 attention | GPU | ~0 | matches its byte cost |
| 4 attention | CPU, long ctx | 1.8x | FA regression, reproducible |
| 5 MoE FFN | GPU, batched | 2.2x | batched expert path at 45% of DRAM wall |
