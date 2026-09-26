# Stage 2 evidence: tokenize and graph build

Baseline for this stage: tg128 15.87 t/s at 8 threads, that is 63.0 ms per token.

## Tokenize

`llama_tokenize` runs once per prompt, in `llama-vocab.cpp` behind the BPF/trie pre-tokenizer for
`qwen35`. A decode step never re-tokenizes: each sampled token id is fed straight back through
`llama_batch`. Detokenization for display is a vocab lookup plus incremental UTF-8 handling.

Conclusion: there is no per-token tokenizer cost to remove.

## Graph build

`llama_context::process_ubatch` (src/llama-context.cpp:1355) builds a new graph and allocates it
only when the previous graph cannot be reused:

- gate: `!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)`
- `res->can_reuse()` is implemented per graph input in src/llama-graph.cpp (embd, pos, out_ids,
  rs, attn_kv, kq_mask and friends) and checks that the new ubatch fits the old topology.
- a steady decode loop (same batch shape every step) therefore builds the graph once and reuses it,
  only re-setting the input tensors.

Measurement, `LLAMA_GRAPH_REUSE_DISABLE=1` against the default, tg128 at 8 threads, r 3:

| setting | tg128 | ms/token |
| --- | --- | --- |
| graph reuse on (default) | 15.87 +- 0.01 | 63.0 |
| graph reuse off | 15.43 +- 0.29 | 64.8 |

So build plus alloc plus reset for a full 40 layer MoE graph is ~1.8 ms per token, 2.8% of the
63 ms. Reuse already removes it, and disabling it is the worst case, not the current state.

## Headroom

Stage 2 can return at most ~2.8%, and only by making the reuse predicate accept more shapes (for
example during speculative or multi-sequence steps where the batch shape changes). Per-token cost
worth chasing in this stage is ~0.

## Instrumentation note

src/llama-context.cpp:1390 has a commented out per-token graph build timer:

```cpp
//const auto t_start_us = ggml_time_us();
gf = model.build_graph(gparams);
//LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
```

It is the right place for finer numbers if a later stage needs them, but the reuse experiment above
already bounds the whole path.

## Knobs

`LLAMA_GRAPH_REUSE_DISABLE=1` (env), `graph_reuse_disable` (member), `res->can_reuse()` per input.

## See also

`docs/research/02-gpu-vulkan-recheck.md` rechecks this stage on the Vulkan backend. Graph reuse is
worth 9.5% per token there against 2.8% on the CPU, so the reuse constraint is stricter on the GPU.
