# Qwen-only fork of llama.cpp

This tree is a dieted llama.cpp: it is the inference engine for Qwen3.5 and up only. Everything
outside the Qwen3.5+ family was removed so the remaining code is small enough to read, measure and
modify as a research base.

Branch: `qwen-only` (forked from upstream `master` at 81bc6b83f).

## Supported architectures

| `general.architecture` | models | graph |
| --- | --- | --- |
| `qwen35` | Qwen3.5 dense, Qwen3.6, text tower of Qwen3.5-VL | hybrid: gated delta net (linear attention) + full attention, MTP head |
| `qwen35moe` | Qwen3.5 MoE (for example 35B-A3B) | same as `qwen35` plus MoE FFN, MTP head |
| `qwen4exp` | Qwen4 experimental | same plus QSA sparse indexer attention, PLE, hyper-connections |
| `clip` | quantize-only stub for mmproj GGUFs | never runs inference |

Related support kept because the kept models need it:

- tokenizer: BPE with pre-tokenizers `qwen35` and `qwen2` (plus the `default` fallback)
- multimodal: only the `QWEN3VL` projector (Qwen3.5-VL and the Qwen4Exp vision tower, which is an
  unmodified Qwen3-VL ViT)
- hybrid memory: plain KV cache, SWA KV cache, hybrid (attention + recurrent) memory and the
  hybrid indexer memory used by Qwen4Exp QSA
- deepstack visual embedding injection (Qwen3.5-VL text tower is arch `qwen35`)

Any other arch name is rejected at load time with `unsupported model architecture`.

## Removed

- the other 150 model implementations in `src/models/`
- the corresponding `enum llm_arch`, tensor and KV entries in `src/llama-arch.h` / `llama-arch.cpp`
- per-arch branches in the loader, context, graph, quantizer, model saver and tokenizer
- the other conversion modules (only `base`, `qwen`, `qwen4exp`, `qwen3vl` remain) and the matching
  GGUF metadata tables in `gguf-py`
- KV cache / memory / graph subsystems that no kept arch can instantiate (DSA, MSA, DSV4,
  pure-recurrent memory, cross-attention encoder-decoder plumbing)
- tokenizer implementations other than BPE, and pre-tokenizers other than `qwen35` / `qwen2` / `default`
- arch-specific components: `tools/tts`, `examples/diffusion`, `examples/convert-llama2c-to-ggml`,
  the non-Qwen `models/ggml-vocab-*.gguf` fixtures, the corresponding test registrations and snapshots
- the non-QWEN3VL multimodal projectors in `tools/mtmd`

## Verify

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DGGML_NATIVE=ON
cmake --build build -j

./build/bin/test-llama-archs -a qwen35
./build/bin/test-llama-archs -a qwen35moe
./build/bin/test-llama-archs -a qwen4exp
./build/bin/test-batch-alloc
./build/bin/test-tokenizer-0 models/ggml-vocab-qwen35.gguf
./build/bin/test-tokenizer-0 models/ggml-vocab-qwen2.gguf

./build/bin/llama-completion -m Qwen3.5-0.8B-Q4_K_M.gguf -p "The capital of France is" -n 24 -no-cnv
```

`test-llama-archs` builds a synthetic model for the requested arch, runs the graph and checks the
output against a reference forward pass, so it is the fastest correctness check when touching the
graph. `models/ggml-vocab-qwen35.gguf` and `models/ggml-vocab-qwen2.gguf` are vocab-only GGUFs used
by the tokenizer tests.

## Verification log (2026-09-26)

Run on the dieted tree, AMD Ryzen 7 6800H, CPU backend:

| check | result |
| --- | --- |
| full build (`cmake --build build -j8`, all targets incl. server, cli, mtmd) | 0 errors |
| `test-llama-archs -a qwen35` / `-a qwen35moe` / `-a qwen4exp` | all pass |
| `ctest -L main` | 34 / 34 pass |
| `test-batch-alloc` | 49 tests, 324 assertions, 0 failures |
| `test-tokenizer-0` on both kept vocab fixtures | pass |
| `test-mtmd-impl` | pass |
| Qwen3.5-0.8B Q4_K_M generation (seed 1, 24 tokens) | byte-identical to the pre-diet build |
| Qwen3.5-0.8B tokenizer ids | identical to the pre-diet build |
| `llama-quantize --allow-requantize Q4_K_M -> Q8_0` then load and generate | works |
| Qwen3.5-VL (0.8B text + `qwen3vl_merger` mmproj) image description | works, output matches the pristine upstream build on the same inputs |

The same generation and tokenizer baselines are what the tests above compare against, so any future
change that alters Qwen3.5 inference will show up as a diff there.

## Deliberate retentions

- `models/templates/*.jinja`: chat/autoparser test corpus, arch independent (the chat parsing tests
  parse arbitrary Jinja templates), so the template-name registry in `src/llama-chat.cpp` stays whole
  as well.
- `conversion/base.py` `get_vocab_base_pre()` hash table: tokenizer fingerprints. A model whose
  fingerprint is missing fails conversion outright, and fingerprints cannot be derived from the arch,
  so the table stays intact.
- public API enum values in `include/llama.h` (for example `llama_vocab_type`): the implementations
  were pruned but the enum values stay so the public headers and ABI stay stable.
- `clip` arch and `src/models/clip.cpp`: `llama-quantize` uses this stub to quantize mmproj GGUFs.

## Adding a model family back

1. `conversion/<mod>.py` with the HF loader/tensor map, registered in `TEXT_MODEL_MAP` and/or
   `MMPROJ_MODEL_MAP` in `conversion/__init__.py`, plus its `MODEL_ARCH` entries in
   `gguf-py/gguf/constants.py`.
2. `src/models/<mod>.cpp` implementing `load_arch_hparams`, `load_arch_tensors` and
   `build_arch_graph`, with the struct declared in `src/models/models.h`.
3. `LLM_ARCH_<NAME>` in `src/llama-arch.h` plus its entry in `LLM_ARCH_NAMES` (`src/llama-arch.cpp`)
   and any new `LLM_TENSOR_*` / `LLM_KV_*` entries.
4. a case in `llama_model_mapping()` in `src/llama-model.cpp`.
5. arch handling in `llm_arch_is_hybrid()` / `llm_arch_supports_rs_rollback()` if it needs hybrid or
   recurrent state, and in `llama_model::create_memory()` if it needs a special cache layout.
6. test registration in `tests/CMakeLists.txt` and, for the graph test, a branch in
   `tests/test-llama-archs.cpp`.
