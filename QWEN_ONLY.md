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

## Backends

This tree builds the CPU and the Vulkan backend and nothing else.

| `GGML_*` option | status |
| --- | --- |
| `GGML_CPU` (default ON) | x86-64 only: `arch/x86` kernels, AMX, tiled k-quant mul-mat, llamafile sgemm, OpenMP |
| `GGML_VULKAN` (default OFF) | full Vulkan backend, SPIR-V generated at build time by `glslc` + `vulkan-shaders-gen` |

```sh
cmake -B build    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake -B build-vk -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_VULKAN=ON
```

Dynamic backends stay: `-DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON` builds
`ggml-cpu` and `ggml-vulkan` as loadable modules, and `-DGGML_CPU_ALL_VARIANTS=ON`
additionally builds one CPU module per x86 feature level.

### Removed backends

- the other 16 backends: `ggml-blas`, `ggml-cann`, `ggml-cuda`, `ggml-et`, `ggml-hexagon`,
  `ggml-hip`, `ggml-metal`, `ggml-musa`, `ggml-opencl`, `ggml-openvino`, `ggml-rpc`,
  `ggml-sycl`, `ggml-virtgpu`, `ggml-webgpu`, `ggml-zdnn`, `ggml-zendnn`
- their headers in `ggml/include`, their entries in `GGML_PUBLIC_HEADERS`, their option
  blocks in `ggml/CMakeLists.txt`, their `ggml_add_backend()` calls, and their
  `register_backend()` and `ggml_backend_load_best()` entries
- the non-x86 CPU sources: `ggml-cpu/spacemit`, `ggml-cpu/kleidiai` and
  `ggml-cpu/arch/{arm,loongarch,powerpc,riscv,s390,wasm}`, plus the CMake branches that
  selected them and the non-x86 `GGML_CPU_ALL_VARIANTS` lists
- CPU feature paths whose options no longer exist: Accelerate/vDSP, CPU HBM (memkind),
  KleidiAI, SpacemiT IME, and `ggml-cpu/hbm.{cpp,h}`
- RPC: the `--rpc` flag, `llama_supports_rpc()`, the RPC device ordering in
  `src/llama.cpp`, `tools/rpc`, and `tests/test-rpc-multi-server.*`
- CI: `.github/workflows`, `.github/actions`, `.github/labeler.yml`, `ci/`, and the
  Dockerfiles and SRPM spec for the removed backends under `.devops/` (`.devops/nix`
  stays, `flake.nix` uses it)
- docs and build presets for the removed backends: `docs/backend/`, the matching
  `docs/ops/*.csv`, `docs/multi-gpu.md`, `docs/docker.md`, `docs/release.md`, the
  `x64-windows-sycl-*` and `arm64-*` presets, and their toolchain files

`ggml-cpu/vec.h`, `quants.c` and the other shared CPU files still carry the `#if`
ladders for non-x86 SIMD. They compile out on x86, and deleting them would touch every
kernel, so they stay.

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

### Backend diet (2026-09-26)

Same machine. Pre-diet reference builds: `build-diet` (CPU) and `build-vk` (CPU + Vulkan), built from
the same commit before the backend diet. `test-llama-archs` is run with `-s 1` so the two runs can be
compared byte for byte. Timestamps in the logs are the only thing stripped before diffing.

| check | result |
| --- | --- |
| CPU build (Release, `GGML_NATIVE=ON`, tests ON) | 0 errors |
| Vulkan build (same plus `GGML_VULKAN=ON`) | 0 errors |
| dynamic backends (`BUILD_SHARED_LIBS=ON`, `GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`, `GGML_VULKAN=ON`, `GGML_NATIVE=OFF`) | 0 errors, and at runtime `libggml-vulkan.so` plus the matching `libggml-cpu-<variant>.so` are loaded |
| `test-llama-archs` qwen35 / qwen35moe / qwen4exp, seed 1, CPU | byte-identical to pre-diet |
| same three with `-b Vulkan0` | byte-identical to pre-diet |
| `ctest -L main` | 34 / 34 (CPU build), 34 / 34 (Vulkan build) |
| `test-backend-ops -b CPU` | 19914 / 19914, same count as pre-diet |
| `test-backend-ops -b Vulkan0` | 18976 / 18976, same count as pre-diet |
| `test-tokenizer-0` on both vocab fixtures | identical to pre-diet |
| Qwen3.6-35B-A3B IQ3_XXS (`qwen35moe`) 16-token greedy generation, CPU | identical text to pre-diet |
| the same on Vulkan0 (`-ngl 99`, all 84 layer assignments to Vulkan0) | identical text to pre-diet |
| grep guard for removed backend headers, dirs and `GGML_*` options | no references left |

### Delta-net output projection (2026-09-27)

The delta-net output projection declares its activation as `[value_dim, n_seq_tokens*n_seqs]` instead
of `[value_dim, n_seq_tokens, n_seqs]`, so the projections read their weight once per step instead of
once per sequence. Same machine, both build trees rebuilt from the change. Full write-up in
`docs/research/06-ssm-out-fix-results.md`.

| check | result |
| --- | --- |
| CPU and Vulkan builds | 0 errors |
| `test-llama-archs -s 1` qwen35 / qwen35moe / qwen4exp, CPU and Vulkan0 | all pass, NMSE 0 to 1.23e-07 vs CPU |
| `test-backend-ops test --test-file` on the 35B graph at `-np 8`, Vulkan0 | both `linear_attn_out` shapes match the CPU reference |
| `linear_attn_out`, 35B, Vulkan0, 1 token x 8 seqs | 823 us to 380 us, 2.2x |
| `linear_attn_out`, 35B, Vulkan0, 64 tokens x 8 seqs | 3280 us, unchanged |
| single stream (`n_seqs = 1`) | same tensor before and after, no change by construction |
| `linear_attn_out`, 35B, CPU, 1 token x 8 seqs | about 1.3x slower, recorded as a known regression |

No full-engine run: the change was verified on the exported graph at the real shapes and on the
synthetic arch models.

### Recurrent state view (2026-09-27)

`build_rs` returns a `ggml_view_2d` of the cache instead of gathering the active recurrent state rows
when they are already the identity mapping `[head, head + n_rs)`. Same machine, both build trees
rebuilt from the change. Full write-up in `docs/research/08-recurrent-state-phase1-results.md`.

| check | result |
| --- | --- |
| CPU build and Vulkan build | 0 errors |
| `ctest -L main`, both build trees | 34 / 34 both |
| `test-llama-archs -s 1` qwen35 / qwen35moe / qwen4exp | all pass, NMSE 0.00e+00 |
| generated text, 0.8B and 35B, single sequence, CPU and Vulkan0 | identical to pre-change |
| generated text, 0.8B, 2 and 4 sequences in one batch, CPU and Vulkan0 | identical to pre-change |
| predicate forced false, gather path | reproduces every pre-change baseline |
| tg128, 35B, Vulkan0 | 22.33 to 24.21 t/s |
| batched decode, S_TG at B=8, 35B, Vulkan0 | 44.30 to 49.11 t/s |
| batched prefill, S_PP at B=8, 35B, Vulkan0 | 239.60 to 241.35 t/s, unchanged |

Note for future checks: `llama-passkey -np N` decodes one sequence only (`n_grp = grp_attn_n`), so it
cannot verify the multi-sequence state mapping; `llama-batched -np N -kvu` does.

## Deliberate retentions

- `models/templates/*.jinja`: chat/autoparser test corpus, arch independent (the chat parsing tests
  parse arbitrary Jinja templates), so the template-name registry in `src/llama-chat.cpp` stays whole
  as well.
- `conversion/base.py` `get_vocab_base_pre()` hash table: tokenizer fingerprints. A model whose
  fingerprint is missing fails conversion outright, and fingerprints cannot be derived from the arch,
  so the table stays intact.
- `GGML_BACKEND_DL` / `GGML_CPU_ALL_VARIANTS`: kept, so `ggml-cpu` (one module per x86
  feature level) and `ggml-vulkan` can be built and loaded as dynamic libraries.
- `test-backend-ops`: generic, so it stays whole and is the engine check for both backends.
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
