# MTP speculative decoding: results

Date: 2026-09-27
Machine: Ryzen 7 6800H, Radeon 680M (RADV REMBRANDT), 27.1 GiB, Vulkan0 + CPU.
Tree: `qwen-only-backends` at 4f5831ba2, no engine change.
Plan: `docs/superpowers/plans/2026-09-26-mtp-speculation.md`.
Spec: `docs/superpowers/specs/2026-09-26-mtp-speculative-design.md`.

Target: `/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`
(qwen35moe, 35B-A3B, IQ3_S 3.4375 bpw, 13.09 GiB, `nextn_predict_layers = 1`, MTP head on disk as
`blk.40.*`).

Verdict: the gate in the plan was a single-stream speedup above about 10%. Measured 1.27x on CPU and
1.36x on Vulkan. The sub-project pays. The extraction tool the plan asks for is not needed.

## The premise was wrong: no separate draft file is required

`docs/speculative.md` lists `draft-mtp` as "Use Multi Token Prediction (MTP) heads from the main
model", and `common/common.cpp` sets `shares_model = !has_draft` with the comment "an MTP context
runs on the weights of the main model". So `--spec-type draft-mtp` on the target alone runs the head
that is already inside the target GGUF. The plan's reasoning ("the implementation requires a separate
draft context", "no MTP sidecar exists, so the head has to be extracted") does not hold for this
fork, and the head does not need `-md`.

Working command, no draft file:

```sh
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
./build-vk/bin/llama-server -m "$M" --spec-type draft-mtp -c 256 -np 1 -t 8 -ngl 99 --port 8127
# then one /completion request with n_predict 32, temperature 0, seed 1
```

The server reports `draft acceptance = 0.91667 (22 accepted / 24 generated), mean len = 3.75` and
`draft_n` / `draft_n_accepted` in the response timings, which is the acceptance data the plan wanted.

Two command corrections that follow from this. The plan's Task 2 and Task 3 commands use
`llama-completion` with `-md`, which this fork rejects: `error: invalid argument: -md`, because the
speculative options are registered only for the `llama-cli`, `llama-server` and `llama-speculative`
examples, and `llama-completion` is not among them. And the plan's `--draft N` flag does not exist;
the draft length is `--spec-draft-n-max N`. `llama-speculative` cannot be used either, it hard
requires `--model-draft` (`E main: --model-draft is required`), so the server is the tool that can
run the main-model MTP route.

## The extraction tool still works

`scripts/research/extract_mtp_head.py` was written and run as Task 1 asked. It produced
`/home/herlanggays/.jcode/scratch/mtp/draft-mtp.gguf`, 754,588,352 bytes, 22 tensors: the 20
`blk.40.*` tensors plus `token_embd.weight` and `output_norm.weight`. Verification against the
source: no metadata key missing, every copied tensor byte identical with the same type and shape, the
only differing key is `GGUF.tensor_count` (22 against the source's 41), which is the draft's own
header.

Three corrections the plan's script needed, all found by running it:

- `raw_shape=list(tensor.shape)` is wrong. `ReaderTensor.shape` is the file dims; the writer's
  `raw_shape` when `raw_dtype` is a quantized type must be the *byte* shape, and it converts it back
  with `quant_shape_from_byte_shape`. Passing the file dims fails with
  `Quantized tensor bytes per row (248320) is not a multiple of Q6_K type size (210)`. The fix is
  `raw_shape=list(tensor.data.shape)`, which is what `gguf-py/scripts/gguf_editor_gui.py` does for
  the same copy, plus `tensor_endianess=reader.endianess`.
- The reader exposes the file header as `GGUF.version` / `GGUF.tensor_count` / `GGUF.kv_count`
  fields. Copying them makes the draft unreadable (`Duplicate GGUF.version already in list`), so
  they are skipped along with `general.architecture`.
- The plan's expected count is 21 tensors; the model actually has 20 `blk.40.*` tensors, so 22.

The tool is not proposed for commit: it is dead weight for this model, and the results above show the
head in the target file is what the engine uses.

## A draft built this way cannot be loaded as a model

Loading the extracted draft as a model segfaults, both standalone and through `-md`:

```
./build-cpu/bin/llama-completion -m <draft> -p hello -n 1 -c 64
0.00.003.590 I llama_completion: load the model and apply lora adapter, if any
Segmentation fault
```

```
#0 ggml_mul_mat
#1 llm_graph_context::build_lora_mm
#2 llama_model_qwen35moe::graph::build_qkvz
#3 llama_model_qwen35moe::graph::build_layer_attn_linear
#5 llama_model_qwen35moe::build_arch_graph
#7 llama_context::graph_reserve
#9 llama_context::sched_reserve
#12 common_get_device_memory_data_impl
#13 common_params_fit_impl
```

The draft declares `block_count = 41` with only layer 40 present, so a context built with a normal
context type walks 40 trunk layers whose tensors are absent and dereferences the null weights in
`ggml_mul_mat`. `llama-model.cpp:1943` says a file that is entirely nextn layers with no trunk is the
supported shape, which means the draft would have to be written with `block_count = 1` and the
tensors renamed to `blk.0.*` for a plain load. That was not worth pursuing once the main-model route
above was measured, but the segfault itself is a finding: the model loader should report a missing
tensor rather than crash, and `common_params_fit_impl` builds this graph for any `-md` model without
the MTP context type.

## Measurements

`llama-server`, `-c 256 -np 1 -t 8`, one `/completion` request per run with
`n_predict 32, temperature 0, seed 1`, prompt `The capital of France is`, three repeats per
configuration. Logs in `/home/herlanggays/.jcode/scratch/mtp/`.

Baseline, no speculation:

| backend | runs, t/s | ms/token |
| --- | --- | --- |
| CPU | 17.63, 17.76, 17.75 | 56.3 to 56.7 |
| Vulkan0 (`-ngl 99`) | 23.85, 23.84, 23.57 | 41.9 to 42.4 |

With `--spec-type draft-mtp`, per max draft length. `draft_n / accepted` is from the last run:

| backend | n_max | runs, t/s | speedup | draft_n / accepted |
| --- | --- | --- | --- | --- |
| CPU | 1 | 19.36, 19.67, 19.97 | +10% | 15 / 15 |
| CPU | 2 | 21.25, 21.27, 21.70 | +20% | 22 / 20 |
| CPU | 3 | 22.44, 22.47, 22.49 | +27% | 24 / 22 |
| CPU | 4 | 20.64, 20.89, 21.96 | +19% | 28 / 23 |
| CPU | 6 | 20.87, 21.29, 21.54 | +19% | 30 / 25 |
| CPU | 8 | 16.57, 16.85, 17.84 | -3% | 34 / 26 |
| Vulkan0 | 1 | 28.57, 28.59, 28.60 | +20% | 16 / 15 |
| Vulkan0 | 2 | 28.29, 28.63, 31.73 | +23% | 23 / 19 |
| Vulkan0 | 3 | 29.92, 30.77, 31.03 | +30% | 27 / 22 |
| Vulkan0 | 4 | 28.06, 28.73, 30.82 | +24% | 28 / 23 |
| Vulkan0 | 6 | 32.19, 32.22, 32.70 | +37% | 30 / 25 |

Best configuration found: CPU `--spec-draft-n-max 3`, Vulkan0 `--spec-draft-n-max 6`. The CPU falls
over at n_max 8, below its own baseline, so the draft work is not free there: the head is read once
per drafted token and the CPU has no spare bandwidth. Vulkan0 is still climbing at 6, which is where
this stopped.

## Verification on the real path: speculation must not change greedy output

Speculative decoding is only correct if greedy decoding still picks the same tokens, and that had not
been checked. Same binary, same prompt, same seed, one flag apart:

```
./build-vk/bin/llama-cli -m "$M" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 256 -ngl 99 -t 8 -st
./build-vk/bin/llama-cli -m "$M" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 256 -ngl 99 -t 8 -st --spec-type draft-mtp
```

The generated text is byte identical. The only line that differs in the whole transcript is the
throughput report: `23,3 t/s` without the MTP head and `28,5 t/s` with it, so the invariant holds and
the speedup reproduces at the default `--spec-draft-n-max` (the 37% in the table above was at n_max 6,
which is why it is larger).

Notes for anyone repeating this. `llama-cli` needs `-st` to run one turn and exit; without it, it sits
in conversation mode and the run hangs until killed. `--spec-type` must be two arguments, not one
string with a space in it, or the parser reports `invalid argument: --spec-type draft-mtp`. Models
without MTP layers fail cleanly rather than silently: the 0.8B Q4_K_M gives
`context type MTP requested but model doesn't contain MTP layers` and exits, which is the right
behaviour and also means a small-model check of this path is impossible.

Caveats: 32 predicted tokens per run is short, so most of the run is the first verification step and
the numbers are noisier than the six significant digits suggest; the plan's gate is met by a wide
margin at every length except CPU n_max 8, so the conclusion does not depend on the noise. The
acceptance rate is 0.92 at n_max 3 and stays near 0.85 at n_max 6, which is what makes it pay.
