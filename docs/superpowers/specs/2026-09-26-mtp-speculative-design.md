# MTP speculative decoding: extract the head, measure, productionize if it pays

Date: 2026-09-26
Status: approved in scope, not yet implemented
Scope: a new extraction tool under `scripts/research/`, no engine changes, plus measurement and, conditionally, documentation
Related: `docs/research/00-baseline-stage1.md` (per-token byte accounting), `docs/research/04-stage5-diagnosis.md` (op level costs)

## Context

At batch 1 the engine streams about 2.08 GB of weights per token and is bandwidth bound: the dense
weights are 83% of that and are read once per token no matter what the router does. The only way to
make the 17% expert share pay off, and the only way to amortize the dense share, is to process more
than one token per weight read. Batching does that for throughput (measured 1.86x at B=8). For a
single stream the lever is speculative decoding, and this model ships an MTP head for exactly that.

Facts on the ground:

- `Qwen3.6-35B-A3B` (Unsloth requant of the same base), `qwen35moe.nextn_predict_layers = 1`, with the
  head on disk as `blk.40.*`, about 300 MB (attention plus MoE experts at Q2_K/Q3_K/IQ4_XS).
- `load_mtp` defaults to false, so the head is skipped and the loader logs it as unused.
- The fork has a working `draft-mtp` speculative implementation (`common/speculative.cpp:1328`) with
  its own acceptance statistics (`common_speculative_print_stats`, line 2951).
- That implementation requires a *separate* draft context: `--spec-draft-model`/`-md` pointing at a
  model whose `n_layer_nextn >= 1` and whose output embedding width matches the target. It asserts
  both (`common/speculative.cpp:1369-1374`).
- `llama-model.cpp:1943` already handles the case of a model file that is entirely nextn layers with
  no trunk, which is the shape an MTP-head-only file has.
- No MTP sidecar exists for this file locally or in the fork's download plan, so the head has to be
  extracted from the file we already have.

Why it can pay: the head costs about 300 MB per step, +14% of per-token traffic, and the target's
2.08 GB of weights then amortize over 1 + drafted tokens. With `--draft 1` and a 70% acceptance rate
the expected win is roughly 1.35x, and it is exact, not lossy: drafted tokens are verified by the
target model, so a rejection only costs time.

## Design

**Tool: `scripts/research/extract_mtp_head.py`** (new file, `gguf-py`, no engine changes).

- Input: the target GGUF path. Output: `draft-mtp.gguf` in a chosen directory.
- Copies metadata: the architecture and hparams keys (`qwen35moe.*`), the tokenizer and vocab keys
  (the draft context creates its own sampler, so it needs a vocab), and `general.*` identity keys,
  with `nextn_predict_layers = 1` and `block_count = 41` unchanged, which keeps the head at `blk.40`
  exactly as it is named in the source file.
- Copies tensors byte for byte, using the source's existing quantized types: the `blk.40.*` set. No
  requantization, so no quality decision is involved.
- The exact tensor set beyond `blk.40.*` is discovered by loading, not guessed: any tensor the loader
  reports missing is added to the copy list by name. Two candidates are known and will show up
  immediately if needed: `token_embd.weight` (417 MB at Q6_K, which would change the arithmetic in
  Context above) and the output/head weights.

**Configuration to test**: `--spec-type draft-mtp -md draft-mtp.gguf --draft N` for N in 1, 2, 3, on
both backends (`build-cpu`, and `build-vk` with `-ngl 99`). Only the 35B MoE has an MTP head; the
0.8B dense model is not part of this sub-project.

**Measurement**: `llama-bench` has no draft support, so:

- Throughput comes from `llama-completion`'s perf summary (eval time per token) with and without
  speculation, same prompt, seed, temperature (0) and token count. Report tokens per second and the
  speedup ratio per (backend, N).
- Acceptance comes from `common_speculative_print_stats`: accepted drafts, accepted tokens, and
  per-position acceptance. Report the accepted-tokens-per-step figure, which is what the arithmetic
  above depends on.
- Correctness: with temperature 0 the generated text must be identical with and without speculation,
  because verification is exact. That is the gate, and it is checked at every N.

**Productionize gate**: if the best measured single-stream speedup is above about 10%, keep the
script, record the recommended backend and `--draft` value, and document the configuration in
`docs/research/` and in the `QWEN_ONLY.md` verification log. If it is at or below 10%, write the
negative result with its acceptance numbers and stop; the script and the finding are still the
deliverable, and nothing else is kept.

## Risks

- **The MTP-only file may not load.** This is the first gate, before any measurement. If the loader
  cannot accept a nextn-only model in practice, C is dropped and the finding recorded. The
  `llama-model.cpp:1943` path suggests it can, but that is not the same as having done it.
- **The head may need a large tensor** such as `token_embd`, which would raise the per-step cost well
  above 300 MB and could turn the arithmetic negative. Measured, not assumed.
- **The fork's `draft-mtp` may assume a head layout this model does not have.** Its assertions
  (`n_embd_out` of draft and target must match) are checked at load; a mismatch fails loudly.
- **Acceptance may be poor** on this model or at these quant levels, in which case the tokens-per-step
  figure stays near 1 and speculation is pure overhead. That is a legitimate, useful result.

## Verification

1. `draft-mtp.gguf` loads: `llama-completion -m TARGET -md draft-mtp.gguf --spec-type draft-mtp -n 8`
   prints the speculative implementation line and completes.
2. Greedy identity: `llama-completion --temp 0 -s 1 -n 32` with and without the draft, both backends,
   byte identical text.
3. Acceptance and throughput table per backend and per `--draft` value, against the no-speculation
   baseline measured in the same session.
4. A second prompt and a longer context (`-c 16384`) to confirm the acceptance rate is not an artifact
   of one prompt.
5. The path not taken is recorded: `test-llama-archs -a qwen35moe -s 1` and the existing suite must be
   unaffected, since this sub-project changes no engine code.

## Out of scope

- Requantizing the MTP head to reduce its 300 MB. Only reconsider if the measurement shows the head's
  bytes are the limiting factor.
- Other speculative types (`dflash`, `eagle3`, `dspark`), and any change to `common/speculative.cpp`
  unless the measurement exposes a specific defect in it.
- Drafting on a separate device, and multi-sequence speculative serving.
- Engine changes. This sub-project is tooling, configuration and measurement.
