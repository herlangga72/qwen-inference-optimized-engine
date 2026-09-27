# Prefill Attention and MoE Implementation Plan

**Outcome: nothing implemented, and the ordering in this plan is wrong.** A profile of the prefill pass
now exists, `docs/research/26-prefill-vulkan-profile.md`, and it puts flash attention at 1.2 percent of
the pass while the MoE expert matmul is 47 percent and runs at about a third of the efficiency the dense
matmuls reach on the same GPU. Task 2 below is the first task and the measured smallest lever; task 4 is
the largest. Re-ordering is the cheapest change available here. The problem statement agreed: a change
that does not beat the in-situ baseline gets reverted, and now the baseline says where to look.
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement
> this plan task by task, or superpowers:executing-plans with review checkpoints between tasks. Steps use
> checkbox syntax for tracking. Do not commit without explicit approval, and never push.

**Goal:** Cut the prefill pass on the Radeon 680M by acting on the two measured targets in order:
flash attention, then the attention projections, then the MoE expert path. Every change has to beat the
in-situ baseline, and every change that does not is recorded and reverted.

**Architecture:** Nothing new is designed until the profile is re-measured with the backend's own in-situ
instrument, because the existing reference is an offline replay whose two artifacts are already known.
Flash attention on this part runs the scalar path with an analytic configuration computed in
`get_fa_tuning_params_scalar`; the work is to find out what that configuration actually is at the prefill
shape, sweep the knobs through a temporary override, and only then decide whether to change the rule. The
MoE fusion comes second and must beat a measured bar, because those kernels already run at 41% of peak.

**Tech Stack:** C++17 and Vulkan in `ggml/src/ggml-vulkan`, GLSL shaders in
`ggml/src/ggml-vulkan/vulkan-shaders`, CMake, `test-backend-ops` and `llama-bench`.

## Global Constraints

- **No quantization or dequantization of model weights.** Hard user constraint. Nothing in this plan
  touches the file or its tensors.
- **Measure in the same session.** Every before/after pair runs back to back with no other work in
  between, as every earlier stage did.
- **Temporary instrumentation is temporary.** The knob override in task 2 and any debug print are reverted
  before the closing commit. The patch text is kept in the stage doc so it can be re-applied.
- **Negative results are kept.** A knob sweep that finds nothing, or a fused kernel that loses, is
  written into the stage doc with its numbers, as stage 5 did.
- **No commit or push without explicit approval per action.** Commit messages use `Assisted-by: Jcode`.

## Reference Conditions

- Machine: Ryzen 7 6800H, Radeon 680M (RDNA2, unified memory, no cooperative matrix support), 27.1 GiB.
- Model: `/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`
  (13.1 GiB, 35B total and 3B active MoE, 256 experts top-8, 41 blocks).
- Baseline to beat: pp512 at 202 t/s (2.53 s per 512-token pass), tg128 at 22.35 t/s, Vulkan0, `-ngl 99`.
- Build tree: `build-vkb` (Vulkan). Rebuild with `cmake --build build-vkb -j` after any shader change.

## File Structure

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` - only file expected to change, and only in
  `get_fa_tuning_params_scalar` (~line 1195) or the matmul tile selection, if a sweep earns it.
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_id.comp` - touched only by task 4, only if the standalone
  fused kernel wins the gate first.
- `docs/research/05-stage6-prefill-attention.md` - new, the stage record: baseline, sweeps, decisions.
- `scripts/research/` - existing harness, reused unchanged. `mmid_bw.py` already parses per-op output and
  gains a FLOP column for the logger output.

---

### Task 1: Re-baseline prefill with the in-situ instrument

The reference table came from exporting the prefill graph and replaying each op offline. The backend can
time its own nodes on the real graph, which removes both known artifacts and adds FLOPs per op.

**Files:**
- Create: `docs/research/05-stage6-prefill-attention.md`

**Steps:**

- [ ] 1.1 Run the prefill once with the backend's perf logger, using the harness invocation plus the env
  var, so the settings match every earlier measurement:

  ```sh
  MODEL="$HOME/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
  GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 \
    build-vkb/bin/llama-bench -m "$MODEL" -ngl 99 -r 3 -p 512 -n 0 -t 8 2>&1 | tee "$JCODE_SCRATCH_DIR/f2-logger-pp512.txt"
  ```

- [ ] 1.2 Do the same for a decode run (`-p 0 -n 128`) and keep both outputs, so the plan can later show
  that a prefill change did not move decode.

- [ ] 1.3 Extract the logger's per-name totals into a table with columns: op name, ms, share of the pass,
  FLOPs, achieved GFLOPS. The logger prints fused nodes as a group name, so note which names are groups.

- [ ] 1.4 Reconcile against the replay table in the spec. For each of the top ten rows, state agreement or
  the reason for disagreement. Two specific checks: the logits row (`result_output`) should now be a
  single-token row and small, and the MoE gate and up rows should now appear separately if the logger
  distinguishes them.

- [ ] 1.5 Print the flash attention configuration actually in use. Try the existing pipeline stats first:
  `GGML_VK_PIPELINE_STATS=1` on the same run. If the flash attention pipeline's subgroup size and spec
  constants are not in that output, add a temporary one-shot log in `get_fa_tuning_params_scalar` that
  prints the returned struct plus `device->subgroup_size`, `device->uma`, and
  `maxComputeSharedMemorySize`, and record the values in the stage doc.

**Verification:**
- The logger's total for the pass is within 15% of 2.53 s, and its per-op ranking is consistent with the
  replay table for at least eight of the top ten rows.
- The stage doc records the FA configuration as measured, not as read from the source.

---

### Task 2: Flash attention configuration

**Files:**
- Modify (temporarily, then revert): `ggml/src/ggml-vulkan/ggml-vulkan.cpp` in `get_fa_tuning_params_scalar`
- Modify: `docs/research/05-stage6-prefill-attention.md`

**Interfaces:**
- Temporary override, read from the environment so no rebuild is needed per variant:
  `GGML_VK_TEST_FA_BR`, `_BC`, `_ROWSPLIT`, `_DSPLIT`, `_WG`, `_SUBGROUP`, each replacing the corresponding
  field of `vk_fa_tuning_params` when set. This is instrumentation, not a feature.

**Steps:**

- [ ] 2.1 Sweep the subgroup size first, since the scalar tuner asks for the device's reported size, 64,
  for four rows or more, while `get_subgroup_size` pins every other RDNA2 pipeline to 32. Test 32 against
  64 with everything else at the measured defaults.

- [ ] 2.2 Sweep the remaining knobs one at a time against the best of 2.1: block_rows (8 against 16, and 4
  against 8), block_cols (32 against 64), row_split (4 against 1), d_split (8 against 4 and 2), and
  workgroup size (256 against 128 and 512). One variable per run, everything else at the best value so far.

- [ ] 2.3 Measure each variant as a pair: the op in isolation, then the pass.
  - Regenerate the real-shape op list first, because the earlier export lives in scratch. The prompt only
    has to be long enough to fill a 512-token batch; the exported shapes are the same for any text:

    ```sh
    PROMPT="$(printf 'word %.0s' {1..600})"
    build-vkb/bin/test-export-graph-ops -m "$MODEL" -ngl 99 -b 512 -ub 512 -n 0 -p "$PROMPT" \
      -o "$JCODE_SCRATCH_DIR/prefill-b512-tests.txt"
    build-vkb/bin/test-backend-ops perf -o FLASH_ATTN_EXT -b Vulkan0 \
      --test-file "$JCODE_SCRATCH_DIR/prefill-b512-tests.txt"
    ```

    Check that the exported rows really are 512 wide before trusting the numbers; the op description's
    third field is the row count.
  - Harness sweep for the isolated op: `test-backend-ops perf -o FLASH_ATTN_EXT -b Vulkan0 -p "hsk=256"`,
    after reading the real variable names off one unfiltered run, since the filter is a regex on the case
    vars string.
  - Pass: the task 1 command, reading only the flash attention rows.
  - Report both: a knob that helps the isolated op but not the pass is not a win, because occupancy and
    cache effects only show up in place.

- [ ] 2.4 Confirm the winner on the pass with the logger on, and check decode too, since FA at one row
  takes a different branch of the tuner.

- [ ] 2.5 If and only if the pass gains at least 5% (about 127 ms), turn the winning configuration into a
  rule in `get_fa_tuning_params_scalar`: the narrowest condition that covers this case, written in the
  style of the surrounding branches, with a comment saying what was measured and on what. Then revert the
  override, rebuild, and re-run the pass to confirm the rule alone reproduces the win.

- [ ] 2.6 Correctness: `build-vkb/bin/test-backend-ops test -o FLASH_ATTN_EXT -b Vulkan0` must pass, and a
  prefill output comparison against the pre-change build must be identical, using the same method as
  stage 4.

**Verification:**
- Keep decision is backed by a same-session pp512 pair, plus an unchanged tg128 within noise (1%).
- If nothing wins, the stage doc records the whole sweep as a negative result and no source change lands.

---

### Task 3: Attention projections

**Files:**
- Modify: `docs/research/05-stage6-prefill-attention.md`
- Possibly modify: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` in the matmul tile selection

**Steps:**

- [ ] 3.1 Establish which path the 680M takes for the quantized matmuls at 512 rows. Read the selection in
  `ggml_vk_matmul` first, then confirm empirically by running the pass with each of
  `GGML_VK_DISABLE_MMVQ=1`, `GGML_VK_FORCE_MMVQ=1`, and `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`, recording
  which one changes the `attn_qkv` row and by how much. That answers both the path question and the
  sensitivity question in one run each.

- [ ] 3.2 If the path is a quantized kernel with a tile choice, sweep the tile and staging options that
  `ggml_vk_matmul` already enumerates, using the same pair method as task 2: the op in isolation
  (`test-backend-ops perf -o MUL_MAT -b Vulkan0`, filtered with `-p` to the layer's shape) and the pass.

- [ ] 3.3 Check the shared-kernel risk before landing anything: `attn_qkv`, the MoE gate, up and down, and
  the dense projections all route through the same `mul_mm` and `mul_mmq` shaders on this backend. A tile
  change that helps `attn_qkv` must be shown not to hurt the MoE rows, which are currently well utilized.

**Verification:**
- A keep decision needs at least 3% on pp512 (about 76 ms) and no row in the logger table worse by more
  than its run-to-run noise.
- `test-backend-ops test -o MUL_MAT -b Vulkan0` passes for the affected types.

---

### Task 4: MoE expert fusion, second

This is the larger change and it goes last, against a bar rather than against a hope. The measured MoE
segment is 513 ms of the 2.53 s pass, running at 41% of the GPU's FP32 peak.

**Files:**
- Modify: `docs/research/05-stage6-prefill-attention.md`
- Possibly create: a fused kernel in `ggml/src/ggml-vulkan/vulkan-shaders/`, plus its host-side setup

**Steps:**

- [ ] 4.1 Re-measure the MoE segment in situ from the task 1 logger output, so the bar is a number from the
  same instrument that will judge the replacement, and write it into the stage doc.

- [ ] 4.2 Build the fused path standalone before touching the graph. The unit is one layer's expert work for
  a 512-token batch: rows grouped by expert, gate and up projections through the existing quantized
  weights, SiLU between them, weighted sum, down projection. Measure it as a standalone kernel against the
  three separate kernels running the same work through the existing op, on the same shapes.

- [ ] 4.3 Go or no-go on the standalone number. The fused kernel is only worth wiring into the graph if it
  beats the three-kernel path by at least 15% at the segment level, because integration costs correctness
  risk in a path that currently produces correct output at 41% of peak. A no-go here is a success for the
  plan and is recorded as one.

- [ ] 4.4 If it is a go, wire it behind the narrowest condition that matches the layer shapes, keep the
  existing path as the fallback for every other shape, and verify correctness the way stage 5 did: same
  output bytes on a fixed prompt, then the pass.

- [ ] 4.5 Note in the stage doc, without acting on them, the two small-op findings carried from the
  profile: a bare `CONCAT` costs 125 ms across the layers and `RMS_NORM` about 1 ms per call. If a later
  stage touches those fused groups, this is the recorded size of the prize.

**Verification:**
- Fused path: pp512 gain at least 5%, `test-backend-ops test -o MUL_MAT_ID -b Vulkan0` passes, and a fixed
  prompt decodes to identical text against the unmodified build.
- No-go path: the standalone numbers are in the stage doc and the tree is unchanged.

---

### Task 5: Gates, documentation, cleanup

**Steps:**

- [ ] 5.1 Revert every temporary change (`git diff` must show only deliberate changes) and rebuild from
  clean, then re-run the final pp512 and tg128 pair.
- [ ] 5.2 Complete `docs/research/05-stage6-prefill-attention.md`: the in-situ baseline, each sweep with its
  table, each keep or drop decision with the number behind it, the surviving source changes, and the
  reverted instrumentation patch text.
- [ ] 5.3 Report the final numbers against the baseline table: pp512 202 t/s, tg128 22.35 t/s, and the
  logger's top rows before and after.
- [ ] 5.4 Ask for commit approval. Nothing in these tasks is committed without it.

**Verification:**
- A clean rebuild plus the final pair reproduces the reported numbers.
- `git status` shows no stray files, and the diff is limited to the files listed in File Structure.
