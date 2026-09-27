# Recurrent state view (phase 1) Implementation Plan

**Outcome: done and measured.** `build_rs` returns a view of the cache when the active rows are
already contiguous, which removes a per-token gather and its write-back, with byte identical outputs.
Results in `docs/research/08-recurrent-state-phase1-results.md`. The checkboxes below were never
ticked as the work landed.
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the active recurrent state rows are already `[head, head+n_rs)` in order, alias them instead of gathering them into a graph buffer, removing one 2 MB copy per delta-net layer per token.

**Architecture:** Add a contiguity predicate to the recurrent memory, carry it on the graph input next to the existing `head` and `rs_z` fields, use it in `build_rs` to return a `ggml_view_2d` of the cache and to skip the extra-state copy, and include it in `can_reuse` so a graph built for one mapping is never reused for another.

**Tech Stack:** C++17, CMake, the fork's test binaries (`test-llama-archs`, `llama-bench`, `llama-batched-bench`, `llama-passkey`, `llama-completion`).

**Spec:** `docs/superpowers/specs/2026-09-26-recurrent-state-copies-design.md` (phase 1)

## Global Constraints

- Commits need explicit human approval for each action (`AGENTS.md`). Every commit step is a checkpoint: ask, and if approval is not given, leave the tree uncommitted and report the diff. Never push.
- Do not add files under `tests/`. Reuse the existing binaries.
- ASCII only, no em dashes. Comments concise, simple wording, no restating the code.
- Builds: `build-cpu` and `build-vk`, both Release with `GGML_NATIVE=ON`, already configured.
- Models: MoE `/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`, dense `/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf`.
- Scratch dir: `/home/herlanggays/.jcode/scratch/rs-view/`
- Generation output from `llama-completion` and `llama-passkey` goes through the log callback to stderr, so capture with `2>&1` and strip timestamped lines before diffing.
- `llama-parallel` aborts on this architecture, do not use it. Use `llama-passkey -np N` for multi-sequence functional checks.
- Predicted effect from the ablation (spec evidence table): tg128 22.35 to about 24.0 t/s, batch 8 aggregate 31.61 to about 44.5 t/s. Those are the numbers to compare against, not a pass/fail threshold.

---

### Task 1: Record pre-change baselines

**Files:**
- Create: `docs/research/07-recurrent-state-phase1-baselines.md`
- Create (scratch): `/home/herlanggays/.jcode/scratch/rs-view/*.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: the reference files and numbers Tasks 2 and 3 diff against.

- [ ] **Step 1: Scratch dir, single sequence baselines**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
S=/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf
D=/home/herlanggays/.jcode/scratch/rs-view
mkdir -p "$D"

./build-cpu/bin/llama-completion -m "$S" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 64 > "$D/gen-0.8b-cpu.txt" 2>&1
./build-vk/bin/llama-completion -m "$S" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 64 -ngl 99 > "$D/gen-0.8b-vk.txt" 2>&1
./build-vk/bin/llama-completion -m "$M" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 64 -ngl 99 > "$D/gen-35b-vk.txt" 2>&1
```

Expected: three non-empty files ending in generated text.

- [ ] **Step 2: Multi-sequence functional baseline**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
S=/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf
D=/home/herlanggays/.jcode/scratch/rs-view

./build-cpu/bin/llama-passkey -m "$S" -np 2 -n 16 -c 4096 -s 1 > "$D/passkey-0.8b-cpu.txt" 2>&1
./build-vk/bin/llama-passkey -m "$S" -np 2 -n 16 -c 4096 -s 1 -ngl 99 > "$D/passkey-0.8b-vk.txt" 2>&1
```

Expected: each run prints `passkey = <N>` and then the retrieved answer containing the same `<N>`. The model is small, so if a retrieval fails, record it: the check is identity between before and after, not retrieval quality.

- [ ] **Step 3: Throughput baseline**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/rs-view

./build-vk/bin/llama-bench -m "$M" -r 3 -p 0 -n 128 -t 8 -ngl 99 > "$D/bench-tg-35b-vk.txt" 2>&1
./build-vk/bin/llama-batched-bench -m "$M" -ngl 99 -t 8 -c 4096 -npp 128 -ntg 32 -npl 1,2,4,8 > "$D/bb-35b-vk.txt" 2>&1
```

Expected: tg128 about 22.3 t/s; in the batched table, S_TG at B=8 about 31.6 t/s with `-npp 128`.

- [ ] **Step 4: Write the baseline record**

Create `docs/research/07-recurrent-state-phase1-baselines.md` with the machine and build line, the generation baselines (prompt, seed, temperature, tokens, resulting text), the passkey runs (inserted and retrieved key per sequence), and the throughput tables. State that these are the pre-change references for phase 1.

- [ ] **Step 5: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add docs/research/07-recurrent-state-phase1-baselines.md
git commit -m "docs: record pre-change baselines for the recurrent state view"
```

---

### Task 2: Contiguity predicate and the view

**Files:**
- Modify: `src/llama-memory-recurrent.h` (context class, around line 174)
- Modify: `src/llama-memory-recurrent.cpp` (around line 1332, after `get_size`)
- Modify: `src/llama-graph.h` (struct `llm_graph_input_rs` at line 241, and the `build_rs` declaration at line 1032)
- Modify: `src/llama-graph.cpp` (`build_rs_inp_impl` around line 2604, `llm_graph_input_rs::can_reuse` around line 340, `build_rs` at line 2551 and its wrapper at line 2618)

**Interfaces:**
- Consumes: `llama_memory_recurrent::cells` (`std::vector<mem_cell>`, `src0` member), `mem->n_rs_seq`, existing `get_head()`, `get_n_rs()`.
- Produces: `bool llama_memory_recurrent_context::state_rows_are_contiguous() const`; `bool llm_graph_input_rs::rows_contiguous`; a new trailing parameter `bool state_contiguous` on the first `build_rs` overload (declared with a default of `false` so only the wrapper needs updating).

- [ ] **Step 1: Declare the predicate**

In `src/llama-memory-recurrent.h`, inside `class llama_memory_recurrent_context`, after `uint32_t get_size() const;`:

```cpp
    // true when the active state rows are exactly [head, head + n_rs) in order
    bool state_rows_are_contiguous() const;
```

- [ ] **Step 2: Implement the predicate**

In `src/llama-memory-recurrent.cpp`, after the `get_size()` definition:

```cpp
bool llama_memory_recurrent_context::state_rows_are_contiguous() const {
    // with rollback the row index carries a snapshot offset, so the mapping is never the identity
    if (mem->n_rs_seq != 0) {
        return false;
    }

    const uint32_t head = get_head();
    const uint32_t n_rs = get_n_rs();

    for (uint32_t i = 0; i < n_rs; ++i) {
        if (mem->cells[head + i].src0 != (int32_t) (head + i)) {
            return false;
        }
    }

    return true;
}
```

- [ ] **Step 3: Carry the predicate on the graph input**

In `src/llama-graph.h`, in `class llm_graph_input_rs`, after `int32_t rs_z;`:

```cpp
    // true when the active state rows are contiguous and in order
    bool rows_contiguous;
```

In `src/llama-graph.cpp`, in `build_rs_inp_impl`, after `inp->rs_z = mctx_cur->get_rs_z();`:

```cpp
    inp->rows_contiguous = mctx_cur->state_rows_are_contiguous();
```

- [ ] **Step 4: Invalidate a reused graph when the predicate changes**

In `src/llama-graph.cpp`, in `llm_graph_input_rs::can_reuse`, after `res &= rs_z == mctx->get_rs_z();`:

```cpp
    res &= rows_contiguous == mctx->state_rows_are_contiguous();
```

- [ ] **Step 5: Use the view when contiguous**

In `src/llama-graph.cpp`, in the first `build_rs` overload, add the parameter:

```cpp
ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy_main,
        ggml_tensor * state_copy_extra,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
            bool      state_contiguous,
        const llm_graph_get_rows_fn & get_state_rows) const {
    GGML_UNUSED(rs_size);

    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    if (state_contiguous) {
        // the active rows are already in place: alias them, and the extra rows need no copy either
        return ggml_view_2d(ctx0, s, state_size, n_seqs, s->nb[1], (size_t) rs_head * s->nb[1]);
    }

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // {state_size, rs_size} -> {state_size, n_seqs}
    ggml_tensor * output_states = get_state_rows(ctx0, states, state_copy_main);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_get_rows(ctx0, states, state_copy_extra);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])));

    return output_states;
}
```

In `src/llama-graph.h`, add the parameter to the declaration with a default so no other caller changes:

```cpp
    ggml_tensor * build_rs(
            ggml_tensor * s,
            ggml_tensor * state_copy_main,
            ggml_tensor * state_copy_extra,
                int32_t   state_size,
                int32_t   n_seqs,
               uint32_t   n_rs,
               uint32_t   rs_head,
               uint32_t   rs_size,
                int32_t   rs_zero,
                bool      state_contiguous = false,
            const llm_graph_get_rows_fn & get_state_rows = ggml_get_rows) const;
```

In `src/llama-graph.cpp`, in the wrapper, pass the flag:

```cpp
    return build_rs(s, inp->s_copy_main, inp->s_copy_extra, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    inp->rows_contiguous,
                    get_state_rows);
```

- [ ] **Step 6: Build both trees**

```bash
cd /home/herlanggays/RISET/llama.cpp
cmake --build build-cpu -j8 2>&1 | tail -3
cmake --build build-vk  -j8 2>&1 | tail -3
```

Expected: both clean. A compile error here means a signature or member name is off; fix it before continuing.

- [ ] **Step 7: Architecture tests**

```bash
cd /home/herlanggays/RISET/llama.cpp
./build-cpu/bin/test-llama-archs -a qwen35    -s 1
./build-cpu/bin/test-llama-archs -a qwen35moe -s 1
./build-cpu/bin/test-llama-archs -a qwen4exp  -s 1
```

Expected: all pass, as in `QWEN_ONLY.md`.

- [ ] **Step 8: Single sequence output identity**

```bash
cd /home/herlanggays/RISET/llama.cpp
S=/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/rs-view

./build-cpu/bin/llama-completion -m "$S" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 64 > "$D/gen-0.8b-cpu-after.txt" 2>&1
./build-vk/bin/llama-completion -m "$S" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 64 -ngl 99 > "$D/gen-0.8b-vk-after.txt" 2>&1
./build-vk/bin/llama-completion -m "$M" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 64 -ngl 99 > "$D/gen-35b-vk-after.txt" 2>&1

diff <(grep -v "^0\." "$D/gen-0.8b-cpu.txt") <(grep -v "^0\." "$D/gen-0.8b-cpu-after.txt") && echo CPU_IDENTICAL
diff <(grep -v "^0\." "$D/gen-0.8b-vk.txt")  <(grep -v "^0\." "$D/gen-0.8b-vk-after.txt")  && echo VK_IDENTICAL
diff <(grep -v "^0\." "$D/gen-35b-vk.txt")  <(grep -v "^0\." "$D/gen-35b-vk-after.txt")  && echo MOE_IDENTICAL
```

Expected: all three identical. `grep -v "^0\."` strips timestamped log lines; also strip timing lines if they differ and nothing else does.

- [ ] **Step 9: Multi-sequence functional identity**

```bash
cd /home/herlanggays/RISET/llama.cpp
S=/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf
D=/home/herlanggays/.jcode/scratch/rs-view

./build-cpu/bin/llama-passkey -m "$S" -np 2 -n 16 -c 4096 -s 1 > "$D/passkey-0.8b-cpu-after.txt" 2>&1
./build-vk/bin/llama-passkey -m "$S" -np 2 -n 16 -c 4096 -s 1 -ngl 99 > "$D/passkey-0.8b-vk-after.txt" 2>&1

diff <(grep "^0\..*passkey =" "$D/passkey-0.8b-cpu.txt" | sed 's/^[0-9.]* //') \
     <(grep "^0\..*passkey =" "$D/passkey-0.8b-cpu-after.txt" | sed 's/^[0-9.]* //') && echo PASSKEY_CPU_SAME
diff <(grep "^0\..*passkey =" "$D/passkey-0.8b-vk.txt" | sed 's/^[0-9.]* //') \
     <(grep "^0\..*passkey =" "$D/passkey-0.8b-vk-after.txt" | sed 's/^[0-9.]* //') && echo PASSKEY_VK_SAME
```

Expected: the inserted keys and the retrieved answers match the baseline. This is the check that the multi-sequence state mapping did not change meaning.

- [ ] **Step 10: Measure**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/rs-view

./build-vk/bin/llama-bench -m "$M" -r 3 -p 0 -n 128 -t 8 -ngl 99 > "$D/bench-tg-35b-vk-after.txt" 2>&1
./build-vk/bin/llama-batched-bench -m "$M" -ngl 99 -t 8 -c 4096 -npp 128 -ntg 32 -npl 1,2,4,8 > "$D/bb-35b-vk-after.txt" 2>&1
grep tg128 "$D/bench-tg-35b-vk-after.txt"
tail -6 "$D/bb-35b-vk-after.txt"
```

Expected: tg128 rises from about 22.3 toward 24.0 t/s, and S_TG at B=8 rises from about 31.6 toward 44.5. If nothing moves, the predicate is returning false in this configuration: log it (temporarily add `LLAMA_LOG_DEBUG` in `build_rs_inp_impl`) and find out why before concluding anything.

- [ ] **Step 11: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add src/llama-memory-recurrent.h src/llama-memory-recurrent.cpp src/llama-graph.h src/llama-graph.cpp
git commit -m "graph : alias contiguous recurrent state rows instead of gathering them"
```

---

### Task 3: Prove the slow path, then sweep

**Files:**
- Temporarily modify: `src/llama-memory-recurrent.cpp` (the predicate body), reverted at the end of the task
- Create: `docs/research/08-recurrent-state-phase1-results.md`
- Modify: `QWEN_ONLY.md` (verification log)

**Interfaces:**
- Consumes: everything from Task 2.
- Produces: evidence that the gather path is unchanged, and the recorded result.

- [ ] **Step 1: Force the predicate false**

Temporarily replace the body of `state_rows_are_contiguous` so it returns `false` immediately, keeping the rest of the function commented out beneath it:

```cpp
bool llama_memory_recurrent_context::state_rows_are_contiguous() const {
    return false; // TEMPORARY: force the gather path for this verification
```

- [ ] **Step 2: Rebuild and re-run the identity checks**

```bash
cd /home/herlanggays/RISET/llama.cpp
cmake --build build-cpu -j8 2>&1 | tail -2
S=/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf
D=/home/herlanggays/.jcode/scratch/rs-view

./build-cpu/bin/llama-completion -m "$S" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 64 > "$D/slowpath-gen-0.8b-cpu.txt" 2>&1
./build-cpu/bin/llama-passkey -m "$S" -np 2 -n 16 -c 4096 -s 1 > "$D/slowpath-passkey-0.8b-cpu.txt" 2>&1

diff <(grep -v "^0\." "$D/gen-0.8b-cpu.txt") <(grep -v "^0\." "$D/slowpath-gen-0.8b-cpu.txt") && echo SLOWPATH_GEN_IDENTICAL
diff <(grep "^0\..*passkey =" "$D/passkey-0.8b-cpu.txt" | sed 's/^[0-9.]* //') \
     <(grep "^0\..*passkey =" "$D/slowpath-passkey-0.8b-cpu.txt" | sed 's/^[0-9.]* //') && echo SLOWPATH_PASSKEY_IDENTICAL
```

Expected: both identical. That is the proof that the gather path still produces exactly the pre-change result, on both a single and a multi-sequence workload.

- [ ] **Step 3: Restore the predicate and rebuild**

Put the full body back, rebuild both trees, and confirm the tree is clean:

```bash
cd /home/herlanggays/RISET/llama.cpp
cmake --build build-cpu -j8 2>&1 | tail -2
cmake --build build-vk  -j8 2>&1 | tail -2
git diff --stat src/llama-memory-recurrent.cpp
```

Expected: the diff for that file is empty, meaning only the intended Task 2 change remains.

- [ ] **Step 4: Full sweep**

```bash
cd /home/herlanggays/RISET/llama.cpp
ctest --test-dir build-cpu -L main --output-on-failure 2>&1 | tail -8
ctest --test-dir build-vk  -L main --output-on-failure 2>&1 | tail -8
./build-vk/bin/test-backend-ops -b CPU     2>&1 | tail -2
./build-vk/bin/test-backend-ops -b Vulkan0 2>&1 | tail -2
```

Expected: `ctest` all pass on both builds; `test-backend-ops` counts match `QWEN_ONLY.md` (19914 CPU, 18976 Vulkan0). This change adds no ops and no kernels, so no counts should move.

- [ ] **Step 5: Write the results**

Create `docs/research/08-recurrent-state-phase1-results.md`: the before and after tg128 and batched tables, the passkey results before and after, the slow-path forcing result, and a plain statement of whether the predicted numbers held. If the measured gain is well below the prediction, say so and record what the predicate actually returned.

- [ ] **Step 6: Update the fork verification log**

Add a dated entry to `QWEN_ONLY.md` in the same table format as the existing entries: arch tests, `ctest -L main`, `test-backend-ops` counts, the output identity checks (single and multi sequence, fast and forced-slow path), and the throughput delta.

- [ ] **Step 7: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add docs/research/08-recurrent-state-phase1-results.md QWEN_ONLY.md
git commit -m "docs: record the recurrent state view phase 1 results"
```

---

## Self-Review

**Spec coverage:** the spec's phase 1 requirement (view when the active rows are `[head, head+n_seqs)`, predicate exposed, `can_reuse` updated, no kernels) maps to Task 2 steps 1 to 5. Its correctness condition maps to Task 2 step 9 and Task 3 steps 1 to 3. Its expected result maps to Task 2 step 10. The spec's phase 2 is deliberately not in this plan; its mechanism is decided by a spike that has not run.

**Placeholder scan:** no TBD, no "similar to Task N", every code step carries the full replacement text, every command carries its expected result.

**Type consistency:** `state_rows_are_contiguous()` is declared const and used as such in `build_rs_inp_impl` and `can_reuse`; `rows_contiguous` is a `bool` on the input, next to `head` and `rs_z`; the new `build_rs` parameter is `bool state_contiguous` and is passed from the wrapper only, since the declaration gives it a default.

**Known gap, stated rather than hidden:** a genuine non-contiguous mapping cannot be constructed reliably from the command line on this model, because the batch window always covers every cell the batch uses. The predicate is therefore covered by construction, by the forced-false slow path run, and by the observation that the fast path and the gather agree in every configuration we can run. If the predicate is wrong, it is wrong in the direction of returning true when the mapping is not the identity, which the forced-false run cannot catch; the review of step 2's loop against `s_copy` is what covers that.
