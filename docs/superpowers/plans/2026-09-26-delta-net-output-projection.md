# Delta-net output projection layout: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the delta-net output projection read its weight once per step instead of once per sequence, by declaring the activation as 2D before the matmul.

**Architecture:** Three call sites in `src/models/` reshape the gated delta net output to 3D `[value_dim, n_seq_tokens, n_seqs]` immediately before `build_lora_mm`. ggml treats dims 2 and 3 of `src1` as a batch dimension, so `MUL_MAT` walks the whole weight once per batch index. Changing the same view to 2D `[value_dim, n_seq_tokens*n_seqs]` is layout identical and turns it into one GEMM over all tokens. No weights, kernels or memory layouts change.

**Tech Stack:** C++17, CMake, the fork's own test binaries (`test-llama-archs`, `test-backend-ops`, `test-export-graph-ops`, `llama-batched-bench`, `llama-bench`, `llama-completion`).

**Spec:** `docs/superpowers/specs/2026-09-26-delta-net-output-projection-design.md`

## Global Constraints

- Commits need explicit human approval for each action (`AGENTS.md`). Every commit step below is a checkpoint: ask, and if approval is not given, leave the tree uncommitted and report the diff instead. Never push.
- Do not add files under `tests/`. Reuse the existing binaries and the fork's verification practice.
- ASCII only, no em dashes. Comments concise, simple wording, no restating the code.
- Any change that alters Qwen3.5 inference must show up as a diff in the `QWEN_ONLY.md` verification log.
- Builds used: `build-cpu` (CPU) and `build-vk` (CPU + Vulkan), both Release with `GGML_NATIVE=ON`.
- Test model paths:
  - MoE: `/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`
  - dense: `/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf`
- Scratch dir for captured output: `/home/herlanggays/.jcode/scratch/ssm-out-fix/`
- Reference numbers from the pre-change diagnosis (`docs/research/04-stage5-diagnosis.md`): Vulkan `linear_attn_out` at 8 tokens 823.85 us, CPU 279.14 us, Vulkan aggregate TG at B=8 39.47 t/s, single stream tg128 22.46 t/s (`-ngl 99`).

---

### Task 1: Record pre-change baselines

Nothing is edited in this task. Its deliverable is a committed record of what the engine did before the change, so every later task can be diffed against it.

**Files:**
- Create: `docs/research/05-ssm-out-fix-baselines.md`
- Create (scratch artifacts): `/home/herlanggays/.jcode/scratch/ssm-out-fix/*.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: the baseline files and the numbers quoted in Tasks 2 to 5.

- [ ] **Step 1: Create the scratch dir and capture deterministic generation**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
S=/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf
D=/home/herlanggays/.jcode/scratch/ssm-out-fix
mkdir -p "$D"

./build-cpu/bin/llama-completion -m "$S" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 64 > "$D/gen-0.8b-cpu.txt" 2>&1
./build-vk/bin/llama-completion -m "$S" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 64 -ngl 99 > "$D/gen-0.8b-vk.txt" 2>&1
./build-cpu/bin/llama-completion -m "$M" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 64 > "$D/gen-35b-cpu.txt" 2>&1
./build-vk/bin/llama-completion -m "$M" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 64 -ngl 99 > "$D/gen-35b-vk.txt" 2>&1
```

Expected: four non-empty files, each ending with generated text. `llama-completion` reads stdin, so it exits on EOF without a prompt.

- [ ] **Step 2: Capture the batched scaling numbers**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
S=/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf
D=/home/herlanggays/.jcode/scratch/ssm-out-fix

./build-vk/bin/llama-batched-bench -m "$M" -ngl 99 -t 8 -c 16384 -npp 512 -ntg 32 -npl 1,2,4,8,16 > "$D/bb-35b-vk.txt" 2>&1
./build-cpu/bin/llama-batched-bench -m "$M" -t 8 -c 16384 -npp 512 -ntg 32 -npl 1,2,4,8,16 > "$D/bb-35b-cpu.txt" 2>&1
./build-vk/bin/llama-batched-bench -m "$S" -ngl 99 -t 8 -c 4096 -npp 128 -ntg 32 -npl 1,2,4,8,16 > "$D/bb-0.8b-vk.txt" 2>&1
./build-cpu/bin/llama-batched-bench -m "$S" -t 8 -c 4096 -npp 128 -ntg 32 -npl 1,2,4,8,16 > "$D/bb-0.8b-cpu.txt" 2>&1
```

Expected for `bb-35b-vk.txt` (from the diagnosis run): S_TG 21.19 / 30.35 / 37.40 / 39.47 / 34.76 t/s at B = 1 / 2 / 4 / 8 / 16.

- [ ] **Step 3: Capture single stream and per-op numbers**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/ssm-out-fix

./build-vk/bin/llama-bench -m "$M" -r 3 -p 256 -n 128 -t 8 -ngl 99 > "$D/bench-35b-vk.txt" 2>&1
./build-cpu/bin/llama-bench -m "$M" -r 3 -p 256 -n 128 -t 8 > "$D/bench-35b-cpu.txt" 2>&1

cd "$D" && /home/herlanggays/RISET/llama.cpp/build-vk/bin/test-export-graph-ops -m "$M" -np 8 -c 4096 > /dev/null 2>&1
cd /home/herlanggays/RISET/llama.cpp && ./build-vk/bin/test-backend-ops perf --test-file "$D/tests.txt" -b Vulkan0 > "$D/perf-35b-vk-np8.txt" 2>&1
```

Expected: `bench-35b-vk.txt` shows tg128 about 22.5 t/s and pp256 about 200 t/s. In `perf-35b-vk-np8.txt`, `MUL_MAT(name=linear_attn_out-0,ne=[2048,1,8,1]...)` reports about 823 us.

- [ ] **Step 4: Write the baseline record**

Create `docs/research/05-ssm-out-fix-baselines.md` containing: the machine and build lines, the four generation baselines (prompt, seed, temperature, token count, and the resulting text), the batched scaling tables for both models and both backends, the single stream table, and the `linear_attn_out` before numbers (823.85 us Vulkan 8 tokens, 279.14 us CPU 8 tokens, 109.57 us Vulkan 1 token, 39.35 us CPU 1 token). State that these are the pre-change references for the tasks that follow.

- [ ] **Step 5: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add docs/research/05-ssm-out-fix-baselines.md
git commit -m "docs: record pre-change baselines for the delta-net output projection fix"
```

If approval is not given, report the file and leave it uncommitted.

---

### Task 2: Fix `src/models/qwen35.cpp`

The dense Qwen3.5 path, exercised by the 0.8B model.

**Files:**
- Modify: `src/models/qwen35.cpp:456-457`

**Interfaces:**
- Consumes: `head_v_dim`, `num_v_heads`, `n_seq_tokens`, `n_seqs`, `attn_out_norm` from the surrounding delta-net builder; `build_lora_mm(weight, src1, scale)`.
- Produces: `final_output`, a 2D view `[head_v_dim*num_v_heads, n_seq_tokens*n_seqs]`, consumed by `build_lora_mm`.

- [ ] **Step 1: Reproduce the failure mode as a measurement, not a test**

There is no unit test for this; the repo forbids adding files under `tests/`. The observable is the op cost and the batched throughput. Confirm the baseline is as recorded:

```bash
grep -E "linear_attn_out" /home/herlanggays/.jcode/scratch/ssm-out-fix/perf-35b-vk-np8.txt | sed 's/\x1b\[[0-9;]*m//g' | sed 's/  */ /g'
```

Expected: a row with `ne=[2048,1,8,1]` and about 823 us, and a row with `ne=[2048,64,8,1]`.

- [ ] **Step 2: Apply the change**

Replace:

```cpp
    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
```

with:

```cpp
    // flatten tokens and sequences so the projection is one GEMM over all of them
    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
```

- [ ] **Step 3: Build both trees**

```bash
cd /home/herlanggays/RISET/llama.cpp
cmake --build build-cpu -j8 2>&1 | tail -3
cmake --build build-vk  -j8 2>&1 | tail -3
```

Expected: both finish with no errors.

- [ ] **Step 4: Run the architecture test**

```bash
cd /home/herlanggays/RISET/llama.cpp
./build-cpu/bin/test-llama-archs -a qwen35 -s 1
```

Expected: pass, same as the pre-change run in `QWEN_ONLY.md`.

- [ ] **Step 5: Check generation is byte identical**

```bash
cd /home/herlanggays/RISET/llama.cpp
S=/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf
D=/home/herlanggays/.jcode/scratch/ssm-out-fix
./build-cpu/bin/llama-completion -m "$S" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 64 > "$D/gen-0.8b-cpu-after.txt" 2>&1
./build-vk/bin/llama-completion -m "$S" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 64 -ngl 99 > "$D/gen-0.8b-vk-after.txt" 2>&1
diff <(grep -v "^0\." "$D/gen-0.8b-cpu.txt") <(grep -v "^0\." "$D/gen-0.8b-cpu-after.txt") && echo CPU_IDENTICAL
diff <(grep -v "^0\." "$D/gen-0.8b-vk.txt") <(grep -v "^0\." "$D/gen-0.8b-vk-after.txt") && echo VK_IDENTICAL
```

Expected: both `CPU_IDENTICAL` and `VK_IDENTICAL`. The `grep -v "^0\."` strips timestamped log lines. If the diff shows only timing lines, strip those too rather than accepting a content difference.

- [ ] **Step 6: Re-run the batched scaling for this model**

```bash
cd /home/herlanggays/RISET/llama.cpp
S=/home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf
D=/home/herlanggays/.jcode/scratch/ssm-out-fix
./build-vk/bin/llama-batched-bench -m "$S" -ngl 99 -t 8 -c 4096 -npp 128 -ntg 32 -npl 1,2,4,8,16 > "$D/bb-0.8b-vk-after.txt" 2>&1
diff "$D/bb-0.8b-vk.txt" "$D/bb-0.8b-vk-after.txt"
```

Expected: B=1 unchanged (86.5 t/s), B=8 and B=16 at or above the before numbers (230.02 and 249.57 t/s). Do not expect a large jump: the 0.8B `ssm_out` weight is small and largely cache resident on the CPU side.

- [ ] **Step 7: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add src/models/qwen35.cpp
git commit -m "qwen35 : one weight read per step in the delta-net output projection"
```

---

### Task 3: Fix `src/models/qwen35moe.cpp`

The MoE path, exercised by the 35B model and the main payoff.

**Files:**
- Modify: `src/models/qwen35moe.cpp:480-481`

**Interfaces:**
- Consumes: same locals as Task 2, in the MoE model's delta-net builder.
- Produces: `final_output` as a 2D view, consumed by `build_lora_mm`.

- [ ] **Step 1: Apply the change**

Replace:

```cpp
    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
```

with:

```cpp
    // flatten tokens and sequences so the projection is one GEMM over all of them
    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
```

- [ ] **Step 2: Build both trees**

```bash
cd /home/herlanggays/RISET/llama.cpp
cmake --build build-cpu -j8 2>&1 | tail -3
cmake --build build-vk  -j8 2>&1 | tail -3
```

Expected: both finish with no errors.

- [ ] **Step 3: Run the architecture tests for the touched paths**

```bash
cd /home/herlanggays/RISET/llama.cpp
./build-cpu/bin/test-llama-archs -a qwen35moe -s 1
./build-cpu/bin/test-llama-archs -a qwen35  -s 1
```

Expected: both pass.

- [ ] **Step 4: Check 35B generation is byte identical**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/ssm-out-fix
./build-cpu/bin/llama-completion -m "$M" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 64 > "$D/gen-35b-cpu-after.txt" 2>&1
./build-vk/bin/llama-completion -m "$M" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 64 -ngl 99 > "$D/gen-35b-vk-after.txt" 2>&1
diff <(grep -v "^0\." "$D/gen-35b-cpu.txt") <(grep -v "^0\." "$D/gen-35b-cpu-after.txt") && echo CPU_IDENTICAL
diff <(grep -v "^0\." "$D/gen-35b-vk.txt") <(grep -v "^0\." "$D/gen-35b-vk-after.txt") && echo VK_IDENTICAL
```

Expected: both identical. This is the correctness gate for the whole change.

- [ ] **Step 5: Confirm the op dropped, at the real shape**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/ssm-out-fix
rm -rf "$D/after" && mkdir -p "$D/after"
cd "$D/after" && /home/herlanggays/RISET/llama.cpp/build-vk/bin/test-export-graph-ops -m "$M" -np 8 -c 4096 > /dev/null 2>&1
cd /home/herlanggays/RISET/llama.cpp
./build-vk/bin/test-backend-ops perf --test-file "$D/after/tests.txt" -b Vulkan0 > "$D/perf-35b-vk-np8-after.txt" 2>&1
grep -E "linear_attn_out" "$D/perf-35b-vk-np8-after.txt" | sed 's/\x1b\[[0-9;]*m//g' | sed 's/  */ /g'
```

Expected: the 8 token row now reads `ne=[2048,8,1,1]` and about 110 us, against 823.85 us before. If it is still about 823 us, the reshape did not take effect and the graph is still batched: stop and investigate before proceeding.

- [ ] **Step 6: Measure the payoff**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/ssm-out-fix
./build-vk/bin/llama-batched-bench -m "$M" -ngl 99 -t 8 -c 16384 -npp 512 -ntg 32 -npl 1,2,4,8,16 > "$D/bb-35b-vk-after.txt" 2>&1
./build-cpu/bin/llama-batched-bench -m "$M" -t 8 -c 16384 -npp 512 -ntg 32 -npl 1,2,4,8,16 > "$D/bb-35b-cpu-after.txt" 2>&1
./build-vk/bin/llama-bench -m "$M" -r 3 -p 256 -n 128 -t 8 -ngl 99 > "$D/bench-35b-vk-after.txt" 2>&1
```

Expected: B=8 aggregate TG up from 39.47 t/s to about 43.5 t/s, B=1 unchanged at about 21.2 t/s, tg128 unchanged at about 22.5 t/s. If B=8 does not improve, the op was not the bottleneck in situ and the spec's prediction is wrong; report that rather than tuning to fit.

- [ ] **Step 7: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add src/models/qwen35moe.cpp
git commit -m "qwen35moe : one weight read per step in the delta-net output projection"
```

---

### Task 4: Fix `src/models/qwen4exp.cpp`

Same code path, different arch. No GGUF for this arch exists on this machine, so verification is the graph test only.

**Files:**
- Modify: `src/models/qwen4exp.cpp:977`

**Interfaces:**
- Consumes: same locals as Task 2, in the qwen4exp delta-net builder.
- Produces: `final_output` as a 2D view, consumed by `build_lora_mm`.

- [ ] **Step 1: Apply the change**

Replace:

```cpp
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
```

with:

```cpp
    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
```

- [ ] **Step 2: Build and run the arch test**

```bash
cd /home/herlanggays/RISET/llama.cpp
cmake --build build-cpu -j8 2>&1 | tail -3
./build-cpu/bin/test-llama-archs -a qwen4exp -s 1
```

Expected: pass. If it fails, revert this file only (`git checkout -- src/models/qwen4exp.cpp`) and report that qwen4exp needs its own investigation, as the spec allows.

- [ ] **Step 3: Confirm the other archs still pass**

```bash
cd /home/herlanggays/RISET/llama.cpp
./build-cpu/bin/test-llama-archs -a qwen35 -s 1
./build-cpu/bin/test-llama-archs -a qwen35moe -s 1
```

Expected: both pass.

- [ ] **Step 4: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add src/models/qwen4exp.cpp
git commit -m "qwen4exp : one weight read per step in the delta-net output projection"
```

---

### Task 5: Full verification sweep and documentation

**Files:**
- Modify: `QWEN_ONLY.md` (verification log section)
- Create: `docs/research/06-ssm-out-fix-results.md`

**Interfaces:**
- Consumes: the before and after artifacts from Tasks 1 to 4.
- Produces: the recorded result of the change.

- [ ] **Step 1: Run the full test suite**

```bash
cd /home/herlanggays/RISET/llama.cpp
ctest --test-dir build-cpu -L main --output-on-failure 2>&1 | tail -15
ctest --test-dir build-vk  -L main --output-on-failure 2>&1 | tail -15
```

Expected: all tests pass on both builds, same counts as recorded in `QWEN_ONLY.md` (34 main tests).

- [ ] **Step 2: Run the backend op suites**

```bash
cd /home/herlanggays/RISET/llama.cpp
./build-vk/bin/test-backend-ops -b CPU     2>&1 | tail -3
./build-vk/bin/test-backend-ops -b Vulkan0 2>&1 | tail -3
```

Expected: the same pass counts as `QWEN_ONLY.md` records (19914 CPU, 18976 Vulkan0).

- [ ] **Step 3: Write the results doc**

Create `docs/research/06-ssm-out-fix-results.md` with: the diff of each op number before and after, the batched scaling tables before and after for both models and both backends, single stream before and after, and a plain statement of whether the spec's predictions held. If a prediction did not hold, say so with the numbers and leave the change in place only if correctness and single stream are unaffected.

- [ ] **Step 4: Update the fork's verification log**

Add a dated entry to the verification log in `QWEN_ONLY.md` in the same table format as the existing entries, covering the arch tests, `ctest -L main`, `test-backend-ops` counts, the byte identical generation checks, and the batched throughput change.

- [ ] **Step 5: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add QWEN_ONLY.md docs/research/06-ssm-out-fix-results.md
git commit -m "docs: record the delta-net output projection results"
```

---

## Self-Review

**Spec coverage:** the spec's change is covered by Tasks 2, 3 and 4 (one file each). The spec's verification list maps to Task 1 (baselines), Task 2 steps 4 to 6, Task 3 steps 3 to 6, Task 4 step 2, and Task 5 steps 1 and 2. The spec's fallback (`ggml_cont_2d`) is not a task because it is only reachable if `ggml_reshape_2d` asserts; Task 2 step 3 and Task 3 step 2 are where that would surface. The spec's out of scope items are not tasks.

**Placeholder scan:** no TBD, no "add error handling", no "similar to Task N". Every step carries its command and expected output, or its exact code.

**Type consistency:** `final_output` keeps its name and its role as the `build_lora_mm` input in all three tasks; only its shape changes from `[value_dim, n_seq_tokens, n_seqs]` to `[value_dim, n_seq_tokens*n_seqs]`. The trailing `ggml_reshape_2d(cur, n_embd, n_seq_tokens*n_seqs)` is left untouched in all three files, and consumes the same `cur` name as before.
