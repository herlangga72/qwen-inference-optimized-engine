# KV planar3 V cache (plan 2, Vulkan) Implementation Plan

**Outcome: implemented, with the speed gate still open.** The type runs on Vulkan and CPU and Vulkan
produce byte identical text with it. The plan understated the work: it covered the read side only, and
the write side needs a GPU quantizer, because V is computed on the device. Results, including the
decode speeds at four depths, are in `docs/research/25-kv-planar3-vulkan-results.md`. The f16 baseline
that the speed gate is written against was started and stopped on request, so no speedup is claimed.
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Vulkan flash attention path consume a `planar3_0` V cache, then measure the speed premise at 16384 and 32768 depth, where the byte saving actually lands.

**Architecture:** The Vulkan FA is already built for asymmetric K/V types: spec constants `FaTypeK`/`FaTypeV` select a branch of an uber `dequantize4()` in `flash_attn_dequant.glsl`, with aliased SSBO views per type. Adding planar3 means one new shader branch, one block struct, one entry in the block-size map, and the host-side acceptance of the type. The inverse rotation is a plain `ggml_mul_mat` with a host-built matrix, so it needs no Vulkan work at all: the graph-level change from plan 1 already runs on this backend.

**Tech Stack:** GLSL (Vulkan shaders, generated to SPIR-V at build time), C++ (`ggml-vulkan.cpp`), CMake, `test-backend-ops`, `llama-bench`, `llama-perplexity`.

**Spec:** `docs/superpowers/specs/2026-09-26-kv-planar3-design.md`
**Depends on:** `docs/superpowers/plans/2026-09-26-kv-planar3-cpu.md`, which defines the type, the tables and the quality gate. Do not start this plan unless plan 1's gate passed.

## Global Constraints

- Commits need explicit human approval for each action (`AGENTS.md`). Every commit step is a checkpoint: ask, and if approval is not given, leave the tree uncommitted and report. Never push.
- The type is KV only, never a weight type.
- The codebook in the shader must match the host's exactly. The 8 values come from the host implementation, dumped once after plan 1 builds, pasted as literals with a comment recording where they came from. The shader does not solve Lloyd-Max.
- The shader needs no Givens table: the insert-side rotation happens in the host quantizer, and the read-side inverse rotation is the graph matmul.
- Builds: `build-vk` (CPU + Vulkan), Release, `GGML_NATIVE=ON`.
- Model: `/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`
- Scratch dir: `/home/herlanggays/.jcode/scratch/kvplanar3/`
- Speed gate: Vulkan decode at 16384 depth must improve measurably against the f16-V baseline, with 32768 reported as well. The expected figure from the spec is about 8% at 16k and 14% at 32k. Below about 3% at 16k, the type is not worth its maintenance cost and the work stops with the number recorded.
- Correctness gate: CPU and Vulkan must agree. The same prompt, seed and cache types must produce identical text on both backends, since the FA path is backend specific and the dequant arithmetic must be identical.

---

### Task 1: The shader type

**Files:**
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/types.glsl` (block structs around line 183)
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/fa_types.glsl` (`fa_block_elems` at line 9)

**Interfaces:**
- Consumes: the type id and block layout from plan 1 (`GGML_TYPE_PLANAR3_0`, 256 coordinates, 96 B of codes, f16 norm).
- Produces: `block_planar3_0_packed16` and a `fa_block_elems` case, so the FA shaders can size their tiles.

- [ ] **Step 1: Add the block struct**

In `types.glsl`, after the Q8_0 family:

```glsl
// KV only: 256 coordinates per block, one f16 norm, 32 groups of 8 coordinates
// packed little-endian into 3 bytes. Mirrors block_planar3_0 in ggml-common.h.
struct block_planar3_0_packed16
{
    float16_t norm;
    uint8_t   qs[96];
};
```

Confirm the struct stride the SPIR-V ends up with (std430 aligns the stride to the largest member alignment, which is 2 here, so 98 B is already aligned). If the driver reports a stride mismatch, pad to 100 B with a `uint8_t pad[2]` and note it, because the host block is 98 B and any padding must match what the KV cache stores. The safest check is the correctness gate at the end of Task 3.

- [ ] **Step 2: Register the type id and block size**

Where the shaders get their `GGML_TYPE_*` constants (the same place `GGML_TYPE_Q8_0` comes from in `types.glsl`), add:

```glsl
#define GGML_TYPE_PLANAR3_0 43
```

and in `fa_types.glsl`:

```glsl
        case GGML_TYPE_PLANAR3_0: return 256u;
```

- [ ] **Step 3: Build and confirm the shaders still compile**

```bash
cd /home/herlanggays/RISET/llama.cpp
cmake --build build-vk -j8 2>&1 | tail -5
```

Expected: clean. A GLSL error here is a struct or constant mistake, not a host problem.

- [ ] **Step 4: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add ggml/src/ggml-vulkan/vulkan-shaders/types.glsl ggml/src/ggml-vulkan/vulkan-shaders/fa_types.glsl
git commit -m "vulkan : declare the planar3_0 KV block for flash attention"
```

---

### Task 2: The dequant branch

**Files:**
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_dequant.glsl` (aliased views around line 30, `dequantize4()` around line 123)

**Interfaces:**
- Consumes: `block_planar3_0_packed16` from Task 1, the codebook values from the host, the block index the FA already passes to its dequant path.
- Produces: `dequantize4()` returning 4 values in rotated space for `FaTypeV == 43`, so the FA kernel needs no other change.

- [ ] **Step 1: Dump the codebook from the host**

After plan 1 builds, print the 8 centroids from the C implementation once:

```bash
cd /home/herlanggays/RISET/llama.cpp
# temporary main or a debug print in planar3_0_codebook(); capture the 8 values in order
```

Record them in the shader with a comment naming the source, for example:

```glsl
// Lloyd-Max centroids for sigma = 1/16, from ggml-quants.c planar3_0_codebook()
const float PLANAR3_0_CB[8] = { -0.1..., ..., 0.1... };
```

- [ ] **Step 2: Add the aliased V view**

In `flash_attn_dequant.glsl`, next to the other views:

```glsl
layout (binding = 2) readonly buffer V_PACKED_PLANAR3_0 { block_planar3_0_packed16 data[]; } v_packed_planar3_0;
```

- [ ] **Step 3: Add the dequant branch**

In `dequantize4()`, add a branch for the new type id. The shape of the branch: given a block index `ib` and a coordinate offset within the block, load the group's 3 bytes as one `uint`, shift out the four 3-bit indices, look them up in `PLANAR3_0_CB`, scale by `float(v_packed_planar3_0.data[ib].norm)`, and return a `vec4`.

```glsl
    case GGML_TYPE_PLANAR3_0: {
        const block_planar3_0_packed16 blk = v_packed_planar3_0.data[ib];
        const float norm = float(blk.norm);
        const uint g = i / 8u;          // group of 8 coordinates
        const uint w = uint(blk.qs[3u*g + 0u])
                     | (uint(blk.qs[3u*g + 1u]) << 8)
                     | (uint(blk.qs[3u*g + 2u]) << 16);
        const uint base = i % 8u;
        vec4 r;
        for (uint t = 0u; t < 4u; ++t) {
            r[t] = PLANAR3_0_CB[(w >> (3u*(base + t))) & 7u] * norm;
        }
        return r;
    }
```

The exact variable names (`ib`, `i`) must match the surrounding `dequantize4()` signature; adapt them while implementing, and keep the packed little-endian order identical to `quantize_row_planar3_0_ref`.

Note that this returns values in *rotated* space, which is what the FA kernel needs; the inverse rotation happens later in the graph.

- [ ] **Step 4: Build**

```bash
cd /home/herlanggays/RISET/llama.cpp
cmake --build build-vk -j8 2>&1 | tail -5
```

Expected: clean.

- [ ] **Step 5: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_dequant.glsl
git commit -m "vulkan : dequantize planar3_0 V in flash attention"
```

---

### Task 3: Host acceptance and correctness on the GPU

**Files:**
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (the FA type support check and the K/V type mapping used when creating the FA pipelines)

**Interfaces:**
- Consumes: the shader branches from Tasks 1 and 2.
- Produces: `--cache-type-v planar3_0` usable with `-ngl 99`, producing output identical to the CPU path.

- [ ] **Step 1: Accept the type**

Where the Vulkan backend decides which K/V types the flash attention op supports, add `GGML_TYPE_PLANAR3_0` for the V side. Where the host maps a `ggml_type` to the `FaTypeV` spec constant value, make sure the constant carries the ggml type id (43) as it does for the other quantized types.

- [ ] **Step 2: Build and check that FA is selected**

```bash
cd /home/herlanggays/RISET/llama.cpp
cmake --build build-vk -j8 2>&1 | tail -3
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
./build-vk/bin/llama-completion -m "$M" -p "hi" -n 1 -c 64 -ngl 99 --cache-type-k q8_0 --cache-type-v planar3_0 -v 2>&1 | grep -iE "flash|not supported|falling back|KV buffer size" | head -8
```

Expected: no "not supported" fallback, and the KV buffer size is about 2.7x smaller for the V half than f16. If the op falls back to the CPU, the support check is incomplete.

- [ ] **Step 3: Backend op check**

```bash
cd /home/herlanggays/RISET/llama.cpp
./build-vk/bin/test-backend-ops -b Vulkan0 -o FLASH_ATTN_EXT 2>&1 | tail -4
./build-vk/bin/test-backend-ops -b CPU -o FLASH_ATTN_EXT 2>&1 | tail -4
```

Expected: the existing cases still pass on both backends. The new type has no test case of its own; it is exercised by the runs in Task 4.

- [ ] **Step 4: CPU and Vulkan must agree**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/kvplanar3
./build-cpu/bin/llama-completion -m "$M" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 2048 --cache-type-k q8_0 --cache-type-v planar3_0 > "$D/agree-cpu.txt" 2>&1
./build-vk/bin/llama-completion  -m "$M" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 2048 -ngl 99 --cache-type-k q8_0 --cache-type-v planar3_0 > "$D/agree-vk.txt" 2>&1
diff <(grep -v "^0\." "$D/agree-cpu.txt") <(grep -v "^0\." "$D/agree-vk.txt") && echo AGREEMENT
```

Expected: identical text. A difference means the shader's dequant arithmetic differs from the host's, most likely the codebook literals, the packing order, or the norm handling. Compare the two dequantizers on a synthetic block before touching anything else.

- [ ] **Step 5: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add ggml/src/ggml-vulkan/ggml-vulkan.cpp
git commit -m "vulkan : accept planar3_0 V in flash attention"
```

---

### Task 4: The speed gate

**Files:**
- Create: `docs/research/11-kv-planar3-vulkan-results.md`
- Create (scratch): `/home/herlanggays/.jcode/scratch/kvplanar3/*.txt`

**Interfaces:**
- Consumes: a working Vulkan planar3 V cache from Task 3.
- Produces: the measured speed at depth and the decision on keeping the type.

- [ ] **Step 1: Decode speed at depth**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/kvplanar3
for d in 16384 32768; do
  ./build-vk/bin/llama-bench -m "$M" -r 2 -p 0 -n 64 -t 8 -ngl 99 -d $d --cache-type-k q8_0 --cache-type-v f16      > "$D/bench-f16-$d.txt" 2>&1
  ./build-vk/bin/llama-bench -m "$M" -r 2 -p 0 -n 64 -t 8 -ngl 99 -d $d --cache-type-k q8_0 --cache-type-v planar3_0 > "$D/bench-planar3-$d.txt" 2>&1
done
grep -H tg64 "$D"/bench-*.txt
```

Expected: tg64 improves at both depths, about 8% at 16k and 14% at 32k. Confirm llama-bench passes the cache-type flags through to the context; if it does not, use `llama-completion` with a long context prefix and read its eval time instead.

- [ ] **Step 2: Confirm quality is unchanged from plan 1**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/kvplanar3
./build-vk/bin/llama-perplexity -m "$M" -f "$D/corpus-512k.txt" -c 2048 -ngl 99 --cache-type-k q8_0 --cache-type-v planar3_0 > "$D/ppl-vk-planar3.txt" 2>&1
grep -E "Final estimate" "$D/ppl-vk-planar3.txt" "$D/ppl-planar3-2048.txt"
```

Expected: the same perplexity within noise as the CPU planar3 number from plan 1. A materially different number on Vulkan means the shader path is not equivalent, which is a bug, not a speed difference.

- [ ] **Step 3: Write the results and apply the gate**

Create `docs/research/11-kv-planar3-vulkan-results.md` with both bench tables, the speed percentages, the perplexity comparison, and the decision. If the 16k gain is below about 3%, record that the type is not worth keeping and stop, leaving the code in place only if you intend to revisit it; otherwise say plainly that it is kept and why.

- [ ] **Step 4: Update the fork verification log**

If the type is kept, add an entry to the verification log in `QWEN_ONLY.md`: the arch tests, `ctest -L main`, `test-backend-ops` counts on both backends, the CPU and Vulkan agreement check, the perplexity numbers, and the speed deltas.

- [ ] **Step 5: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add docs/research/11-kv-planar3-vulkan-results.md QWEN_ONLY.md
git commit -m "docs: record the planar3 V cache Vulkan results"
```

---

## Self-Review

**Spec coverage:** the spec's plan 2 is "add the type to the Vulkan flash attention path, speed gated". Task 1 sizes the tiles, Task 2 dequantizes, Task 3 makes the backend accept it and proves agreement with the CPU, Task 4 measures the gate. The spec's note that the inverse rotation is a graph matmul is honoured by not adding any Vulkan work for it.

**Placeholder scan:** no TBD. Task 2 step 1 requires a value dump from the host rather than inventing numbers, and says where they go and how they are labelled; step 3 gives the branch body and names the surrounding variables to adapt.

**Type consistency:** `block_planar3_0_packed16`, `PLANAR3_0_CB`, `GGML_TYPE_PLANAR3_0` (43), 256 coordinates, 96 B of codes and the 8-coordinates-into-3-bytes little-endian packing match plan 1's host implementation exactly. That match is what Task 3 step 4 verifies.

**Known gap, stated rather than hidden:** the SSBO struct stride for a 98 B block is asserted rather than measured, because the driver decides it. Task 1 step 1 names the check and the pad-if-needed fallback, and Task 3 step 4 would catch a mismatch as a correctness difference between backends.
