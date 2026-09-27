# KV planar3 V cache (plan 1, CPU) Implementation Plan

**Outcome: done and verified.** `GGML_TYPE_PLANAR3_0` is in the tree as type 43, with the CPU
quantizer verified byte for byte against the Rust reference. Results in
`docs/research/10-kv-planar3-results.md`. The checkboxes below were never ticked as the work landed.
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `planar3_0` KV type (per-row norm, fixed Givens rotation, Lloyd-Max codebook, grouped packing) and use it for V with K at `q8_0`, then gate it on measured quality at 2048 and 16384 depth.

**Architecture:** A new quantized ggml type, mirroring the existing `Q2_0`/`Q4_0` plumbing (block struct, type traits, `ggml_quantize_chunk` case), ported exactly from the reference implementation in `~/ternary-bonsai-inference/src/kvquant.rs`. The insert path quantizes V rows through the existing `ggml_quantize_chunk` call in the KV cache; the attention output is inverse-rotated once per query in the graph, which is where the reference implementation also does it.

**Tech Stack:** C11 ggml (`ggml-common.h`, `ggml-quants.c`, `ggml.c`), C++17 (`llama-kv-cache.cpp`, the attention graph builder), CMake, the fork's test binaries.

**Spec:** `docs/superpowers/specs/2026-09-26-kv-planar3-design.md`

## Global Constraints

- Commits need explicit human approval for each action (`AGENTS.md`). Every commit step is a checkpoint: ask, and if approval is not given, leave the tree uncommitted and report. Never push.
- Do not add files under `tests/`. Reuse the existing binaries.
- ASCII only, no em dashes. Comments concise, simple wording, no restating the code.
- The type is KV only. It must never be selectable for model weights.
- Port the tables exactly: Givens angles from `splitmix64` with seed 42, `u = (r >> 11) * 2^-53`, `t = u * 2*pi` in double then cast to f32; codebook from the Lloyd-Max solve for `sigma = 1/sqrt(256)`, with the Abramowitz-Stegun 7.1.26 `erf`.
- Blocks are 256 coordinates. `head_dim` is 256 for both K and V in this fork's models, so one block is one row; other shapes must fail loudly, not miscompute.
- Builds: `build-cpu` and `build-vk`, Release, `GGML_NATIVE=ON`.
- Model: `/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`
- Scratch dir: `/home/herlanggays/.jcode/scratch/kvplanar3/`
- Gate (from the spec): perplexity relative increase at or below 2% against the f16-V baseline at both depths; above 3% the premise fails.

---

### Task 1: The type and its reference quantizer

**Files:**
- Modify: `ggml/include/ggml.h` (type enum, around line 429)
- Modify: `ggml/src/ggml-common.h` (block struct, near `block_q2_0` around line 189)
- Modify: `ggml/src/ggml-quants.h` (declarations)
- Modify: `ggml/src/ggml-quants.c` (implementation)
- Modify: `ggml/src/ggml.c` (type traits entry near line 668, and the `ggml_quantize_chunk` switch near line 8062)

**Interfaces:**
- Consumes: `ggml_half`, `GGML_FP32_TO_FP16`/`GGML_FP16_TO_FP32`.
- Produces: `GGML_TYPE_PLANAR3_0 = 43`, `block_planar3_0` (98 bytes, 256 coordinates), `quantize_row_planar3_0_ref(const float *, block_planar3_0 *, int64_t)`, `dequantize_row_planar3_0(const block_planar3_0 *, float *, int64_t)`.

- [ ] **Step 1: Add the enum value**

In `ggml/include/ggml.h`, next to the other type ids:

```c
        GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
        GGML_TYPE_PLANAR3_0 = 43, // KV only: 256 coords, f16 norm, 3 bit codes
```

and bump `GGML_TYPE_COUNT` from 43 to 44.

- [ ] **Step 2: Add the block struct**

In `ggml/src/ggml-common.h`, after `block_q2_0`:

```c
// 3-bit planar codes for the KV cache, one f16 norm per 256 coordinates.
// Codes are 8 coordinates packed little-endian into 3 bytes, 32 groups per block.
#define QK_PLANAR3_0 256
#define PLANAR3_0_LEVELS 8
#define PLANAR3_0_GROUPS (QK_PLANAR3_0/8)
typedef struct {
    ggml_half norm;
    uint8_t qs[3*PLANAR3_0_GROUPS];
} block_planar3_0;
static_assert(sizeof(block_planar3_0) == sizeof(ggml_half) + 3*QK_PLANAR3_0/8, "wrong planar3_0 block size/padding");
```

- [ ] **Step 3: Declare the reference functions**

In `ggml/src/ggml-quants.h`:

```c
void quantize_row_planar3_0_ref(const float * GGML_RESTRICT x, block_planar3_0 * GGML_RESTRICT y, int64_t k);
void dequantize_row_planar3_0(const block_planar3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
```

- [ ] **Step 4: Implement the tables and the quantizer**

In `ggml/src/ggml-quants.c`:

```c
// ---------------------------------------------------------------------------
// planar3_0: rotation and Lloyd-Max codebook for the KV cache
// ported from ternary-bonsai-inference src/kvquant.rs
// ---------------------------------------------------------------------------

static uint64_t planar3_0_splitmix64(uint64_t * state) {
    *state += 0x9E3779B97F4A7C15ull;
    uint64_t z = *state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static double planar3_0_erf(double x) {
    // Abramowitz & Stegun 7.1.26
    const double sign = x < 0.0 ? -1.0 : 1.0;
    x = fabs(x);
    const double t = 1.0 / (1.0 + 0.3275911 * x);
    const double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return sign * y;
}

static double planar3_0_norm_pdf(double x, double sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma)) / (sigma * sqrt(2.0 * M_PI));
}

static double planar3_0_norm_cdf(double x, double sigma) {
    return 0.5 * (1.0 + planar3_0_erf(x / (sigma * M_SQRT2)));
}

static void planar3_0_solve_lloyd_max(double sigma, int levels, float * out) {
    const double lo = -3.5 * sigma;
    const double hi =  3.5 * sigma;
    const double outer = 12.0 * sigma;
    double c[PLANAR3_0_LEVELS];
    for (int i = 0; i < levels; ++i) {
        c[i] = lo + (hi - lo) * (i + 0.5) / levels;
    }
    for (int it = 0; it < 300; ++it) {
        double next[PLANAR3_0_LEVELS];
        double shift = 0.0;
        for (int i = 0; i < levels; ++i) {
            const double a = i == 0 ? -outer : 0.5 * (c[i - 1] + c[i]);
            const double b = i == levels - 1 ? outer : 0.5 * (c[i] + c[i + 1]);
            const double mass = planar3_0_norm_cdf(b, sigma) - planar3_0_norm_cdf(a, sigma);
            if (mass > 1e-15) {
                // integral of x*pdf over [a,b] is sigma^2 * (pdf(a) - pdf(b))
                next[i] = sigma * sigma * (planar3_0_norm_pdf(a, sigma) - planar3_0_norm_pdf(b, sigma)) / mass;
            } else {
                next[i] = c[i];
            }
            const double d = fabs(next[i] - c[i]);
            if (d > shift) {
                shift = d;
            }
        }
        memcpy(c, next, sizeof(c));
        if (shift < 1e-12) {
            break;
        }
    }
    for (int i = 0; i < levels; ++i) {
        out[i] = (float) c[i];
    }
}

static const float * planar3_0_givens(void) {
    static float table[QK_PLANAR3_0]; // cos, sin per pair
    static bool init = false;
    if (!init) {
        uint64_t state = 42;
        for (int i = 0; i < QK_PLANAR3_0/2; ++i) {
            const uint64_t r = planar3_0_splitmix64(&state);
            const double u = (double) (r >> 11) / (double) (1ull << 53);
            const double t = u * 2.0 * M_PI;
            table[2*i + 0] = (float) cos(t);
            table[2*i + 1] = (float) sin(t);
        }
        init = true;
    }
    return table;
}

static const float * planar3_0_codebook(void) {
    static float cb[PLANAR3_0_LEVELS];
    static bool init = false;
    if (!init) {
        planar3_0_solve_lloyd_max(1.0 / sqrt((double) QK_PLANAR3_0), PLANAR3_0_LEVELS, cb);
        init = true;
    }
    return cb;
}

void quantize_row_planar3_0_ref(const float * GGML_RESTRICT x, block_planar3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PLANAR3_0 == 0);

    const float * cs = planar3_0_givens();
    const float * cb = planar3_0_codebook();

    for (int64_t ib = 0; ib < k/QK_PLANAR3_0; ++ib) {
        const float * xb = x + ib*QK_PLANAR3_0;
        block_planar3_0 * yb = y + ib;

        float norm = 0.0f;
        for (int j = 0; j < QK_PLANAR3_0; ++j) {
            norm += xb[j]*xb[j];
        }
        norm = sqrtf(norm);
        const float inv = norm > 1e-12f ? 1.0f/norm : 0.0f;
        yb->norm = GGML_FP32_TO_FP16(norm);

        uint8_t idx[QK_PLANAR3_0];
        for (int i = 0; i < QK_PLANAR3_0/2; ++i) {
            const float c = cs[2*i + 0];
            const float s = cs[2*i + 1];
            const float a = xb[2*i + 0]*inv;
            const float b = xb[2*i + 1]*inv;
            const float r0 = c*a - s*b;
            const float r1 = s*a + c*b;

            for (int t = 0; t < 2; ++t) {
                const float v = t == 0 ? r0 : r1;
                int best = 0;
                float bd = fabsf(v - cb[0]);
                for (int l = 1; l < PLANAR3_0_LEVELS; ++l) {
                    const float d = fabsf(v - cb[l]);
                    if (d < bd) {
                        bd = d;
                        best = l;
                    }
                }
                idx[2*i + t] = (uint8_t) best;
            }
        }

        for (int g = 0; g < PLANAR3_0_GROUPS; ++g) {
            const uint8_t * ig = idx + 8*g;
            const uint32_t w = (uint32_t) ig[0]
                             | ((uint32_t) ig[1] << 3)
                             | ((uint32_t) ig[2] << 6)
                             | ((uint32_t) ig[3] << 9)
                             | ((uint32_t) ig[4] << 12)
                             | ((uint32_t) ig[5] << 15)
                             | ((uint32_t) ig[6] << 18)
                             | ((uint32_t) ig[7] << 21);
            yb->qs[3*g + 0] = (uint8_t) ( w        & 0xFF);
            yb->qs[3*g + 1] = (uint8_t) ((w >>  8) & 0xFF);
            yb->qs[3*g + 2] = (uint8_t) ((w >> 16) & 0xFF);
        }
    }
}

void dequantize_row_planar3_0(const block_planar3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PLANAR3_0 == 0);

    const float * cs = planar3_0_givens();
    const float * cb = planar3_0_codebook();

    for (int64_t ib = 0; ib < k/QK_PLANAR3_0; ++ib) {
        const block_planar3_0 * xb = x + ib;
        float * yb = y + ib*QK_PLANAR3_0;

        const float norm = GGML_FP16_TO_FP32(xb->norm);

        float rot[QK_PLANAR3_0];
        for (int g = 0; g < PLANAR3_0_GROUPS; ++g) {
            const uint32_t w = (uint32_t) xb->qs[3*g + 0]
                             | ((uint32_t) xb->qs[3*g + 1] << 8)
                             | ((uint32_t) xb->qs[3*g + 2] << 16);
            for (int t = 0; t < 8; ++t) {
                const uint8_t i8 = (uint8_t) ((w >> (3*t)) & 0x7);
                rot[8*g + t] = cb[i8];
            }
        }

        for (int i = 0; i < QK_PLANAR3_0/2; ++i) {
            const float c = cs[2*i + 0];
            const float s = cs[2*i + 1];
            const float a = rot[2*i + 0];
            const float b = rot[2*i + 1];
            yb[2*i + 0] = (c*a + s*b)*norm;
            yb[2*i + 1] = (-s*a + c*b)*norm;
        }
    }
}
```

Note on the inverse rotation here: the last loop applies the transposed Givens rotation, matching `rotate_inv` in the reference. If the graph applies the inverse rotation instead (Task 3), then `dequantize_row_planar3_0` must return rotated-space values and this loop is removed. Task 3 decides which; keep it here for the round-trip test and remove it in Task 3 if the graph takes over.

- [ ] **Step 5: Register the type**

In `ggml/src/ggml.c`, in the `type_traits` table after the `Q2_0` entry:

```c
    [GGML_TYPE_PLANAR3_0] = {
        .type_name                = "planar3_0",
        .blck_size                = QK_PLANAR3_0,
        .type_size                = sizeof(block_planar3_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_planar3_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_planar3_0_ref,
    },
```

and in `ggml_quantize_chunk`, in the type switch:

```c
        case GGML_TYPE_PLANAR3_0: result = quantize_planar3_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
```

with, next to the other `quantize_*` helpers in `ggml-quants.c`:

```c
size_t quantize_planar3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    GGML_ASSERT(n_per_row % QK_PLANAR3_0 == 0);
    quantize_row_planar3_0_ref(src, (block_planar3_0 *) dst, nrows*n_per_row);
    return nrows * ggml_row_size(GGML_TYPE_PLANAR3_0, n_per_row);
}
```

and its prototype in `ggml-quants.h`.

- [ ] **Step 6: Build, and round-trip the reference**

```bash
cd /home/herlanggays/RISET/llama.cpp
cmake --build build-cpu -j8 2>&1 | tail -3
```

Expected: clean. Then confirm the type is known:

```bash
./build-cpu/bin/test-backend-ops support -o CPY 2>&1 | head -3
./build-cpu/bin/test-llama-archs -a qwen35 -s 1
```

Expected: the arch test passes (nothing uses the new type yet), and the type appears in `ggml_type_name` output if you print it.

- [ ] **Step 7: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add ggml/include/ggml.h ggml/src/ggml-common.h ggml/src/ggml-quants.h ggml/src/ggml-quants.c ggml/src/ggml.c
git commit -m "ggml : add planar3_0, a rotated Lloyd-Max type for the KV cache"
```

---

### Task 2: KV plumbing and the rotation exclusion

**Files:**
- Modify: `common/arg.cpp` (the `kv_cache_types` list at line 304)
- Modify: `src/llama-kv-cache.cpp` (the attn_rot flags at line 320)

**Interfaces:**
- Consumes: `GGML_TYPE_PLANAR3_0` from Task 1.
- Produces: `--cache-type-v planar3_0` accepted; `attn_rot_v` false for that type.

- [ ] **Step 1: Accept the type on the command line**

In `common/arg.cpp`:

```cpp
const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_PLANAR3_0,
};
```

- [ ] **Step 2: Keep the Hadamard path off this type**

In `src/llama-kv-cache.cpp`, where `attn_rot_k` and `attn_rot_v` are computed, add the exclusion:

```cpp
        attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            type_k != GGML_TYPE_PLANAR3_0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k() % 64 == 0;

        attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            type_v != GGML_TYPE_PLANAR3_0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v() % 64 == 0;
```

- [ ] **Step 3: Refuse the type for weights**

In `llama-quantize`'s option parsing (or where `--output-tensor-type` is validated), reject `planar3_0` with a clear message, for example in `common/arg.cpp` next to the type parsing used by `llama-quantize`:

```cpp
    if (type == GGML_TYPE_PLANAR3_0) {
        throw std::invalid_argument("planar3_0 is a KV cache type, not a weight type");
    }
```

- [ ] **Step 4: Build and check the flag path**

```bash
cd /home/herlanggays/RISET/llama.cpp
cmake --build build-cpu -j8 2>&1 | tail -3
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
./build-cpu/bin/llama-completion -m "$M" -p "hi" -n 1 -c 64 --cache-type-v planar3_0 2>&1 | grep -iE "cache type|attn_rot|error|KV buffer" | head -6
```

Expected: the cache type is accepted, `attn_rot_v = 0` is logged, and the KV buffer size shrinks for the V half. Loading still works because nothing reads V through the new type's dequantizer yet in a way that matters for correctness of this step; if attention output is obviously broken here, that is expected until Task 3 lands, so only check the flags and sizes at this step.

- [ ] **Step 5: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add common/arg.cpp src/llama-kv-cache.cpp
git commit -m "kv : accept planar3_0 for V and keep the Hadamard path off it"
```

---

### Task 3: Inverse rotation of the attention output

**Files:**
- Modify: `src/llama-kv-cache.cpp` (expose the inverse rotation matrix as a graph input, next to `build_input_v_rot`)
- Modify: `src/llama-graph.cpp` (apply it to the attention output when the V type is planar3)
- Modify: `ggml/src/ggml-quants.c` (drop the inverse rotation from `dequantize_row_planar3_0` if the graph owns it)

**Interfaces:**
- Consumes: the Givens table from Task 1 (the same table, so the two paths cannot drift).
- Produces: attention output in unrotated space for a planar3 V cache.

- [ ] **Step 1: Build the inverse rotation matrix on the host**

In the KV cache, next to `attn_rot_hadamard`, build the dense inverse Givens matrix for the head size: a `[head_dim, head_dim]` f32 matrix that is block diagonal with the 2x2 transpose of each Givens pair, and expose it through a `build_input_v_rot_inv`/`set_input_v_rot_inv` pair mirroring `build_input_v_rot` and `set_input_v_rot`.

```cpp
// inverse Givens, same table the quantizer uses: block diagonal 2x2 [c s; -s c]
```

The matrix is `head_dim x head_dim` f32, 256 KB for head_dim 256, built once per context.

- [ ] **Step 2: Apply it after attention**

In the attention builder, after the attention op produces the output and before the output projection, when `type_v == GGML_TYPE_PLANAR3_0`, multiply the per-head `head_dim` vectors by the inverse matrix:

```cpp
    // planar3 V is stored rotated, so the weighted sum comes back rotated
    if (type_v == GGML_TYPE_PLANAR3_0) {
        cur = ggml_mul_mat(ctx0, v_rot_inv, cur); // [head_dim, head_dim] x [head_dim, n_head*n_tokens]
    }
```

placed so that both the flash attention and the non-flash paths pass through it. Confirm the exact tensor shape at that point while implementing; the reference implementation does this once per token on the weighted sum, not per cached position.

- [ ] **Step 3: Decide where the inverse lives and remove the duplicate**

Either the graph applies it (as above) and `dequantize_row_planar3_0` returns rotated-space values, or the dequantizer applies it and the graph does nothing. The graph is preferred because the dequantizer runs per cached position, which is exactly the cost the rotation is supposed to avoid. Whichever is chosen, exactly one of the two applies it, and the comment says which.

- [ ] **Step 4: Check correctness on a short run**

```bash
cd /home/herlanggays/RISET/llama.cpp
cmake --build build-cpu -j8 2>&1 | tail -3
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/kvplanar3
mkdir -p "$D"
./build-cpu/bin/llama-completion -m "$M" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 2048 --cache-type-k q8_0 --cache-type-v f16 > "$D/base-2k.txt" 2>&1
./build-cpu/bin/llama-completion -m "$M" -p "The capital of France is" -n 32 -s 1 --temp 0 -c 2048 --cache-type-k q8_0 --cache-type-v planar3_0 > "$D/cand-2k.txt" 2>&1
tail -3 "$D/base-2k.txt"; echo ---; tail -3 "$D/cand-2k.txt"
diff <(grep -v "^0\." "$D/base-2k.txt") <(grep -v "^0\." "$D/cand-2k.txt") > /dev/null && echo IDENTICAL || echo DIFFERS
```

Expected: both runs produce coherent text. `DIFFERS` is acceptable at 3 bits (quantization perturbs logits), but the candidate text must be coherent, not garbage. Garbage means the rotation direction or the packing order is wrong; debug the round trip first (quantize then dequantize a synthetic row and compare against the input within the expected 3-bit error).

- [ ] **Step 5: Round-trip check before anything else**

```bash
cd /home/herlanggays/RISET/llama.cpp
./build-cpu/bin/test-backend-ops test -o CPY -b CPU 2>&1 | tail -3
./build-cpu/bin/test-llama-archs -a qwen35 -s 1
./build-cpu/bin/test-llama-archs -a qwen35moe -s 1
```

Expected: all pass. If the arch tests pass with an f16 cache but the planar3 run is garbage, the bug is in the new type, not in the graph.

- [ ] **Step 6: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add src/llama-kv-cache.cpp src/llama-kv-cache.h src/llama-graph.cpp ggml/src/ggml-quants.c
git commit -m "graph : inverse-rotate the attention output for a planar3 V cache"
```

---

### Task 4: The quality gate

**Files:**
- Create: `docs/research/10-kv-planar3-results.md`
- Create (scratch): `/home/herlanggays/.jcode/scratch/kvplanar3/*.txt`

**Interfaces:**
- Consumes: a working planar3 V cache from Task 3.
- Produces: the measured quality numbers that decide whether the Vulkan plan proceeds.

- [ ] **Step 1: Build the fixed corpus**

```bash
cd /home/herlanggays/RISET/llama.cpp
D=/home/herlanggays/.jcode/scratch/kvplanar3
mkdir -p "$D"
cat src/*.cpp src/models/*.cpp common/*.cpp ggml/src/ggml-cpu/ops.cpp > "$D/corpus.txt"
head -c 524288 "$D/corpus.txt" > "$D/corpus-512k.txt"
wc -c "$D/corpus-512k.txt"
```

Expected: 524288 bytes.

- [ ] **Step 2: Perplexity, both configuration, both depths**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/kvplanar3
for ctx in 2048 16384; do
  ./build-cpu/bin/llama-perplexity -m "$M" -f "$D/corpus-512k.txt" -c $ctx --cache-type-k q8_0 --cache-type-v f16      > "$D/ppl-f16-$ctx.txt" 2>&1
  ./build-cpu/bin/llama-perplexity -m "$M" -f "$D/corpus-512k.txt" -c $ctx --cache-type-k q8_0 --cache-type-v planar3_0 > "$D/ppl-planar3-$ctx.txt" 2>&1
done
grep -E "Final estimate|perplexity" "$D"/ppl-*.txt | tail -8
```

Expected: four perplexity numbers. Compute the relative increase for each depth. Gate: at or below 2% at both depths passes; between 2% and 3% is a judgement call; above 3% fails and the Vulkan plan does not proceed.

- [ ] **Step 3: Greedy continuation**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/kvplanar3
./build-cpu/bin/llama-completion -m "$M" -p "The capital of France is" -n 64 -s 1 --temp 0 -c 2048 --cache-type-k q8_0 --cache-type-v f16      > "$D/greedy-f16.txt" 2>&1
./build-cpu/bin/llama-completion -m "$M" -p "The capital of France is" -n 64 -s 1 --temp 0 -c 2048 --cache-type-k q8_0 --cache-type-v planar3_0 > "$D/greedy-planar3.txt" 2>&1
diff <(grep -v "^0\." "$D/greedy-f16.txt") <(grep -v "^0\." "$D/greedy-planar3.txt") | head -10
```

Expected: identical for 64 tokens. A divergence is recorded with its position; a divergence inside the first 16 tokens fails the gate.

- [ ] **Step 4: Memory, state round trip, rotation stacking**

```bash
cd /home/herlanggays/RISET/llama.cpp
M="/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
D=/home/herlanggays/.jcode/scratch/kvplanar3
grep -i "KV buffer size" "$D/greedy-f16.txt" "$D/greedy-planar3.txt"
LLAMA_ATTN_ROT_DISABLE=1 ./build-cpu/bin/llama-completion -m "$M" -p "The capital of France is" -n 8 -s 1 --temp 0 -c 2048 --cache-type-k q8_0 --cache-type-v planar3_0 > "$D/rot-disabled.txt" 2>&1
diff <(grep -v "^0\." "$D/greedy-planar3.txt" | head -3) <(grep -v "^0\." "$D/rot-disabled.txt" | head -3) > /dev/null && echo ROT_NOT_STACKED || echo ROT_STACKED_CHECK
```

Expected: the V half of the KV buffer is about 2.7x smaller than f16; with the Hadamard path disabled the output is unchanged, proving the two rotations are not stacked. The state round trip is checked with the same command pair as the spec requires (`llama_state_seq` save and restore, output identical).

- [ ] **Step 5: Write the results and apply the gate**

Create `docs/research/10-kv-planar3-results.md` with the four perplexity numbers and their relative increases, the greedy comparison, the KV memory figures, the rotation-stacking result, and a plain statement of whether the 2% gate passed. If it failed, record that the Vulkan plan is dropped.

- [ ] **Step 6: Commit (checkpoint)**

Ask for approval, then:

```bash
cd /home/herlanggays/RISET/llama.cpp
git add docs/research/10-kv-planar3-results.md
git commit -m "docs: record the planar3 KV cache quality measurements"
```

---

## Self-Review

**Spec coverage:** the spec's type and block format map to Task 1; its rotation interaction and KV plumbing map to Task 2; its note that the inverse rotation runs once per query maps to Task 3; its validation list maps to Task 4 (perplexity, greedy, memory, state round trip, rotation stacking). The spec's speed items are deliberately absent: the spec makes plan 1 quality-gated and defers speed to the Vulkan plan.

**Placeholder scan:** no TBD. Task 1 carries complete code for the block, tables, quantizer and dequantizer. Task 3 step 1 describes the matrix construction rather than pasting it, because the KV cache's existing rotation matrix plumbing defines the shape it must match; the step says what to mirror and how large it is.

**Type consistency:** `GGML_TYPE_PLANAR3_0`, `block_planar3_0`, `QK_PLANAR3_0` (256), `PLANAR3_0_LEVELS` (8), `quantize_row_planar3_0_ref`, `dequantize_row_planar3_0` and `quantize_planar3_0` are used with the same names and signatures in every task. The 8-coordinates-into-3-bytes packing is written once in the quantizer and read back in the same order in the dequantizer.

**Known gap, stated rather than hidden:** Task 3 step 3 leaves one decision to the implementer, which of the graph or the dequantizer applies the inverse rotation. Both alternatives are named, the constraint (exactly one, commented) is explicit, and the verification in Task 4 would fail if the rotation were applied twice or not at all. The reference implementation applies it to the weighted sum, which is why the graph is preferred.
