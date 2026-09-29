# Prefill nondeterminism: it is the Vulkan driver, not this fork

Date: 2026-09-28
Tree: `qwen-only-backends`, working tree on top of `397b643f4`
Machine: Ryzen 7 6800H (8 cores, 16 threads), 27.1 GiB, AC connected
Model: `Qwen3.5-0.8B-Q4_K_M.gguf`, CPU only
Harness: `scripts/research/ssdcache/memcheck_self.c`, `scripts/research/ssdcache/perturb_probe.c`,
         and the `ubatch_determinism` harness that doc 28 describes
Open question from: `docs/research/28-ubatch-determinism-results.md`, which this answers
Related: `docs/research/26-prefill-vulkan-profile.md` (the GPU side of the same box, and the
         advice to run a memory test)

**Verdict: the nondeterminism is the Vulkan driver being in the process, not this fork. With the
driver hidden by pointing `VK_ICD_FILENAMES` at nothing, the same binary, at one thread and at
four, returns the identical state `f863c3cce8b2ef52` on every run and for every ubatch and chunk
split, including two fresh contexts back to back, and a tree built with `GGML_VULKAN=OFF` returns
that same value. With the driver visible, four runs of one configuration returned four different
states. Along the way an uninitialized heap read, an uninitialized stack read, the virtual address
layout and the FPU mode were each eliminated by a separate measurement, and the MemorySanitizer
route was shown to be unusable here by a fifteen line repro rather than by argument. So none of the
three routes this document originally ranked, a reboot, valgrind, or an instrumented libc++, is
needed to explain this. What the driver does to the arithmetic is still open, and that is a new
question with a new place to look.**

## What was already known

Doc 28 established that two runs of the same prompt differ, that the control (serializing one
unchanged state twice) is clean, and that the difference is in the computation that fills the
state. It ruled out thread count and OpenMP scheduling, and it named three candidates it could not
separate:

1. an uninitialized read in a kernel, depending on the contents of the context's memory
2. an address or alignment dependent code path
3. a genuine arithmetic ordering defect in this fork

It also left box health open, and doc 26 had asked for a memory test because non-deterministic
arithmetic is a signature of failing RAM.

## Test 1: is the box's arithmetic and memory reliable at all

`scripts/research/ssdcache/memcheck_self.c`. Two workloads, each repeated and compared bit for
bit:

- fill 512 MiB with a pattern derived from the byte index, then read it back and fold it into an
  FNV-1a 64 checksum, four rounds in one process
- a serial `fmaf` chain over a 4096 element array, 2000 iterations, folded the same way

```
  round 0  memory 9aee3aa52d350383  fma eb4d2bd473ccf799  0.36 s
  round 1  memory 9aee3aa52d350383  fma eb4d2bd473ccf799  0.14 s
  round 2  memory 9aee3aa52d350383  fma eb4d2bd473ccf799  0.15 s
  round 3  memory 9aee3aa52d350383  fma eb4d2bd473ccf799  0.14 s

  memory checksum stable: yes
  fma checksum stable:    yes
```

Across three separate processes, the same two checksums each time, `2066940cc8440383` for the 64
MiB case and `eb4d2bd473ccf799` for the fma chain.

So for a 512 MiB working set, a byte pattern round trip, and a tight floating point reduction, this
box computes and stores the same answer every time and across processes.

What that does and does not say. It does not cover a 13 GiB working set with heavy interleaved
multi-threaded access, it does not cover the GPU, and it does not cover the box under the thermal
and power load a large model puts on it. It does say that a simple ALU or DRAM fault is not the
first thing to suspect, and it makes candidate 3 less likely for simple reductions.

## Test 2: MemorySanitizer, which is the tool for candidate 1

Uninitialized reads are what MSan is for, and this box has no valgrind, so a build was configured
for it:

```sh
cmake -B build-msan -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF \
  -DCMAKE_C_FLAGS="-fsanitize=memory -fno-omit-frame-pointer -g -O1" \
  -DCMAKE_CXX_FLAGS="-fsanitize=memory -fno-omit-frame-pointer -g -O1" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=memory" -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=memory" \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
cmake --build build-msan -j7 --target llama-cli
```

That builds, in about two minutes, and the binary reports this before `main`:

```
==3003445==WARNING: MemorySanitizer: use-of-uninitialized-value
    #0 std::_Rb_tree<llm_kv, ...>::_M_get_insert_unique_pos(llm_kv const&) bits/stl_tree.h:2808
    ...
    #4 std::map<llm_kv, char const*, ...>::map(std::initializer_list<...>) bits/stl_map.h:248
    #5 __cxx_global_var_init.5 src/llama-arch.cpp:16:60
SUMMARY: MemorySanitizer: use-of-uninitialized-value src/llama-arch.cpp:16:60 in __cxx_global_var_init.5
Exiting
```

The process then exits, before the model is loaded, so the run covers no engine code at all.
`MSAN_OPTIONS=halt_on_error=0` does not get past it: the report is in a global initializer, and the
process dies there either way. The options are not the problem, the timing is.

### The repro that shows this is not this tree's bug

A fifteen line program containing nothing but the same construct:

```cpp
enum kv { KV_A = 0, KV_B = 1, KV_C = 2 };
static const std::map<kv, const char *> names = { { KV_A, "a" }, { KV_B, "b" }, { KV_C, "c" } };
int main() { printf("size %zu\n", names.size()); }
```

compiled with the same flags and the same compiler:

```
SUMMARY: MemorySanitizer: use-of-uninitialized-value .../msan_artifact.c:16:54 in main
```

So a global `std::map` built from an initializer list reports under MSan on this system whatever
it is inside. That is the documented shape of the problem: MSan needs *every* dependency
instrumented, libstdc++ here is not, and the shadow memory for values it copies is left poisoned.
This is a false positive class, not a finding.

The consequence is that MSan on this box needs an MSan-instrumented C++ standard library, which
means building libc++ from source, and that is a project rather than a step. A clean MSan run with
an instrumented libc++ would be strong evidence for or against candidate 1, because uninstrumented
code can only hide reads, not invent them, so a hit would be real.

## Test 3: is it an uninitialized heap read, without needing MSan

There is a way to test candidate 1 that needs no sanitizer and no new build. glibc's
`MALLOC_PERTURB_` fills every fresh allocation with a known byte and every freed block with its
complement, so setting it makes the contents of uninitialized heap memory deterministic and
different per value. If the engine reads uninitialized heap, then at a fixed value two runs should
agree with each other, and different values should give different answers.

First, whether the fill reaches the sizes ggml uses. `scripts/research/ssdcache/perturb_probe.c`,
which allocates and prints the first bytes:

```
perturb=0   malloc 1048576:  first bytes ab 00 00 00 00 00 00 00
perturb=85  malloc 1048576:  first bytes ab aa aa aa aa aa aa aa
perturb=85  malloc 67108864: first bytes ab aa aa aa aa aa aa aa
```

`MALLOC_PERTURB_=85` gives `0xAA` from the second byte on at 1 MiB and at 64 MiB, which are the
sizes of the tensors and buffers involved, so the test is meaningful.

Then the harness, config A, 1024 token prompt, three processes per value:

| `MALLOC_PERTURB_` | state hashes | logits hashes | control diff |
| --- | --- | --- | --- |
| 0, disabled | `9169ae90...`, `042496da...`, `97e40662...` | `c047982e...`, `3e797aab...`, `0b969473...` | 0 in all three |
| 85, heap filled with `0xAA` | `42d1c010...`, `15262006...`, `910e750e...` | `a5a708b4...`, `7c6bfe2c...`, `78193794...` | 0 in all three |

Three processes, three different states and three different logits, at a value that should have
made the heap deterministic. **So it is not an uninitialized heap read.** That is the first of the
three candidates to be eliminated by measurement rather than left open.

What it does not cover, and neither would the MSan run have covered it:

- **The stack.** `MALLOC_PERTURB_` does not touch it. An uninitialized stack read is not excluded.
- **Anything not allocated through `malloc`.** The model is mmap'd from the file and is fully
  initialised, but a buffer obtained by `mmap` rather than `malloc` is not perturbed either.
- `MALLOC_PERTURB_` is glibc specific. It is in use here, so this run is valid on this box only.

## Test 4: is it the address layout

Every run above already varies its address layout, because ASLR is on. If an address or alignment
dependent code path were the cause, then pinning the layout should make runs agree. `setarch -R`
turns ASLR off for the process and needs no root.

```
--- ASLR disabled (setarch -R), config A, 1024 tokens ---
  run1 hash=a95f8fbe... logits=d457966d... first_tok=2180
  run2 hash=facb23ee... logits=64acb955... first_tok=2180
  run3 hash=68998f46... logits=85f32fa0... first_tok=2180
--- control: ASLR on ---
  run1 hash=0a5257c9... logits=38b12398... first_tok=2180
  run2 hash=2ac73999... logits=421af89d... first_tok=2180
```

Three runs at one fixed layout, three different states, control diff zero throughout. **So it is not
the virtual address layout.** Together with test 3 that removes both of the cheap explanations, and
it is worth noting what the two have in common: neither of them constrains which *physical* pages
the process gets, nor the state of the box at the time.

## Test 5: is it an uninitialized stack read

The stack counterpart of test 3, because `MALLOC_PERTURB_` cannot reach the stack. The harness paints
it: `decode_chunk` recurses 128 frames of 16 KiB before every `llama_decode`, covering 2 MiB below
its caller, and `UBDET_PAINT=<byte>` turns it on.

Whether the paint reaches the computation decides whether the test means anything, and at one thread
it does. In `ggml-cpu.c` the non-OpenMP path runs `ggml_graph_compute_thread(&threadpool->workers[0])`
on the calling thread (line 3441), and the OpenMP branch does the same when `n_threads == 1`
(line 3429), so the whole graph is computed on the stack that was just painted. Three runs per
condition, at the default thread count and at one:

| n_threads | condition | states | control |
| --- | --- | --- | --- |
| default | none | `00f8e87f...`, `c9890063...`, `a416a77c...` | 0, 0, 0 |
| default | paint `0xAA` | `c2cf2baa...`, `8c9b48b9...`, `c9d2e025...` | 0, 0, 0 |
| default | paint `0x55` | `199bcc92...`, `99ce7f49...`, `913a9943...` | 0, 0, 0 |
| 1 | none | `51ca778c...`, `d2ddbd46...`, `3648cb45...` | 0, 0, 0 |
| 1 | paint `0xAA` | `21ebc053...`, `16d879f6...`, `4f0060e9...` | 0, 0, 0 |
| 1 | paint `0x55` | `92d6d919...`, `49b43568...`, `7c2cbf7e...` | 0, 0, 0 |

Eighteen runs, eighteen states, control zero throughout, and no relation between the painted byte and
the outcome. **So it is not an uninitialized stack read either**, and candidate 1 is gone in both of
its forms.

One detail in that table says where the difference lives. The logits hash `7c6bfe2c...` appears twice,
in two runs whose states differ, and `first_tok` is 2180 in most runs and 0 in some. So two runs can
agree bit for bit on the last position's logits and still differ elsewhere in the 32.8 MB state. The
divergence is not confined to the final token, and it is not always visible in the sampled token.

## Test 6: the driver, and one environment variable

There was a confound sitting in both documents, and the thread list showed it. Even at
`n_threads = 1` the harness is not single threaded:

```
threads during a 1-thread decode: 3
  ubatch_determin       (the main thread)
  ubatch_:disk$0        futex_do_wait
  ubatch_:disk$0        futex_do_wait
```

Two Mesa shader disk cache threads, and `/proc/<pid>/maps` holds 25 mappings of
`libvulkan`/`libLLVM`/`libdrm`. `GGML_VULKAN` is compiled into `build-vk`, and the backend registry
initialises the device at startup even though the harness asks for `n_gpu_layers = 0`. So "CPU only"
in the headers of doc 28 and this document means "no tensor on the GPU", not "no Vulkan driver in
the process". Removing the confound takes one variable and no rebuild: point the loader at no ICD,
and ggml's Vulkan backend initialises no device. Same binary, same model, same prompt, interleaved:

| build | driver | threads | states |
| --- | --- | --- | --- |
| `build-cpu`, `GGML_VULKAN=OFF` | not compiled in | 1 | `f863c3ce...` |
| `build-cpu`, all five configs | not compiled in | 1 | A, B, C, D and A2 all `f863c3ce...` |
| `build-vk` | hidden | 1 | `f863c3ce...` five times |
| `build-vk` | hidden | 4 | `f863c3ce...` twice |
| `build-vk` | visible | 1 | `51a15fa1...`, `7a045dfc...` |
| `build-vk` | visible | 4 | `2ec07a1b...`, `1f6b6484...` |

Eleven runs with the driver hidden, all `f863c3ce8b2ef52` and all with logits `8bddb7bdd7c1a58b`,
seven of them at one thread and four at four. Eight runs with it visible, eight different states and
eight different logits, none of them the clean value. **So the prefill nondeterminism is not in this
tree.** The same binary disagrees with itself only while the driver is loaded, and the driver's
absence does not merely remove noise from a wobbling computation: every visible run differs from the
clean value, so the driver changes the arithmetic rather than adding variance to it.

Is the culprit ggml's Vulkan backend or the driver behind it? The mappings answer that, because the
loader is present in both cases:

```
driver hidden    libvulkan.so.1.4.357                          5 mappings of vulkan, llvm, drm
driver visible   libvulkan.so.1.4.357, libvulkan_radeon.so,   25 mappings
                 libLLVM.so.22.1, libdrm_amdgpu.so.1.134.0
```

`GGML_VULKAN` is compiled into and loaded by `build-vk` either way, so the ggml side is constant.
What hiding the ICD removes is Mesa's RADV driver, its LLVM, and `libdrm`'s access to the device.

### The FPU mode is not the mechanism

MXCSR is per thread, and its flush to zero, denormals are zero and rounding bits change results, so it
is the first thing to check. `UBDET_CSR=1` prints it:

```
  mxcsr main entry             0x00001f80 ftz=0 daz=0
  mxcsr after backend_init     0x00001fbb ftz=0 daz=0
  mxcsr after prompt A         0x00001fbb ftz=0 daz=0
```

Identical in both conditions, at both thread counts, at every sample point. `backend_init` masks the
SSE exception bits (0x3b) in both, which cannot change a value. So flush to zero, denormals are zero
and the rounding mode are excluded by measurement rather than by assumption, and the mechanism is
somewhere else.

### What is left, as hypotheses

Doc 28 measured roughly 66 percent of the state's bytes differing between two runs, so real values
differ, not only unwritten padding. With the heap, the stack, the addresses and the FPU mode all
excluded, and distinct processes agreeing exactly whenever the driver is absent, the shapes that
remain are a race whose timing the driver's own threads perturb, or a memory path that only the
driver opens. Both are questions about the driver's initialisation rather than about this fork's
compute, so the next instrument to reach for is one that watches `llama_backend_init`.

Three of the ways the driver could matter are already ruled out, each by setting one variable and
running the visible build three times at one thread. All twelve runs gave twelve distinct states, so
none of the three changes anything:

| condition | states |
| --- | --- |
| baseline, driver visible | `09fca05f...`, `52e11963...`, `44517f1a...` |
| `MESA_SHADER_CACHE_DISABLE=true` | `2981abde...`, `9f0cb7dd...`, `5d71ce65...` |
| `LD_BIND_NOW=1` | `f1f543e1...`, `e7e26b8a...`, `858fb931...` |
| `RADV_DEBUG=nocache` | `90a42341...`, `8c1849fb...`, `c6d666e0...` |

What each would have meant: the two `ubatch_:disk$0` threads are Mesa's shader disk cache workers, so
disabling that cache would have pointed at them; eager binding would have pointed at a symbol
resolution whose timing follows the order `libLLVM` and the rest are loaded in; and
`RADV_DEBUG=nocache` would have pointed at the driver's own cache path. So the shader cache, its
worker threads and dynamic binding are out, and what remains is reached by no environment variable
found so far.

## The same thing at the server level, on the 35B

Doc 28 measured the engine through the state API. The same disagreement shows up through
`llama-server`, and it is worth recording because it is the form that a user meets. These runs have
the Vulkan driver loaded, as any run with layers on the GPU does, so read them with test 6 in hand:
they show what a user sees on a machine where the driver is present, which is all of them that use
the GPU.

`cache_hit_check.py` sends one prompt, gets the answer from a state restored from disk, and then
sends the identical prompt again in the same process with `cache_prompt: false`, which forces
`n_past` to 0 and reprocesses everything from scratch. Both answers come out of one process, one
context, one machine state, and the restore is byte exact by `store_roundtrip`. Measured on
Qwen3.6-35B-A3B on Vulkan, `--ngl 99`, 16 greedy tokens:

| run | restored | recomputed | same |
| --- | --- | --- | --- |
| 1 | 34 chars | 57 chars | no |
| 2 | 57 chars | 57 chars | no |
| 3 | 57 chars | 57 chars | yes |

And across two fresh servers with no cache at all, one prompt:

| run | process 1 | process 2 | same |
| --- | --- | --- | --- |
| 1 | 57 chars | 57 chars | yes |
| 2 | 57 chars | 57 chars | no |
| 3 | 57 chars | 16 chars | no |

The text level is a weaker detector than the byte level, and the 0.8B says by how much. At 200
generated tokens, 812 characters, a restored state and a full recomputation in one process produced
identical text, and two uncached processes did too, with the driver visible as well as with it
hidden. The sampled token is the argmax of a large distribution and it rarely moves when the last
bits of the logits move: in the harness nearly every pair of runs has disagreeing logits hashes while
`first_tok` stays 2180, and the run where it did not was the exception. So a text comparison is not
the instrument for this question in either direction, and the 35B rows above are six single draws of
it rather than a measurement.

So the disagreement is not a property of the cache. It appears between two runs that both have no
cache, and it appears between a restore and a recomputation inside one process. The consequence for
the cache is stated in `docs/research/30-ssd-prompt-cache-results.md`: a hit reproduces the
computation that wrote the entry, and on this build that is a weaker claim than "a hit equals a
recomputation" because nothing here equals a recomputation.

It also has a practical consequence worth knowing before enabling the cache: **the cache can change
the text a client sees relative to running without it.** Not because the cache returns something
wrong, but because it pins the answer to the state that was saved while a no-cache run draws a new
one. On a build whose prefill is reproducible the two would agree.

## What the candidates came to

| candidate | outcome |
| --- | --- |
| 1. an uninitialized read, heap | **eliminated, test 3.** `MALLOC_PERTURB_` reaches `posix_memalign` too, which is what ggml's buffers use, verified at 4 KiB, 1 MiB and 64 MiB |
| 1. an uninitialized read, stack | **eliminated, test 5.** At one thread the painted stack is the stack the graph runs on |
| 2. an address or alignment dependent path | **eliminated, tests 4 and 6.** Pinning the virtual layout does not help, and with the driver hidden, distinct processes with distinct layouts agree exactly |
| 3. an arithmetic ordering defect | **eliminated, test 6.** A Vulkan-free build of the same source is reproducible, and the same binary stops being reproducible the moment the driver is loaded |
| 4. the box | **not supported.** Test 1 found the arithmetic and bulk memory bit stable, and test 6 has eleven runs agree bit for bit |
| the FPU mode | **eliminated, test 6.** `0x1fbb` with ftz 0 and daz 0 in both conditions |
| 5. the Vulkan driver in the process | **the finding** |

## What to do with it

1. For a reproducible prefill on this fork, either build with `GGML_VULKAN=OFF` (`build-cpu`) or run
   the Vulkan build with `VK_ICD_FILENAMES=/nonexistent/icd.json`. Both give `f863c3ce8b2ef52`. That
   is only sensible on the CPU paths, which is what doc 28, doc 30 and this document measure: with
   the driver hidden there is no GPU at all.
2. None of the three routes this document ranked earlier is needed. Not the reboot, not valgrind, not
   an instrumented libc++. The reboot is still worth doing for the box's other symptoms, doc 26's
   drift and the seven hard resets with nothing in the kernel log, but it is no longer the
   explanation for this.
3. The mechanism inside the driver is a new question, and a well posed one now that the fork, the
   memory and the FPU mode are all out of it. The next instrument to reach for is one that watches
   the driver's initialisation.

## What this does not change

Nothing here affects the SSD prompt cache. The cache does not need two computations of a prefix to
agree; it needs a save and a load to agree, and that is measured byte for byte on the 0.8B on CPU
and on Vulkan and on the 35B, `differing=0` in all three. The nondeterminism narrows the claim from
"a hit equals a recomputation" to "a hit reproduces the computation that wrote the entry", which is
what `docs/research/30-ssd-prompt-cache-results.md` says.

It does affect anything in this tree that compares two runs byte for byte, or compares a cached
result against a fresh one. `cache_hit_check.py` was changed for exactly that reason: it now runs
the uncached baseline twice and only counts the cached/uncached comparison as evidence when those
two agree.

## Reproduce

```sh
gcc -O2 -march=native -o memcheck_self scripts/research/ssdcache/memcheck_self.c -lm
./memcheck_self 4 512

# test 3: uninitialized heap. perturb_probe prints a malloc line and a memalign line,
# and the memalign one is what matters, because that is where ggml's buffers come from
gcc -O1 -o perturb_probe scripts/research/ssdcache/perturb_probe.c
MALLOC_PERTURB_=85 ./perturb_probe 1048576

# tests 3, 4 and 5 in one harness, built against build-vk as doc 28 describes.
# config A is ubatch 512 chunk 512; arg 3 is n_threads, 0 leaves the library default
for R in 1 2 3; do ./ubatch_determinism <0.8B> 1024 0 2048 A | grep "^config A "; done
for R in 1 2 3; do MALLOC_PERTURB_=85 ./ubatch_determinism <0.8B> 1024 1 2048 A | grep "^config A "; done
for R in 1 2 3; do UBDET_PAINT=0xAA ./ubatch_determinism <0.8B> 1024 1 2048 A | grep "^config A "; done
for R in 1 2 3; do setarch -R ./ubatch_determinism <0.8B> 1024 0 2048 A | grep "^config A "; done

# test 6: the same binary with the driver hidden, and the control with it visible
VK_ICD_FILENAMES=/nonexistent/icd.json UBDET_CSR=1 ./ubatch_determinism <0.8B> 1024 1 2048 | grep -E "mxcsr|^config "
UBDET_CSR=1 ./ubatch_determinism <0.8B> 1024 1 2048 | grep -E "mxcsr|^config "

# the three mechanisms ruled out for the driver's effect, each with the driver visible
for V in MESA_SHADER_CACHE_DISABLE=true LD_BIND_NOW=1 RADV_DEBUG=nocache; do
  for R in 1 2 3; do env $V ./ubatch_determinism <0.8B> 1024 1 2048 A | grep "^config A "; done
done

# the Vulkan free tree, which lands on the same state as the driver hidden
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
cmake --build build-cpu -j7 --target llama
g++ -O2 -std=c++17 -I include -I ggml/include scripts/research/ssdcache/ubatch_determinism.cpp \
  -L build-cpu/bin -lllama -lggml-base -lggml -Wl,-rpath,$PWD/build-cpu/bin -o ubatch_cpu
./ubatch_cpu <0.8B> 1024 1 2048   # all five configs, including A against A2, all f863c3ce8b2ef52

# the MSan build and the artifact repro
cmake -B build-msan -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF \
  -DCMAKE_C_FLAGS="-fsanitize=memory -fno-omit-frame-pointer -g -O1" \
  -DCMAKE_CXX_FLAGS="-fsanitize=memory -fno-omit-frame-pointer -g -O1" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=memory" -DLLAMA_SHARED_LINKER_FLAGS="-fsanitize=memory" \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
cmake --build build-msan -j7 --target llama-cli
MSAN_OPTIONS=halt_on_error=0 build-msan/bin/llama-cli -m <0.8B> -n 2 -t 1 -c 2048 -no-cnv -p alpha
```
