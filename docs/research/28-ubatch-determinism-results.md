# Ubatch and chunk-split determinism of the serialized sequence state: results

Date: 2026-09-28
Tree: `qwen-only-backends`, commit `397b643f4`
Machine: Ryzen 7 6800H (8 cores, 16 threads), Radeon 680M, 27.1 GiB, CPU only
         (`n_gpu_layers = 0`), AC connected
Model: `Qwen3.5-0.8B-Q4_K_M.gguf`, one 1500-token synthetic word-list prompt,
       16 threads reported by `nproc`, 4 ggml threads in the library default
Build: `build-vk`, the only build directory, `GGML_OPENMP = ON`, no BLAS,
       `GGML_NATIVE = ON`, `Release`
Harness: `scripts/research/ssdcache/ubatch_determinism.cpp`
Gates: the transparency claim in
       `docs/superpowers/specs/2026-09-28-ssd-prompt-cache-design.md`
Related: `docs/research/27-ssd-prompt-cache-feasibility.md` (the open risk this
         test was written for), `docs/research/22-state-serialization-not-reproducible.md`
         (why a control is mandatory)

**Corrected the same day by `docs/research/31-prefill-nondeterminism-localisation.md`. The
nondeterminism recorded below is the Vulkan driver being in the process, not this fork. `build-vk`
initialises the Vulkan device at startup even at `n_gpu_layers = 0`, so "CPU only" in the header
above means no tensor on the GPU, not no driver in the process, and the process has Mesa shader disk
cache threads running during the decode. Re-running this harness with the driver kept out of the
process, and running it against a tree built with `GGML_VULKAN=OFF`, returns one state for every
configuration, every thread count and every process, `f863c3ce8b2ef52`. Read every "the states
differ" below as a property of the environment this was run in, not of the build. The harness,
the control, the byte accounting and the finding that the control is clean all stand.**

**Verdict: the states differ, and the gate fails. It fails by more than the
question asked. Two fresh contexts running the identical configuration over the
identical 1500-token prefix produce different serialized states, 25441008 of
38670092 bytes differ, 65.79 percent, and different logits. The stock
`llama-cli`, greedy, same seed, produced three different continuations for the
same prompt in three runs. So on this build the engine's prompt processing is
not reproducible above about 24 prompt tokens, and an SSD prompt cache cannot
claim byte equality or behavioural equality for a hit at any ubatch setting. The
split comparison A vs B, A vs C, A vs D is dominated by that run to run
nondeterminism, which means the answer to the posed question is not "splits
differ" but "this build does not recompute the same prefix at all".**

The control required by document 22 is clean: serializing the same unchanged
state twice inside one context gave 0 differing bytes in all 145 configuration
rows of the 42 runs behind this document. The serialization path is stable. The
instability is in the computation that fills the state, and it is visible in the
logits alone, before any serialization is involved.

## The configurations

All four share one prompt of 1500 tokens, decoded with explicit
`llama_decode` calls of a given size, positions increasing, one fresh context
per configuration, `n_ctx = 2048`, `n_batch = 2048`, `n_seq_max = 1`.

| config | n_ubatch | chunk | note |
| --- | --- | --- | --- |
| A | 512 | 512 | each decode call is one full ubatch |
| B | 512 | 128 | decode call is a quarter ubatch |
| C | 128 | 512 | each decode call splits into four ubatches |
| D | 256 | 256 | each decode call is one half ubatch |
| A2 | 512 | 512 | A repeated in a second fresh context, the rerun control |

## Result at 1500 tokens

Canonical run, log at
`/home/herlanggays/.jcode/scratch/ssd-cache/ubatch-determinism.txt`. Hash is
FNV-1a 64 over the serialized buffer, `state_size` and `len` are equal in every
row, `control diff` is the byte count between two back to back serializations in
the same context.

| config | state size | state hash | control diff | logits hash | first greedy token |
| --- | --- | --- | --- | --- | --- |
| A | 38670092 | 70dd64030267944d | 0 | d075fcddbe0acecf | 26813 |
| B | 38670092 | 2ab0d7b33b89678e | 0 | 323998b717267265 | 26813 |
| C | 38670092 | 6361dc43dfac970e | 0 | 661c937925af6b8b | 26813 |
| D | 38670092 | 42bc06e1f5874815 | 0 | 31c47cf80301eed8 | 26813 |
| A2 | 38670092 | 335c04786b64553f | 0 | cc153fcbb968bd38 | 26813 |

Five distinct state hashes, five distinct logits hashes, same input, same
process. The logits hash is what makes the control argument complete: the used
computation, not just the serialized blob, produced different numbers.

Pair differences, a byte by byte scan of the two buffers:

| pair | differing bytes | of bytes | fraction | first differing offset |
| --- | --- | --- | --- | --- |
| A-B | 24806715 | 38670092 | 64.149615% | 106692 |
| A-C | 26306181 | 38670092 | 68.027200% | 106692 |
| A-D | 24773085 | 38670092 | 64.062648% | 106692 |
| A-A2 | 25441008 | 38670092 | 65.789882% | 48324 |

The A-A2 row is the one that decides how to read the other three. A and A2 are
the same configuration, the same prompt, the same process, back to back. They
differ in 65.79 percent of the state. A split comparison cannot be cleaner than
the rerun it is compared against, and the rerun is not clean.

The differences are spread over the whole blob, not localized: every 1 MiB
bucket of the 37 contains differing bytes in most pairs. A-A2 has 37 of 37
nonzero buckets. So this is not one tensor region being skipped or one padding
area being uninitialized. It is the stored values themselves.

### The behavioural probe is insensitive, do not rely on it

In the canonical run all five configurations emitted the same eight greedy
tokens, 26813 39200 3760 201179 175660 8029 235212 1122, while their states
differed by 50 to 68 percent. In other runs at the same length the first token
split between 26813 and 0, for example the `n_threads = 1` run where B and A2
produced logits hash `7c6bfe2c727ae383`, first token 0, and A, C, D produced
26813. Eight greedy tokens is therefore not a usable equality test here. It
agrees when the perturbation stays under the argmax margin, and the states still
differ. The logits hash is the probe that always shows the difference.

## The rerun control, across processes

Config A run alone in separate processes, same parameters, 1500 tokens:

| run | state hash | logits hash | first greedy token |
| --- | --- | --- | --- |
| A1 | c3b23b6020cc0abb | e6dc5d9489ae1dad | 26813 |
| A2 | 7ae76d3d2663a945 | 753ff8f687551fd6 | 1659 |
| A3 | 8b19b7be830bb7cc | cfaa47c6457c1023 | 26813 |
| A4 | c3305cea973c7128 | c490a9503c2a26c6 | 26813 |

Four runs, four state hashes, four logits hashes. The same holds with one
thread: `n_threads = 1` produced A = 39d9fd1432dc3baf and A = bcdb64e3524dcd4a
in two processes, and `OMP_NUM_THREADS = 1` with `n_threads = 1` still differed.
So this is not the ggml thread count and not OpenMP scheduling.

The output space is small and discrete, which is why partial agreement keeps
appearing. `logits_hash = 7c6bfe2c727ae383` with first token 0 recurs in
configs B, C and others, and several runs landed on state hash
`bedaa6964ed48df7` for different configuration labels. That is why one batch of
isolated runs showed B, C and D agreeing bit for bit and looked like a clean
split-invariance result. Repeating B and C in new processes broke it: B gave
b4548847720fbf3d, then 109e134c8487ef53, C gave 2789a0cfaea6aec5, then
fe8ff0559f30a9ce. The agreement was a small attractor hit twice, not stability.

## The stock CLI reproduces nothing either

This is the independent check that the harness is not the cause. `llama-cli`
from the same build, `-ngl 0 -c 2048 -n 32 --temp 0 -s 1 -st`, on 100
repetitions of the same word list, about 1608 tokens:

```
run 1, 4 threads:  "A long string of text."  then "Text Content:"
run 2, 4 threads:  "A massive block of text."
run 3, 1 thread:   "A long string of text."  then "Text content:"
```

Three runs, three outputs, greedy sampling, fixed seed. The nondeterminism is in
the build, not in the harness's context handling or its state reads. A first CLI
check at 8 output tokens was not discriminating because the generated text was
identical there, which is the same insensitivity the harness shows.

## Length threshold

Same harness, sweep over prompt length, all five configurations per run:

| prompt tokens | verdict | config A hash across processes |
| --- | --- | --- |
| 8 | IDENTICAL | 2f4a9eed4ec7c185 in both of 2 processes |
| 12 | IDENTICAL | - |
| 16 | IDENTICAL | - |
| 24 | IDENTICAL | af9ea53e6b15ffad in both of 2 processes, logits 0be076fb38a4f6c9 |
| 32 | DIFFERS | not stable, 2 of 2 processes DIFFERS |
| 48 | DIFFERS | - |
| 64 | DIFFERS | - |
| 128, 256, 384, 512, 513, 700, 1024, 1500 | DIFFERS | not stable |

Below the threshold the harness is bit exact, including across separate
processes, and all four split configurations agree. That is the strongest
available evidence that the harness itself is correct and that the differences
above the threshold are real. The transition is between 24 and 32 tokens, which
is not a chunk boundary for any of the four configurations and not the point
where a second decode call appears in config A, so it is not a boundary in this
harness.

## State size is set by tokens used, not by the context

`n_ctx = 4096` with the same 1500-token prompt gives state size 38670092, the
same as `n_ctx = 2048`. The size scales with the tokens actually decoded:

| tokens | state size |
| --- | --- |
| 8 | 20300588 |
| 24 | 20497580 |
| 64 | 20990060 |
| 128 | 21778028 |
| 256 | 23353964 |
| 384 | 24929900 |
| 512 | 26505836 |
| 513 | 26518148 |
| 700 | 28820492 |
| 1024 | 32809580 |
| 1500 | 38670092 |

The slope is 12312 bytes per token and the fixed term is about 20.2 MB. So the
blob does not contain the unused cells of the allocated context, and "the
serializer wrote never-written memory" does not explain the differences. The
padding beyond the last written token is small and the logits difference rules
this out independently in any case.

## What this means for the prompt cache

The design's transparency claim was: a cache hit yields the same state a
recomputation would, so the continuation cannot differ. On this build that claim
cannot be made at any strength, because a recomputation does not agree with
another recomputation.

1. **Byte equality is dead.** Even two identical recomputations differ in 65
   percent of the state at 1500 tokens.
2. **Behavioural equality is not available either.** The logits differ, and the
   greedy continuation differs across runs in `llama-cli`. Restoring an exact
   cached state would make a hit *more* self-consistent than a recomputation,
   not equal to it, so a hit would not reproduce the no-cache run.
3. **The split question the gate asked is currently unanswerable.** Different
   ubatch and chunk sizes cannot be compared while the same setting is unstable
   against itself. Any A vs B number in this document is two nondeterministic
   draws, not a split effect.
4. **Below 24 tokens the whole pipeline is bit exact**, so this is not a
   permanent property of the engine's arithmetic. It is a defect that appears
   with length.

The honest summary for the design is that the prompt cache is blocked on a
determinism bug in prefill, not on the cache's own format. Fixing the bug is now
on the critical path, and until it is fixed no cache-hit correctness experiment
with the 0.8B on this build can produce a meaningful pass or fail.

## What this does not cover

- **The mechanism.** I did not find it. The candidates I could not separate are
  an uninitialized read in a kernel that depends on context memory contents, an
  address or alignment dependent code path, and a genuine arithmetic ordering
  defect in this fork. Distinguishing them needs instrumentation inside `src/`,
  which this task does not touch.
- **Whether the bug is this fork or ggml.** `build-vk` is the only build
  directory, so there was no second build to compare against. Upstream
  `llama.cpp` was not tested. A one run comparison against a CPU only build at
  the same commit would settle it.
- **The 35B and the GPU.** Only `Qwen3.5-0.8B-Q4_K_M.gguf` on CPU was run. The
  GPU path is the degraded one from `docs/research/26-prefill-vulkan-profile.md`
  and was deliberately not used.
- **The cache itself.** Nothing here writes, reads, parks or restores an entry.
  This is the prefill computation the cache would store, measured through the
  same state API a cache would use.
- **Hardware or box health.** This document collected no evidence on it.
  `docs/research/31-prefill-nondeterminism-localisation.md` followed up: a 512 MiB pattern round
  trip and a tight `fmaf` reduction are bit stable across rounds and across processes, so a simple
  ALU or DRAM fault does not explain this. It also found that MemorySanitizer, the instrument for
  the leading candidate, is blocked on this system by an uninstrumented libstdc++, demonstrated
  with a fifteen line repro, so candidate 1 is still untested.
- **The exact threshold.** It is between 24 and 32 tokens on this prompt. The
  boundary was not refined further and may depend on the prompt and the model.

## Reproduce

```sh
# build, from the tree root
g++ -O2 -std=c++17 -I include -I ggml/include \
  scripts/research/ssdcache/ubatch_determinism.cpp \
  -L build-vk/bin -lllama -lggml-base -lggml \
  -Wl,-rpath,$PWD/build-vk/bin -o /home/herlanggays/.jcode/scratch/ssd-cache/ubatch_determinism

# the canonical run, under the watchdog
LOG=/home/herlanggays/.jcode/scratch/ssd-cache/ubatch-determinism.txt \
  scripts/research/ssdcache/run_guard.sh \
  /home/herlanggays/.jcode/scratch/ssd-cache/ubatch_determinism

# usage: model n_prompt n_threads n_ctx only_cfg
# the length threshold
for n in 8 12 16 24 32 48 64; do
  LOG=/home/herlanggays/.jcode/scratch/ssd-cache/ubdet-$n.txt \
    scripts/research/ssdcache/run_guard.sh \
    /home/herlanggays/.jcode/scratch/ssd-cache/ubatch_determinism \
    /home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf $n 0 2048
done

# one configuration alone, for the across-process rerun check
for i in 1 2 3; do
  LOG=/home/herlanggays/.jcode/scratch/ssd-cache/solo-A$i.txt \
    scripts/research/ssdcache/run_guard.sh \
    /home/herlanggays/.jcode/scratch/ssd-cache/ubatch_determinism \
    /home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf 1500 0 2048 A
done

# the independent check: stock CLI, greedy, three runs must agree and do not
P=""
for i in $(seq 1 100); do
  P="$P alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"
done
for i in 1 2 3; do
  build-vk/bin/llama-cli -m /home/herlanggays/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf \
    -p "$P" -n 32 --temp 0 -ngl 0 -c 2048 -s 1 -st -co off < /dev/null
done
```
