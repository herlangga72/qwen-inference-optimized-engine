# Tokenizer: SIMD and threaded rebuild

Date: 2026-09-26
Status: approved design, not yet implemented
Scope: `src/llama-vocab.cpp` (the BPE tokenizer internals) and the detokenize path. No change to vocabularies, ids, or the tokenizer's external behaviour.
Target: at least 30x wall-clock throughput on this box, with byte identical ids.

## Context

Two hot spots, both single-threaded, neither SIMD anywhere in the 2373 lines of `llama-vocab.cpp`:

1. **Pre-tokenizer.** `llm_tokenizer_bpe` holds one regex per pre-type (`regex_exprs`,
   `llama-vocab.cpp:82-105`). For `qwen35` it is one expression combining `[^\r\n\p{L}\p{N}]?\p{L}+`,
   `\p{N}`, ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`, `\s*[\r\n]+`, `\s+(?!\S)` and `\s+`. `tokenize()` runs the
   whole prompt through `unicode_regex_split(text, regex_exprs)` once.
2. **Merge loop.** The classic shape: a linked list of `llm_symbol`s plus a priority queue of
   `llm_bigram_bpe` ranked by merge rank, with the merge table as
   `std::unordered_map<std::pair<std::string,std::string>, int>` (line 388). Each candidate merge
   builds two strings and hashes both; each accepted merge rewrites its neighbours.

Measured on this box, `llama-tokenize`, `qwen35` pre-type, 2.68 MB corpus, 791,777 tokens:
**342k tokens/s, 2.9 MB/s**, against 8 cores and 16 threads with AVX2 and no AVX-512.
`qwen2` measures 3.5 MB/s on the same corpus.

Where that sits in the engine, honestly: prefill on this model runs at 202 t/s with the whole model
on the GPU, so a 100k-token prompt spends about 0.29 s tokenizing and about 495 s prefilling.
Tokenization is 0.06% of prompt ingest today and 0.002% at the 30x target. What the work buys is
prompt-ingest latency for very long prompts and a lower CPU cost per request when serving many short
prompts, about 0.34 ms per KB today. This sub-project does not change decode or prefill throughput,
and the spec says so rather than implying otherwise.

## Design

**Units.** Four, each understandable and testable on its own:

1. **`pre_tokenizer`** (new): given a byte range, yields the word ranges the pre-type would produce,
   for `qwen35` and `qwen2`. Scalar UTF-8 decoding with a Unicode table lookup for non-ASCII, plus an
   AVX2 path that classifies 32 bytes at a time into letter, digit, space and other masks and
   resolves ASCII runs without per-character branching. The lookahead rules (`\s+(?!\S)`,
   `\s*[\r\n]+`) are handled explicitly, since they are the part a naive scanner gets wrong. The
   generic `std::regex` implementation stays as the fallback for `default` and unknown pre-types, so
   the change is additive and cannot regress the paths it does not cover.
2. **`bpe_merge`** (reworked internals): merge ranks keyed by packed ids, `(uint64_t) left << 32 |
   right`, in a flat table built once at vocab load, replacing the string-pair map. No string is
   constructed per candidate. The linked list and heap stay, because they are the right shape; what
   changes is that a candidate's identity is two integers and a rank lookup is an array index.
3. **`chunked driver`** (new): splits the prompt at positions that are provably boundaries of the
   pre-tokenizer's own output, dispatches chunks to a pool, and concatenates ids in order. The pool
   is owned by the vocab, because `llama_tokenize` is a free function with no context to borrow one
   from; it is created lazily on the first large prompt, and never used for prompts below 4 KB, which
   stay on the calling thread to avoid paying for a pool on short inputs. Pool size is 8 on this box,
   not 16: the machine is 8 physical cores with SMT, and both the engine threads and the earlier
   thread sweeps measured 16 threads worse than 8. Elsewhere it is `min(hardware_concurrency, 8)`.
4. **`detokenizer`** (touched): `llama_token_to_piece` gains an ASCII fast path that writes into the
   caller's buffer without building an intermediate string, which is the common case for streaming
   output.

**Data flow.** Bytes are split at safe boundaries, pre-tokenized and BPE-merged per chunk in
parallel, and ids are concatenated in chunk order. Special-token parsing, BOS/EOS handling and the
public API are unchanged, so every caller sees exactly the ids it sees today.

**Correctness is the invariant, not a target.** Two mechanisms carry it, both implemented outside
`tests/`, because this repository does not accept new files there:

- **Differential comparison.** The legacy implementation stays selectable through a development-only
  switch (an environment variable that forces the `std::regex` pre-tokenizer and the string-keyed
  merge table) for as long as the rewrite is in progress. A script in `scripts/research/` tokenizes a
  corpus twice, once per path, and diffs the id lists. The corpus is the whole repository source
  tree plus a multilingual sample with Cyrillic, CJK, Arabic, emoji, combining marks, mixed
  whitespace and CRLF. The switch and the script are removed or narrowed to the generic pre-types
  before the change is finalised, so no debug path ships on the hot route.
- **Boundary test.** For every chunk the driver produces, tokenizing the chunk alone and concatenating
  must equal the single-chunk run. This makes the split rule a verified property rather than an
  assumption, and it is the test that protects the threaded path.

The existing `test-tokenizer-0` is also run against both kept vocab fixtures, and it is the check
that the public behaviour is unchanged.

**Verification and gates.**

- Throughput: `llama-tokenize` on the same 2.68 MB corpus, at least 10M tokens/s, which is 30x from
  342k. Reported for 1, 4 and 8 threads, so the scaling is visible.
- Latency: a 100k-token prompt end to end, reported as the tokenization share of prompt ingest.
- Correctness: the differential and boundary tests above, plus `ctest -L main`.
- Engine impact: `llama-bench` pp256 and tg128 recorded before and after, expected unchanged, since
  the tokenizer is not in the decode loop. A regression there would mean the change leaked into the
  engine, which is itself the finding.

## Risks

- **Unicode edge cases.** The scalar scanner must reproduce `std::regex` semantics exactly, including
  the negative lookahead on trailing whitespace. The differential corpus is the mitigation, and any
  divergence is a bug, never a tolerance.
- **Chunk boundaries.** A wrong boundary changes ids silently. Mitigated by restricting splits to
  positions the pre-tokenizer would split at, and by the boundary test.
- **Pool interaction.** The engine has its own thread pools; a tokenizer pool must not oversubscribe
  the machine during prefill. Sized to 8 with lazy creation, and unused for short prompts.
- **API latency shape.** Threaded tokenization makes a long call faster and a short call unchanged;
  the 4 KB threshold is what keeps the second half true.
- **The honest one:** none of this moves engine throughput. If the goal were engine speed, the
  measurement says to spend the effort elsewhere; this spec exists because a tokenizer that takes
  30x less time is worth having on its own terms, and because prompt ingest for very long inputs is
  user-visible.

## Out of scope

- **GPU tokenization.** Rejected: the merge loop is a sequential priority queue, so only
  classification and boundary finding are GPU-able; on UMA the data needs no transfer but the launch
  and sync per prompt dominates for ordinary prompts, and the CPU stays on the critical path for
  merges. It would add shaders in both backends for no measured gain.
- Changing vocabularies, merges, ids or the GGUF format.
- The Jinja template and chat parsing path, which is a separate subsystem; it benefits from a faster
  tokenizer without being part of this change.
- Other pre-types beyond `qwen35` and `qwen2`, which stay on the generic path.
- `llama_tokenize` API changes: the signature and the public behaviour stay as they are.
