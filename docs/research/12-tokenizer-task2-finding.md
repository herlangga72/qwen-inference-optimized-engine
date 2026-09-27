# Tokenizer SIMD and threads: Task 1 done, Task 2 blocker

Date: 2026-09-27
Tree: `qwen-only-backends` at 4f5831ba2 plus the plan E change.
Plan: `docs/superpowers/plans/2026-09-26-tokenizer-simd-threads.md`.
Spec: `docs/superpowers/specs/2026-09-26-tokenizer-simd-threads-design.md`.

## Task 1, the differential harness: done and verified

- `src/llama-vocab.cpp`: `llm_tokenizer_use_legacy()`, reading `LLAMA_TOKENIZER_LEGACY` once, and the
  seam in `llm_tokenizer_bpe_session::tokenize` where the fast scanner will land. Both settings take
  the reference path today, which is what the plan says to expect at this point.
- `scripts/research/tok_corpus_gen.py` and `scripts/research/tok_diff.sh` as specified, plus the
  generated `scripts/research/tok_corpus/` (12 files, 276 kB, multilingual plus whitespace, digits,
  punctuation, repetition and a long line).
- `scripts/research/tok_diff.sh scripts/research/tok_corpus` reports `all identical` over the corpus
  plus every `src/*.cpp` and `common/*.cpp` (113 inputs, two tokenizations each) in 29.4 s wall. That
  is the regression tool every later task leans on.
- The count baseline reproduces: `llama-tokenize -m models/ggml-vocab-qwen35.gguf -f
  /home/herlanggays/.jcode/scratch/tokbench/corpus.txt --show-count` gives 791,777 tokens, matching the
  plan's number, on a 2.68 MB corpus.

## Task 2, keying merge ranks by token ids: the plan's recipe cannot work as written

The plan adds `uint32_t id` to `llm_symbol`, "initialised to UINT32_MAX for a byte-level symbol, and to
the merged id when a merge is accepted", and looks ranks up with `bpe_rank(symbols[i].id,
symbols[j].id)`.

The first-level lookup has no id to use. The symbols the merge loop works on are produced by
`llm_tokenizer_bpe_session::tokenize`, which walks the word by *character*, not by byte:

```cpp
                size_t char_len = std::min(word.size() - offset, (size_t) unicode_len_utf8(word[offset]));
```

so a symbol is one to four bytes of the word. The rank table's keys are the model's merge strings, and
the current lookup is `vocab.find_bpe_rank(left_token, right_token)` on the two symbol *texts*
(`add_new_bigram` builds both strings for every candidate and `llm_bigram_bpe::text` is their
concatenation, used for the stale check). Setting the initial id to UINT32_MAX and looking up
`bpe_rank(UINT32_MAX, UINT32_MAX)` finds nothing, so no first merge would ever be accepted and every
word would fall back to one token per character. The differential harness would catch that, but the
recipe has to be corrected before it is worth running.

The fix is not mechanical, because a character symbol is not always a token. Measured on the qwen35
vocab over the Task 1 corpus (`scratch/tokbench/char_token.cpp`, 143 unique codepoints): **135 map to
exactly one token and 8 do not**, so for those 8 there is no id to key anything on.

| not one token | pieces |
| --- | --- |
| U+0303 combining tilde | 2 |
| U+0308 combining diaeresis | 2 |
| U+0661..U+0665 Arabic-Indic digits | 2 each |
| U+200D zero width joiner | 2 |

Combining marks, ZWJ and non-ASCII digits are ordinary text, so this is not a corner of a corner. Any
id-keyed design has to carry a second path for them, and the spec's "a candidate's identity is two
integers and a rank lookup is an array index" cannot hold as stated.

Three ways forward, none of which the plan chose:

1. Hybrid: `id = text_token(symbol text)` when the string is a token, otherwise a sentinel; a rank
   lookup involving a sentinel falls back to the string pair. Keeps ids bit-identical to today by
   construction and gets the fast path for the common case. Costs a second lookup path.
2. Keep the strings and remove the allocations: key the table by a 64-bit hash of the pair computed
   from the two symbol texts in place, with the strings still available for the collision check, and
   compare against `bigram.hash` for the stale check instead of concatenating. Semantics unchanged,
   no per-candidate allocation, and it is a strictly smaller change than the plan's.
3. Change the initial split to one symbol per byte. That makes every initial id a byte token id, which
   is what the plan assumes, but it changes tokenization itself rather than only its speed, so it needs
   a deliberate decision and the harness has to show the ids are unchanged before and after.

Recommendation: 2 first. It removes the per-candidate allocations from the lookup and the stale check
without touching the symbol identity at all, so the 8 non-token characters need no special case and
the ids can only change if the rewrite is wrong, which the harness decides in 30 seconds. Then 1 if a
profile says the id lookup is what matters. The spec's own budget puts the pre-tokenizer (Task 3) and
the merge ranks (Task 2) at different sizes, so measure both before committing to either.

A concretely cheaper variant of 2, since the project is C++17: store the merge table as
`unordered_map<pair<string_view, string_view>, int>` with the halves pointing into the GGUF metadata,
which is stable for the vocabulary's lifetime. A lookup then builds two views into the symbol texts
with no allocation, and the comparison stays byte exact, so no hash collision can pick a wrong rank.

## Where the time is

A rewrite is only worth doing if the tokenizer is on a hot path, so this was measured before anything
else. Temporary `ggml_time_us()` marks around the three phases of the BPE session (since removed; the
harness reports all identical again afterwards), over the 250 kB Task 1 corpus in one call:

| phase | time |
| --- | --- |
| split (`unicode_regex_split`) | 0.009 s |
| merge loop | 0.051 s |
| symbol text to id (the emit loop) | 0.001 s |
| total | 0.061 s |

Two readings. The merge loop is 84% of tokenization, so Task 2 is where a tokenizer rewrite would pay.
But the whole tokenizer runs at about 4 MB/s, which is 0.24 s per MB of prompt: a 10 kB prompt costs
2.4 ms of tokenization in front of a prefill that takes about a second, under 1%, and Task 3 is the
same order. The plan budgets the tokenizer as a unit of work; on this machine it is not a unit of end
to end time. Whoever picks either task up should first name what it is supposed to move, then measure
that, not the tokenizer in isolation.

## Not done

Tasks 2 through 7 of the plan. Task 1 is a clean checkpoint on its own: the harness, the corpus, the
switch and the reverted-free baseline are all in place and verified, and nothing about tokenization
behaviour changed.
