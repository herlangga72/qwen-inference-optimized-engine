# Tokenizer, plan G task 2: the merge table

Status: implemented and verified. Task 1 (`scripts/research/tok_diff.sh`, the corpus and the
`LLAMA_TOKENIZER_LEGACY` switch) is the baseline this builds on, and the design question is written up
in `12-tokenizer-task2-finding.md`.

## What the plan asked for and what was built

The plan wanted the merge rank table keyed on token ids, with a candidate's identity being two
integers and a rank lookup an array index. Measurement killed that: 8 of the 143 codepoints in the
corpus are not a single token in this vocabulary, so there is no id to key them on.

What is built instead keeps the rank table defined over strings, byte exact, and removes the
allocations:

- `bpe_ranks` is keyed by `std::pair<std::string_view, std::string_view>`. The halves point into
  `merge_strings`, a vector owned by the vocab that holds the merge table's bytes. The loader fills
  that vector first and builds the map afterwards, so no view can outlive a reallocation.
- `find_bpe_rank` takes views. A caller with strings converts for free, and the merge loop passes views
  into the symbol texts, so a lookup costs no allocation.
- The stale check no longer rebuilds the candidate's text. A symbol keeps its start pointer and only
  ever grows by absorbing its right neighbour, so the pair is unchanged exactly when
  `left.n + right.n == bigram.size`. A candidate whose symbols moved has a different total length, so
  the check is equivalent, not approximate.
- `get_bpe_merges` was adapted; it still rebuilds the `"first second"` strings by rank, which is what
  `llama-model-saver` writes.

## Verification

| check | result |
| --- | --- |
| reference merge path vs view merge path, 2,939,758 bytes, 859,228 tokens | byte identical, sha256 `e0e4d242ae098bd5` both sides |
| id checksum over the Task 1 corpus, 20 repetitions, before vs after the change | `ba424148b2873e23` both |
| `bpe_ranks.size()` after load | 151387, equal to an independent count of unique merge pairs in the gguf |
| `scripts/research/tok_diff.sh`, qwen35 and qwen2 vocabs | all identical, 113 inputs |
| `ctest -L main` | 34 of 34 |
| `build-cpu` and `build-vk` | build clean |

The first row is the real differential. The `LLAMA_TOKENIZER_LEGACY` switch is inert for the merge
loop in the final tree, so the harness compares a path with itself there; the reference used for that
row was a temporary second merge path (string candidates plus the concatenated-text stale check) kept
in the session only long enough to run it, then removed. Compare hashes, not the harness label, if
this is ever re-run: `tok_bench.sh` prints the same style of checksum.

## Speed

`scripts/research/tok_bench.sh`, 250,820 bytes per repetition, 20 repetitions, three runs each:

| build | ms per repetition | MB/s |
| --- | --- | --- |
| before | 60.4, 60.7, 62.7 | 4.00 - 4.16 |
| after | 33.0, 33.4, 33.5 | 7.48 - 7.60 |

1.82x on the whole tokenizer over prose and source text. As `12-tokenizer-task2-finding.md` records,
tokenization is about 0.24 s per MB, so this moves under 1% of end to end time on this machine. It is
a real tokenizer win, not a real inference win.

Measured aside worth knowing before touching this again: keeping the reference merge path in the tree
cost 25% of the fast path, 44.4 ms against 33.4 ms. The branch is not the expensive part, the
`std::string text` member in `llm_bigram_bpe` is, because every candidate is copied through the
priority queue. Anything added to that struct is paid for on every bigram.

## Task 3 and the rest are not worth starting

Decide against the workload, not against the code. Same vocabulary, same machine, `tok_bench.sh`:

| input | bytes | tokens | per call | MB/s |
| --- | --- | --- | --- | --- |
| chat style prompt | 166 | 33 | 19 us | 8.65 |
| source excerpt | 1024 | 379 | 99 us | 10.31 |
| Task 1 corpus | 250820 | 65720 | 31.9 ms | 7.86 |

Cost is linear in the input, with no fixed per call overhead to attack. Task 2 left the regex split at
about 9 ms of the 32 ms a 250 kB input now costs, so the split is roughly a quarter of the tokenizer
and a little under 1% of it is the symbol to id lookup. A hand written scanner replacing
`unicode_regex_split` therefore has about 5 us of a 19 us short prompt call to win, against a prefill
of that prompt that takes on the order of a tenth of a second. That is not a target, and neither is
the remaining lookup.

This closes the plan's tokenizer work. Tasks 2 and 3 were both budgeted as units of work; on this
machine the writable part of the tokenizer is a rounding error on a request, and only a bulk
tokenization workload, where 8 to 10 MB/s might matter, would change that.

## Not done

Tasks 3 to 7 of the plan, dropped above with the numbers behind the decision. Task 1 and task 2 are
clean checkpoints: the harness, the corpus, the switch and the reference baseline in place, and the
merge table rewritten with ids byte identical to the reference over 2.94 MB of input.
