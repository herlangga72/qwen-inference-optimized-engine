# The two state file APIs do not interoperate: results

Date: 2026-09-27
Tree: `qwen-only-backends`
Code: `scripts/research/kvstore/state_api_compat_check.cpp`
Model: `Qwen3.5-0.8B-Q4_K_M`, 303 prompt tokens, n_ctx 2048, n_seq_max 1

**Verdict: the claim that the store format is a strict superset of `LLAMA_STATE_SEQ` v3, so that
`--slot-save-path` and the prompt cache keep working, is false. A file written by
`llama_state_seq_save_file` cannot be read by `llama_state_seq_load_file_direct` and a file written
by the direct pair cannot be read by the shipping pair. Both directions return 0.**

The direct pair itself is sound. Restoring into a fresh context and continuing gives a continuation
identical to a run that never parked, byte for byte.

## What the check does

Saves the same sequence twice, once through each public API, then loads each file through each
loader, four combinations. Every load gets a fresh context, restores into an empty sequence, then
decodes one fresh token and samples 8 tokens. The reference is the same sequence never parked, with
the same probe decode and the same 8 samples, so a restore only passes if the continuation matches.

## Results

| save | load | returned | continuation |
| --- | --- | --- | --- |
| shipping | shipping | 23933844 | `235212 1122 11234 9197 1627 11419 2180 4489` |
| direct | direct | 23933832 | `235212 1122 11234 9197 1627 11419 2180 4489` |
| shipping | direct | 0 | not loaded |
| direct | shipping | 0 | not loaded |

The reference continuation is also `235212 1122 11234 9197 1627 11419 2180 4489`. Both working
combinations match it exactly, which is the first time the store path has been checked against a
never-parked continuation through a fresh context rather than a reused one.

The boundary rows are the finding. The formats are not interchangeable, so an existing slot file
cannot be loaded by the store, and a store file cannot be handed to the existing slot loader.

## What this means for the design

The compat claim was one of the stated reasons for the format choice, so it needs a decision:

- Detect the format on read. A framed file has a block 0 prefix and framed blocks after it; a v3
  file starts with its own magic. Either loader could sniff and dispatch, which would make the
  superset claim true in the direction that matters, reading old files.
- Or drop the claim and state plainly that the store owns its own files, and that migrating an
  existing `--slot-save-path` file is a conversion step.

Reading old files is the direction worth keeping, since it decides whether a user's existing saved
slots survive. Nothing in the data says it is hard: the payload is the same v3 stream in both
cases, only the container differs.

## Return values disagree

`llama_state_seq_save_file` returns the bytes written, which for an unframed file is the file size.
The direct save returns `io.n_bytes()`, which is the payload length, so for a framed file the file
on disk is larger than the returned value by the framing overhead, about 1.6 percent at this block
size. That is 54875 blocks against 27640948 payload bytes on the park file. A caller that uses the
return to size a buffer or to verify disk usage will be wrong. Either return the file size or
document that the store's return is a payload length.

## Four wrong versions of this check, and what each looked like

Every one of these first read as an engine failure and was mine:

1. Clearing with `llama_memory_clear(mem, true)` before a restore. That resets the memory module
   itself, and the restore then fails for both APIs. Freeing the sequence and reinitialising the
   context are the two things that work.
2. Saving after generating. The save records the token count it is handed while the state had
   advanced past it, and the shipping loader rejected its own file with `n_stream mismatch`. The
   sequence has to be back at the prompt before saving.
3. Reusing one context for successive restores. The second restore in the same context either
   returned 0 or came back with a shifted continuation. A fresh context per restore is the honest
   way to ask whether a file is good.
4. Sampling straight after a restore. A restore produces no logits, so a fresh context has none and
   `generate` returned nothing, which looks like a failed restore. Decoding one fresh token first
   is both the fix and the more faithful test.

## What is still not verified

**First, a correction. An earlier version of this section said there is no in-tree test for the
server park path. That was wrong.** `tools/server/tests/` is a pytest suite and it contains
`unit/test_kv_keep_only_active.py`, which drives `--cache-ram` and `--cache-idle-slots` with two
slots and unified KV, and `unit/test_slot_save.py`, which drives `--slot-save-path`. Both are
exactly the acceptance path this document is about. I had only looked in `tests/`, which is the C
test directory.

I tried to run them against a freshly built server and could not:

- The suite needs Python packages that are not installed (`wget`, `openai`, and the rest of
  `tools/server/tests/requirements.txt`). A venv install ran for sixteen minutes and did not get
  past `pip`, because `numpy` has no wheel for this interpreter and pip falls back to a source
  build. That path was abandoned rather than left running.
- I then started the server directly with `-np 2 -c 2048 --cache-ram 64 --cache-idle-slots --slots`.
  It came up with two slots and `/slots` reachable, but `kv_unified` was false, and the idle slot
  behavior is specified in terms of unified KV, so that run did not match the tested config and I
  did not drive a park and restore cycle through it. That server has been stopped.

One thing that run did turn up matters more than it did: **the user already has a live server on
port 8741 running the real workload with the park configuration**, `Qwen3.6-35B-A3B-UD-IQ3_XXS`,
`-ngl 99 -np 2 -kvu --cache-ram 512 --cache-idle-slots`. I read `/health` and `/slots` and
confirmed it is healthy with idle slots, and did not send it any prompts, since a live workload the
user is running is not mine to perturb with test traffic.

So layers 2 and 3 remain unverified end to end. The reason is now precise rather than vague: the
test suite exists and is the right harness, it is blocked on a dependency build on this box, and
the alternative, driving the live server, would mean sending test prompts into a workload that is
currently in use. Either the dependency build needs to be fixed, or the run needs to happen on a
machine that is not serving anything.
