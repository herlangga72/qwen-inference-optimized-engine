# Park and unpark to disk: results

Date: 2026-09-27
Machine: Ryzen 7 6800H, CPU backend, `/dev/nvme1n1p2` f2fs
Model: Qwen3.5-0.8B-Q4_K_M (hybrid, `n_layer = 24`)
Code: `src/llama-io.{h,cpp}`, `src/llama-context.{h,cpp}`, `include/llama.h`
Check: `scripts/research/kvstore/park_disk_check.cpp`
Spec: `docs/superpowers/specs/2026-09-27-session-kv-store-design.md`
Plan: Task 4, steps 2 and 3

**Verdict: parking a real sequence to disk and unparking it works, but the check fails about
one run in five, and I could not root-cause it in this session. When it passes, the 8 token
greedy continuation after an unpark is identical to never having parked and the host RSS
delta across the round trip is 0 KiB while the state is 26.4 MiB. When it fails, the unpark
cannot restore the recurrent state, and the failure is a real open bug, not a flake in the
check.**

An earlier version of this document reported a single run as a verdict. That was wrong: the
failure is intermittent, and it showed up only when the check was run repeatedly. Details in
"Known intermittent failure" below, before the rest of the results, because it is the most
important thing here.

## What was added

| piece | note |
| --- | --- |
| `llama_state_seq_save_file_direct` | public, mirrors `llama_state_seq_save_file` |
| `llama_state_seq_load_file_direct` | public, mirrors `llama_state_seq_load_file` |
| `llama_context::state_seq_*_file_direct` | members, so the state streams through the sink |
| `llama_io_read_direct::read_prefix` | reads block 0, the file header |

The point of the new pair rather than reusing the existing one is that the existing pair hands
the whole state to the caller in a host buffer. A host buffer the size of the session makes the
RSS assertion meaningless, so the state has to stream device to file through the sink, which is
what `state_seq_write_data(io, ...)` does when `io` is the direct writer.

File layout, two files' worth of rules in one file:

    block 0        [u32 magic][u32 version][u32 n_token_count][u32 crc]  padded to a block
    block 1..      framed stream: tokens[], then the serialized state

## Known intermittent failure

Roughly one run in five, the unpark fails:

```
state_read_meta: cell_count = 604, dest_seq_id = 0
state_read_data: mismatched s row size (1048576 != 0, layer 18)
llama_state_seq_load_file_direct: error loading sequence state file: failed to restore kv cache
```

The destination computes a row size of 1048576 for layer 18 of the recurrent state, and the
value read from the file is 0. What was established:

| observation | value |
| --- | --- |
| failure rate | 3 of 12 runs, and 1 of 6 in another batch |
| file size, passing vs failing | identical, 28096000 |
| block framing, failing run | every block header valid, 54874 blocks |
| structure across runs | identical block count and identical used byte total |
| arena before the unpark | empty, `pos_max = -1` |
| retrying the same file | **also fails**, so the file is bad, not the destination |
| the failure is a row size mismatch, not a type mismatch | the stream stays aligned, so it is not a bad block |

The last two rows are the interesting ones. A retry from the same file failing means the bytes
on disk are wrong for that run, not that the destination was transiently occupied. And the
mismatch being a *row size* rather than a *type* means the reader is consuming the stream in
the right places: a desync or a rejected block would trip the type check first, or set the
reader's own error.

**Where the wrong bytes come from was established later, and it is not the engine.** Serializing
the same state twice into a host buffer is always identical, while writing it twice to a file
through the sink is not (`docs/research/22-state-serialization-not-reproducible.md`). So the
corruption enters on the file path. This document originally attributed it to the engine's
recurrent serialization; that was wrong. The park feature is still blocked, because the park
path *is* the file path.

What is not established is which of the two file path suspects it is: the sink, or the buffered
read-back used to compare the files, since the comparison itself reads `O_DIRECT` written files
through `fopen`. The test that separates them is in
`docs/research/22-state-serialization-not-reproducible.md`: write a fixed buffer through the sink
twice and compare, which needs no model.

This blocks the feature. An intermittent restore failure that loses a session is worse than no
restore at all, so the next step is to root-cause it rather than to build on top of it.

## Result, on the runs that pass

604 token prompt, 8 greedy tokens generated after each run:

| check | result |
| --- | --- |
| park wrote state | ok |
| store file padded to blocks | ok |
| unpark read state | ok |
| token count round tripped | ok |
| **tokens round tripped** | ok |
| **continuation identical to never parking** | **ok** |
| **rss delta across park and unpark** | **0 KiB**, while the state is 26993 KiB |

| quantity | value |
| --- | --- |
| logical bytes parked | 27640948 (26.36 MiB) |
| bytes on disk | 28096000 (26.79 MiB) |
| framing overhead | 1.6% |
| prompt tokens | 604 |

The last row of the first table is the acceptance test. Greedy decoding means the comparison is
exact rather than statistical: the same token ids, 8 of 8, with the KV, the SSM convolution
state and the recurrent state all having made a round trip through the disk in between.

## The size law is confirmed independently

`docs/research/17-device-to-disk-path-results.md` derived a law from two lengths on the same
model: `19.3 MiB + 12.0 KB per token`. At 604 tokens that predicts 26.38 MiB. Measured here:
26.36 MiB.

The law was fitted on a different code path, the shipping prompt cache, and this is a different
implementation writing through a different medium. Agreement to 0.1% is a real cross check on
both.

## What this does not cover

- **One sequence, one context, no concurrency.** The parking is not yet driven by a policy or
  by the server.
- **The unpark still needs the whole state to fit in the arena**, which is the capacity finding
  from E0. Parking frees memory, but restoring a long session still requires room for it, and
  nothing here changes that.
- No eviction policy, no disk budget enforcement, and no reclaim.
- The pages file path (the sink's plain mode) is still unexercised, as are `write_tensor` and
  `read_tensor` directly.
- f2fs and Linux only. Windows is a hard `#error`.
- Two bugs in this check itself, both mine and both worth naming because they cost time:
  `llama_batch_init(n, ...)` treats its first argument as the allocation size and leaves
  `n_tokens` at 0, and a batch larger than `n_batch` is rejected. Neither was a fault in the
  code under test.
- **The intermittent failure above is not fixed.** The check is therefore not yet a usable
  regression test: it will fail roughly one run in five for reasons unrelated to any change
  under test.

## Reproduce

```sh
g++ -O2 -std=c++17 -I include -I ggml/include \
    scripts/research/kvstore/park_disk_check.cpp \
    -L build-cpu/bin -lllama -lggml-base -lggml \
    -Wl,-rpath,$PWD/build-cpu/bin -o /tmp/park_disk_check

/tmp/park_disk_check ~/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf \
                    /tmp/park.bin 400
```

About 8 seconds. Set `PARK_VERBOSE=1` to see the engine logs.
