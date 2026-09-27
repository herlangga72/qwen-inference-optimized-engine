# State serialization is not reproducible: results

Date: 2026-09-27
Tree: `qwen-only-backends`
Code: `src/llama-memory-recurrent.cpp` (`state_write_data`), reached from
`llama_state_seq_get_data_ext` and from `llama_state_seq_save_file_direct`
Check: `scripts/research/kvstore/park_disk_check.cpp`
Corrects: `docs/research/15-e0-park-unpark-results.md`, which attributed an identical failure to
arena capacity

**Verdict: serializing the same unchanged state twice to a host buffer is reproducible, 0
differing bytes in 5 of 5 runs. The same state written twice to a file through
`llama_io_write_direct` differs, 4 of 5 runs, at a few KB starting at 4 KiB aligned offsets,
with an identical total length. So the instability is on the file path: the sink, or the
read-back this check uses to compare the two files. It is not in the engine's serialization.**

This reverses the first version of this document, which said the engine's recurrent state
serialization was at fault and that the sink was cleared. The measurement that reverses it is
direct: `llama_state_seq_get_data_ext` twice into two host buffers, nothing else involved. Earlier
I inferred the attribution from cross-process file comparisons, which could not separate the
engine from the file path. That inference was wrong and this document replaces it.

**Then a further measurement made even that conclusion unsafe.** Comparing the two files by the
*payload area of each block only*, ignoring the padding the reader does not read, showed 1955
differing payload bytes between two saves. So the difference is real data, not ignored padding,
and in that same run the unpark failed. That puts the sink back in scope, and it also raises a
third possibility that explains the host result: the host comparison makes two serializations
back to back in microseconds, while the file comparison makes three separated by disk I/O. If
the state mutates with time rather than with the medium, only the slower test would catch it.

So the honest position is: the engine's serialization is stable when measured back to back, the
file payload is not stable when measured across disk writes, and the test that separates
"the sink is wrong" from "the state changes over time" has not been run yet. Read the sections
below knowing that I asserted the first two verdicts of this document in turn and had to
withdraw each.

**The deciding test has now been run, and it settles it.**

I instrumented the sink itself: for every tensor window it is handed, keep the first bytes and
compare them when the same window is handed again on a later save. The result:

```
SINK INPUT DIFFERS: tensor 0x… off 0 size 73728 n 504 (diff 36)
```

36 differences, once per window, in every run. **The sink is handed different bytes for the same
window on different saves of an unchanged state.** So the sink is cleared, and the instability is
upstream of it: the tensor contents that the serialization reads are not stable between saves.

## Why the host comparison could not see this

`llama_io_write_host::write_tensor` does not read the tensor during the walk at all. It queues the
window and gathers every window at once in its destructor. So every host comparison in this
document sampled the state at one quiet instant after a walk, and the two instants agreed. The
sink reads each window inline, spread across a walk that also writes about 27 MB to disk, and
that is the only reader that sees the tensor change.

That also means the earlier "host serialization is reproducible, 5 of 5" rows are true but
narrow: they prove the state is the same at the end of two walks, not that it is stable during one.

## Confirmed: the tensors change during the walk

The next instrument captured each window as it was serialized and re-read it at the end of the
same walk, in `flush()`:

```
WALK CHANGED: tensor 0x… off 0 size 73728 n 504
WALK CHANGED TOTAL: 36 of 60 windows
```

36 of 60 windows differ between the read during the walk and the re-read at the end of the same
walk. Every run, without exception. So the tensors are being written while the state that
contains them is being serialized.

That is the root cause, and it explains the whole shape of this investigation: the file bytes
are a mixture of before and after a mid-walk write, the length is unaffected because sizes come
from the structure rather than the content, and any reader that defers its reads, like the host
path, sees a consistent snapshot and reports no problem.

The changed windows have `size 73728`, which is the R tensor window from the earlier read trace,
so the recurrent state tensors are among those being written. 36 of the 60 captured windows is
more than the 24 layers, so it is not only R.

What is not yet known is who writes them. The candidates, in order of how well they fit: an
in-flight compute or copy that the save path does not wait for, and the recurrent rollback
snapshot machinery the writer's own comment mentions.

## Test B result: the recurrent tensors are NOT the ones changing, and this narrows it a lot

The instrument moved into `state_write_data`, keyed by `(tag, layer, offset, size)`, comparing
against the previous walk:

```
RCUR CHANGED tag=r il=0 off=0 size=73728
RCUR CHANGED TOTAL: 0 of 36 windows      <- 10 walks
RCUR CHANGED TOTAL: 36 of 36 windows     <- 1 walk
```

Read that carefully, because the first conclusion I drew from it was wrong and the numbers say
the opposite. **Ten walks report no change at all in the recurrent windows.** The single walk
that reports all 36 is the one that follows the unpark in the same process, and the unpark
legitimately rewrites the whole state from the file. So that 36 of 36 is correct behaviour, not a
finding.

Two things follow:

1. **The recurrent tensors are stable across consecutive saves.** That rules them out, and it also
   rules out the rollback snapshot machinery that I had been treating as the leading candidate.
2. The changed layers are only `r`, only the delta-net ones (0, 1, 2, 4, 5, 6, 8, 9, 10: the
   full-attention layers 3, 7, 11 are absent, matching `full_attention_interval = 4`), which is a
   useful map of which tensors hold recurrent state but is not where the instability is.

So the within-walk change reported by the sink-side instrument must be in windows this
test does not cover: the **attention KV** tensors, written by `llama_kv_cache::state_write_data`.
The sink-side instrument covered 60 windows and 36 changed; this one covers only the 36 recurrent
windows of the same walk and none of them changed in the ordinary case.

Next instrument, one build: the same capture and compare inside
`llama_kv_cache::state_write_data`, which will name the attention layers exactly as this named the
recurrent ones.

## Test A, still open: is the writer time based or I/O based?

The walk that shows the change takes about
50 ms because it writes 27 MB to disk; the host path's walk takes about 1 ms and defers its
gathers. If the mutation simply continues with time, then a slow walk sees it and a fast one does
not, with no writer tied to the I/O. Test: in `write_tensor`, sleep 50 ms once, after the 30th
window, then let the existing end-of-walk comparison run. If the windows captured before the
sleep have also changed, the mutation continues during an idle sleep and is time based. If only
windows after the sleep changed, it is tied to the writing.

**Test B, which tensors, exactly?** The changed windows report `size 73728`, which is the R
window from the read trace. Print the tag and layer alongside the offset in the end-of-walk
comparison, which the sink does not know but the writer does: do the capture and the comparison
in `state_write_data` instead of in the sink, keyed by `(tag, il, off, size)`. That names the
layer and the tensor kind in one run and costs the same as the instrument already run.

Both tests are edits to the code described above and take one build and one run each. I stopped
here rather than start them, because the location is already recorded and the value of another
half finished instrument cycle is low.

## Instrument defect found, which invalidates the two results above

Both capturing instruments, the sink-side one and the two writer-side ones, stored only the
**first 504 bytes** of each window (`take = min(504, size)`) and compared only those. A window is
up to 1 MiB. Both would therefore report zero for any change past the first chunk.

And the file difference is still real. With the instrumentation reverted and the tree clean, 6
runs of the check:

| run | raw difference 1v2 / 2v3 | payload difference 1v2 / 2v3 |
| --- | --- | --- |
| 1, 2, 4, 5 | 0 / 0 | 0 / 0 |
| 3 | 0 / 6099 | 0 / 12 |
| 6 | 3560 / 6101 | 7 / 12 |

So the difference persists, it is small in the payload (7 and 12 bytes) and large in the raw
comparison (thousands of bytes), which means most of it is outside the used payload area and some
of it is inside.

Therefore: **`RCUR CHANGED TOTAL: 0 of 36` and `KV CHANGED TOTAL: 0 of 12` do not rule out those
tensors. They only rule out the first 504 bytes of each window.** I reported them as eliminations
and that was wrong, for the same reason as everything else in this document: I read a count
without checking what the instrument actually covered.

## The instrument that would have worked

Hash the whole window, not its first chunk. In `write_tensor`, run a rolling hash over every
chunk of the window as it is written and store `(tensor, off, size) -> hash`, then compare hashes
at the start of the next walk. Eight bytes per window, full coverage, and cheap. Every instrument
so far has had this hole and it is why four separate hypotheses were "confirmed" and then died.

## Whole-window hash result: the source is still stable, and the contradiction stands

The whole-window hash ran, and it reports 36 differences per run. The arithmetic matters here:
the registry reports each differing window once per comparison round, and there are 36 R windows,
so 36 diffs is **one round**, not four. The rounds are: save 1 to 2, 2 to 3, 3 to 4, and 4 to 5.
The only round with a state change in between is the last one, because the unpark runs before
save 5 and legitimately rewrites the whole state. So this is the post-unpark round again, and the
three rounds that matter, consecutive saves with nothing between them, are clean.

So after four instruments and one correction of the method:

| fact | how it was measured |
| --- | --- |
| the source windows are stable between consecutive saves | whole-window hash, 3 consecutive rounds |
| the sink is deterministic for a fixed buffer | 200 records written twice, byte identical, 6 of 6 |
| the file still differs between saves sometimes | 2 of 6 runs, payload 7 and 12 bytes, raw thousands |
| the `write_tensor` path is not covered by the determinism test | that test calls `write` directly with 40 byte records |

Those four cannot all hold. Identical source bytes through a deterministic sink must give
identical files. Since the first two are measured and the third is measured, the error must be in
my premise about what "the source bytes" are, and the untested component is exactly the one the
table names: **`write_tensor`**. The determinism test proves `write` is deterministic for a fixed
buffer; it says nothing about `write_tensor`, which is the only path the park actually uses.

That is the next test, it needs no model and no engine code: add a `write_tensor` case to
`direct_sink_check.cpp` that writes the same tensor twice and compares, which requires a tensor,
so use a small CPU tensor built with ggml directly.

## write_tensor is deterministic, so both sink paths are, and the hash count was confounded

Added to `direct_sink_check.cpp`: build a 400 KB CPU tensor with ggml, write it twice through the
sink via `write_tensor`, compare block by block.

```
same tensor written twice    blocks=794 first_diff=-1 differing=0 (tensor is 400000 bytes)
```

Six of six runs, zero differing bytes, on a window that spans 794 blocks and therefore exercises
the chunking and the block boundary logic. **Both sink paths are deterministic.**

That kills the last hypothesis and forces the opposite reading of the whole-window hash result.
Recall the counts: four comparison rounds, and 36 diffs equals one round because there are 36 R
windows. I said that round was the last one, the post-unpark one. But the file difference in the
failing runs is between `2v3`, which is the round between save 2 and save 3, with no unpark
anywhere near it. So the single 36-diff round could be either the 2v3 round or the 4v5 round, and
**I assumed the one that let me keep my conclusion.**

That is the fifth time on this bug that I read a count as agreeing with my hypothesis without
checking which case it covered. The instrument was right; the reading was not.

## The experiment that removes the ambiguity

Run the hash instrument with **no unpark in the process at all**: a check that only decodes once
and then saves four times. Then every comparison round is a consecutive save, and the diff count
directly answers whether the source changes between saves. If it reports 0 across all rounds, the
source is stable and the file difference has to be in the medium or the comparison. If it reports
36 in some round, the source is unstable and the round tells you where.

That is a small edit to `park_disk_check.cpp`: guard the unpark and the isolated RSS block behind
a flag, and run with it off.

## An earlier proposal, now superseded

This section used to propose comparing the sink's bytes for a window against a host snapshot of
the same window. That has since been done better: the end-of-walk comparison above is the same
idea without needing to locate the window in the host blob, and it is what produced the 36 of 60
result. Kept as a note so the ordering of the work is readable.

An account of how many times the attribution changed, since it affects how much weight to put on
any single claim here: engine serialization, then the file path, then the sink, then upstream
again. Each change was a measurement replacing an inference that could not separate two things.
The confirmed facts are the walk-changed result and the ruling-out table; the attributions
between them were mine and were wrong more than once.

Rate over batches, which varies with environment: 5 of 6, 6 of 12, 3 of 12, 4 of 4, 7 of 8 file
differences; the host-buffer comparison has been clean every time it has run.

## The repro

The check saves the same state twice into two files and compares them byte for byte:

| run | sizes | first difference | differing bytes |
| --- | --- | --- | --- |
| 1 | 27640948, 27640948 | none | 0 |
| 2 | 27640948, 27640948 | 6901764 | 5025 |
| 3 | 27640948, 27640948 | 204804 | 14666 |
| 4 | 27640948, 27640948 | 5242880 | 5600 |
| 5 | 27640948, 27640948 | 8384512 | 10159 |
| 6 | 27640948, 27640948 | 25653252 | 2470 |

Nothing between the two saves mutates the state. Same context, same tokens, same KV, same
recurrent state. The length is identical every time, so the writer is emitting the right
number of bytes and the wrong content.

Note the first differences: 204804, 5242880, 8384512, 25653252 and 6901764 are all multiples
of 4096 plus 4. The plus 4 is the block header crc field, which differs first because it
covers the payload. So the differing region begins on a 4096 byte boundary, which is the
allocation granularity, not the 512 byte block granularity of the file.

Saving three times, mostly all three differ, and the pairs differ in different places:

| run | 1 vs 2 | 2 vs 3 |
| --- | --- | --- |
| 2 | first 16297988 | first 8921092 |
| 3 | first 4173828 | first 4173828 |
| 5 | first 7548932 | first 7548932 |
| 8 | first 135172 | identical |

So it is not a first-save side effect that later saves settle into: each save independently
picks up different bytes. Run 8 also shows two adjacent saves agreeing while the earlier pair
did not, so the nondeterminism is not simply "every save differs".

## What this explains

Across processes, the same effect shows up as a clean correlation:

| observation | value |
| --- | --- |
| runs whose file matched the deterministic good bytes | 6 of 6 passed |
| runs whose file differed | 6 of 6 failed |

So the load failures are fully explained by the file, and the read path is not at fault. The
observed failure is:

```
state_read_data: mismatched s row size (1048576 != 0, layer 18)
```

The destination computes a 1 MiB row for layer 18 and the value read from the file is 0. A
desync remains the likely mechanism rather than a stored zero: `GGML_TYPE_F32` is 0 and zeros
are common in the stream, so a misaligned read passes the type check and then finds a zero
where the row size should be. An earlier write-up of mine claimed the type check passing ruled
out a desync. That was wrong.

## Why it matters beyond this feature

`state_seq_write_data` is the same path behind `llama_state_seq_get_data_ext` and
`llama_state_seq_set_data_ext`. Those back the server's `--cache-ram` prompt cache and
`--slot-save-path`. So the shipping session parking has the same latent unreliability, and it
would show up as a random failure to restore a slot rather than a clean error.

It also invalidates an earlier conclusion of mine. E0 compared a failing 2048 context against a
single successful 8192 context and concluded the limit was arena capacity, not the mechanism.
With a failure rate near 50% from this unrelated bug, one successful run at 8192 is not evidence
of anything. **E0's capacity conclusion is not established and needs repeating runs.** The
2048 arithmetic (1274 + 1339 > 2048) is still true as arithmetic; it is just no longer
established as the reason the restore failed.

## Where it is

`llama_memory_recurrent::state_write_data`. The shapes that matter:

```cpp
for (uint32_t il = 0; il < n_layer; ++il) {
    if (s_l[il] == nullptr) continue;            // null layers are skipped
    const uint64_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
    io.write(&s_size_row, sizeof(s_size_row));
    for (const auto & range : cell_ranges) {
        io.write_tensor(s_l[il], range.first * s_size_row, (range.second - range.first) * s_size_row);
    }
}
```

The header comment says the logical current state may live in a rollback snapshot plane, and
`cell.src` is used to pick the source row:

```cpp
const uint32_t cell_id = rs_idx_cur * size + (cell.src >= 0 ? cell.src : (int32_t) i);
```

## What has been ruled out on the way to the cause

Each of these is a hypothesis that was tested and killed, with the instrument noted, so the
remaining search space is smaller than the list of guesses it replaced.

| hypothesis | test | result |
| --- | --- | --- |
| the source row is outside the tensor, so the writer serializes whatever follows the buffer | bounds check on every `write_tensor` in the recurrent writer, 6 runs including 4 failures | never fired, so every read is in range |
| the tensor is being written while it is copied | read the same 512 bytes twice inside one call and compare | zero differences in 12 calls across 4 runs |
| the range selection differs between saves | log `(tag, layer, offset, size)` for every read, then diff the two saves | identical, 36 reads, same offsets and sizes and order in all four saves |
| a first save has a side effect that later saves settle into | save three times and compare every pair | no, the pairs differ in different places |

That leaves one shape: **the same offsets of the same tensors yield different bytes across
saves, and something changes the tensor contents between the save calls.** The recurrent writer
has been cleared of reading the wrong place or racing the copy, so the next test is to capture
the bytes of a few reads during the first save and compare them on the next save. If they
change, the memory is being modified between calls. If they do not, the difference lives in the
attention writer, which was not instrumented and which the earlier offset analysis also
implicated (one differing region fell at 4.2 MiB, inside the attention part rather than the
recurrent part).

## What is not at fault

| candidate | ruled out by |
| --- | --- |
| the sink | it is deterministic for fixed input, 6 of 6, and the input trace shows it is handed different bytes |
| the read path of a restore | a retry of the same file also fails |
| arena capacity | the arena is empty before the unpark, `pos_max = -1` |
| the recurrent writer reading the wrong place | bounds check on every serialized read, silent over 6 runs |
| the tensor being written during a single copy | double read inside one call, 0 differences |
| the range selection differing between saves | per read trace identical across 4 saves |
| the write not being durable when compared | `fsync` before the comparison changed nothing |
| page cache incoherence on the read-back | `fadvise(DONTNEED)` changed nothing |
| the sink's payload assembly | fixed input written twice is byte identical |
| the state drifting over seconds | host serializations 2 s apart identical |
| the saves mutating the state permanently | host serialization before and after the saves identical |

## The test that has not been run, and should be next

Serialize to two host buffers with a delay or a disk write between the two calls. If they then
differ, the state changes with time and the medium is irrelevant, which would explain why the
host comparison has been clean every time: it is simply too fast to catch it. If they still
agree, the sink is the fault and the fixed-buffer test below decides it in a second.

That single test resolves the contradiction between "host buffers are reproducible 5 of 5" and
"file payload differs".

## The suspects that remain

`llama_io_write_direct`, confirmed by the table above. The state is provably unchanged across
the writes that differ, so nothing upstream is implicated. Every other candidate has been killed
with an instrument: see the ruled-out table.

One incidental discovery on the way, which is why the host path looked so much calmer than the
sink: `llama_io_write_host::write_tensor` does not read the tensor at all during the walk. It
queues the window and gathers every window at once in its destructor. So the host path takes a
single tight snapshot at the end, while the sink reads interleaved with writing tens of MB to
disk. That difference in timing does not explain this bug, since the state is provably constant,
but it is worth knowing when comparing the two paths again.

## The next step, which is now a one second debug loop

In `direct_sink_check.cpp`, which already compiles `src/llama-io.cpp` and needs no model: write
the same fixed byte buffer to two paths through the sink and compare, then walk the two files
block by block and print the first block whose payload differs together with both `used` values
and both crcs. A fixed buffer removes the tensor path from the question and turns this into a
reproducible case that can be stepped through, which is what this investigation has been missing
all along. The observed signature to explain is a payload difference of a few bytes to a few KB
at a random position, with an identical total length.

## If it is the sink

The most likely shape would be the reused block buffer not being fully overwritten between
flushes, since everything after the first differing block in a file would then be shifted or
stale. Note that this is a guess and the first test above is what decides it.

## Reproduce

```sh
g++ -O2 -std=c++17 -I include -I ggml/include \
    scripts/research/kvstore/park_disk_check.cpp \
    -L build-cpu/bin -lllama -lggml-base -lggml \
    -Wl,-rpath,$PWD/build-cpu/bin -o /tmp/park_disk_check

for i in 1 2 3 4 5; do
  /tmp/park_disk_check ~/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf /tmp/p.bin 400 \
    | grep 'two saves'
done
```

Roughly five of the five will report a non zero `first_diff`. When all five agree, the state
serialization is reproducible and this bug is gone.

---

## Correction: the writer is exact, and the file has holes

Date: 2026-09-27 (later the same day)
Method: `scripts/research/kvstore/validate_file.py`, which shares no code with the C++ reader

Everything above this line is now superseded. The file difference is real, but it is not a
framing difference and it is not in the engine. It is whole blocks of zeros on disk at offsets
the writer wrote and accounted for.

### What the writer does

With a per save stream stamp and a running count of framed bytes, over 4 saves x 4 runs:

    stream gen=1..4  hash=798381547129944045  logical=27640948  framed=27640948  match=1

Every save is handed the same bytes, and the writer frames every one of them. The writer is not
losing anything and it reports no error.

### What the file contains

`validate_file.py` parses the file independently and recomputes every block crc with `zlib`:

| run | file | payload | crc failures | zero frames | zero runs |
| --- | --- | --- | --- | --- | --- |
| 1 | all three | 27640948 | 0 of 54875 | 0 | - |
| 3 | `zy2.bin` | 27635404 | 0 of 54875 | 11 | `[9761,9765]`, `[12961,12966]` |
| 3 | `zy2.bin.third` | 27637420 | 0 of 54875 | 7 | `[19472,19478]` |
| 4 | all three | 27640948 | 0 of 54875 | 0 | - |

Two separate runs of 5 and 6 consecutive zero blocks in one file, one run of 7 in another. The
payload is short by exactly `504 x zero_frames`, and 504 is one full frame. Every remaining block
passes crc.

### Why this hid for so long

A zero block parses as `used=0, crc=0`, and `crc32("") == 0`, so **a hole in the file is a valid
empty frame**. It passes validation. It only shows up as a payload that is short by a multiple of
504, and by then whatever reported it was blamed instead.

The difference is whole blocks, not bytes. That also explains the byte comparison results: when a
run of blocks is empty, every later byte is shifted, so a 3 KB hole reports as ~19 MB of differing
bytes. Those numbers are misalignment, not damage.

### What the earlier instruments got wrong

Three readings in this document were artifacts and each sent the search the wrong way:

- An in process read back added to `flush()` opened a second reader on the file being written. That
  is what produced the payloads that were short by 504 and the 24 MB diffs. A diagnostic was
  corrupting the file it was measuring.
- The window hash covered only `write_tensor`, so the metadata written through `write()` was never
  hashed. It reported reproducibility it could not see.
- The stream hash accumulated across saves because it was never reset, so it reported a difference
  on a run whose files were identical.

### Consequence for the format

**The reader already handles this and the first version of this section was wrong.** `next_block()` in
`src/llama-io.cpp` rejects `used == 0` with `EBADMSG` and validates the crc before the block is used,
so a hole is already a loud error at the exact block where it occurs. Nothing accepts it as data.

The Python validator hid this because it parses the format literally and treats `used=0, crc=0` as a
legitimate empty frame. That is a gap in the validator, not in the store, and it is why the holes
were only visible as a short payload.

So the earlier restore failures with `row size 0` were the engine reporting a hole correctly. The end
to end behavior is right: the data was damaged on the way to disk, the store refused to restore it,
and the failure surfaced as an error instead of a wrong answer. That is the right outcome and it is
worth stating clearly, since several earlier sections of this document read as though the store
accepted the damage.

What is left is policy, not format. When a save hits a hole the store currently just fails. The
options are to retry the save, to keep the previous good checkpoint, or to treat the slot as lost.
Because the writer is exact, a retry is worth trying first.

### Where the loss happens

The writer is exact and reports no error; the blocks are zeros on disk. The remaining suspect was
the `O_DIRECT` write path, so that was tested directly in
`docs/research/23-write-path-loses-blocks-results.md`. The short answer: it is real but rare, and it
did not reproduce under controlled conditions, so it is still unattributed.
