# The write path loses blocks: results

Date: 2026-09-27
Tree: `qwen-only-backends`
Code: `scripts/research/kvstore/post_write_check.py`
Follows: `docs/research/22-state-serialization-not-reproducible.md`
Volume: `/dev/nvme1n1p2`, 461G, 426G used, 35G free, 4 KB native block size

**Verdict: the loss is real and intermittent, and it did not reproduce under controlled
conditions. Over 180 files of 28 MB written by the probe, every one came back byte exact. During
the original park runs the same file lost contiguous runs of blocks in roughly one save in three.
The probe therefore cannot attribute the loss to `O_DIRECT`, to `fsync`, or to the block size, and
the first two probe rounds that did show damage were confounded with time.**

## Method

`post_write_check.py` writes a 28 MB file the way the sink does, one aligned `pwrite` per block,
then reads it back with buffered reads and compares every block against a per block pattern. It
runs each case N times and reports runs that came back different, classifying each damaged block
as zeros, the previous block, the next block, or a partial match.

## Results

| round | case | damaged | blocks |
| --- | --- | --- | --- |
| 1 | 512 direct, fsync | 1 of 10 | 4 |
| 1 | 512 direct, no fsync | 4 of 10 | 28 |
| 1 | 512 buffered | 0 of 10 | 0 |
| 1 | 4096 buffered | 0 of 10 | 0 |
| 2 | 512 direct, fsync | 3 of 10 | 32 |
| 2 | 512 direct, no fsync | 0 of 10 | 0 |
| 2 | 4096 direct, fsync | 0 of 10 | 0 |
| 2 | 4096 direct, no fsync | 0 of 10 | 0 |
| 2 | 512 buffered | 0 of 10 | 0 |
| 2 | 4096 buffered | 0 of 10 | 0 |
| 3 | 512 direct, fsync | 0 of 30 | 0 |
| 3 | 512 direct, no fsync | 0 of 30 | 0 |
| 3 | 4096 direct, fsync | 0 of 30 | 0 |
| 3 | 4096 direct, no fsync | 0 of 30 | 0 |
| 3 | 512 buffered | 0 of 30 | 0 |
| 3 | 4096 buffered | 0 of 30 | 0 |

The damage in rounds 1 and 2 forms contiguous runs, for example `41808-41811` and `49112-49118`,
matching the shape seen in the park files. The first probe report found 3 zero blocks in one run
and, in another run, 11 blocks whose content was neither zero nor the expected pattern, most of
them matching only a few bytes out of 512.

`fsync` does not help: damage appears with it and without it, in both rounds.

## Why I cannot conclude anything from this

The case order is fixed, and buffered runs last. In rounds 1 and 2, every damaged run is in an
early case and every buffered case is late and clean. So mode and time are confounded: the damage
may have simply stopped, and the clean buffered results may reflect when they ran rather than what
they did.

Round 3 is the counterexample. Thirty iterations per case, same order, 180 files, and nothing was
damaged anywhere, including `512 direct` which showed the most damage earlier. So whatever caused
it was not present during round 3.

Total damage across all rounds: 8 of 70 runs for 512 direct, 0 of 50 for 4096 direct, 0 of 50 for
buffered. Those counts look like a signal for `O_DIRECT` at 512, and the confound makes them
untrustworthy. Reporting a conclusion from them would repeat the mistake this document exists to
avoid.

## What is actually established

- The engine's writer is exact. It frames every byte it is handed and reports no error. Proven by
  a per save stream stamp and a framed byte count over 4 saves x 4 runs, all identical.
- Files on disk did contain contiguous zero runs. Proven independently in Python, with run
  positions, in `docs/research/22-state-serialization-not-reproducible.md`.
- The store detects this. The reader rejects `used == 0` with `EBADMSG` and checks every crc, so a
  hole is a loud error at the exact block and never restored as data.
- The loss is rare, silent, and not attributable to block size, `fsync`, or `O_DIRECT` with the
  evidence gathered so far.

## What to do about it

The store already fails safe, so this is a policy question rather than a correctness one. Saving
twice and comparing would turn a silent loss into a detected one at the cost of double the write
bandwidth, which is acceptable for a park that happens once per session eviction. A retry on the
`EBADMSG` path is cheaper and probably enough, given the rate.

To attribute it properly, the probe needs to interleave the cases, run the cases in a random order
per iteration, and run it during the same load that showed the loss, since round 3 was idle and
clean. Anything else risks the same confound.

---

## The lost blocks contain a deleted file's data

Date: 2026-09-27 (later again)
Repro: `/tmp/park_disk_check <model> <out> 0 8192`, which fills one context, then
`scripts/research/kvstore/validate_file.py <out>`

A max context run reproduced the loss and, for the first time, showed what the damaged blocks
already contained. They are not zeros:

    bad block bytes 0..48 : b'":"endif","file":"/usr/share/cmake/Modules/CMake'
    bad  used=1696741922 crc=0x6669646e   good used=504 crc=0x8c80605d

That is CMake build output, from `CMakeDetermineCompilerId.cmake`. No process involved in this
test ever had that text in memory, so it was not copied from the writer's buffer.

### What it means

The write is lost, and a block that was never written reads back as whatever it held before. The
filesystem does not zero blocks when it allocates them, so a freshly created file whose write is
dropped reveals the previous occupant of that block, which is typically a deleted file.

This is consistent with everything measured so far and explains the shape of it:

- Earlier runs showed runs of zero blocks. Those were the same failure where the previous occupant
  of the block happened to be zeroed.
- One damaged file in three, whole blocks at a time, contiguous runs of five to seven, never a torn
  prefix of a frame.
- The good blocks always pass crc, so the damage is never a byte level tear inside a written frame.

So this is a storage layer failure, not a buffer or a framing bug, and it is now pinned by
content rather than by inference.

### Two consequences

1. Zero is not a sentinel. A damaged region cannot be detected by looking for zeros, because the
   previous occupant decides what it looks like. The store's detection has to stay structural, which
   it already is: `used == 0` and every crc are both rejected.
2. The store file can contain bytes from other files that were deleted earlier. On a shared volume
   that is a disclosure path, and it is worth saying out loud even though it is a property of the
   filesystem rather than of the store.

### Observed end to end at max context

With one session at `n_ctx 8192`, `K q8_0`, `V planar3_0`:

| check | result |
| --- | --- |
| park size | 56732452 logical bytes, 57646080 on disk |
| three saves | 56732452, 56732452, 56732452 |
| saving the same state is reproducible | FAIL, `1v2 first=8859648 diff=2551` |
| validator on the park file | `bad_blk=5 holes=5`, run `[17304,17308]` |
| validator on the other two | `bad_blk=0`, identical to each other |
| unpark read state | FAIL, load and retry both returned 0 |
| continuation identical to never parking | FAIL |
| rss delta, isolated park and unpark | 0 KiB on a 19795 KiB state |

So the sequence is: a write is lost, the reader rejects the file, the retry fails on the same file,
and the unpark fails. The store fails safe, refusing to restore damaged data rather than restoring
something wrong, but the session is lost. The KV types are not implicated: the two clean saves
round trip byte for byte.

The RSS result is the one thing here that is good news and it is the hard requirement: parking a
state keeps the resident set flat, at full context as well as at 400 words.

---

## The fix

Date: 2026-09-27
Check: `scripts/research/kvstore/park_disk_check.cpp <model> <out> 0 8192`, then
`scripts/research/kvstore/validate_file.py <out> <out>.second <out>.third`

Three changes, all in the store. Together they make a save either produce an intact file or fail,
which was not true before.

### 1. Block size 512 to 4096, which is what removed the loss

The volume has a 4 KB native block size, and the store was writing 512 bytes at a time through
`O_DIRECT`, which forces the filesystem into a read-modify-write of a whole device block per 512
byte write. With 512 byte blocks the generation check below reported **about ten damaged blocks out
of 112590 on every attempt, every time**. That is a loss rate near one in eleven thousand blocks,
which is high enough that a clean attempt has probability roughly `exp(-10)`: retrying could never
have converged. That is why three attempts failed back to back before.

At 4096 bytes the same check reports zero, in every run since the change. `docs/research/18` chose
512 for alignment reasons, and 4096 satisfies those too, being a multiple of 512. It also removes
most of the write amplification that document measured.

### 2. A per-save generation in the frame header

This is the part that fixes silent corruption rather than just detecting it. A lost write leaves the
block holding the previous occupant, and because the store rewrites the same path with the same
layout, that occupant is usually **an earlier save of the same file**, whose bytes form a valid frame
with a matching crc. Per-block crc cannot tell it apart, so the restore succeeded and returned partly
stale state, which is exactly the wrong continuation observed earlier.

Every frame now carries `[u32 used][u32 crc32][u32 gen]`, and the file header carries the same `gen`.
A block from a different save is rejected on read. Zero is never a valid generation.

### 3. A verify pass after every save

`llama_io_verify_direct` reads the file back block by block through `O_DIRECT`, checking `used`, the
crc, and the generation, using one reused block buffer so the check stays O(1) in host memory. If any
block is damaged the whole file is rewritten, up to three attempts, and if it still fails the save
reports 0 rather than returning success over a damaged file.

### Format change

| | before | after |
| --- | --- | --- |
| block size | 512 | 4096 |
| frame header | `[u32 used][u32 crc]` | `[u32 used][u32 crc][u32 gen]` |
| file header | 4 x u32 | 5 x u32, generation added |
| `LLAMA_STATE_SEQ_VERSION` | 3 | 4 |

Store files written before this are not readable. The payload stream is unchanged, so only the
container moves.

### One thing not to do

An `fsync` in the writer's `flush` was tried, so that the verify would read back what had actually
landed. It made things worse: with it, every attempt reported damaged blocks and no save completed in
three attempts. Without it, and at 512 blocks, ten blocks per file were still lost. So `fsync` after
`O_DIRECT` writes appears to lose blocks on this volume, and the store does not use it. Durability
rests on `O_DIRECT` plus the verify pass.

### Result

| check | before | after |
| --- | --- | --- |
| saves reproducible | FAIL | ok |
| unpark read state | FAIL | ok |
| continuation identical to never parking | FAIL | ok |
| rss delta, isolated park and unpark | - | 0 KiB on a 55437 KiB state |
| damaged blocks reported by the engine | 5 to 16 per attempt | 0 |
| independent Python validation | `bad_blk=5 holes=5` | `bad_blk=0 holes=0 gens=1 stale=0` |
| payload across the three saves | 6 blocks differed | identical, `payload_crc=50086f76` |
| overall | FAIL | pass |

The independent validator was updated for the new layout, so it parses the generation too and reports
mixed generations as stale blocks rather than trusting the crc alone.

### What is not proven

The loss is intermittent, so one clean run is weak evidence, and a confirmation set is running. The
rate and the 4096 result also need the interleaved probe that this document keeps asking for, because
every probe round so far has been confounded in some way.
