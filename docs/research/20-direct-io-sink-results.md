# Direct I/O sink: results

Date: 2026-09-27
Machine: Ryzen 7 6800H, `/dev/nvme1n1p2` f2fs
Code: `src/llama-io.h`, `src/llama-io.cpp` (`llama_io_write_direct`, `llama_io_read_direct`,
`llama_io_crc32`)
Checks: `scripts/research/kvstore/direct_sink_check.cpp` (C++),
`scripts/research/kvstore/cross_check.py` (Python)
Spec: `docs/superpowers/specs/2026-09-27-session-kv-store-design.md`
Plan: Task 4, step 1

**Verdict: the sink exists, it obeys the residency policy (RSS delta 0 KiB over 200 records),
and two independent implementations agree on the bytes it produces. The Python cross check
found a real bug in the first version: the sink split a logical record across a block
boundary, which makes the record unreadable.**

## What was added

| piece | note |
| --- | --- |
| `llama_io_crc32` | zlib polynomial and initial and final xor, so C++ and Python produce identical bytes |
| `llama_io_write_direct(path, framed)` | one reused aligned block, 512 bytes |
| `llama_io_read_direct(path, framed)` | streams one block at a time |

Two layouts, because the two files need different things:

- `framed = true` packs a byte stream into blocks with `[u32 used][u32 crc32(payload)][payload]`.
  This is the status log, where a bare record stream cannot tell its own padding apart from a
  torn tail.
- `framed = false` writes whole blocks as given. This is the pages file, where each slot is
  already padded to a block multiple by its own layout.

The sink owns the block framing because under `O_DIRECT` the medium's write granularity and
the format's framing granularity are the same thing. There is no smaller unit to hand out.

`write_tensor` and `read_tensor` are chunked at one block, so a tensor transfer does not
allocate host memory proportional to the tensor.

## Result

`direct_sink_check`, 200 records through the sink:

| check | result |
| --- | --- |
| writer good after flush | ok |
| logical bytes match | ok, 8000 |
| sink output | 200 records, 17 blocks |
| **rss delta over the write** | **0 KiB** while the block buffer is 512 bytes |
| records read back byte exact | ok |
| read byte count matches | ok |
| zeroed block header refused | ok |
| short tail detected as short | ok |

The RSS number is the residency policy check, and it is the one that fails silently: a sink
that accumulates the stream in memory first would pass every other check here.

`cross_check.py` then reads the file the C++ sink wrote, using `sim.parse_blocks`:

| check | result |
| --- | --- |
| size is a whole number of blocks | ok |
| status base parsed from the C++ header | ok |
| block stream parsed to the end | ok |
| record count matches | ok |
| epochs, ops and args match Python | ok |
| payload stream equals the records, byte for byte | ok |

That last row is the strongest form: the concatenated payload areas of every block equal the
record bytes, in order, with no gap and no extra. Two implementations, one format.

## The bug the cross check found

The first C++ version filled each block until it was byte full and then carried the remainder
into the next block. A 40 byte record landing 24 bytes from the end was therefore split:

| | first version | after the fix |
| --- | --- | --- |
| blocks for 200 records | 16 | 17 |
| payload bytes | 8000 | 8000 |
| `parse_blocks` clean | **no** | yes |
| records recovered | fewer than 200 | 200 |

The stream was intact, so the naive payload comparison passed, and a per record comparison
over the parsed prefix passed too. Only the block walk exposed it: `used` ended mid record,
so the reader hit a record that ran past the end of its block and stopped there.

The fix is that a logical write never straddles a block boundary: if a record does not fit in
the space left, the block is sealed first. That is exactly what the Python harness already
did, and after the fix the two pack 200 records into the same 17 blocks.

Worth stating plainly: the byte level check I would have written by hand passed on the broken
version. It was the second implementation, not the first, that caught it.

## What this does not cover

- **Not wired into the engine.** Nothing calls the sink yet. Task 4 steps 2 and 3 are open:
  park a real sequence to disk, unpark it, and assert generation is unchanged and RSS does not
  grow with context length. Until that runs, this is a verified component, not a feature.
- The plain (pages file) mode is implemented but not exercised by any check yet.
- `write_tensor` and `read_tensor` are not exercised.
- Windows is a hard `#error`, deliberately, rather than a silent fall back to buffered I/O
  that would break the residency policy.
- Multi-session concurrency, and behaviour under sustained write load.

## Reproduce

```sh
# C++ sink and its controls
g++ -O2 -std=c++17 -I src -I include -I ggml/include \
    scripts/research/kvstore/direct_sink_check.cpp src/llama-io.cpp \
    -L build-cpu/bin -lggml-base -lggml -Wl,-rpath,$PWD/build-cpu/bin \
    -o /tmp/direct_sink_check
/tmp/direct_sink_check

# C++ writer, Python reader
python3 scripts/research/kvstore/cross_check.py
```
