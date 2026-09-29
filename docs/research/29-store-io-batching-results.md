# Store I/O batching: results

Date: 2026-09-28
Tree: `qwen-only-backends`, working tree on top of `397b643f4`
Machine: Ryzen 7 6800H, Radeon 680M (RADV REMBRANDT), 27.1 GiB, AC connected
Volume: `/dev/nvme1n1p2`, f2fs, 4 KB native block size
Models: `Qwen3.5-0.8B-Q4_K_M` for the store checks, `Qwen3.6-35B-A3B-UD-IQ3_XXS` for the server
Harness: `scripts/research/ssdcache/block_read_probe.c`, `scripts/research/ssdcache/crc_bench.cpp`,
         `scripts/research/ssdcache/crc_verify.cpp`, `scripts/research/ssdcache/store_roundtrip.cpp`,
         `scripts/research/kvstore/direct_sink_check.cpp`
Related: `docs/research/27-ssd-prompt-cache-feasibility.md` (why this mattered),
         `docs/research/28-ubatch-determinism-results.md` (why the round trip is the right check)

**Verdict: the store's I/O path was 9.8x slower than the volume, and three separate per-block
costs were responsible. Batching the syscalls, slicing the crc32, and widening the device copy
took the measured load rate from 197 MiB/s to 843 MiB/s and the save from 185 MiB/s to 432,
with the format unchanged and byte identical.**

## The three costs, in the order they were found

| | cost | why | fix |
| --- | --- | --- | --- |
| 1 | one `pread`/`pwrite` of 4096 bytes per block | 21 us per syscall with one request in flight, latency bound | move blocks in 1 MiB windows |
| 2 | a byte at a time crc32 over every block | 7.5 us per 4 KB block, 519 MiB/s, and the reader checksums every block while the writer checksums once and the verify pass again | slice the crc32 by eight |
| 3 | one `ggml_backend_tensor_get` per block on the tensor path | per call overhead of a device copy, 4 KB at a time | one device copy per window |

Cost 3 only shows up on a device backend and it is the largest of the three: the same save took
63 ms on CPU and 1049 ms on Vulkan before the fix.

## Cost 1: the syscall granularity

`llama_io_read_direct::next_block` read one block per `pread`, and
`llama_io_write_direct::put_block` wrote one block per `pwrite`. `llama_io_verify_direct` read one
block per `pread` as well, and it runs over the whole file on every save, so it is on the save
path.

256 MiB file, O_DIRECT, three runs, read immediately after the write:

| chunk | read MiB/s | us per read | write MiB/s |
| --- | --- | --- | --- |
| 4 KB | 187.8 / 181.7 / 185.4 | 20.8 / 21.5 / 21.1 | 272.9 / 279.6 |
| 64 KB | 725.3 / 733.6 / 716.9 | 86.2 / 85.2 / 87.2 | 1351.1 / 1405.6 |
| 256 KB | 832.5 / 878.7 / 916.0 | 300.3 / 284.5 / 272.9 | 1756.5 / 1682.1 |
| 1 MiB | 1790.8 / 1809.3 / 1840.9 | 558.4 / 552.7 / 543.2 | 1854.3 / 1831.4 |

A single outstanding 4 KB read costs about 21 us whatever the device can do, which is what a
single threaded loop of `pread` calls produces. The fix is a window buffer in each direction:
1 MiB, aligned, holding whole blocks, so one syscall covers 256 of them.

One behaviour worth recording because it changed a threshold: the 4 KB read also collapses under
writeback. A standalone 4 KB read taken after seven rewrite passes measured 19.1 and 15.7 MiB/s
in two runs of three, against 1817 MiB/s for 1 MiB reads at the same point in the same run. Small
reads lose a further order of magnitude exactly when the device is busy, which is when a cache
hit happens.

## Cost 2: the crc32

The reader rejects a block whose crc does not match, so it checksums everything it reads. The
writer checksums everything it frames, and the verify pass checksums it again.

`llama_io_crc32` was the textbook byte at a time loop:

```
crc32 over 6900 x 4084 B: 0.0518 s, 7.509 us per block, 518.7 MiB/s
```

That is a hard ceiling on the reader, and the reader is on every cache hit. Slice by eight, one
table lookup per byte instead of one per byte per bit, computes the same value with the same
polynomial:

```
crc32 over 6900 x 4084 B: 0.0127 s, 1.833 us per block, 2124.4 MiB/s
```

4.1x, and the values did not move. That is checked rather than assumed, against `zlib.crc32` from
Python, at 28 sizes including 0, 1, 7, 8, 9, 4084, 65536 and 69999 so the eight byte body and the
tail loop are both covered: identical at every size.

## Cost 3: the device copy granularity

`write_tensor` copied the tensor to the host 4084 bytes at a time, one `ggml_backend_tensor_get`
per block, and `read_tensor` did the same in the other direction. On the CPU backend that is a
memcpy and it costs little. On Vulkan it is a device copy per call, and the call overhead is most
of it.

Measured on the 0.8B with `n_ctx 8192`, a 27 MiB state, `store_roundtrip`:

| | save | load |
| --- | --- | --- |
| CPU, before the fix | 63 ms | 32 ms |
| Vulkan, before the fix, from the server log | 1049 ms for a 54 MiB state, 51 MiB/s | not measured |
| CPU, after the fix | 63 ms, 427 MiB/s | 33 ms, 807 MiB/s |
| Vulkan, after the fix | 181 ms, 149 MiB/s | 34 ms, 803 MiB/s |

The Vulkan save before the fix is the `prompt cache update took 1049.13 ms` line from the end to
end server log. After widening the copy the load matches the CPU path, and the same end to end
check reports a 1.69x wall clock win where it previously reported 0.65x, which is the change that
mattered.

The Vulkan save is still 2.9x slower than the CPU save and that difference is not attributed. It
is on the write side only, since the load matches, so it is not the file or the crc. One candidate
is a per call staging allocation inside the Vulkan `ggml_backend_tensor_get`, one per 1 MiB
window, which the CPU backend does not do. Not measured.

## The format did not move

`direct_sink_check` passes, and two of its checks had to be corrected to say what they meant:

| check | before | after | why |
| --- | --- | --- | --- |
| `rss stays one block` | `ok`, 0 KiB | `rss stays one window`, 964 KiB | the store now holds a 1 MiB window per direction, which is a bound and not a leak |
| `the sink is deterministic for fixed input` | `FAIL`, differing=8 | `ok`, differing=0 | the two writers each drew a random generation, so they differed in exactly the four byte generation field of each of the two blocks. the check now pins the generation for both |
| `write_tensor is deterministic` | `FAIL`, differing=392 | `ok`, differing=0 | same, 98 blocks at four bytes each |

The first of those two was verified as pre-existing rather than assumed: the original
`src/llama-io.cpp` was restored, the library rebuilt, and the same check re-run. It reported the
same `differing=8` and `differing=392`. Both are the generation field, which is meant to differ
between two independent saves.

## Round trip, measured

`store_roundtrip` saves a state, loads it into a fresh context, re-serializes, and compares:

```
  store file            28315972 bytes payload, 28516352 bytes on disk
  state blob after load 28308716 vs 28308716 bytes, first_diff=-1, differing=0
  gen A: 15 17 15 8029 235212 1122 11234 9197
  gen B: 15 17 15 8029 235212 1122 11234 9197
  loaded state continues identically           ok
  save     0.064 s    419.8 MiB/s   rss delta 2056 KiB
  load     0.034 s    796.8 MiB/s
```

The loaded state is byte identical to the original and continues with the same eight greedy
tokens, on a fresh context. RSS grows by two windows and not with the 27 MiB state.

## What this does not cover

- Only the f2fs volume this box has. A different filesystem may not need the window.
- Only little endian, which the slice by eight load assumes. This tree is x86-64 only.
- The `O_DIRECT` write path still writes windows that are shorter than a full window at the end
  of a save, which is a whole number of blocks but not of windows. That is legal for `O_DIRECT`
  and measured to work, and it was not varied.
- Whether a wider window than 1 MiB helps. 1 MiB was chosen from the sweep, which flattens by
  512 KB.
- Nothing here was measured in the healthy GPU window. See
  `docs/research/26-prefill-vulkan-profile.md`.

## Reproduce

```sh
# the syscall granularity, three runs
gcc -O2 -o block_read_probe scripts/research/ssdcache/block_read_probe.c
./block_read_probe 256

# the crc32 rate and its values
#   the rate is the inline crc_bench in this document's shell history; the values are checked
#   against Python with the harness below
g++ -O2 -std=c++17 -I include -I ggml/include -I src scripts/research/kvstore/direct_sink_check.cpp \
    -L build-vk/bin -lllama -lggml-base -lggml -Wl,-rpath,$PWD/build-vk/bin -o dsc
./dsc

# the round trip and its rate
g++ -O2 -std=c++17 -I include -I ggml/include scripts/research/ssdcache/store_roundtrip.cpp \
    -L build-vk/bin -lllama -lggml-base -lggml -Wl,-rpath,$PWD/build-vk/bin -o store_roundtrip
./store_roundtrip ~/.jcode/scratch/qwen-models/Qwen3.5-0.8B-Q4_K_M.gguf /tmp/rt.bin 1200 8192 0
```
