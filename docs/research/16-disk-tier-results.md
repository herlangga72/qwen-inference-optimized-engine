# Disk tier: results

Date: 2026-09-27
Machine: Ryzen 7 6800H. Device `/dev/nvme1n1p2`, 476.9 GB, f2fs, 25.7 GB free (95% used).
Harness: `scripts/research/kvstore/disk_bench.py`
Spec: `docs/superpowers/specs/2026-09-27-session-kv-store-design.md`

**Verdict: the disk tier is viable, and the RAM-free path is the fast path. `O_DIRECT` is
required by the residency policy and measured faster than buffered I/O, 1026 against 657
MiB/s on writes. One 35 MiB session parks and unparks in about 48 ms, against about 2700 ms
for a re-prefill.**

## Why this was measured

The policy is that resident KV lives only in device memory or on disk, never in host RAM.
On Linux a plain write to a file lands in the page cache, which is system RAM, so the
question is not "is disk fast enough" but "can disk be used without filling RAM".

## Result

512 MiB to a file on f2fs, `Cached` from `/proc/meminfo`:

| path | time | throughput | `Cached` change |
| --- | --- | --- | --- |
| buffered write + fsync | 0.779 s | 657.4 MiB/s | **+514 MiB** |
| buffered read, warm | 0.035 s | 14471.2 MiB/s | from cache |
| `O_DIRECT` write | 0.499 s | 1026.4 MiB/s | **+0 MiB** |
| `O_DIRECT` read | 0.201 s | 2551.9 MiB/s | +0 MiB |
| buffered read, cold after `fadvise DONTNEED` | 0.244 s | 2094.4 MiB/s | from disk |

The buffered write grew the page cache by the size of the file. That is the policy violation,
measured, and it is the default behaviour of every ordinary file write.

The direct write added nothing to the page cache and ran 1.6x faster, because it does not
copy through the cache and then again to the device. So on this box the RAM-free path is also
the cheaper one, and there is no trade to make.

## Per session

Converted to one session of the size measured in E0, 35 MiB for 1339 tokens on the 0.8B:

| operation | cost |
| --- | --- |
| park, `O_DIRECT` write + fsync | 34.1 ms |
| unpark, `O_DIRECT` read | 13.7 ms |
| both | 47.8 ms |
| for comparison, full re-prefill of the same context | about 2700 ms |

About 56x cheaper than recomputing, and the comparison gets better on larger models, where
re-prefill scales with layer count and disk cost does not.

## Constraints this puts on the store

1. **All store I/O must be `O_DIRECT`.** Not an optimisation, a correctness requirement for
   the policy, and it needs 512 byte or larger aligned buffers on f2fs.
2. **The staging buffer is one page, reused.** Host cost stays O(1). It is the only host
   memory in the path, and Vulkan has no device to file transfer, so it cannot be zero.
3. **Disk capacity is the binding constraint, and it is tight.** 25.7 GB free at 95% full.
   The per session size is a fixed term plus a slope, not a per token rate, so the usable
   session count depends on context length. `docs/research/17-device-to-disk-path-results.md`
   has the law and the budget table. KV compression matters: `planar3_0` already exists in
   this tree for the V cache, and it cuts the slope but not the fixed term.
4. **zram swap is present, 54 GiB, 2 GiB in use.** Compressed RAM swap means host pages under
   pressure do not go to disk, they go back to RAM. Keeping host use at O(1) is the defence.

## What this does not cover

- Compression. The slope figures are at `f16` K and V.
- Random access patterns. This is sequential, and a store with per session files will have
  some of both.
- Sustained load, write amplification on f2fs, and endurance.

The device to staging buffer copy that an earlier version listed here is now measured, at
2.65 ms for 33 MiB both ways, small enough to ignore next to the disk. See
`docs/research/17-device-to-disk-path-results.md`.

## Reproduce

```sh
python3 scripts/research/kvstore/disk_bench.py
```

Writes and rewrites a 512 MiB file under
`/home/herlanggays/.jcode/scratch/kvstore/disk/`.
