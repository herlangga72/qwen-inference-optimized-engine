# Direct I/O framing: results

Date: 2026-09-27
Machine: Ryzen 7 6800H. `/dev/nvme1n1p2`, f2fs.
Harness: `scripts/research/kvstore/direct_io_probe.py`
Spec: `docs/superpowers/specs/2026-09-27-session-kv-store-design.md`

**Verdict: `O_DIRECT` works, and it forces a change to the layer 1 format. Alignment is 512
bytes, not 4 KiB. One direct write per record costs 93x write amplification, so the status
log must batch records into blocks. And once it does, the block needs its own header: without
one, the padding is indistinguishable from a torn tail and a reader recovers 93 records out
of 2000.**

Two format rules came out of this and are now in the spec. Everything else held.

## Alignment

`O_DIRECT` requires alignment of both the offset and the length. The probe writes and reports
the errno:

| size at offset 0 | result |
| --- | --- |
| 64, 128, 256 | `EINVAL` |
| 512, 1024, 4096 | ok |

| offset of a 4096 byte write | result |
| --- | --- |
| 0 | ok |
| 64 | `EINVAL` |
| 512, 1024, 4096 | ok |

So the minimum unit on this filesystem is 512 bytes for both, not the 4096 a page size would
suggest. A 4096 block is still the sane choice, since it costs about 2% padding for record
shaped data and matches the flash page.

## Write amplification

2000 status log records, 44 bytes each, 88000 bytes of payload:

| layout | bytes on disk | amplification | time |
| --- | --- | --- | --- |
| one `O_DIRECT` write per record, padded | 8192000 | **93.09x** | 0.094 s |
| batched into 4096 blocks, 22 flushes | 90112 | **1.02x** | 0.003 s |

Per record direct writes are 93x because every 44 byte record rounds up to a 4096 byte block.
Batching is not an optimisation here, it is the difference between a usable log and a
93x write amplifier on the disk that is also the smallest resource in the system.

## The framing requirement

Batching introduces padding, and padding ends a naive record scan. Same file, two readers:

| reader | records recovered |
| --- | --- |
| walk records to the first bad frame (the layer 1 rule as originally written) | 93 of 2000 |
| walk blocks, check the block header, then the records | **2000 of 2000, byte exact** |

The naive reader stops at the first block boundary, because the 4 bytes of zero padding after
the last record in a block look exactly like a torn record. That failure is silent and it
recovers the first block only, which is the worst possible outcome: it looks like a partial
write rather than a format bug.

The fix is a block header:

    block := [u32 used][u32 crc32(payload area)][records][zero padding]

`used` says where the records end, and the block crc covers exactly that region, so padding
and a torn tail are different things again.

Torn block behaviour with the header, losing the final block:

| | records |
| --- | --- |
| written before the torn block | 1993 |
| recovered | 1932 |
| lost | 61, all inside the torn block |

A block is rejected whole. That is the coarser durability point the spec now names.

## RAM cost

512 MiB written through one reused 1 MiB aligned buffer:

| | value |
| --- | --- |
| throughput | 2450 MiB/s |
| RSS before | 12280 KiB |
| RSS after | 13304 KiB |
| delta | 1024 KiB |

The RSS delta is exactly the staging buffer, not the data written. That is the residency
policy check passing, and it is the assertion Task 4 must carry into the engine.

## Changes to the spec

1. `kv.status` is block framed. The header is `used` plus a crc over the payload area.
2. The atomic unit for durability is a block, so the crash rule stops at the first bad block
   rather than the first bad record.
3. Slot size in `kv.pages` is padded to a multiple of 512 bytes, so a slot write is a whole
   number of aligned blocks.

## What this does not cover

- **The layer 1 crash matrix has not been re-run against the block framed layout.** The
  atomic unit changed from a record to a block, so the matrix in
  `scripts/research/kvstore/scenarios.py` models a finer granularity than the real file now
  has. The rules it checks are unchanged and the torn block case was verified here, but the
  matrix should be re-pointed at block granularity before the format is called settled.
- The device to staging copy for the write side. Only the file was exercised here.
- f2fs behaviour under sustained load, and whether the 512 minimum holds on other filesystems.

## Reproduce

```sh
python3 scripts/research/kvstore/direct_io_probe.py
```

About half a second. Writes under `/home/herlanggays/.jcode/scratch/kvstore/direct/`.
