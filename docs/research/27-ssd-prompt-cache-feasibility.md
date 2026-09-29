# SSD prompt cache: feasibility and measurements

Date: 2026-09-28
Tree: `qwen-only-backends`, commit `397b643f4`
Machine: Ryzen 7 6800H, Radeon 680M (RADV REMBRANDT), 27.1 GiB, AC connected
Model: `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` (qwen35moe, 41 layers, 10 full attention, 31 state),
       K `q8_0` and V `planar3_0` unless the table says otherwise
Volume: `/dev/nvme1n1p2`, f2fs, 461G with 72G free, 4 KB native block size
Harness: `scripts/research/ssdcache/econ.py`, `scripts/research/ssdcache/block_read_probe.c`,
         `build-vk/bin/llama-bench`
Spec: `docs/superpowers/specs/2026-09-28-ssd-prompt-cache-design.md`

## Superseded in three places, by building it

This document is the feasibility case and it was written before the code. Three of its numbers or
claims have since been measured directly. The list is here rather than in the body so a reader does
not have to diff four documents.

1. **The per token restore cost is optimistic by about 2x.** The tables below model a restore from
   the device's raw rate. The store path now measures 843 MiB/s reading, not the 1810 MiB/s the
   1 MiB `pread` probe suggested, so a cached token costs about 8.8 us rather than 4.25 and the
   ratio to prompt processing is about 950x rather than 1962x. See
   `docs/research/29-store-io-batching-results.md`.
2. **The open risk in "What this does not cover" is answered, and the answer is no.** Two prompt
   processing runs of the same prefix do not produce the same KV. They differ in 65 percent of the
   serialized state and produce different logits, on one thread, in the same process, and stock
   `llama-cli` greedy with a fixed seed gave three different continuations in three runs. See
   `docs/research/28-ubatch-determinism-results.md`. What survives is the narrower property the
   cache actually needs and which is now measured: a save and a load are byte exact, so a hit
   reproduces the computation that wrote the entry.
3. **The projected loop speedup is not what a request pair measures.** The model below predicts a
   16.9x loop speedup from prompt processing alone. The end to end check measures a 3.4x reduction
   in prompt evaluation and a 1.42 to 1.52x wall clock win for one request, because the request
   that displaces a conversation also pays to write the displaced state. See
   `docs/research/30-ssd-prompt-cache-results.md`.

The headline verdict below still holds: restoring a cached prefix is far cheaper per token than
processing it, and the I/O is a small fraction of what it replaces.

**Verdict: yes, and by a wide margin. Restoring a cached prefix costs 4.25 us per token
against 8333 us to process it, a factor of 1962. Over a 20 turn agentic loop that resends its
transcript, prompt processing falls from 175000 token-turns to 10000, and the wall clock from
24 minutes to 1.5. Disk traffic is 0.16 percent of the time saved.**

**Two findings change what has to be built first, and one of them is a correction to this
tree's own numbers.**

1. **The store's I/O path is 9.8x slower than the same volume read in 1 MiB chunks.**
   `llama_io_read_direct` issues one 4 KB `pread` per block and `llama_io_write_direct::put_block`
   one 4 KB `pwrite`. Measured: 185 MiB/s reads and 273 MiB/s writes, against 1810 and 1854 MiB/s
   for the same file at 1 MiB. The gap is per syscall overhead, not the device. Every SSD restore
   cost in this document is given twice for that reason.
2. **`docs/research/16-disk-tier-results.md` measured the device, not the path.** Its 2551 MiB/s
   came from 1 MiB `pread`s in `disk_bench.py`. The store does not read that way, so its 48 ms
   end to end park and unpark is optimistic by roughly the same factor. The conclusion that
   parking beats re-prefill survives, with less margin than the document claims.

The third thing this document does not settle is whether any of it can be verified today: the
box is in the degraded, non-deterministic GPU window recorded in
`docs/research/26-prefill-vulkan-profile.md` at commit `4e94b1937`, and that document's own
advice is to reboot and re-check the gate before trusting a measurement. The numbers below were
taken inside that window.

## Prompt processing on the target model

`llama-bench -ngl 99 -fa on -ctk q8_0 -ctv planar3_0 -p 512,2048,8192 -n 0 -r 2`:

| test | t/s | us/token |
| --- | --- | --- |
| pp512 | 120.07 +/- 0.76 | 8329 |
| pp2048 | 112.92 +/- 0.79 | 8856 |
| pp8192 | 105.21 +/- 0.14 | 9505 |

Doc 26 recorded 230 t/s on this model and tree. This is 120, which is the same degraded window
`4e94b1937` describes as "pp512 down from 254.8 to about 114 to 119". The cache is worth twice
as much in this state as in a healthy one, so the design is written against 120 and the healthy
case is carried alongside.

## The KV types are not why pp is slow

Same command, `-ctk` and `-ctv` crossed, two repetitions each. This was run because the first
suspicion for the 2x gap was the `planar3_0` V cache write path, which is new in this tree and
sits in the prefill graph:

| type_k | type_v | pp512 | pp2048 |
| --- | --- | --- | --- |
| f16 | f16 | 118.76 +/- 0.12 | 116.82 +/- 0.12 |
| f16 | planar3_0 | 116.47 +/- 0.17 | 114.32 +/- 0.30 |
| q8_0 | f16 | 118.71 +/- 0.42 | 117.14 +/- 0.80 |
| q8_0 | planar3_0 | 115.89 +/- 0.22 | 113.52 +/- 0.05 |

The whole spread is 2.4 percent. `planar3_0` V costs about 2 percent of prefill and buys a 5.2x
smaller V cache (1960 against 10240 B/token), which is the right trade for a design whose
binding constraint is disk. The 2x gap to doc 26 is the box, not the cache types.

## Disk at the granularity the store actually uses

256 MiB file, O_DIRECT, three runs. The read sweep runs immediately after the file is written,
which is the quiet-device case.

| chunk | read MiB/s | us per read | write MiB/s |
| --- | --- | --- | --- |
| 4 KB | 187.8 / 181.7 / 185.4 | 20.8 / 21.5 / 21.1 | 272.9 / 279.6 |
| 8 KB | 226.1 / 249.0 / 224.6 | 34.6 / 31.4 / 34.8 | 499.6 / 498.0 |
| 16 KB | 239.1 / 296.9 / 269.6 | 65.4 / 52.6 / 58.0 | 710.8 / 711.8 |
| 32 KB | 453.4 / 479.1 / 428.2 | 68.9 / 65.2 / 73.0 | 1048.9 / 1060.7 |
| 64 KB | 725.3 / 733.6 / 716.9 | 86.2 / 85.2 / 87.2 | 1351.1 / 1405.6 |
| 128 KB | 488.9 / 509.7 / 522.0 | 255.7 / 245.3 / 239.5 | 1565.7 / 1441.6 |
| 256 KB | 832.5 / 878.7 / 916.0 | 300.3 / 284.5 / 272.9 | 1756.5 / 1682.1 |
| 512 KB | 1300.1 / 1340.4 / 1396.0 | 384.6 / 373.0 / 358.2 | 1793.0 / 1831.4 |
| 1 MiB | 1790.8 / 1809.3 / 1840.9 | 558.4 / 552.7 / 543.2 | 1854.3 / 1831.4 |

Three things in that table:

- **4 KB to 1 MiB is 9.8x on reads and 6.8x on writes.** At 4 KB the cost is latency bound at
  about 21 us per request with one request in flight, which is what a single threaded loop of
  `pread` calls produces. Both sweep directions are stable across runs.
- The 128 KB row dips below 64 KB in all three runs. Not explained, not on the critical path.
- Warm buffered reads are 8460 to 12717 MiB/s at 1 MiB, off the same file, so the device is not
  the limit at any chunk size here.

### The 4 KB read collapses under writeback

The last measurement in the probe, a standalone 4 KB read of the whole file taken after seven
`O_TRUNC` rewrite passes, returned 19.1 and 15.7 MiB/s in two runs of three, against 181.7 in
the one clean run. The same three runs read the same file at 1380 to 1836 MiB/s with 1 MiB
chunks at the same point in the sequence. So a small-read path loses a further order of
magnitude when the device is busy flushing, and a large-read path does not.

That matters for a cache more than for a park, because a cache hit is exactly the moment other
slots are writing.

### What a restore costs

Restore payload is `tokens * 7400 * 1.016` for attention plus a fixed 62.8 MiB recurrent state.
7400 B/token is the attention slope for 10 full attention layers at K `q8_0` and V `planar3_0`,
from `scripts/research/kvstore/kv_capacity.py` against the model metadata, and 62.8 MiB is the
same fixed term `docs/research/17-device-to-disk-path-results.md` measured as the intercept.
Device to staging transfer is excluded here and is 0.29 us/token at 24.4 GiB/s.

| prefix tokens | on disk | 4 KB reads, as built | per token | 1 MiB reads | per token | pp / restore, batched |
| --- | --- | --- | --- | --- | --- | --- |
| 4000 | 91.5 MiB | 503 ms | 126 us | 50 ms | 12.5 us | 668x |
| 13500 | 159.6 MiB | 878 ms | 65 us | 87 ms | 6.4 us | 1294x |
| 64000 | 521.7 MiB | 2871 ms | 45 us | 284 ms | 4.4 us | 1877x |
| 262144 | 1942.4 MiB | 10688 ms | 41 us | 1058 ms | 4.0 us | 2064x |

The per token figure falls with length because the recurrent term is fixed, not per token. That
is the same law as the park cost: a prefix restore has a floor.

So batching the store's I/O is worth 9.8x on every cache hit, and it is the difference between
counting a restore in tens of microseconds per token and counting it in single digits.

## The economics, with the measured numbers

`scripts/research/ssdcache/econ.py --pp 120`:

| | per token |
| --- | --- |
| prompt processing, 120 t/s | 8333 us |
| ssd read, 4 KB per syscall | 38.8 us |
| ssd read, 1 MiB per syscall | 3.96 us |
| device transfer | 0.29 us |
| ssd write, 1 MiB per syscall | 3.87 us |
| pp / restore, as built | 213x |
| pp / restore, batched | 1962x |

Agentic loop, 20 turns, prefix 4000 tokens growing 500 per turn, the transcript resent each
turn, 175000 token-turns. Text generation is excluded, so this is the prompt processing part of
the loop only:

| | pp tokens | pp | ssd write | ssd read | xfer | total | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no cache | 175000 | 1458 s | | | | 1458 s | 1x |
| cache, 4 KB per syscall | 10000 | 83 s | 9.2 s | 13.6 s | 0.2 s | 106 s | 13.7x |
| cache, 1 MiB per syscall | 10000 | 83 s | 1.4 s | 1.4 s | 0.2 s | 86 s | 16.9x |

At the healthy 230 t/s the no-cache loop is 761 s and the cached loop is 46 s, 16.4x, so the
speedup is nearly independent of the pp rate: the cache removes 94.3 percent of the token-turns
either way, and the I/O is fixed. Disk traffic is between 0.16 and 0.26 percent of the time
saved, which is what "free" means here.

Two forms of the same cache are modelled, because the choice matters for write volume rather
than for time:

| | cumulative write over the loop | residency at the end | write amplification |
| --- | --- | --- | --- |
| whole state rewritten per turn | 2.45 GiB | 159.6 MiB | 15.7x |
| attention blocks appended, one recurrent blob per turn | 1.30 GiB | 159.6 MiB | 8.3x |

The recurrent state is why these are close. At 13500 tokens the attention part is 101.5 MiB and
the recurrent blob is 62.8 MiB, so rewriting the recurrent blob each turn puts a floor under the
write volume that block granularity cannot remove. It can only remove the attention part.

## Footprint, 20 live conversations

| tokens | attention | recurrent | each | 20 conversations |
| --- | --- | --- | --- | --- |
| 4000 | 28.7 MiB | 62.8 MiB | 91.5 MiB | 1.79 GiB |
| 16000 | 114.7 MiB | 62.8 MiB | 177.5 MiB | 3.47 GiB |
| 64000 | 458.9 MiB | 62.8 MiB | 521.7 MiB | 10.19 GiB |
| 262144 | 1879.6 MiB | 62.8 MiB | 1942.4 MiB | 37.94 GiB |

With 72 GB free, 20 conversations at 64k tokens is 10.2 GiB and fits. At the model's own context
length it does not. The recurrent term is 62.8 MiB per conversation before a single token is
stored, so conversations with short prefixes are dominated by it and a byte budget has to count
entries, not just tokens.

## What this does not cover

- **The reproducibility question this document left open has been answered the other way.** The
  test was run, in `docs/research/28-ubatch-determinism-results.md`: two prompt processing runs of
  the same prefix do **not** produce the same KV, on any split, on one thread, even in the same
  process. The consequence is not that the cache fails but that its claim narrows from "a hit
  equals a recomputation" to "a hit reproduces the computation that wrote the entry", which is the
  property that was measured directly and which holds byte for byte.
- No end to end cache hit. Everything here is components, or a model of them. Nothing in this
  document runs a server, requests a cached prefix, and compares the completion.
- The 35B has never been through the park and unpark check. Docs 21 and 24 verified the direct
  path on the 0.8B only.
- Sustained load, eviction under concurrency, and endurance. The write volumes above are per
  conversation and the loop is not a day.
- The measurements were taken in the degraded window. The pp figure may double, the disk figures
  should not move, and the device read at 4 KB under writeback is the one that may be worse in
  real use than in a quiet probe.

## Reproduce

```sh
# prompt processing, and the KV type A/B
build-vk/bin/llama-bench \
  -m "/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf" \
  -ngl 99 -fa on -ctk q8_0 -ctv planar3_0 -p 512,2048,8192 -n 0 -r 2
build-vk/bin/llama-bench \
  -m "/home/herlanggays/Desktop/Billings That I can Collect/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf" \
  -ngl 99 -fa on -ctk f16,q8_0 -ctv f16,planar3_0 -p 512,2048 -n 0 -r 2

# disk at block and at chunk granularity
gcc -O2 -o block_read_probe scripts/research/ssdcache/block_read_probe.c
./block_read_probe 256

# the economics
python3 scripts/research/ssdcache/econ.py --pp 120
python3 scripts/research/ssdcache/econ.py --pp 230
```

Run the llama-bench commands one at a time. The model is 13.09 GiB against a 13.57 GiB TTM limit
on this box and `docs/research/freeze-notes-2026-09-27.md` records seven hard resets under it.
`scripts/research/ssdcache/run_guard.sh` wraps a child in the `MemAvailable` watchdog used for
the runs here; the minimum observed was 8686 MiB, against a 2000 MiB floor.
