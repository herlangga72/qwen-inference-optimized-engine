# Device to disk path, and the session size law: results

Date: 2026-09-27
Machine: Ryzen 7 6800H. Vulkan build `build-vk`, CPU build `build-cpu`.
Models: Qwen3.5-0.8B-Q4_K_M, Qwen3.6-35B-A3B-UD-IQ3_XXS
Spec: `docs/superpowers/specs/2026-09-27-session-kv-store-design.md`
Plan: Task 5

**Verdict: a session's parked size is not proportional to its context. It is a fixed
per-session term plus a per-token slope, and on the 35B the fixed term is 62.8 MiB against a
slope of 20.5 KB per token. Parking a short session is expensive because of the fixed part.
The device transfer is nearly free: 2.65 ms for 33 MiB both ways.**

Two corrections to earlier numbers, both from the same measurement:

1. The figure "27.4 KB per token" in `15-e0-park-unpark-results.md` conflated a fixed term
   with a slope. The real slope on the 0.8B is 12.0 KB per token.
2. The device copy that `16-disk-tier-results.md` listed as unmeasured is now measured, and
   it is small enough to ignore next to the disk.

## The session size law

Measured from the save path, two lengths per model, so the slope and the intercept separate:

| model | tokens | state size |
| --- | --- | --- |
| 0.8B | 1274 | 34.225 MiB |
| 0.8B | 1339 | 34.988 MiB |
| 35B | 198 | 66.685 MiB |
| 35B | 526 | 73.099 MiB |

Solving each pair:

| model | fixed per session | per token |
| --- | --- | --- |
| Qwen3.5-0.8B | 19.3 MiB | 12.0 KB |
| Qwen3.6-35B-A3B | 62.8 MiB | 20.5 KB |

The fixed term is the recurrent state. The 35B log shows
`llama_memory_recurrent: Vulkan0 RS buffer size = 125.62 MiB` for two sequences, which is
62.8 MiB per sequence, matching the intercept exactly. Qwen3.5 and 3.6 are hybrid models, so
every session carries a gated delta net state that is a running summary, not per token data.

Two consequences that follow directly:

- **Parking has a floor.** On the 35B no session can be parked in less than 62.8 MiB, no
  matter how short. A 200 token session parks at 66 MiB.
- **Context compaction does not shrink a park.** Trimming tokens reduces only the attention
  part. The recurrent part is fixed, and it cannot be derived from a subset of tokens at all,
  which is the same fact that made `seq_rm` need a rewrite in the layer 1 results.

## The device transfer is nearly free

The 3.84 ms figure from E0 was a CPU-only run, where the KV already sits in host memory, so
there was no device transfer in it. Repeating on Vulkan with the KV on the device, for a
33.145 MiB session:

| path | save plus restore |
| --- | --- |
| CPU, `-ngl 0` | 3.84 ms |
| Vulkan, `-ngl 99` | 2.65 ms |

So a full round trip of 33 MiB out of the device and back costs about 2.7 ms. Next to the
48 ms the disk costs, it is noise. The disk is the entire park cost, not the transfer.

## End to end park and unpark

| component | cost |
| --- | --- |
| device gather and scatter, 33 MiB | 2.7 ms |
| disk write, `O_DIRECT` + fsync, 35 MiB | 34.1 ms |
| disk read, `O_DIRECT`, 35 MiB | 13.7 ms |
| total | about 51 ms |

Against recomputing the same context: 1339 tokens on the 0.8B is about 2700 ms on CPU and
about 1000 ms on Vulkan. On the 35B, prefill measured 1489 ms for 198 tokens, which is
7.5 ms per token, so a 4k context is about 30 s. The gap grows with model size, and the park
cost does not.

## The UMA point, which changes what the policy can mean

On Vulkan this box reports:

```
common_params_fit_impl: projected to use 12917 MiB of device memory vs. 26677 MiB of free device memory
```

26677 MiB of "free device memory" is the machine's system RAM. There is no separate video
memory pool of that size on a Radeon 680M. So on this box:

- "KV in VRAM" and "KV in system RAM" are the same physical DRAM. The distinction is which
  heap owns the allocation, not which silicon holds it.
- The residency policy is therefore enforceable as "no KV in the page cache, and no host
  buffer that outlives a transfer". It cannot be enforced as "different memory hardware",
  because there is none.
- The benefit of the device heap on this box is not saved RAM, it is one fewer copy and no
  page cache pressure. The benefit of the disk is real and is the only thing that lowers
  memory footprint.

On a discrete GPU later, the two heaps separate and the original framing applies unchanged.
The policy is written to hold in both cases; only its justification differs.

## Disk budget

From the law, at 25.7 GB free on a 95% full disk:

| session length | 35B parked size | sessions that fit |
| --- | --- | --- |
| 4k tokens | 147 MiB | 175 |
| 32k tokens | 718 MiB | 36 |
| 128k tokens | 2684 MiB | 9 |

So the disk is generous for short sessions and tight for long ones, and the fixed term is
what makes many short parked sessions cost more than expected. `planar3_0`, already in this
tree for the V cache, cuts the attention slope and does nothing for the recurrent term.

## What this does not cover

- Compression. The slope figures are at `f16` K and V.
- The store's own eviction policy and on-disk indexing overhead.
- Sustained multi-session load, and any concurrent session behaviour.

## Reproduce

The two point measurement is two lengths saved through the idle slot path:

```sh
# 35B, Vulkan, three prompts of different lengths, read the prompt_save lines
grep -E 'saving prompt with length' /home/herlanggays/.jcode/scratch/kvstore/e0-35b-b.log
```

Logs for both models are under `/home/herlanggays/.jcode/scratch/kvstore/`.
