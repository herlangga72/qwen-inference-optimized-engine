# Freeze notes: 2026-09-27

The box froze seven times between 06:33 and 10:05 on 2026-09-27. Every one was a hard reset: no
clean shutdown is recorded, the ext4 journal replay left recent files as NUL bytes, and the jcode
sessions that were live at the time (`hamster`, `t-rex`) are unrecoverable, all-zero files.

What follows is what can be established from the journal, and what remains open.

## Every reset was abrupt, and cost data

| boot | start | end | notes |
| --- | --- | --- | --- |
| -6 | 09-26 17:49 | 09-27 06:33 | 12h50m, ended mid capture run |
| -5 | 06:40 | 06:45 | 9 min |
| -4 | 06:50 | 06:52 | 3 min, `capture-baselines.sh` written at the end |
| -2 | 08:52 | 10:05 | plan A committed 09:20, plan B baselines started 09:21 |
| -3/-1 | 09:48, 09:49 | 45 s and 1 min | no heavy work logged |

`last -x` reports `crash` for the graphical session of every one of these boots, and there is one
`shutdown` record in the whole day (08:52, which was a reboot into the 08:52 boot). The kernel
reports the reset source as `x86/amd: Previous system reset reason [0x00080800]: software wrote 0x6
to reset control register 0xCF9`, which is a hard reset, not a graceful shutdown.

Data lost to the resets, none of it recoverable:

- `session_hamster_*.json`, `.bak` and `.journal.corrupt.jsonl`: 100% NUL bytes. This was the
  session that committed plan A and started plan B.
- `session_t-rex_*.journal.corrupt.jsonl`: all NUL bytes.
- `/home/herlanggays/.jcode/scratch/ssm-out-fix/capture-baselines.sh` (3837 bytes of NULs) and the
  first plan B baseline outputs `gen-0.8b-{cpu,vk}.txt` (2099 bytes of NULs each).

## The freeze mechanism, as far as the journal shows

The 06:33 boot has the clearest record. Under load it printed, repeatedly:

```
kswapd0: page allocation failure: order:0, mode:0xc0de0(...)
Node 0 Normal free:1400kB ...
Node 0 Normal: 0*4kB 0*8kB ... 0*4096kB = 0kB
... gpu_active:9900252kB gpu_reclaim:24484kB
systemd-journald[447]: Under memory pressure, flushing caches.
kwin_wayland[1558]: The main thread was hanging temporarily!
```

That is 9.9 GiB of GPU-active memory, 3.5M pages of page cache, and the Normal zone at zero free
pages, with the desktop already starving. The workloads running around those events were the 14 GiB
35B model on Vulkan and its batched benches; the same boot also produced three `llama-parallel`
SIGABRT core dumps, a tool the plan already says aborts on this architecture.

The kernel is configured to make this worse:

```
Command line: ... amdgpu.no_system_mem_limit=1 amdgpu.gttsize=24576
amdgpu 0000:04:00.0: [drm] Configuring gttsize via module parameter is deprecated, please use ttm.pages_limit
amdgpu 0000:04:00.0: [drm] GTT size has been set as 25769803776 but TTM size has been set as 14568730624, this is unusual
```

GTT is set to 24 GiB on a 27.1 GiB box while TTM stays at 13.57 GiB, and the driver calls the
combination out as unusual on every boot. The MoE model is 13.09 GiB and `-ngl 99` places it in
GPU-visible memory, which on this APU is system RAM: the model alone sits at 96% of the TTM limit
before the desktop's own GPU allocations are counted. `no_system_mem_limit=1` removes the check
that would refuse an over-size allocation. The plausible reading is therefore that a GPU allocation
that should fail cleanly instead drives TTM eviction and kswapd at once, which is the state the
`kswapd0: page allocation failure` dumps describe, and the box livelocks rather than returning ran
out of memory.

Not every reset fits that story. The 09:48 boot lived 45 seconds and the 10:07 boot died during a
`sync` plus an `fadvise(DONTNEED)` sweep over the 13 GiB model file, launched by the capture script
that has since dropped both steps. Neither had the 35B model loaded. That leaves a hardware or
driver fault in play that the memory story does not cover, and there is no evidence in the journal
either way: no MCE, no EDAC error, no amdgpu ring timeout, no lockup trace, no pstore entry.

## What was changed as a result

- `capture.sh` serializes every step, logs memory around it, and kills the child when
  `MemAvailable` drops under 1.5 GiB. Killing the child is what releases the GPU buffers.
- The capture script does not `sync` and does not fadvise the model files any more.
- `llama-parallel` is not used.
- All eleven plan B baseline steps then ran to completion with `MemAvailable` between 19.2 and
  19.6 GiB the whole time, which is the evidence that the workload alone is survivable when it is
  serialized and nothing else is holding memory.

## Verification on the real path

Re-run once more to close the loop, not inspected but observed. Two heavy runs of the 13 GiB 35B on
Vulkan, `llama-completion -ngl 99 -p "The capital of France is" -n 32`, with `MemAvailable` sampled
every second by the same watchdog code:

| run | result | min `MemAvailable` | peak `mem_info_gtt_used` |
| --- | --- | --- | --- |
| 1 | rc=0 | 12189 MiB | 10.00 GiB |
| 2 | rc=0 | 12150 MiB | 10.00 GiB |

The GPU side is the interesting part: 10.00 GiB of GTT for the model is the same number the crash
journal showed as `gpu_active:9900252kB`, so a healthy run and a crashing run had the same GPU
footprint. What differs is system free memory. In these runs `MemAvailable` never fell below 12.1 GiB,
which is ten GiB above the watchdog floor, so the floor is an escape hatch that a normal run never
approaches.

The escape hatch itself was tested, with the same code and the floor moved above the machine's
available memory so that any child trips it: the watchdog warned twice, killed the child, reaped it,
and returned 99. Logs: `scratch/rs-view/heavy-35b-vk-trace.log`, `heavy-verify.sh`,
`heavy-35b-vk-verify.txt`.

One correction to the record: `capture.sh` samples memory at the start and end of a step and on
warnings, not during, so its `gtt` and `vram` columns are snapshots of idle moments. They read 0.1 and
0.6 GiB in `capture.log` even for 35B runs. Use the per-second trace for GPU memory, and treat
`MemAvailable` as the signal, which is what the watchdog decides on.

## Not done, and why

`amdgpu.gttsize=24576` and `amdgpu.no_system_mem_limit=1` are still on the kernel command line.
Removing them needs root and a reboot, so it is the user's call. Doing so is the single change most
likely to turn the next over-size Vulkan allocation into a `VK_ERROR_OUT_OF_DEVICE_MEMORY` error
instead of a dead machine.
