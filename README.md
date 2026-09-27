# Qwen Inference Optimized Engine

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp), scoped to the Qwen model family and
optimized for machines whose GPU has no memory of its own. It adds a disk-backed session KV store so
that one box can hold many concurrent Qwen sessions without keeping their KV cache in system RAM.

Upstream llama.cpp is MIT licensed and so is this. The measurements behind every claim on this page
are in `docs/research/`, the designs are in `docs/superpowers/`, and the harnesses are in
`scripts/research/kvstore/`.

## How this differs from llama.cpp

|  | upstream llama.cpp | this fork |
| --- | --- | --- |
| Architectures in `src/models/` | 156 files | 3 (`qwen35`, `qwen35moe`, `qwen4exp`, plus a shared `delta-net-base`) |
| Size of `src/` | 3.7 MB | 1.8 MB |
| KV types | `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `iq4_nl` | the same, plus `planar3_0`, a 3 bit type |
| Park and unpark | host buffered, through `--slot-save-path` | `llama_state_seq_save_file_direct` and `llama_state_seq_load_file_direct`, streamed through `O_DIRECT` |
| Store format | an unframed `LLAMA_STATE_SEQ` v3 stream | 4 KB blocks, 12 byte frame header carrying a per-save generation, `LLAMA_STATE_SEQ_VERSION` 4. Store files written by earlier versions are not readable. |

The dropped architectures are not merely untested here. Their graph code is gone, so the engine only
builds and only reasons about what this project runs. That is most of the difference in source size,
and it is why a change to a shared path is cheaper to reason about here than upstream.

## How inference works

Qwen3.5-0.8B and Qwen3.6-35B-A3B are hybrid models, and the shape of that hybrid decides everything
about memory. In the 35B, of 41 layers, 31 are gated delta net layers that carry a fixed size
recurrent state, and 10 are full attention layers, one every fourth layer.

- A session's footprint is therefore two different things: a fixed recurrent state term, measured at
  62.8 MiB for the 35B against 19.3 MiB for the 0.8B, plus a per-token attention KV term that scales
  with context. Only the second one grows with the conversation.
- That fixed term is why parking has a floor. An otherwise empty 35B session still costs 62.8 MiB,
  and context compaction cannot shrink it, because it is the delta net state and not the context.
- One decode step walks the recurrent layers, which advance their state token by token, and the
  attention layers, which read the session's KV. The MTP head that ships inside the 35B
  (`nextn_predict_layers = 1`) is used on top for speculative decoding.

A session lives in the GPU arena while it is active. When it goes idle it can be moved out to disk and
moved back when it is needed again, and the store that does that is what most of this fork's work is
about. Parking writes the recurrent state and the attention KV to one file through `O_DIRECT`, so a
parked session costs no host memory, and unparking restores it into a context so that generation
continues exactly as if the session had never left.

## What we optimized

| area | change | measured |
| --- | --- | --- |
| KV cache, V | `planar3_0`, a 3 bit type with rotated rows and a Givens table | a 98 byte block per 256 coordinates; cuts the KV slope 2.77x against f16 on the 35B |
| KV cache, K | `q8_0` | 5440 B/token against 10240 at f16, on the 35B |
| Session parking | disk store through `O_DIRECT`, 4 KB blocks, a per-save generation, verify after write | park and unpark with a 0 KiB resident delta on a 55 MiB state |
| delta net output projection | declare `final_output` as `[value_dim, n_seq_tokens*n_seqs]` instead of 3D, at the three delta net output projections | Vulkan token generation, 1 token x 8 sequences: 2.2x faster. Vulkan prefill unchanged. CPU token generation about 1.3x slower, CPU prefill about 5% faster |
| Recurrent state | return a view of the cache instead of gathering rows, when the active rows are already contiguous | removes a per-token gather and its write-back; outputs byte-identical to before |
| MTP speculation | speculate with the model's own MTP head, no separate draft file | 1.27x on CPU, 1.36x on Vulkan, single stream |
| Tokenizer | keep the merge rank table over string views into the vocab's own bytes | byte identical output, and the per-lookup allocations are gone |

Two of these trade CPU for Vulkan, deliberately. The target is a Radeon 680M, where token generation
was the bottleneck, and the delta net output projection change is 2.2x faster on Vulkan token
generation against about 1.3x slower on CPU token generation: a bad trade on a CPU only box, a good one
here.

One thing did not work. Binding the GGUF's own pages to the Vulkan device with
`VK_EXT_external_memory_host`, so that weights exist once in RAM and the GPU reads those pages
directly, fails on this driver: RADV refuses to import a file backed host pointer at any size while it
accepts an anonymous one on the same device. That was dropped and the enabling changes reverted,
written up in `docs/research/11-uma-zero-copy-findings.md`.

## Why these changes exist

The development machine is a Ryzen 7 6800H with a Radeon 680M and 27 GiB of memory, and no dedicated
video memory. The GPU arena and system RAM are the same 27 GiB. A Qwen3.6-35B-A3B at IQ3_XXS takes
about 13 GiB, which leaves roughly 14 GiB for everything else, including the KV cache of every live
session.

The shipping prompt cache does not fit in that budget. It keeps the KV of idle slots in host memory,
so memory used by idle sessions grows with the number of sessions instead of staying flat. On a
machine with a real GPU and 24 GiB of VRAM that is a reasonable trade. Here it is the constraint.

So the rule this fork is built around is: **resident KV lives in device memory or on disk, never in
host RAM**. Host memory holds the model, plus one reused I/O staging block. Most of the design
follows from that, including the use of `O_DIRECT` for the store, and it is why a parked session
costs 0 KiB of resident memory instead of a few hundred MiB.

The second reason is capacity. Ten sessions at Qwen3.6-35B-A3B's own context of 262144 tokens, with K
cached `q8_0` and V cached `planar3_0`, is 1.868 GiB per session and 18.68 GiB for ten, against 50.6
GiB at f16. `scripts/research/kvstore/kv_capacity.py` recomputes that from a GGUF header without
loading the model. Note that this makes disk the binding constraint, not RAM: the box has 35 GB free.

## What is different

- `llama_io_write_direct` / `llama_io_read_direct`: block framed store I/O through `O_DIRECT`, with
  one reused aligned buffer, so host memory stays O(1) in the size of the state.
- `llama_state_seq_save_file_direct` / `llama_state_seq_load_file_direct`: the park and unpark pair.
- A per-save generation in every frame, and a read-back verify after each save. This class of volume
  loses writes silently, and a lost write leaves the block holding whatever was there before, which
  for the store's own path is an earlier save of the same file: a valid frame with a matching crc. So
  a crc alone cannot detect it and a generation can. A save now either writes an intact file or fails.
- `planar3_0`, a 3 bit V cache type, and `q8_0` for K, which together cut the KV slope 2.77x against
  f16 on Qwen3.6-35B-A3B.

## What is measured

- Park a real sequence to disk, restore it, and continue: the continuation is identical to a run that
  never parked. Isolated RSS delta across park and unpark: 0 KiB on a 55 MiB state.
- Saves are reproducible and intact: four consecutive runs with no damaged blocks, cross-checked by a
  Python validator that shares no code with the C++ reader.

## What is not done

- The server level path is not exercised. Two sessions parking and restoring through `llama-server`
  is the acceptance test, and no in-tree automated test covers it (it needs Python dependencies that
  did not build here). The public C API is exercised, on a fresh context into an empty sequence.
- The pages file has no checksum, so it has no way to detect a lost write.
- Store files written before this work are not readable. The format changed: 4 KB blocks, a 12 byte
  frame header carrying a generation, and `LLAMA_STATE_SEQ_VERSION` 4.

Upstream build instructions, model support and the REST API are unchanged and still apply below.

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- CPU kernels for x86-64: AVX, AVX2, AVX512, AMX, BMI2, FMA, F16C
- Vulkan backend support for GPUs
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [CPU](docs/build.md#cpu-build) | All |
| [Vulkan](docs/build.md#vulkan) | GPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Build on Android](docs/android.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
