# llama.cpp (Qwen-only fork: disk-backed session KV store)

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) for running Qwen models on machines whose
GPU has no memory of its own. It adds a layered session KV store that parks idle sessions out of the
compute arena onto disk and restores them on demand.

Everything outside the KV store is upstream llama.cpp. The measurements behind this work are in
`docs/research/`, the design is in `docs/superpowers/`, and the harnesses are in
`scripts/research/kvstore/`.

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
