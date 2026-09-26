# Multimodal support in this fork

This directory provides the multimodal capabilities of this fork: `libmtmd` plus the `clip` encoder.
The fork only supports the **Qwen3.5 family**, so the only supported projector is `QWEN3VL`, which is
what the Qwen3.5-VL vision tower (and the Qwen4Exp vision tower, an unmodified Qwen3-VL ViT) uses.
Audio input is not supported here.

## Pre-quantized models

See the list of pre-quantized models [here](../../docs/multimodal.md)

## How it works and what is `mmproj`?

Multimodal support works by encoding images into embeddings with a separate model component, and then
feeding these embeddings into the language model.

This keeps the multimodal components distinct from the core `libllama` library. Separating them allows
faster, independent development. While many modern vision models are based on Vision Transformers
(ViTs), their specific pre-processing and projection steps vary significantly, and integrating that
complexity directly into `libllama` is challenging.

Consequently, running a multimodal model requires two GGUF files:
1. The standard language model file (for Qwen3.5-VL this is a plain `qwen35` / `qwen35moe` GGUF).
2. A corresponding **multimodal projector (`mmproj`)** file, which handles the image encoding and
   projection.

## What is `libmtmd`?

`libmtmd` is the library that handles multimodal inputs. Built on top of `clip.cpp`, it offers:
- **Unified interface:** one API and one CLI (`llama-mtmd-cli`) for the supported model family.
- **Improved UX/DX:** an API inspired by the `Processor` class in the Hugging Face `transformers`
  library.
- **Flexibility:** image input today, with the encoder/projector split kept generic.

## How to obtain `mmproj`

Multimodal projector files are specific to each model architecture. For Qwen3.5-VL, use
`convert_hf_to_gguf.py --mmproj` on the Hugging Face checkpoint:

- Qwen3.5-VL (dense and MoE, for example `Qwen/Qwen3.5-9B` / `Qwen/Qwen3.5-35B-A3B`)

The resulting mmproj declares `clip.projector_type = qwen3vl_merger`, which this fork maps to
`PROJECTOR_TYPE_QWEN3VL`.

Pre-quantized mmproj files can also be found alongside the text GGUFs on Hugging Face:

- https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending&search=qwen3.5

## Smoke test

`tests.sh` runs `llama-mtmd-cli` on one text GGUF plus its mmproj:

```sh
MODEL=/path/to/qwen35-vl-text.gguf MMPROJ=/path/to/mmproj-qwen35-vl.gguf ./tools/mtmd/tests.sh
```

`IMAGE` (default `test-1.jpeg`) and `VIDEO` can be set to override the inputs.
