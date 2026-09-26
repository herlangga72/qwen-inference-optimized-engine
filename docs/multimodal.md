# Multimodal

This fork supports multimodal input via `libmtmd`, for the Qwen3.5 family only. The supported
projector is `QWEN3VL`, used by the vision tower of Qwen3.5-VL (dense and MoE) and of Qwen4Exp
(which reuses an unmodified Qwen3-VL ViT). Audio input is not supported in this fork.

There are 2 tools that support this feature:
- [llama-cli](../tools/cli/README.md)
- [llama-server](../tools/server/README.md) via OpenAI-compatible `/chat/completions` API
- [llama-mtmd-cli](../tools/mtmd/README.md), for testing and development

To enable it, you can use one of the 2 methods below:

- Use `-hf` option with a supported model
    - To load a model using `-hf` while disabling multimodal, use `--no-mmproj`
    - To load a model using `-hf` while using a custom mmproj file, use `--mmproj local_file.gguf`
- Use `-m model.gguf` option with `--mmproj file.gguf` to specify text and multimodal projector respectively

By default, multimodal projector will be offloaded to GPU. To disable this, add `--no-mmproj-offload`

For example:

```sh
# simple usage with CLI (text GGUF plus its matching mmproj GGUF)
llama-mtmd-cli -m qwen35-vl-Q4_K_M.gguf --mmproj mmproj-qwen35-vl-f16.gguf

# simple usage with server
llama-server -m qwen35-vl-Q4_K_M.gguf --mmproj mmproj-qwen35-vl-f16.gguf

# no GPU offload
llama-server -m qwen35-vl-Q4_K_M.gguf --mmproj mmproj-qwen35-vl-f16.gguf --no-mmproj-offload
```

Notes:

- the text part of Qwen3.5-VL is a plain `qwen35` (or `qwen35moe`) GGUF, so it can also be run
  text-only with any of the other tools
- the projector GGUF must be the one that belongs to the text model, otherwise the image embeddings
  will not line up
- image input can be a URL, a local path or base64, depending on the tool

## Pre-quantized models

GGUF models with vision capabilities for the Qwen3.5 family can be searched on Hugging Face:

- https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending&search=qwen3.5

Pick a repository that ships both the text GGUF and a matching `mmproj-*.gguf` file, then replace
the file names in the examples above with the actual local paths (`-m` for the text model, `--mmproj`
for the projector).
