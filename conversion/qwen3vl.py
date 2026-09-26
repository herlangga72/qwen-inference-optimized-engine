from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf, logger


@ModelBase.register("Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration")
@ModelBase.example("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-35B-A3B")
class Qwen3VLVisionModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.hparams_vision is None:
            logger.info("No vision config found, skipping vision tensor processing")
            return

        # Compute image_size if not present
        if "image_size" not in self.hparams_vision:
            # For Qwen3VL/Qwen3VLMoe, compute from num_position_embeddings
            num_pos = self.hparams_vision.get("num_position_embeddings", 2304)
            patch_size = self.hparams_vision.get("patch_size", 16)
            # num_position_embeddings = (image_size / patch_size) ** 2
            # So image_size = sqrt(num_position_embeddings) * patch_size
            image_size = int(num_pos**0.5 * patch_size)
            self.hparams_vision["image_size"] = image_size

        # Rename config values for compatibility
        self.hparams_vision["num_attention_heads"] = self.hparams_vision.get("num_heads")
        self.hparams_vision["num_hidden_layers"] = self.hparams_vision.get("depth")

        self.is_deepstack_layers = [False] * int(self.hparams_vision["num_hidden_layers"] or 0)
        for idx in self.hparams_vision.get("deepstack_visual_indexes", []):
            self.is_deepstack_layers[idx] = True

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # in case mixed modalities, the arch will be handled by subclass
        if not self.has_audio_encoder:
            self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.QWEN3VL)
        self.gguf_writer.add_vision_use_gelu(True)

        if self.hparams_vision is not None:
            merge_size = self.hparams_vision.get("spatial_merge_size")
            if merge_size is not None:
                self.gguf_writer.add_vision_spatial_merge_size(int(merge_size))

        # Use text config's rms_norm_eps for vision attention layernorm eps
        rms_norm_eps = self.global_config.get("text_config", {}).get("rms_norm_eps", 1e-6)
        self.gguf_writer.add_vision_attention_layernorm_eps(rms_norm_eps)

        if self.is_deepstack_layers:
            self.gguf_writer.add_vision_is_deepstack_layers(self.is_deepstack_layers)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # Skip text model tensors
        if name.startswith("lm_head."):
            return None

        # Skip MTP tensors
        if name.startswith("mtp."):
            return None

        if name.startswith("model.visual."):
            name = name.replace("model.visual.", "visual.", 1)

        if not name.startswith("visual."):
            return None

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        assert self.hparams_vision is not None

        if name.startswith("visual.deepstack_merger_list."):
            prefix, rest = name.split(".", maxsplit=3)[2:]
            # prefix is the layer index, convert to absolute clip layer index!
            idx = self.hparams_vision.get("deepstack_visual_indexes", [])[int(prefix)]
            target = rest

            tensor_type: gguf.MODEL_TENSOR
            if target.startswith("norm."):
                tensor_type = gguf.MODEL_TENSOR.V_DS_NORM
                suffix = target.split(".", 1)[1]
            elif target.startswith("linear_fc1."):
                tensor_type = gguf.MODEL_TENSOR.V_DS_FC1
                suffix = target.split(".", 1)[1]
            elif target.startswith("linear_fc2."):
                tensor_type = gguf.MODEL_TENSOR.V_DS_FC2
                suffix = target.split(".", 1)[1]
            else:
                raise ValueError(f"Unexpected deepstack tensor: {name}")

            new_name = self.format_tensor_name(tensor_type, idx, suffix=f".{suffix}")
            yield from super().modify_tensors(data_torch, new_name, bid)
            return

        if name.startswith("visual.merger."):
            suffix = name.split(".", 2)[2]
            if suffix.startswith("linear_fc"):
                fc_idx_str, tail = suffix.split(".", 1)
                fc_num = int(fc_idx_str.replace("linear_fc", ""))
                # Qwen3VL has linear_fc1 and linear_fc2
                # Map to indices 0 and 2 (matching Qwen2VL which uses indices 0 and 2)
                if fc_num == 1:
                    fc_idx = 0
                elif fc_num == 2:
                    fc_idx = 2
                else:
                    raise ValueError(f"unexpected fc index {fc_num} in {name}")
                new_name = self.format_tensor_name(gguf.MODEL_TENSOR.V_MMPROJ, fc_idx, suffix=f".{tail}")
            elif suffix.startswith("norm."):
                new_name = self.format_tensor_name(gguf.MODEL_TENSOR.V_POST_NORM, suffix=f".{suffix.split('.', 1)[1]}")
            else:
                raise ValueError(f"Unexpected merger tensor: {name}")
            yield (new_name, data_torch)
            return

        if name == "visual.patch_embed.proj.weight":
            # split Conv3D into Conv2Ds along temporal dimension
            c1, c2, kt, _, _ = data_torch.shape
            del c1, c2
            if kt != 2:
                raise ValueError("Current implementation only supports temporal_patch_size of 2")
            yield (gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH] + ".weight", data_torch[:, :, 0, ...])
            yield (gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH] + ".weight.1", data_torch[:, :, 1, ...])
            return

        if name == "visual.patch_embed.proj.bias":
            # Include the bias - it's used by the C++ code
            yield (gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH] + ".bias", data_torch)
            return

        yield from MmprojModel.modify_tensors(self, data_torch, name, bid)


