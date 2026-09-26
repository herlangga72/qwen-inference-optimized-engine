from __future__ import annotations

from .base import (
    ModelBase, TextModel, MmprojModel, ModelType, SentencePieceTokenTypes,
    logger, _mistral_common_installed, _mistral_import_error_msg,
    get_model_architecture, LazyTorchTensor,
)
from typing import Type


__all__ = [
    "ModelBase", "TextModel", "MmprojModel", "ModelType", "SentencePieceTokenTypes",
    "get_model_architecture", "LazyTorchTensor", "logger",
    "_mistral_common_installed", "_mistral_import_error_msg",
    "get_model_class", "print_registered_models", "load_all_models",
]


TEXT_MODEL_MAP: dict[str, str] = {
    "Qwen3_5ForCausalLM": "qwen",
    "Qwen3_5ForConditionalGeneration": "qwen",
    "Qwen3_5MoeForCausalLM": "qwen",
    "Qwen3_5MoeForConditionalGeneration": "qwen",
    "Qwen4ExpForCausalLM": "qwen4exp",
    "Qwen4ExpForConditionalGeneration": "qwen4exp",
}


MMPROJ_MODEL_MAP: dict[str, str] = {
    "Qwen3_5ForConditionalGeneration": "qwen3vl",
    "Qwen3_5MoeForConditionalGeneration": "qwen3vl",
    "Qwen4ExpForConditionalGeneration": "qwen4exp",
}


_TEXT_MODEL_MODULES = sorted(set(TEXT_MODEL_MAP.values()))
_MMPROJ_MODEL_MODULES = sorted(set(MMPROJ_MODEL_MAP.values()))


_loaded_text_modules: set[str] = set()
_loaded_mmproj_modules: set[str] = set()


def load_all_models() -> None:
    """Import all model modules to trigger @ModelBase.register() decorators."""
    if len(_loaded_text_modules) != len(_TEXT_MODEL_MODULES):
        for module_name in _TEXT_MODEL_MODULES:
            if module_name not in _loaded_text_modules:
                try:
                    __import__(f"conversion.{module_name}")
                    _loaded_text_modules.add(module_name)
                except Exception as e:
                    logger.warning(f"Failed to load model module {module_name}: {e}")

    if len(_loaded_mmproj_modules) != len(_MMPROJ_MODEL_MODULES):
        for module_name in _MMPROJ_MODEL_MODULES:
            if module_name not in _loaded_mmproj_modules:
                try:
                    __import__(f"conversion.{module_name}")
                    _loaded_mmproj_modules.add(module_name)
                except Exception as e:
                    logger.warning(f"Failed to load model module {module_name}: {e}")


def get_model_class(name: str, mmproj: bool = False) -> Type[ModelBase]:
    """Dynamically import and return a model class by its HuggingFace architecture name."""
    relevant_map = MMPROJ_MODEL_MAP if mmproj else TEXT_MODEL_MAP
    if name not in relevant_map:
        raise NotImplementedError(f"Architecture {name!r} not supported!")
    module_name = relevant_map[name]
    __import__(f"conversion.{module_name}")
    model_type = ModelType.MMPROJ if mmproj else ModelType.TEXT
    return ModelBase._model_classes[model_type][name]


def print_registered_models() -> None:
    load_all_models()
    logger.error("TEXT models:")
    for name in sorted(TEXT_MODEL_MAP.keys()):
        logger.error(f"  - {name}")
    logger.error("MMPROJ models:")
    for name in sorted(MMPROJ_MODEL_MAP.keys()):
        logger.error(f"  - {name}")
