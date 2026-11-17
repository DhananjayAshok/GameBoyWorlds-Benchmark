# Lazy (PEP 562): eager imports here would make every utils submodule import load the whole VLM stack.

import importlib

_EXPORTS = {
    "utils.parameter_handling": ("load_parameters", "compute_secondary_parameters"),
    "utils.log_handling": ("log_error", "log_info", "log_warn"),
    "utils.fundamental": ("file_makedir", "depathify"),
    "utils.vlm": (
        "ExecutorVLM",
        "convert_numpy_greyscale_to_pillow",
        "VLM",
        "ocr",
        "object_detection",
        "identify_matches",
    ),
    "utils.huggingface_inference": ("HuggingFaceModel",),
    "utils.lm_inference": (
        "InferenceModel",
        "OpenAIModel",
        "OpenRouterModel",
        "AnthropicModel",
        "vLLMModel",
        "parse_key_value",
        "parse_yes_no",
        "sum_meta",
        "sum_optional",
        "zero_meta",
    ),
    "utils.parsing": (
        "PLAN_SEPARATOR",
        "parse_action_line",
        "parse_int",
        "parse_list",
        "parse_plan",
        "parse_steps",
    ),
}

_SYMBOL_SOURCE = {
    name: module for module, names in _EXPORTS.items() for name in names
}

__all__ = sorted(_SYMBOL_SOURCE)


def __getattr__(name: str):
    module_name = _SYMBOL_SOURCE.get(name)
    if module_name is None:
        raise AttributeError(f"module 'utils' has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return __all__
