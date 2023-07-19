"""Common utils for the chat module."""
#! pylint: disable=unused-import, invalid-name
from enum import Enum

import tvm

# functions used in chat module
chat_mod_function_names = [
    "reload",
    "unload",
    "evaluate",
    "prefill",
    "embed",
    "prefill_with_embed",
    "decode",
    "reset_chat",
    "get_role0",
    "get_role1",
    "stopped",
    "get_message",
    "runtime_stats_text",
    "reset_runtime_stats",
    "get_config_json",
    "process_system_prompts",
]

# functions used in image module
image_mod_function_names = [
    "reload",
    "unload",
    "embed",
    "reset",
    "runtime_stats_text",
    "reset_runtime_stats",
]


class PlaceInPrompt(Enum):
    """The place of an input message in a prompt."""

    # The input message should have role names and corresponding seperators appended both prior to it and after it,
    # making it a complete prompt.
    All = 0
    # The input message is only the beginning part of a prompt, no role name and separator should be appended after
    # the message since there will be future messages appended after the message.
    Begin = 1
    # The input message is in the middle of a prompt, nothing should be appended before or after the message.
    Middle = 2
    # The input message is the ending part of a prompt, no role name and separator should be appended prior to it
    # since the message is concatenated to some prior messages.
    End = 3


def get_device(device_name: str = "auto", device_id: int = 0) -> tvm.runtime.Device:
    """Get the device based on the device name."""
    suggested_input = ["llvm", "cuda", "vulkan", "metal", "rocm", "opencl", "auto"]
    valid_input = suggested_input + ["cpu", "gpu"]
    if device_name not in valid_input:
        raise ValueError(
            f"device name is not valid, please enter one of the following: {suggested_input}. \
                If using 'auto', device will be automatically detected."
        )
    device = None
    if device_name in ["llvm", "cpu"]:
        device = tvm.cpu(device_id)
    elif device_name in ["cuda", "gpu"]:
        device = tvm.cuda(device_id)
    elif device_name == "vulkan":
        device = tvm.vulkan(device_id)
    elif device_name == "metal":
        device = tvm.metal(device_id)
    elif device_name == "rocm":
        device = tvm.rocm(device_id)
    elif device_name == "opencl":
        device = tvm.opencl(device_id)
    elif device_name == "auto":
        device = detect_local_device()

    return device


def detect_local_device(device_id: int = 0):
    """automatically detect local device when it is not specified."""
    if tvm.cuda().exist:
        return tvm.cuda(device_id)
    elif tvm.vulkan().exist:
        return tvm.vulkan(device_id)
    elif tvm.metal().exist:
        return tvm.metal(device_id)
    elif tvm.opencl().exist:
        return tvm.opencl(device_id)

    print("Failed to detect local GPU, falling back to CPU as a target")
    return tvm.cpu(device_id)


def first_idx_mismatch(str1: str, str2: str) -> int:
    """Find the first index that mismatch in two strings."""
    for i, (char1, char2) in enumerate(zip(str1, str2)):
        if char1 != char2:
            return i
    return min(len(str1), len(str2))


def quantization_keys():
    """The keys of available quantization methods."""
    return [
        "autogptq_llama_q4f16_0",
        "q0f16",
        "q0f32",
        "q3f16_0",
        "q3f16_1",
        "q4f16_0",
        "q4f16_1",
        "q4f32_0",
        "q4f32_1",
        "q8f16_0",
    ]
