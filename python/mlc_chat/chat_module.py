"""Chat module for MLC chat in a standalone file, including image module for multimodal-purposes."""
#! pylint: disable=unused-import, invalid-name, no-member, too-many-public-methods
import ctypes
import os
import sys

import tvm
import tvm._ffi.base

from . import libinfo, PlaceInPrompt
from .utils import (
    first_idx_mismatch,
    get_device,
    chat_mod_function_names,
    image_mod_function_names,
)


def _load_mlc_llm_lib():
    """Load mlc llm lib"""
    if sys.platform.startswith("win32") and sys.version_info >= (3, 8):
        for path in libinfo.get_dll_directories():
            os.add_dll_directory(path)
    lib_name = "mlc_llm" if tvm._ffi.base._RUNTIME_ONLY else "mlc_llm_module"
    lib_path = libinfo.find_lib_path(lib_name, optional=False)
    return ctypes.CDLL(lib_path[0]), lib_path[0]


if os.environ.get("SKIP_LOADING_MLCLLM_SO", "0") == "0":
    _LIB, _LIB_PATH = _load_mlc_llm_lib()


class ChatModule:
    """A central module supporting MLC-LLM chatting in python, and comes with multimodality.
    It provides both high-level and low-level python interfaces."""

    def __init__(self, device_name: str = "auto", device_id: int = 0):
        r"""Initialize the chat module.

        Parameters
        ----------
        device_name : str
            The name of the device to deploy.
        device_id : int
            The device id.
        """
        self.device = get_device(device_name, device_id)
        self.device_type = self.device.device_type

        fcreate_chat_mod = tvm.get_global_func("mlc.llm_chat_create")
        assert fcreate_chat_mod is not None
        chat_mod = fcreate_chat_mod(self.device_type, device_id)
        fcreate_image_mod = tvm.get_global_func("mlc.llm_image_module_create")
        assert fcreate_image_mod is not None
        image_mod = fcreate_image_mod(self.device_type, device_id)

        for func_name in chat_mod_function_names:
            setattr(self, func_name + "_func", chat_mod[func_name])
        for func_name in image_mod_function_names:
            setattr(self, "image_" + func_name + "_func", image_mod[func_name])

    def generate(self):
        """High-level function."""

    def get_text_embedding(
        self,
        input: str,
        place_in_prompt: PlaceInPrompt = PlaceInPrompt.All,
    ):
        r"""Given a text input, get the embedding of the tokenized prompt.
        User can decide where to place the input in the prompt.

        Parameters
        ----------
        input : str
            The user input string.
        place_in_prompt: PlaceInPrompt
            The place of the input message in the prompt.
        """
        return self.embed_func(input, place_in_prompt.value)

    def get_image_embedding(
        self,
        image: tvm.runtime.NDArray,
    ):
        r"""Given an image of type NDArray, get the embedding of the image.

        Parameters
        ----------
        image : tvm.runtime.NDArray
            The user uploaded image.
        """
        return self.embed_func(image)

    def get_runtime_stats(self) -> str:
        r"""Get the runtime stats text (encoding speed and decoding speed).

        Returns
        -------
        stats : str
            The runtime stats text.
        """
        return self.runtime_stats_text_func()

    def reload_chat_module(
        self, lib: tvm.runtime.Module, model_path: str, app_config_json: str = ""
    ):
        r"""Low-level function. Reload the chat module from the given compiled executable and model path,
        and optionally overwrite the json configuration.

        Parameters
        ----------
        lib : tvm.runtime.Module
            The compiled executable of the chat model.
        model_path : str
            The path to the model parameter folder.
        app_config_json: str
            The json config that is used to partially override the model configuration.
        """
        self.reload_func(lib, model_path, app_config_json)

    def unload_chat_module(self):
        r"""Low-level function. Unload the chat module and clear the global memory."""
        self.unload_func()

    def evaluate(self):
        """Low-level function. For testing purposes only."""
        self.evaluate_func()

    def prefill(
        self,
        input: str,
        decode_next_token: bool = True,
        place_in_prompt: PlaceInPrompt = PlaceInPrompt.All,
    ):
        r"""Low-level function. Run prefill stage for a given input and optionally decode the first output token.
        User can decide where to place the input in the prompt.

        Parameters
        ----------
        input : str
            The user input string.
        decode_next_token : bool
            Whether to decode the next token after prefilling.
        place_in_prompt: PlaceInPrompt
            The place of the input message in the prompt.
        """
        self.prefill_func(input, decode_next_token, place_in_prompt.value)

    def prefill_with_embed(
        self, embedding: tvm.runtime.NDArray, decode_next_token: bool = True
    ):
        r"""Low-level function. Given an embedding, run the prefill stage and optionally decode the first output token.

        Parameters
        ----------
        embedding : tvm.runtime.NDArray
            The embedding of user input.
        decode_next_token : bool
            Whether to decode the next token after prefilling.
        """
        self.prefill_with_embed_func(embedding, decode_next_token)

    def decode(self):
        r"""Low-level function. Decode the next token, the decoding result is stored in a buffer and
        can be retrieved by :func:`get_message`.
        """
        self.decode_func()

    def reset_chat_module(self):
        r"""Low-level function. Reset the chat session and clear all chat history.

        Note
        ----
        The model remains the same after :func:`reset_chat`.
        To reload module, please use :func:`reload` instead.
        """
        self.reset_chat_func()

    def get_role_0(self):
        pass

    def get_role_1(self):
        pass

    def stopped(self) -> bool:
        r"""Low-level function. Check if the stop condition is met for the current round.

        Returns
        -------
        stopped : bool
        """
        return self.stopped_func() != 0

    def get_message(self) -> str:
        r"""Low-level function. Get the output message in the current round.

        Returns
        -------
        message : str

        Note
        ----
        This function returns the message that corresponds to
        all the tokens decoded so far.
        """
        return self.get_message_func()

    def get_chat_module_runtime_stats(self):
        pass

    def reset_chat_module_runtime_stats(self):
        r"""Low-level function. Reset the runtime stats."""
        self.reset_runtime_stats_func()

    def get_chat_module_config_json(self):
        pass

    def process_system_prompts(self):
        r"""Low-level function. Pre-process by prefilling the system prompts, running prior to any user input."""
        self.process_system_prompts_func()

    def reload_image_module(self, lib: str, model_path: str):
        r"""Low-level function. Reload the image module from the given library and model path.

        Parameters
        ----------
        lib : str
            The library path.
        model_path : str
            The model path.
        """
        self.reload_func(lib, model_path)

    def unload_image_module(self):
        pass

    def reset_image_module(self):
        r"""Low-level function. Reset the image module, clear its performance record.

        Note
        ----
        The model remains the same after :func:`reset_image_module`.
        To reload module, please use :func:`reload` instead.
        """
        self.reset_image_module_func()

    def get_image_module_runtime_stats(self) -> str:
        r"""Low-level function. Get the runtime stats text (image encoding speed).

        Returns
        -------
        stats : str
            The runtime stats text.
        """
        return self.runtime_stats_text_func()

    def reset_image_module_runtime_stats(self):
        r"""Low-level function. Reset the runtime stats."""
        self.reset_runtime_stats_func()
