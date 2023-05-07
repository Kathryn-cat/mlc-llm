import argparse
import os
import pickle
from platform import system
from typing import List

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm import relax

import mlc_llm
from mlc_llm import utils
from mlc_llm.relax_model import gpt_neox, llama, minigpt, moss


def _parse_args():
    args = argparse.ArgumentParser()
    utils.argparse_add_common(args)
    args.add_argument("--quantization-sym", action="store_true", default=False)
    args.add_argument(
        "--quantization-mode", type=str, choices=["int4", "int3", "fp4"], default="int4"
    )
    args.add_argument(
        "--quantization-storage-nbit", type=int, choices=[32, 16], default=32
    )
    args.add_argument("--no-quantize", action="store_true", default=False)
    args.add_argument("--max-seq-len", type=int, default=-1)
    args.add_argument("--target", type=str, default="auto")
    args.add_argument("--db-path", type=str, default="log_db/")
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument(
        "--use-cache",
        type=int,
        default=1,
        help="Whether to use previously pickled IRModule and skip trace.",
    )
    args.add_argument("--debug-dump", action="store_true", default=False)
    args.add_argument("--debug-load-script", action="store_true", default=False)

    args.add_argument(
        "--llvm-mingw",
        type=str,
        default="",
        help="/path/to/llvm-mingw-root, use llvm-mingw to cross compile to windows",
    )
    args.add_argument("--system-lib", action="store_true", default=False)

    parsed = args.parse_args()
    parsed.model_path = os.path.join(parsed.artifact_path, "models", parsed.model)
    parsed.artifact_path = os.path.join(
        parsed.artifact_path, parsed.model, parsed.dtype
    )
    parsed.export_kwargs = {}
    assert parsed.max_seq_len == -1 or parsed.max_seq_len > 0
    parsed.lib_format = "so"

    if parsed.target == "auto":
        if system() == "Darwin":
            target = tvm.target.Target("apple/m1-gpu")
        else:
            has_gpu = tvm.cuda().exist
            target = tvm.target.Target("cuda" if has_gpu else "llvm")
        print(f"Automatically configuring target: {target}")
        parsed.target = tvm.target.Target(target, host="llvm")
        parsed.target_kind = parsed.target.kind.default_keys[0]
    elif parsed.target == "webgpu":
        parsed.target = tvm.target.Target(
            "webgpu", host="llvm -mtriple=wasm32-unknown-unknown-wasm"
        )
        parsed.target_kind = "webgpu"
        parsed.lib_format = "wasm"
    elif parsed.target.startswith("iphone"):
        from tvm.contrib import cc, xcode

        # override
        @tvm.register_func("tvm_callback_metal_compile")
        def compile_metal(src):
            return xcode.compile_metal(src, sdk="iphoneos")

        dylib = parsed.target == "iphone-dylib"

        parsed.target = tvm.target.Target(
            tvm.target.Target(
                {
                    "kind": "metal",
                    "max_threads_per_block": 256,
                    "max_shared_memory_per_block": 32768,
                    "thread_warp_size": 1,
                }
            ),
            host="llvm -mtriple=arm64-apple-darwin",
        )
        parsed.target_kind = "iphone"
        parsed.export_kwargs = {
            "fcompile": cc.create_staticlib,
        }
        parsed.lib_format = "a"

        if dylib:
            parsed.export_kwargs = {
                "fcompile": xcode.create_dylib,
                "sdk": "iphoneos",
                "arch": "arm64",
            }
            parsed.lib_format = "dylib"
        else:
            parsed.system_lib = True

    elif parsed.target == "vulkan":
        parsed.target = tvm.target.Target(
            tvm.target.Target(
                {
                    "kind": "vulkan",
                    "max_threads_per_block": 256,
                    "max_shared_memory_per_block": 32768,
                    "thread_warp_size": 1,
                    "supports_float16": 1,
                    "supports_int16": 1,
                    "supports_16bit_buffer": 1,
                }
            ),
            host="llvm",
        )
        parsed.target_kind = parsed.target.kind.default_keys[0]
    elif parsed.target == "metal_x86_64":
        from tvm.contrib import xcode

        parsed.target = tvm.target.Target(
            tvm.target.Target(
                {
                    "kind": "metal",
                    "max_threads_per_block": 256,
                    "max_shared_memory_per_block": 32768,
                    "thread_warp_size": 1,
                }
            ),
            host="llvm -mtriple=x86_64-apple-darwin",
        )
        parsed.target_kind = "metal_x86_64"
        parsed.export_kwargs = {
            "fcompile": xcode.create_dylib,
            "sdk": "macosx",
            "arch": "x86_64",
        }
        parsed.lib_format = "dylib"
    else:
        parsed.target = tvm.target.Target(parsed.target, host="llvm")
        parsed.target_kind = parsed.target.kind.default_keys[0]

    # use mingw to cross compile windows
    if parsed.llvm_mingw != "":
        from tvm.contrib.cc import cross_compiler

        parsed.export_kwargs = {
            "fcompile": cross_compiler(
                os.path.join(parsed.llvm_mingw, "bin", "x86_64-w64-mingw32-clang++"),
                output_format="dll",
            ),
        }
        parsed.target = parsed.target.with_host("llvm -mtriple=x86_64-w64-windows-gnu")
        parsed.lib_format = "dll"

    utils.argparse_postproc_common(parsed)
    return parsed


def debug_dump_script(mod, name, args):
    """Debug dump mode"""
    if not args.debug_dump:
        return
    dump_path = os.path.join(args.artifact_path, "debug", name)
    with open(dump_path, "w") as outfile:
        outfile.write(mod.script(show_meta=True))
    print(f"Dump mod to {dump_path}")


def debug_load_script(name, args):
    input_path = os.path.join(args.artifact_path, "debug", name)
    lib = {"__file__": input_path}
    exec(compile(open(input_path, "rb").read(), input_path, "exec"), lib, lib)
    return lib["Module"]


def debug_dump_shader(ex, name, args):
    """Debug dump mode"""
    if not args.debug_dump:
        return
    target_kind = args.target.kind.default_keys[0]
    suffix_map = {
        "webgpu": ".wgsl",
        "cuda": ".cu",
        "metal": ".mtl",
    }
    suffix = suffix_map.get(target_kind, ".txt")
    dump_path = os.path.join(args.artifact_path, "debug", name + suffix)
    source = ex.mod.imported_modules[0].imported_modules[0].get_source()
    with open(dump_path, "w") as outfile:
        outfile.write(source)
    print(f"Dump shader to {dump_path}")


def mod_transform_before_build(
    mod: tvm.IRModule,
    model_params: List[tvm.nd.NDArray],
    args: argparse.Namespace,
) -> tvm.IRModule:
    """First-stage: Legalize ops and trace"""
    # model_names = ["encoding", "decoding", "create_kv_cache"]
    model_names = ["encoding"]

    if not args.no_quantize:
        mod = mlc_llm.transform.GroupQuantize(
            group_size=40 if args.quantization_mode.endswith("3") else 32,
            sym=args.quantization_sym,
            mode=args.quantization_mode,
            storage_nbit=args.quantization_storage_nbit,
            dtype=args.dtype,
        )(mod)
    mod = mlc_llm.transform.FuseTransposeMatmul()(mod)

    mod = relax.pipeline.get_pipeline()(mod)
    mod = mlc_llm.transform.FuseDecodeMatmulEwise(args.dtype)(mod)
    mod = relax.transform.DeadCodeElimination(model_names)(mod)
    mod = relax.transform.LiftTransformParams()(mod)
    mod_transform, mod_deploy = utils.split_transform_deploy_mod(mod, model_names)

    debug_dump_script(mod_transform, "mod_lift_params.py", args)

    new_params = utils.transform_params(mod_transform, model_params)
    utils.save_params(new_params, args.artifact_path)
    return mod_deploy, new_params


def build(mod_deploy: tvm.IRModule, args: argparse.Namespace) -> None:
    target_kind = args.target_kind
    debug_dump_script(mod_deploy, "mod_before_build.py", args)
    if target_kind != "cpu":
        from tvm import meta_schedule as ms

        db = ms.database.create(work_dir=args.db_path)
        with db, tvm.target.Target("apple/m1-gpu-restricted"):
            mod_deploy = relax.transform.MetaScheduleApplyDatabase()(mod_deploy)
            mod_deploy = mlc_llm.transform.DispatchTIROperator()(mod_deploy)
            mod_deploy = tvm.tir.transform.DefaultGPUSchedule()(mod_deploy)
            mod_deploy = tvm.tir.transform.ForceNarrowIndexToInt32()(mod_deploy)

    if args.debug_load_script:
        mod_deploy = debug_load_script("mod_build_stage_debug.py", args)

    debug_dump_script(mod_deploy, "mod_build_stage.py", args)

    ex = relax.build(mod_deploy, args.target, system_lib=args.system_lib)

    output_filename = f"{args.model}_{target_kind}_{args.dtype}.{args.lib_format}"

    debug_dump_shader(ex, f"{args.model}_{target_kind}_{args.dtype}", args)
    lib_path = os.path.join(args.artifact_path, output_filename)
    ex.export_library(lib_path, **args.export_kwargs)
    print(f"Finish exporting to {lib_path}")

    # TODO: for testing correctness
    executable = tvm.runtime.load_module(lib_path)
    vm = relax.vm.VirtualMachine(executable, tvm.cuda())

    return vm


if __name__ == "__main__":
    ARGS = _parse_args()
    os.makedirs(ARGS.artifact_path, exist_ok=True)
    os.makedirs(os.path.join(ARGS.artifact_path, "debug"), exist_ok=True)
    cache_path = os.path.join(
        ARGS.artifact_path, f"mod_cache_before_build_{ARGS.dtype}.pkl"
    )
    use_cache = ARGS.use_cache and os.path.isfile(cache_path)
    if not use_cache:
        if ARGS.model.startswith("vicuna-") or ARGS.model.startswith("llama-"):
            mod, params = llama.get_model(ARGS)
        elif ARGS.model.startswith("dolly-") or ARGS.model.startswith("stablelm-"):
            mod, params = gpt_neox.get_model(ARGS.model, ARGS.model_path, ARGS.dtype)
        elif ARGS.model.startswith("moss-"):
            mod, params = moss.get_model(ARGS)
        elif ARGS.model.startswith("minigpt4-"):
            mod, params = minigpt.get_model(ARGS)
        else:
            raise ValueError(f"Model {ARGS.model} not supported")
        mod, new_params = mod_transform_before_build(mod, params, ARGS)
        with open(cache_path, "wb") as outfile:
            pickle.dump(mod, outfile)
        print(f"Save a cached module to {cache_path}.")
    else:
        print(
            f"Load cached module from {cache_path} and skip tracing. "
            "You can use --use-cache=0 to retrace"
        )
        mod = pickle.load(open(cache_path, "rb"))
    vm = build(mod, ARGS)

    # TODO: for testing correctness
    import torch

    torch.manual_seed(0)
    input = torch.rand(1, 3, 224, 224)
    new_params_cuda = [tvm.nd.array(x, tvm.cuda()) for x in new_params]
    args_tvm = [tvm.nd.array(input, tvm.cuda()), new_params_cuda]

    res = vm["encoding"](*args_tvm)

    print(f"res: {res}")

    # apply tuning to get an optimized version
    target = tvm.target.Target("nvidia/geforce-rtx-3090")
    ms.relax_integration.tune_relax(
        mod=mod,
        target=target,
        params={},
        work_dir="tuned_minigpt4",
        num_trials_per_iter=32,
        max_trials_global=40000,
        max_trials_per_task=8,
    )
