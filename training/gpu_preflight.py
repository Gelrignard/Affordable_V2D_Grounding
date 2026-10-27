"""Compile a CUDA kernel and execute both frozen SONIC models before scene setup."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from .bundles import sha256
from .runtime import runtime_source


@wp.kernel
def _increment(values: wp.array(dtype=float)):
    i = wp.tid()
    values[i] = values[i] + 1.0


def preflight(runtime):
    import onnxruntime as ort

    source = runtime_source(runtime)
    assets = source / "flash_chord/assets/policies/sonic"
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot use CUDA")
    wp.init()
    # This invokes a JIT-compiled CUDA kernel, beyond merely allocating memory.
    values = wp.array(np.arange(16, dtype=np.float32), device="cuda:0")
    wp.launch(_increment, dim=16, inputs=[values], device="cuda:0")
    wp.synchronize()
    np.testing.assert_allclose(values.numpy(), np.arange(16) + 1)
    models = {}
    for name, width, output_width, expected in (
        (
            "encoder_batched.onnx",
            1762,
            64,
            "6071c943492cb6c91e99b392200a288e829f5fea9491b530b34451fcf9e8d02a",
        ),
        (
            "decoder_batched.onnx",
            994,
            29,
            "de629511bc3a110a26827a54389e82d35a722ab9b7cc4232aab865b8451d2a18",
        ),
    ):
        path = assets / name
        if sha256(path) != expected:
            raise ValueError(f"SONIC model hash mismatch: {path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CUDAExecutionProvider"]
        )
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                "ONNX Runtime fell back to CPU. Version 1.28's default PyPI wheel "
                "uses CUDA 13; a CUDA 12 host needs Microsoft's CUDA 12 feed build."
            )
        inputs = session.get_inputs()[0]
        result = session.run(None, {inputs.name: np.zeros((1, width), np.float32)})[0]
        if result.shape != (1, output_width) or not np.isfinite(result).all():
            raise RuntimeError(f"invalid SONIC output: {name}")
        models[name] = {
            "sha256": expected,
            "provider": session.get_providers()[0],
            "output_shape": list(result.shape),
            "finite": True,
        }
        del session
    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch_cuda": torch.version.cuda,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "warp-lang",
                "newton",
                "mujoco",
                "mujoco-warp",
                "onnxruntime-gpu",
            )
        },
        "warp_cuda_kernel": "passed",
        "sonic": models,
        "scope": "GPU dependency/model inference check, not a physical rollout",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = json.dumps(preflight(args.runtime), indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
