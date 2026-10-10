# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run a short, genuine PyPTO NPU generation or capture a partial network."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from pypto.runtime import ChipWorker, RunConfig
from safetensors import safe_open
from transformers import AutoTokenizer

if __package__:
    from .runner import LAYERS, MAX_SEQ, Qwen3Runner, kernel_functions, kernels, validate_config
else:
    from runner import LAYERS, MAX_SEQ, Qwen3Runner, kernel_functions, kernels, validate_config


def probe_lm_head(model_root: str, reference_path: Path, device: int) -> dict:
    """Feed the head an independently captured HF final-normalization vector."""
    reference = torch.load(reference_path, weights_only=True)
    hidden = reference["hidden_states"][-1].reshape(-1, 2560)
    if tuple(hidden.shape) != (1, 2560) or hidden.dtype != torch.bfloat16:
        raise ValueError("Head reference must contain one BF16 final-normalization vector")
    expected = reference["logits"].reshape(-1).float()
    model_path = Path(model_root).resolve()
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    shard_path = (model_path / index["weight_map"]["model.embed_tokens.weight"]).resolve(strict=True)
    if not shard_path.is_relative_to(model_path):
        raise ValueError("Embedding shard escapes MODEL_ROOT")
    config = RunConfig(platform="a2a3", device_id=device)
    compile_config = RunConfig(platform="a2a3", device_id=device, save_kernels=True,
                               save_kernels_dir=str(Path(os.environ["PYPTO_PROG_BUILD_DIR"]) / "head_probe"))
    with safe_open(shard_path, framework="pt", device="cpu") as shard, ChipWorker(config=config) as worker:
        weight = shard.get_tensor("model.embed_tokens.weight")
        if tuple(weight.shape) != (151936, 2560) or weight.dtype != torch.bfloat16:
            raise ValueError("Tied embedding does not match the fixed head ABI")
        x = worker.alloc_tensor((16, 2560), torch.bfloat16, init=hidden.repeat(16, 1))
        w = worker.alloc_tensor(weight.shape, weight.dtype, init=weight)
        y = worker.alloc_tensor((16, 151936), torch.float32)
        compiled = kernels.lm_head.compile(config=compile_config)
        worker.run(compiled, x, w, y, config=config)
        actual = torch.empty(151936, dtype=torch.float32)
        worker.copy_from(actual.data_ptr(), y.data_ptr, actual.numel() * actual.element_size())
    if not torch.isfinite(actual).all().item():
        raise FloatingPointError("LM head probe returned non-finite logits")
    difference = actual - expected
    return {
        "backend": "pypto_npu", "diagnostic_only": True, "component": "lm_head",
        "reference": str(reference_path.resolve()), "model_root": str(model_path),
        "device_id": device, "kernel_path": str(Path(kernels.__file__).resolve()),
        "kernel_sha256": hashlib.sha256(Path(kernels.__file__).read_bytes()).hexdigest(),
        "max_abs": difference.abs().max().item(),
        "relative_l2": (difference.norm() / expected.norm().clamp_min(1e-12)).item(),
        "cosine": torch.nn.functional.cosine_similarity(actual, expected, dim=0).item(),
        "actual_top5": actual.topk(5).indices.tolist(),
        "expected_top5": expected.topk(5).indices.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", default=os.environ.get("MODEL_ROOT"))
    parser.add_argument("--prompt", default="Reply with exactly: Hello world.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--head-reference", type=Path)
    parser.add_argument("--diagnostic-layers", type=int, default=LAYERS)
    parser.add_argument("--diagnostic-token-count", type=int, default=1)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--verbose-layers", action="store_true")
    args = parser.parse_args()
    if not args.model_root:
        parser.error("MODEL_ROOT or --model-root is required")
    workspace = Path(os.environ["QWEN_WORKSPACE"]).resolve()
    output = (args.output or Path(os.environ["QWEN_RUN_ROOT"]) / "generation.json").resolve()
    for path in (output, args.capture_dir):
        if path is not None and not path.resolve().is_relative_to(workspace):
            parser.error("Output and capture paths must stay inside the workspace")
    config = validate_config(args.model_root)

    def progress(event: dict) -> None:
        if args.verbose_layers or event["event"] != "layer_done":
            print(json.dumps(event), flush=True)

    started = time.monotonic()
    if args.head_reference is not None:
        result = probe_lm_head(args.model_root, args.head_reference, args.device)
        result["seconds"] = time.monotonic() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return
    if args.compile_only:
        for name, function in kernel_functions().items():
            run_config = RunConfig(platform="a2a3", device_id=args.device, save_kernels=True,
                                   save_kernels_dir=str(Path(os.environ["PYPTO_PROG_BUILD_DIR"]) / "qwen3_4b" / name))
            progress({"event": "compile_start", "kernel": name})
            artifact = function.compile(config=run_config)
            progress({"event": "compile_end", "kernel": name, "output_dir": str(artifact.output_dir)})
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, local_files_only=True)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=True,
        add_generation_prompt=True, return_dict=True,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            parser.error("Only batch size one is supported")
        input_ids = input_ids[0]
    if not 0 < len(input_ids) <= MAX_SEQ:
        parser.error(f"Prompt must have between 1 and {MAX_SEQ} tokens")
    if args.diagnostic_layers != LAYERS and args.capture_dir is None:
        parser.error("Partial-network diagnostics require --capture-dir")
    if args.capture_dir is None and len(input_ids) + args.max_new_tokens > MAX_SEQ:
        parser.error(f"Prompt plus generation exceeds {MAX_SEQ} tokens")
    result = {
        "backend": "pypto_npu", "platform": "a2a3", "device_id": args.device,
        "model_root": str(Path(args.model_root).resolve()), "prompt": args.prompt,
        "input_ids": input_ids, "layers": args.diagnostic_layers,
        "runner_path": str(Path(inspect.getfile(Qwen3Runner)).resolve()),
        "kernel_path": str(Path(kernels.__file__).resolve()),
        "source_revision": subprocess.check_output(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("runner.py", "generate.py", "kernels.py")},
    }
    with Qwen3Runner(args.model_root, device_id=args.device, layers=args.diagnostic_layers,
                     progress=progress) as runner:
        if args.capture_dir is not None:
            args.capture_dir.mkdir(parents=True, exist_ok=True)
            if not 1 <= args.diagnostic_token_count <= len(input_ids):
                parser.error("diagnostic-token-count must fit the tokenized prompt")
            for index, token_id in enumerate(input_ids[:args.diagnostic_token_count]):
                capture = {}
                runner.forward_token(token_id, compute_logits=args.diagnostic_layers == LAYERS, capture=capture)
                torch.save({"token_id": token_id, "position": index, "tensors": capture},
                           args.capture_dir / f"token_{index:03d}.pt")
            result["capture_dir"] = str(args.capture_dir.resolve())
            result["diagnostic_only"] = True
        else:
            eos = config.get("eos_token_id", tokenizer.eos_token_id)
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            generated_ids = runner.generate(input_ids, args.max_new_tokens, eos_ids)
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            result.update(generated_ids=generated_ids, text=text, diagnostic_only=False)
            if not text.strip():
                raise RuntimeError("PyPTO generated no readable non-special text")
    result["seconds"] = time.monotonic() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
