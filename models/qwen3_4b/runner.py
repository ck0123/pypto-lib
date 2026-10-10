# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-request Qwen3-4B runner with persistent PyPTO device tensors."""

from __future__ import annotations

import ctypes
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

import torch
from pypto.runtime import ChipWorker, DeviceTensor, RunConfig
from safetensors import safe_open

if __package__:
    from . import kernels
else:
    import kernels

if Path(kernels.__file__).resolve() != Path(__file__).with_name("kernels.py").resolve():
    raise ImportError("Qwen3-4B runner loaded kernels from a different model directory")

BATCH = 16
MAX_SEQ = 128
HIDDEN = 2560
INTERMEDIATE = 9728
Q_WIDTH = 4096
KV_WIDTH = 1024
VOCAB = 151936
LAYERS = 36


def validate_config(model_root: str | Path) -> dict[str, Any]:
    """Reject checkpoints that do not implement this fixed network ABI."""
    root = Path(model_root).resolve(strict=True)
    config = json.loads((root / "config.json").read_text())
    expected = {
        "model_type": "qwen3",
        "num_hidden_layers": LAYERS,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": VOCAB,
        "rope_theta": 5000000,
        "tie_word_embeddings": True,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "hidden_act": "silu",
    }
    mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    dtype = config.get("dtype", config.get("torch_dtype"))
    if dtype != "bfloat16":
        mismatch["dtype"] = (dtype, "bfloat16")
    if config.get("rope_scaling") is not None or config.get("use_sliding_window", False):
        mismatch["attention_variant"] = ("scaled RoPE or sliding window", "full attention")
    if mismatch:
        raise ValueError(f"Checkpoint does not match the Qwen3-4B kernel ABI: {mismatch}")
    return config


def kernel_functions() -> dict[str, Any]:
    """Return the fixed primitive kernels, shared across every layer."""
    return {
        "embedding": kernels.embedding,
        "rms_norm": kernels.rms_norm,
        "q_proj": kernels.make_linear(HIDDEN, Q_WIDTH),
        "kv_proj": kernels.make_linear(HIDDEN, KV_WIDTH),
        "o_proj": kernels.make_linear(Q_WIDTH, HIDDEN),
        "mlp_up": kernels.make_linear(HIDDEN, INTERMEDIATE),
        "mlp_down": kernels.make_linear(INTERMEDIATE, HIDDEN),
        "lm_head": kernels.lm_head,
        "q_rope": kernels.q_rope,
        "append_kv": kernels.append_kv,
        "attention": kernels.attention,
        "add_residual": kernels.add_residual,
        "swiglu": kernels.swiglu,
    }


class Qwen3Runner:
    """Run one causal token at a time; prompt tokens populate the same KV cache.

    Host Torch operations only prepare immutable weights, static RoPE tables,
    transport buffers, and select tokens. Every network primitive runs in PyPTO.
    A runner owns one request's cache and must not be used concurrently.
    """

    def __init__(
        self,
        model_root: str | Path,
        *,
        device_id: int = 0,
        layers: int = LAYERS,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.model_root = Path(model_root).resolve(strict=True)
        self.model_config = validate_config(self.model_root)
        if isinstance(layers, bool) or not 1 <= layers <= LAYERS:
            raise ValueError(f"layers must be in [1, {LAYERS}]")
        self.num_layers = layers
        self.progress = progress
        self.artifact_root = Path(os.environ["PYPTO_PROG_BUILD_DIR"]) / "qwen3_4b"
        self.config = RunConfig(platform="a2a3", device_id=device_id)
        self.position = 0
        self._closed = False
        self._failed = False
        self._handles: dict[str, Any] = {}
        self._functions = kernel_functions()
        self.worker = ChipWorker(config=self.config)
        self.layers: list[dict[str, DeviceTensor]] = []
        try:
            self._load_weights()
            self._allocate_state()
        except BaseException:
            self.close()
            raise

    def _event(self, event: str, **fields: Any) -> None:
        if self.progress is not None:
            self.progress({"event": event, **fields})

    def _upload(self, tensor: torch.Tensor) -> DeviceTensor:
        tensor = tensor.contiguous().cpu()
        return self.worker.alloc_tensor(tensor.shape, tensor.dtype, init=tensor)

    def _load_weights(self) -> None:
        index = json.loads((self.model_root / "model.safetensors.index.json").read_text())
        weight_map = index["weight_map"]
        with ExitStack() as stack:
            shards = {}

            def tensor(name: str, shape: tuple[int, ...]) -> torch.Tensor:
                if name not in weight_map:
                    raise ValueError(f"Checkpoint is missing {name}")
                shard = (self.model_root / weight_map[name]).resolve(strict=True)
                if not shard.is_relative_to(self.model_root):
                    raise ValueError(f"Shard path escapes MODEL_ROOT: {weight_map[name]}")
                if shard not in shards:
                    shards[shard] = stack.enter_context(safe_open(shard, framework="pt", device="cpu"))
                value = shards[shard].get_tensor(name)
                if tuple(value.shape) != shape or value.dtype != torch.bfloat16:
                    raise ValueError(f"{name}: got {tuple(value.shape)}/{value.dtype}, expected {shape}/BF16")
                return value

            embedding = tensor("model.embed_tokens.weight", (VOCAB, HIDDEN))
            self.embedding = self._upload(embedding)
            # The dedicated head reads the original [vocab, hidden] layout.
            self.lm_head = self.embedding
            self.final_norm = self._upload(tensor("model.norm.weight", (HIDDEN,)).float().view(1, HIDDEN))
            del embedding
            specs = {
                "q": ("self_attn.q_proj.weight", (Q_WIDTH, HIDDEN)),
                "k": ("self_attn.k_proj.weight", (KV_WIDTH, HIDDEN)),
                "v": ("self_attn.v_proj.weight", (KV_WIDTH, HIDDEN)),
                "o": ("self_attn.o_proj.weight", (HIDDEN, Q_WIDTH)),
                "gate": ("mlp.gate_proj.weight", (INTERMEDIATE, HIDDEN)),
                "up": ("mlp.up_proj.weight", (INTERMEDIATE, HIDDEN)),
                "down": ("mlp.down_proj.weight", (HIDDEN, INTERMEDIATE)),
            }
            norms = {
                "input_norm": ("input_layernorm.weight", HIDDEN),
                "post_norm": ("post_attention_layernorm.weight", HIDDEN),
                "q_norm": ("self_attn.q_norm.weight", 128),
                "k_norm": ("self_attn.k_norm.weight", 128),
            }
            for layer_id in range(self.num_layers):
                prefix = f"model.layers.{layer_id}."
                layer = {key: self._upload(tensor(prefix + name, shape).t()) for key, (name, shape) in specs.items()}
                layer.update({key: self._upload(tensor(prefix + name, (size,)).float().view(1, size))
                              for key, (name, size) in norms.items()})
                self.layers.append(layer)
                self._event("weights_loaded", layer=layer_id)

    def _allocate_state(self) -> None:
        # RoPE is immutable positional metadata, constructed once in FP32 as HF
        # does. Neither these tables nor converted weights are written to disk.
        inv_freq = 1.0 / (float(self.model_config["rope_theta"]) ** (torch.arange(0, 128, 2).float() / 128))
        frequencies = torch.arange(MAX_SEQ).float().view(-1, 1) * inv_freq.view(1, -1)
        self.cos = self._upload(frequencies.cos())
        self.sin = self._upload(frequencies.sin())
        for layer in self.layers:
            layer["k_cache"] = self._upload(torch.zeros((8 * MAX_SEQ, 128), dtype=torch.bfloat16))
            layer["v_cache"] = self._upload(torch.zeros((8 * MAX_SEQ, 128), dtype=torch.bfloat16))
        shapes = {
            "hidden": HIDDEN, "norm": HIDDEN, "q": Q_WIDTH, "q_rope": Q_WIDTH,
            "k": KV_WIDTH, "v": KV_WIDTH, "attention": Q_WIDTH, "projection": HIDDEN,
            "residual": HIDDEN, "post_norm": HIDDEN, "gate": INTERMEDIATE,
            "up": INTERMEDIATE, "activation": INTERMEDIATE, "down": HIDDEN,
            "final_norm": HIDDEN,
        }
        self.state = {name: self.worker.alloc_tensor((BATCH, width), torch.bfloat16)
                      for name, width in shapes.items()}
        self.logits = self.worker.alloc_tensor((BATCH, VOCAB), torch.float32)

    def _run(self, name: str, *args: Any) -> None:
        if name not in self._handles:
            started = time.monotonic()
            self._event("compile_start", kernel=name)
            compile_config = RunConfig(platform="a2a3", device_id=self.config.device_id,
                                       save_kernels=True, save_kernels_dir=str(self.artifact_root / name))
            compiled = self._functions[name].compile(config=compile_config)
            self._handles[name] = self.worker.register(compiled)
            self._event("compile_end", kernel=name, seconds=time.monotonic() - started,
                        output_dir=str(compiled.output_dir))
        self._handles[name](*args, config=self.config)

    def read_tensor(self, tensor: DeviceTensor, *, first_row: bool = False) -> torch.Tensor:
        shape = (1, tensor.shape[1]) if first_row else tensor.shape
        result = torch.empty(shape, dtype=tensor.dtype)
        self.worker.copy_from(result.data_ptr(), tensor.data_ptr, result.numel() * result.element_size())
        return result

    def reset(self) -> None:
        if self._closed:
            raise RuntimeError("Runner is closed")
        # Attention masks positions above the current token. Every readable
        # prefix slot is overwritten before it can be read in a new request.
        self.position = 0
        self._failed = False

    def forward_token(
        self,
        token_id: int,
        *,
        compute_logits: bool = True,
        capture: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        if self._closed or self._failed:
            raise RuntimeError("Runner is closed or the previous forward failed; reset before reuse")
        if isinstance(token_id, bool) or not isinstance(token_id, int) or not 0 <= token_id < VOCAB:
            raise ValueError(f"token_id must be an integer in [0, {VOCAB})")
        if self.position >= MAX_SEQ:
            raise ValueError(f"Context capacity {MAX_SEQ} exceeded")
        position = ctypes.c_int32(self.position)
        state = self.state

        def save(name: str, value: DeviceTensor, *, full: bool = False) -> None:
            if capture is not None:
                capture[name] = self.read_tensor(value, first_row=not full)

        try:
            self._run("embedding", self.embedding, ctypes.c_int32(token_id), state["hidden"])
            save("embedding", state["hidden"])
            for layer_id, weight in enumerate(self.layers):
                self._run("rms_norm", state["hidden"], weight["input_norm"], state["norm"])
                self._run("q_proj", state["norm"], weight["q"], state["q"])
                self._run("kv_proj", state["norm"], weight["k"], state["k"])
                self._run("kv_proj", state["norm"], weight["v"], state["v"])
                self._run("q_rope", state["q"], weight["q_norm"], self.cos, self.sin, position, state["q_rope"])
                self._run("append_kv", state["k"], state["v"], weight["k_norm"], self.cos, self.sin,
                          position, weight["k_cache"], weight["v_cache"])
                self._run("attention", state["q_rope"], weight["k_cache"], weight["v_cache"],
                          position, state["attention"])
                self._run("o_proj", state["attention"], weight["o"], state["projection"])
                self._run("add_residual", state["hidden"], state["projection"], state["residual"])
                self._run("rms_norm", state["residual"], weight["post_norm"], state["post_norm"])
                self._run("mlp_up", state["post_norm"], weight["gate"], state["gate"])
                self._run("mlp_up", state["post_norm"], weight["up"], state["up"])
                self._run("swiglu", state["gate"], state["up"], state["activation"])
                self._run("mlp_down", state["activation"], weight["down"], state["down"])
                self._run("add_residual", state["residual"], state["down"], state["hidden"])
                if capture is not None:
                    for name, value in state.items():
                        if name != "final_norm":
                            save(f"layer_{layer_id}.{name}", value)
                    save(f"layer_{layer_id}.k_cache", weight["k_cache"], full=True)
                    save(f"layer_{layer_id}.v_cache", weight["v_cache"], full=True)
                self._event("layer_done", position=self.position, layer=layer_id)
            result = None
            if compute_logits:
                self._run("rms_norm", state["hidden"], self.final_norm, state["final_norm"])
                self._run("lm_head", state["final_norm"], self.lm_head, self.logits)
                result = self.read_tensor(self.logits, first_row=True)[0]
                if not torch.isfinite(result).all().item():
                    raise FloatingPointError("PyPTO produced non-finite logits")
                save("final_norm", state["final_norm"])
                if capture is not None:
                    capture["logits"] = result
            self.position += 1
            return result
        except BaseException:
            self._failed = True
            raise

    def prefill(self, token_ids: list[int]) -> torch.Tensor:
        if not token_ids:
            raise ValueError("Prompt must contain at least one token")
        if len(token_ids) > MAX_SEQ:
            raise ValueError(f"Prompt exceeds context capacity {MAX_SEQ}")
        self.reset()
        result = None
        for index, token_id in enumerate(token_ids):
            result = self.forward_token(token_id, compute_logits=index == len(token_ids) - 1)
        assert result is not None
        return result

    def generate(self, token_ids: list[int], max_new_tokens: int, eos_token_ids: set[int]) -> list[int]:
        if self.num_layers != LAYERS:
            raise ValueError("Text generation requires all 36 model layers")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if len(token_ids) + max_new_tokens > MAX_SEQ:
            raise ValueError(f"Prompt plus max_new_tokens exceeds context capacity {MAX_SEQ}")
        logits = self.prefill(token_ids)
        generated = []
        for step in range(max_new_tokens):
            token_id = int(torch.argmax(logits).item())
            generated.append(token_id)
            self._event("token", step=step, token_id=token_id)
            if token_id in eos_token_ids or step + 1 == max_new_tokens:
                break
            logits = self.forward_token(token_id)
            assert logits is not None
        return generated

    def close(self) -> None:
        if not self._closed:
            self.worker.close()
            self._closed = True

    def __enter__(self) -> Qwen3Runner:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
