# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-token Qwen3-4B primitives for resident L2 execution.

The sixteen activation rows contain replicas of one token. Each layer owns
eight contiguous 128-position KV heads, with one cache row per actual token.
Prompt ingestion and autoregressive decoding use the same causal token path.
"""

import pypto.language as pl

BATCH = 16
MAX_SEQ = 128
HIDDEN = 2560
INTERMEDIATE = 9728
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
HALF_DIM = 64
Q_WIDTH = 4096
KV_WIDTH = 1024
VOCAB = 151936
EPS = 1e-6
LINEAR_K_TILE = 128
LINEAR_N_TILE = 128
VECTOR_TILE = 256


@pl.jit
def embedding(
    weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    token_id: pl.Scalar[pl.INT32],
    out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
):
    for chunk in pl.spmd(HIDDEN // VECTOR_TILE, name_hint="token_embedding"):
        row = pl.cast(
            pl.slice(weight, [1, VECTOR_TILE], [token_id, chunk * VECTOR_TILE]), target_type=pl.FP32
        )
        zero = pl.full([BATCH, VECTOR_TILE], dtype=pl.FP32, value=0.0)
        repeated = pl.col_expand_add(zero, row)
        out = pl.assemble(out, pl.cast(repeated, target_type=pl.BF16, mode="rint"), [0, chunk * VECTOR_TILE])
    return out


def make_linear(in_features: int, out_features: int, fp32_out: bool = False):
    """Build a shape-specialized BF16 linear projection with transposed weights."""
    if in_features % LINEAR_K_TILE or out_features % LINEAR_N_TILE:
        raise ValueError("Linear dimensions must be multiples of 128")
    output_dtype = pl.FP32 if fp32_out else pl.BF16

    @pl.jit
    def linear(
        x: pl.Tensor[[BATCH, in_features], pl.BF16],
        weight: pl.Tensor[[in_features, out_features], pl.BF16],
        out: pl.Out[pl.Tensor[[BATCH, out_features], output_dtype]],
    ):
        for chunk in pl.spmd(out_features // LINEAR_N_TILE, name_hint="linear"):
            acc = pl.create_tensor([BATCH, LINEAR_N_TILE], dtype=pl.FP32)
            for kb in pl.pipeline(in_features // LINEAR_K_TILE, stage=2):
                lhs = pl.slice(x, [BATCH, LINEAR_K_TILE], [0, kb * LINEAR_K_TILE])
                rhs = pl.slice(
                    weight,
                    [LINEAR_K_TILE, LINEAR_N_TILE],
                    [kb * LINEAR_K_TILE, chunk * LINEAR_N_TILE],
                )
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(kb == 0))
            if fp32_out:
                out = pl.assemble(out, acc, [0, chunk * LINEAR_N_TILE])
            else:
                projected = pl.cast(acc, target_type=pl.BF16, mode="rint")
                out = pl.assemble(out, projected, [0, chunk * LINEAR_N_TILE])
        return out

    return linear


@pl.jit
def lm_head(
    x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    embedding_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[BATCH, VOCAB], pl.FP32]],
):
    """Project through the original tied embedding layout without a weight copy."""
    for chunk in pl.spmd(VOCAB // LINEAR_N_TILE, name_hint="lm_head"):
        acc = pl.create_tensor([BATCH, LINEAR_N_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(HIDDEN // LINEAR_K_TILE, stage=2):
            lhs = pl.slice(x, [BATCH, LINEAR_K_TILE], [0, kb * LINEAR_K_TILE])
            rhs = pl.slice(
                embedding_weight,
                [LINEAR_N_TILE, LINEAR_K_TILE],
                [chunk * LINEAR_N_TILE, kb * LINEAR_K_TILE],
            )
            acc = pl.matmul_acc(acc, lhs, rhs, b_trans=True, init_cond=(kb == 0))
        out = pl.assemble(out, acc, [0, chunk * LINEAR_N_TILE])
    return out


@pl.jit
def rms_norm(
    x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    gamma: pl.Tensor[[1, HIDDEN], pl.FP32],
    out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rms_norm"):
        squares = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
        for chunk in pl.range(HIDDEN // VECTOR_TILE):
            values = pl.cast(
                pl.slice(x, [BATCH, VECTOR_TILE], [0, chunk * VECTOR_TILE]), target_type=pl.FP32
            )
            reduced = pl.reshape(pl.row_sum(pl.mul(values, values)), [1, BATCH])
            squares = pl.add(squares, reduced)
        inverse = pl.reshape(pl.rsqrt(pl.add(pl.mul(squares, 1.0 / HIDDEN), EPS)), [BATCH, 1])
        for chunk in pl.range(HIDDEN // VECTOR_TILE):
            values = pl.cast(
                pl.slice(x, [BATCH, VECTOR_TILE], [0, chunk * VECTOR_TILE]), target_type=pl.FP32
            )
            scale = pl.slice(gamma, [1, VECTOR_TILE], [0, chunk * VECTOR_TILE])
            normalized = pl.cast(pl.row_expand_mul(values, inverse), target_type=pl.BF16, mode="rint")
            result = pl.col_expand_mul(pl.cast(normalized, target_type=pl.FP32), scale)
            out = pl.assemble(out, pl.cast(result, target_type=pl.BF16, mode="rint"), [0, chunk * VECTOR_TILE])
    return out


@pl.jit
def add_residual(
    a: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    b: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
):
    for chunk in pl.spmd(HIDDEN // VECTOR_TILE, name_hint="add_residual"):
        lhs = pl.cast(pl.slice(a, [BATCH, VECTOR_TILE], [0, chunk * VECTOR_TILE]), target_type=pl.FP32)
        rhs = pl.cast(pl.slice(b, [BATCH, VECTOR_TILE], [0, chunk * VECTOR_TILE]), target_type=pl.FP32)
        out = pl.assemble(out, pl.cast(pl.add(lhs, rhs), target_type=pl.BF16, mode="rint"), [0, chunk * VECTOR_TILE])
    return out


@pl.jit
def swiglu(
    gate: pl.Tensor[[BATCH, INTERMEDIATE], pl.BF16],
    up: pl.Tensor[[BATCH, INTERMEDIATE], pl.BF16],
    out: pl.Out[pl.Tensor[[BATCH, INTERMEDIATE], pl.BF16]],
):
    for chunk in pl.spmd(INTERMEDIATE // VECTOR_TILE, name_hint="swiglu"):
        gate_tile = pl.cast(
            pl.slice(gate, [BATCH, VECTOR_TILE], [0, chunk * VECTOR_TILE]), target_type=pl.FP32
        )
        up_tile = pl.cast(
            pl.slice(up, [BATCH, VECTOR_TILE], [0, chunk * VECTOR_TILE]), target_type=pl.FP32
        )
        activated = pl.mul(gate_tile, pl.recip(pl.add(pl.exp(pl.neg(gate_tile)), 1.0)))
        silu_bf16 = pl.cast(activated, target_type=pl.BF16, mode="rint")
        product = pl.mul(pl.cast(silu_bf16, target_type=pl.FP32), up_tile)
        out = pl.assemble(out, pl.cast(product, target_type=pl.BF16, mode="rint"), [0, chunk * VECTOR_TILE])
    return out


@pl.jit
def q_rope(
    q: pl.Tensor[[BATCH, Q_WIDTH], pl.BF16],
    gamma: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[MAX_SEQ, HALF_DIM], pl.FP32],
    sin: pl.Tensor[[MAX_SEQ, HALF_DIM], pl.FP32],
    position: pl.Scalar[pl.INT32],
    out: pl.Out[pl.Tensor[[BATCH, Q_WIDTH], pl.BF16]],
):
    for head in pl.spmd(NUM_HEADS, name_hint="q_norm_rope"):
        values = pl.cast(pl.slice(q, [BATCH, HEAD_DIM], [0, head * HEAD_DIM]), target_type=pl.FP32)
        squared = pl.row_sum(pl.mul(values, values))
        inverse = pl.rsqrt(pl.add(pl.mul(squared, 1.0 / HEAD_DIM), EPS))
        normalized = pl.cast(pl.row_expand_mul(values, inverse), target_type=pl.BF16, mode="rint")
        gamma_tile = pl.slice(gamma, [1, HEAD_DIM], [0, 0])
        scaled = pl.cast(
            pl.col_expand_mul(pl.cast(normalized, target_type=pl.FP32), gamma_tile),
            target_type=pl.BF16, mode="rint",
        )
        left = pl.cast(pl.slice(scaled, [BATCH, HALF_DIM], [0, 0]), target_type=pl.FP32)
        right = pl.cast(pl.slice(scaled, [BATCH, HALF_DIM], [0, HALF_DIM]), target_type=pl.FP32)
        cos_bf16 = pl.cast(pl.slice(cos, [1, HALF_DIM], [position, 0]), target_type=pl.BF16, mode="rint")
        sin_bf16 = pl.cast(pl.slice(sin, [1, HALF_DIM], [position, 0]), target_type=pl.BF16, mode="rint")
        cosine = pl.cast(cos_bf16, target_type=pl.FP32)
        sine = pl.cast(sin_bf16, target_type=pl.FP32)
        left_cos = pl.cast(pl.col_expand_mul(left, cosine), target_type=pl.BF16, mode="rint")
        right_sin = pl.cast(pl.col_expand_mul(right, sine), target_type=pl.BF16, mode="rint")
        right_cos = pl.cast(pl.col_expand_mul(right, cosine), target_type=pl.BF16, mode="rint")
        left_sin = pl.cast(pl.col_expand_mul(left, sine), target_type=pl.BF16, mode="rint")
        rotated_left = pl.sub(pl.cast(left_cos, target_type=pl.FP32), pl.cast(right_sin, target_type=pl.FP32))
        rotated_right = pl.add(pl.cast(right_cos, target_type=pl.FP32), pl.cast(left_sin, target_type=pl.FP32))
        out = pl.assemble(out, pl.cast(rotated_left, target_type=pl.BF16, mode="rint"), [0, head * HEAD_DIM])
        out = pl.assemble(out, pl.cast(rotated_right, target_type=pl.BF16, mode="rint"), [0, head * HEAD_DIM + HALF_DIM])
    return out


@pl.jit
def append_kv(
    k: pl.Tensor[[BATCH, KV_WIDTH], pl.BF16],
    v: pl.Tensor[[BATCH, KV_WIDTH], pl.BF16],
    gamma: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[MAX_SEQ, HALF_DIM], pl.FP32],
    sin: pl.Tensor[[MAX_SEQ, HALF_DIM], pl.FP32],
    position: pl.Scalar[pl.INT32],
    k_cache: pl.InOut[pl.Tensor[[NUM_KV_HEADS * MAX_SEQ, HEAD_DIM], pl.BF16]],
    v_cache: pl.InOut[pl.Tensor[[NUM_KV_HEADS * MAX_SEQ, HEAD_DIM], pl.BF16]],
):
    for head in pl.spmd(NUM_KV_HEADS, name_hint="k_norm_rope_cache"):
        values = pl.cast(pl.slice(k, [BATCH, HEAD_DIM], [0, head * HEAD_DIM]), target_type=pl.FP32)
        squared = pl.row_sum(pl.mul(values, values))
        inverse = pl.rsqrt(pl.add(pl.mul(squared, 1.0 / HEAD_DIM), EPS))
        normalized = pl.cast(pl.row_expand_mul(values, inverse), target_type=pl.BF16, mode="rint")
        gamma_tile = pl.slice(gamma, [1, HEAD_DIM], [0, 0])
        scaled = pl.cast(
            pl.col_expand_mul(pl.cast(normalized, target_type=pl.FP32), gamma_tile),
            target_type=pl.BF16, mode="rint",
        )
        left = pl.cast(pl.slice(scaled, [BATCH, HALF_DIM], [0, 0]), target_type=pl.FP32)
        right = pl.cast(pl.slice(scaled, [BATCH, HALF_DIM], [0, HALF_DIM]), target_type=pl.FP32)
        cos_bf16 = pl.cast(pl.slice(cos, [1, HALF_DIM], [position, 0]), target_type=pl.BF16, mode="rint")
        sin_bf16 = pl.cast(pl.slice(sin, [1, HALF_DIM], [position, 0]), target_type=pl.BF16, mode="rint")
        cosine = pl.cast(cos_bf16, target_type=pl.FP32)
        sine = pl.cast(sin_bf16, target_type=pl.FP32)
        left_cos = pl.cast(pl.col_expand_mul(left, cosine), target_type=pl.BF16, mode="rint")
        right_sin = pl.cast(pl.col_expand_mul(right, sine), target_type=pl.BF16, mode="rint")
        right_cos = pl.cast(pl.col_expand_mul(right, cosine), target_type=pl.BF16, mode="rint")
        left_sin = pl.cast(pl.col_expand_mul(left, sine), target_type=pl.BF16, mode="rint")
        rotated_left = pl.cast(
            pl.sub(pl.cast(left_cos, target_type=pl.FP32), pl.cast(right_sin, target_type=pl.FP32)),
            target_type=pl.BF16, mode="rint",
        )
        rotated_right = pl.cast(
            pl.add(pl.cast(right_cos, target_type=pl.FP32), pl.cast(left_sin, target_type=pl.FP32)),
            target_type=pl.BF16, mode="rint",
        )
        k_cache = pl.assemble(
            k_cache, pl.slice(rotated_left, [1, HALF_DIM], [0, 0]), [head * MAX_SEQ + position, 0]
        )
        k_cache = pl.assemble(
            k_cache, pl.slice(rotated_right, [1, HALF_DIM], [0, 0]), [head * MAX_SEQ + position, HALF_DIM]
        )
        v_row = pl.slice(v, [1, HEAD_DIM], [0, head * HEAD_DIM])
        v_cache = pl.assemble(v_cache, v_row, [head * MAX_SEQ + position, 0])
    return k_cache, v_cache


@pl.jit
def attention(
    q: pl.Tensor[[BATCH, Q_WIDTH], pl.BF16],
    k_cache: pl.Tensor[[NUM_KV_HEADS * MAX_SEQ, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[NUM_KV_HEADS * MAX_SEQ, HEAD_DIM], pl.BF16],
    position: pl.Scalar[pl.INT32],
    out: pl.Out[pl.Tensor[[BATCH, Q_WIDTH], pl.BF16]],
):
    scores = pl.create_tensor([NUM_HEADS * BATCH, MAX_SEQ], dtype=pl.FP32)
    for head in pl.spmd(NUM_HEADS, name_hint="attention_qk"):
        query = pl.slice(q, [BATCH, HEAD_DIM], [0, head * HEAD_DIM])
        keys = pl.slice(k_cache, [MAX_SEQ, HEAD_DIM], [(head // 4) * MAX_SEQ, 0])
        dot = pl.matmul(query, keys, out_dtype=pl.FP32, b_trans=True)
        scores = pl.assemble(scores, dot, [head * BATCH, 0])

    probabilities = pl.create_tensor([NUM_HEADS * BATCH, MAX_SEQ], dtype=pl.BF16)
    for head in pl.spmd(NUM_HEADS, name_hint="attention_softmax"):
        dot = pl.slice(scores, [BATCH, MAX_SEQ], [head * BATCH, 0])
        weighted = pl.mul(dot, HEAD_DIM ** -0.5)
        masked = pl.fillpad(pl.set_validshape(weighted, BATCH, position + 1), pad_value=pl.PadValue.min)
        maximum = pl.row_max(masked)
        exponentials = pl.exp(pl.row_expand_sub(masked, maximum))
        denominator = pl.row_sum(exponentials)
        probs = pl.row_expand_div(exponentials, denominator)
        probabilities = pl.assemble(
            probabilities, pl.cast(probs, target_type=pl.BF16, mode="rint"), [head * BATCH, 0]
        )

    for head in pl.spmd(NUM_HEADS, name_hint="attention_pv"):
        probabilities_bf16 = pl.slice(probabilities, [BATCH, MAX_SEQ], [head * BATCH, 0])
        values = pl.slice(v_cache, [MAX_SEQ, HEAD_DIM], [(head // 4) * MAX_SEQ, 0])
        context = pl.matmul(probabilities_bf16, values, out_dtype=pl.FP32)
        out = pl.assemble(out, pl.cast(context, target_type=pl.BF16, mode="rint"), [0, head * HEAD_DIM])
    return out
