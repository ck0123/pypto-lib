# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill index-query projection, sparse controls and dispatch scheduling."""

import pypto.language as pl
import torch
from .gemm import make_dot
from .indexer import INDEXER_SELECT_KEY_CAPACITY as KEY_CAPACITY
from .qkv_proj_rope import make_bf16_projection_staged
from .quant import make_dequantize, make_prefill_projection_quantize, make_quantize
from .rope import make_adjacent_rope as make_rope
from .utils import copy_runtime_tensor


# Model configuration.
INDEX_HEADS = 32
INDEX_WIDTH = 128
INDEX_TOPK = 512
WINDOW_SIZE = 128

# Tiling.
QUANT_ROWS_TILE = 16
WEIGHT_ROWS_TILE = 8


def make_indexer_projection(capacity, config):
    """Project and rotate index queries and emit scaled FP16 head weights."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    rope_width = config["qk_rope_head_dim"]
    pairs = rope_width // 2
    index_width = INDEX_HEADS * INDEX_WIDTH
    weights_scale = INDEX_WIDTH ** -0.5 * INDEX_HEADS ** -0.5
    quantize = make_quantize(capacity, latent)
    dot = make_dot(capacity, latent, index_width)
    dequantize = make_dequantize(capacity, index_width)
    rotate = make_rope(capacity, INDEX_HEADS, INDEX_WIDTH, rope_width)
    project_weights = pl.jit.inline(make_bf16_projection_staged(hidden, INDEX_HEADS)._func)

    @pl.jit
    def indexer_projection(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        query_latent: pl.Tensor[[capacity, latent], pl.BF16],
        index_wq_b: pl.Tensor[[latent * (index_width // 64), 64], pl.INT8],
        index_wq_b_scale: pl.Tensor[[index_width], pl.FP32],
        weights_proj: pl.Tensor[[hidden, INDEX_HEADS], pl.BF16],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        query: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS, INDEX_WIDTH], pl.BF16]],
        weights: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        latent_i8 = pl.create_tensor([capacity, latent], dtype=pl.INT8)
        latent_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(query_latent, latent_i8, latent_scale, active)
        accumulator = pl.create_tensor([capacity, index_width], dtype=pl.INT32)
        dot(latent_i8, index_wq_b, accumulator, active)
        projected = pl.create_tensor([capacity, index_width], dtype=pl.BF16)
        dequantize(accumulator, index_wq_b_scale, latent_scale, projected, active)
        query_flat = pl.reshape(query, [capacity, index_width])
        rotate(projected, cosine, sine, query_flat, active)

        projected_weights = pl.create_tensor([capacity, INDEX_HEADS], dtype=pl.BF16)
        project_weights(x, weights_proj, projected_weights, active)
        with pl.spmd((active + WEIGHT_ROWS_TILE - 1) // WEIGHT_ROWS_TILE, name_hint="a2_index_weights"):
            block = pl.tile.get_block_idx()
            row = block * WEIGHT_ROWS_TILE
            rows = pl.min(WEIGHT_ROWS_TILE, active - row)
            value = pl.slice(projected_weights, [WEIGHT_ROWS_TILE, INDEX_HEADS], [row, 0], valid_shape=[rows, INDEX_HEADS])
            scaled = pl.mul(pl.cast(value, pl.FP32), weights_scale)
            narrowed = pl.cast(scaled, pl.FP16, mode="rint")
            weights[row:row + WEIGHT_ROWS_TILE, :] = pl.set_validshape(narrowed, rows, INDEX_HEADS)
        return query, weights

    return indexer_projection


def make_indexer_query_quantize(capacity):
    """Quantize each index head using its rounded FP16 scale."""

    @pl.jit
    def indexer_query_quantize(
        query: pl.Tensor[[capacity, INDEX_HEADS, INDEX_WIDTH], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS, INDEX_WIDTH], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        source = pl.reshape(query, [capacity * INDEX_HEADS, INDEX_WIDTH])
        output = pl.reshape(payload, [capacity * INDEX_HEADS, INDEX_WIDTH])
        blocks = (active * INDEX_HEADS + QUANT_ROWS_TILE - 1) // QUANT_ROWS_TILE
        with pl.spmd(blocks, name_hint="a2_index_query_quant", allow_early_resolve=True):
            block = pl.tile.get_block_idx()
            row = block * QUANT_ROWS_TILE
            rows = pl.min(QUANT_ROWS_TILE, active * INDEX_HEADS - row)
            q = pl.slice(source, [QUANT_ROWS_TILE, INDEX_WIDTH], [row, 0], valid_shape=[rows, INDEX_WIDTH])
            q = pl.set_validshape(pl.fillpad(q, pad_value=pl.PadValue.zero), QUANT_ROWS_TILE, INDEX_WIDTH)
            value = pl.cast(q, pl.FP32)
            maximum = pl.row_max(pl.abs(value))
            raw_scale = pl.div(maximum, 127.0)
            rounded_scale = pl.cast(pl.cast(raw_scale, pl.FP16, mode="rint"), pl.FP32)
            scale = pl.maximum(rounded_scale, 2.0 ** -24)
            normalized = pl.row_expand_div(value, scale)
            rounded = pl.cast(normalized, pl.INT32, mode="rint")
            bounded = pl.minimum(pl.maximum(rounded, -127), 127)
            quantized = pl.cast(bounded, pl.INT8, mode="trunc")
            output[row:row + QUANT_ROWS_TILE, :] = pl.set_validshape(quantized, rows, INDEX_WIDTH)
            scale_row = pl.reshape(pl.cast(scale, pl.FP16, mode="rint"), [1, QUANT_ROWS_TILE])
            token = row // INDEX_HEADS
            head = row % INDEX_HEADS
            scales[token:token + 1, head:head + QUANT_ROWS_TILE] = scale_row
        return payload, scales

    return indexer_query_quantize


def make_indexer_query_rope(capacity):
    """Rotate the trailing channels of the Indexer query heads."""
    rotate = make_rope(
        capacity, INDEX_HEADS, INDEX_WIDTH, 64, name_hint="index_query_rope_heads", allow_early_resolve=True,
    )
    return pl.jit(rotate._func)


def make_index_query_projection(capacity):
    """Project packed queries and apply combined scales in the same task."""
    if capacity <= 0 or capacity % 16:
        raise ValueError("Use a positive capacity divisible by 16")
    quantize = make_prefill_projection_quantize(capacity, 1280)
    QUERY_ROWS_TILE = 16
    QUERY_COLUMNS_TILE = 64
    QUERY_GROUP_TILE = 4

    @pl.jit
    def index_query_projection(
        x: pl.Tensor[[capacity, 1280], pl.BF16],
        weight: pl.Tensor[[1280 * 64, 64], pl.INT8],
        channel_scale: pl.Tensor[[4096], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, 4096], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, 1280], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(x, payload, token_scale, active)
        row_blocks = (active + QUERY_ROWS_TILE - 1) // QUERY_ROWS_TILE
        column_blocks = 4096 // QUERY_COLUMNS_TILE // QUERY_GROUP_TILE
        with pl.spmd(row_blocks * column_blocks, name_hint="index_query_project_scaled", allow_early_resolve=True):
            block = pl.tile.get_block_idx()
            row = block // column_blocks * QUERY_ROWS_TILE
            first_column = block % column_blocks * QUERY_GROUP_TILE
            rows = pl.min(QUERY_ROWS_TILE, active - row)
            source = pl.slice(payload, [QUERY_ROWS_TILE, 1280], [row, 0], valid_shape=[rows, 1280])
            activation = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), QUERY_ROWS_TILE, 1280)
            scale_source = pl.slice(token_scale, [QUERY_ROWS_TILE, 8], [row, 0], valid_shape=[rows, 8])
            scale_source = pl.set_validshape(pl.fillpad(scale_source, pad_value=pl.PadValue.zero), QUERY_ROWS_TILE, 8)
            scale = pl.row_min(scale_source)
            for column in pl.pipeline(first_column, first_column + QUERY_GROUP_TILE, stage=2):
                col = column * QUERY_COLUMNS_TILE
                packed_row = column * 1280
                packed_weight = weight[packed_row:packed_row + 1280, :]
                product = pl.matmul(activation, packed_weight, out_dtype=pl.INT32)
                value = pl.cast(product, pl.FP32, mode="rint")
                channels = pl.reshape(channel_scale[col:col + QUERY_COLUMNS_TILE], [1, QUERY_COLUMNS_TILE])
                ones = pl.full([QUERY_ROWS_TILE, QUERY_COLUMNS_TILE], dtype=pl.FP32, value=1.0)
                channel_rows = pl.col_expand_mul(ones, channels)
                combined = pl.row_expand_mul(channel_rows, scale)
                scaled = pl.mul(value, combined)
                narrowed = pl.cast(scaled, pl.BF16, mode="rint")
                output[row:row + QUERY_ROWS_TILE, col:col + QUERY_COLUMNS_TILE] = pl.set_validshape(narrowed, rows, QUERY_COLUMNS_TILE)
        return output

    return index_query_projection


def make_empty_selection(capacity):
    """Publish the source empty-cache selection sentinel."""

    @pl.jit
    def empty_selection(
        raw_selected: pl.InOut[pl.Tensor[[capacity, INDEX_TOPK], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(active, name_hint="a2_empty_sparse_selection"):
            token = pl.tile.get_block_idx()
            raw_selected[token:token + 1, :] = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
        return raw_selected

    return empty_selection


def make_sparse_controls(capacity, requests, long_pages, window_pages, long_page_rows=64, *,
                         compress_ratio=2, long_page_stride=None, selection_ready=False):
    """Prepare producer QLI or readonly consumer IDs and map both caches to physical rows."""
    if long_page_rows not in (64, 128):
        raise ValueError("A2 compressed page rows must match the actual 64/128-row cache ABI")
    if compress_ratio not in (1, 2):
        raise ValueError("Sparse controls require the physical source compression ratio 1 or 2")

    long_page_stride = long_page_rows if long_page_stride is None else long_page_stride

    @pl.jit
    def sparse_controls(
        raw_selected: pl.Tensor[[capacity, INDEX_TOPK], pl.INT32],
        positions: pl.Tensor[[capacity], pl.INT64],
        request_ids: pl.Tensor[[capacity], pl.INT32],
        long_block_table: pl.Tensor[[requests, long_pages], pl.INT32],
        window_block_table: pl.Tensor[[requests, window_pages], pl.INT32],
        selected: pl.InOut[pl.Tensor[[capacity, INDEX_TOPK], pl.INT32]],
        selected_lengths: pl.InOut[pl.Tensor[[capacity, 1], pl.INT32]],
        compressed_indices: pl.InOut[pl.Tensor[[capacity, INDEX_TOPK], pl.INT32]],
        window_indices: pl.InOut[pl.Tensor[[capacity, WINDOW_SIZE], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(active, name_hint="a2_prepare_sparse_indices") as controls_tid:
            token = pl.tile.get_block_idx()
            position = pl.cast(pl.read(positions, [token]), pl.INT32)
            visible = (position + 1) // compress_ratio
            request = pl.read(request_ids, [token])
            if selection_ready:
                logical_ids = pl.load(raw_selected, [token, 0], [1, INDEX_TOPK])
            else:
                sort_scores = pl.tile.full([1, INDEX_TOPK], dtype=pl.FP32, value=-1e30)
                sort_ids = pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
                for lane in pl.range(INDEX_TOPK):
                    index = pl.read(raw_selected, [token, lane])
                    if index >= 0:
                        if index < visible:
                            pl.tile.write(sort_scores, [0, lane], -pl.cast(index, pl.FP32))
                            pl.tile.write(sort_ids, [0, lane], index)
                sorted_pairs = pl.tile.sort32(sort_scores, pl.reinterpret_view(sort_ids, pl.UINT32))
                sorted_pairs = pl.tile.mrgsort(sorted_pairs, block_len=64)
                sorted_pairs = pl.tile.mrgsort(sorted_pairs, block_len=256)
                logical_ids = pl.tile.gather_mask(sorted_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            pl.store(logical_ids, [token, 0], selected)
            physical_ids = pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
            for lane in pl.range(INDEX_TOPK):
                logical = pl.tile.read(logical_ids, [0, lane])
                if logical >= 0:
                    page = logical // long_page_rows
                    offset = logical % long_page_rows
                    block = pl.read(long_block_table, [request, page])
                    physical = pl.cast(block * long_page_stride + offset, pl.INT32)
                    pl.tile.write(physical_ids, [0, lane], physical)
            pl.store(physical_ids, [token, 0], compressed_indices)

            left = pl.max(0, position + 1 - WINDOW_SIZE)
            window_count = position + 1 - left
            window_ids = pl.tile.full([1, WINDOW_SIZE], dtype=pl.INT32, value=-1)
            for lane in pl.range(WINDOW_SIZE):
                if lane < window_count:
                    window_logical = left + lane
                    window_page = window_logical // WINDOW_SIZE
                    window_offset = window_logical % WINDOW_SIZE
                    window_block = pl.read(window_block_table, [request, window_page])
                    window_physical = pl.cast(window_block * WINDOW_SIZE + window_offset, pl.INT32)
                    pl.tile.write(window_ids, [0, lane], window_physical)
            pl.store(window_ids, [token, 0], window_indices)
        lengths_flat = pl.reshape(selected_lengths, [1, capacity])
        for block in pl.spmd((active + 7) // 8, name_hint="a2_sparse_lengths_publish", deps=[controls_tid]):
            first = block * 8
            rows = pl.min(8, active - first)
            ids = pl.slice(selected, [8, INDEX_TOPK], [first, 0], valid_shape=[rows, INDEX_TOPK])
            values = pl.cast(ids, pl.FP32)
            valid = pl.cmp(values, 0.0, cmp_type=5)
            counts = pl.cast(pl.row_sum(valid), pl.INT32, mode="rint")
            flat_counts = pl.reshape(counts, [1, 8])
            lengths_flat[0:1, first:first + 8] = pl.set_validshape(flat_counts, 1, rows)
        return selected, selected_lengths, compressed_indices, window_indices

    return sparse_controls


def make_index_weights(capacity):
    """Scale BF16 projected head weights into the QLI FP16 boundary."""

    @pl.jit
    def index_weights(
        projected: pl.Tensor[[capacity, INDEX_HEADS], pl.BF16],
        weights: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd((active + WEIGHT_ROWS_TILE - 1) // WEIGHT_ROWS_TILE, name_hint="index_weights_scale", allow_early_resolve=True):
            row = pl.tile.get_block_idx() * WEIGHT_ROWS_TILE
            rows = pl.min(WEIGHT_ROWS_TILE, active - row)
            source = pl.slice(projected, [WEIGHT_ROWS_TILE, INDEX_HEADS], [row, 0], valid_shape=[rows, INDEX_HEADS])
            scaled = pl.mul(pl.cast(source, pl.FP32), 1.0 / 64.0)
            weights[row:row + WEIGHT_ROWS_TILE, :] = pl.set_validshape(pl.cast(scaled, pl.FP16), rows, INDEX_HEADS)
        return weights

    return index_weights


def make_index_key_gather(cache_blocks, requests, table_pages, page_rows=64, *, key_capacity=KEY_CAPACITY):
    """Pack one request's paged INT8/FP16 index cache into the fixed QLI arena."""
    KEY_ROWS_TILE = 64

    @pl.jit
    def index_key_gather(
        cache: pl.Tensor[[cache_blocks, page_rows, 1, INDEX_WIDTH], pl.INT8],
        scales: pl.Tensor[[cache_blocks, page_rows, 1, 1], pl.FP16],
        table: pl.Tensor[[requests, table_pages], pl.INT32],
        key: pl.InOut[pl.Tensor[[key_capacity, INDEX_WIDTH], pl.INT8]],
        key_scale: pl.InOut[pl.Tensor[[key_capacity], pl.FP16]],
        request: pl.Scalar[pl.INT32],
        length: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(cache, [1, cache_blocks * page_rows * INDEX_WIDTH])
        key_output = pl.reshape(key, [1, key_capacity * INDEX_WIDTH])
        scale_flat = pl.reshape(scales, [1, cache_blocks * page_rows])
        scale_output = pl.reshape(key_scale, [1, key_capacity])
        data_blocks = (length + KEY_ROWS_TILE - 1) // KEY_ROWS_TILE
        with pl.spmd(data_blocks, name_hint="index_key_pack", allow_early_resolve=True):
            start = pl.tile.get_block_idx() * KEY_ROWS_TILE
            page = start // page_rows
            offset = start % page_rows
            block = pl.cast(pl.read(table, [request, page]), pl.INDEX)
            valid = pl.min(KEY_ROWS_TILE, length - start)
            source_offset = (block * page_rows + offset) * INDEX_WIDTH
            source = pl.slice(flat, [1, KEY_ROWS_TILE * INDEX_WIDTH], [0, source_offset], valid_shape=[1, valid * INDEX_WIDTH])
            values = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 1, KEY_ROWS_TILE * INDEX_WIDTH)
            scale_source = pl.slice(scale_flat, [1, KEY_ROWS_TILE], [0, block * page_rows + offset], valid_shape=[1, valid])
            scale_values = pl.set_validshape(pl.fillpad(scale_source, pad_value=pl.PadValue.zero), 1, KEY_ROWS_TILE)
            output_start = start * INDEX_WIDTH
            key_output[0:1, output_start:output_start + KEY_ROWS_TILE * INDEX_WIDTH] = values
            scale_output[0:1, start:start + KEY_ROWS_TILE] = scale_values

        zero_rows_tile = 256
        zero_blocks = (key_capacity - length + zero_rows_tile - 1) // zero_rows_tile
        with pl.spmd(zero_blocks, name_hint="index_key_zero_tail", allow_early_resolve=True):
            start = length + pl.tile.get_block_idx() * zero_rows_tile
            valid = pl.min(zero_rows_tile, key_capacity - start)
            zero_values = pl.full([1, zero_rows_tile * INDEX_WIDTH], dtype=pl.FP16, value=0.0)
            values = pl.cast(zero_values, pl.INT8, mode="trunc")
            key_output[0:1, start * INDEX_WIDTH:(start + zero_rows_tile) * INDEX_WIDTH] = pl.set_validshape(
                values, 1, valid * INDEX_WIDTH,
            )
            scale_values = pl.full([1, zero_rows_tile], dtype=pl.FP16, value=0.0)
            scale_output[0:1, start:start + zero_rows_tile] = pl.set_validshape(scale_values, 1, valid)
        return key, key_scale

    return index_key_gather


def make_sparse_index_pack(capacity):
    """Pack the window and compressed physical row IDs into the attention arena."""

    @pl.jit
    def sparse_index_pack(
        window: pl.Tensor[[capacity, WINDOW_SIZE], pl.INT32],
        compressed: pl.Tensor[[capacity, INDEX_TOPK], pl.INT32],
        indices: pl.InOut[pl.Tensor[[capacity, WINDOW_SIZE + INDEX_TOPK], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(active, name_hint="sparse_physical_index_pack"):
            row = pl.tile.get_block_idx()
            indices[row:row + 1, :WINDOW_SIZE] = window[row:row + 1, :]
            indices[row:row + 1, WINDOW_SIZE:] = compressed[row:row + 1, :]
        return indices

    return sparse_index_pack


def make_shared_index_publish(capacity, shared_rows=8192):
    """Publish actual CP selection rows while preserving shared history and tails."""

    @pl.jit
    def shared_index_publish(
        indices: pl.Tensor[[capacity, INDEX_TOPK], pl.INT32],
        lengths: pl.Tensor[[capacity, 1], pl.INT32],
        shared_indices: pl.InOut[pl.Tensor[[shared_rows, INDEX_TOPK], pl.INT32]],
        shared_lengths: pl.InOut[pl.Tensor[[shared_rows, 1], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(active, name_hint="shared_index_rows"):
            row = pl.tile.get_block_idx()
            shared_indices[row:row + 1, :] = indices[row:row + 1, :]
        with pl.spmd(1, name_hint="shared_index_lengths"):
            _writer = pl.tile.get_block_idx()
            for row in pl.range(active):
                pl.write(shared_lengths, [row, 0], pl.read(lengths, [row, 0]))
        return shared_indices, shared_lengths

    return shared_index_publish


def make_physical_index_query_entries(capacity, config, cache_blocks, requests, table_pages, compress_ratio=2,
                                      *, key_capacity=KEY_CAPACITY, packed_keys=False):
    """Build index query, quantization, paged scoring and selection publication entries."""
    from .gemm import make_bf16_linear
    from .indexer import make_prefill_indexer_select as make_indexer_select

    return dict(
        index_wqb=make_index_query_projection(capacity),
        index_weights_proj=make_bf16_linear(capacity, 5120, 32),
        index_query_rope=make_indexer_query_rope(capacity),
        index_query_quant=make_indexer_query_quantize(capacity),
        index_weights_scale=make_index_weights(capacity),
        index_key_gather=make_index_key_gather(cache_blocks, requests, table_pages,
                                              page_rows=128 // compress_ratio, key_capacity=key_capacity),
        index_select=make_indexer_select(
            capacity, key_capacity, compress_ratio=compress_ratio, packed_keys=packed_keys,
        ),
        sparse_controls=make_sparse_controls(capacity, requests, table_pages, table_pages,
                                            long_page_rows=128 // compress_ratio, compress_ratio=compress_ratio,
                                            long_page_stride=128),
        shared_index_publish=make_shared_index_publish(capacity),
    )


def make_prefill_indexer_chain(capacity, key_capacity, compress_ratio, cache_blocks, requests, table_pages):
    """Compose the physical query entries through causal scores and Top-512."""
    entries = make_physical_index_query_entries(
        capacity, {}, cache_blocks, requests, table_pages, compress_ratio, key_capacity=key_capacity,
    )
    project_query = pl.jit.inline(entries["index_wqb"]._func)
    project_weights = pl.jit.inline(entries["index_weights_proj"]._func)
    rotate = pl.jit.inline(entries["index_query_rope"]._func)
    quantize_query = pl.jit.inline(entries["index_query_quant"]._func)
    scale_weights = pl.jit.inline(entries["index_weights_scale"]._func)
    gather = pl.jit.inline(entries["index_key_gather"]._func)
    select = pl.jit.inline(entries["index_select"]._func)
    page_rows = 128 // compress_ratio

    @pl.jit
    def prefill_indexer_chain(
        latent: pl.Tensor[[capacity, 1280], pl.BF16],
        hidden: pl.Tensor[[capacity, 5120], pl.BF16],
        query_weight: pl.Tensor[[1280 * 64, 64], pl.INT8],
        channel_scale: pl.Tensor[[4096], pl.FP32],
        head_weight: pl.Tensor[[5120, 32], pl.BF16],
        cosine: pl.Tensor[[capacity, 32], pl.FP32],
        sine: pl.Tensor[[capacity, 32], pl.FP32],
        cache: pl.Tensor[[cache_blocks, page_rows, 1, 128], pl.INT8],
        cache_scale: pl.Tensor[[cache_blocks, page_rows, 1, 1], pl.FP16],
        table: pl.Tensor[[requests, table_pages], pl.INT32],
        positions: pl.Tensor[[capacity], pl.INT32],
        projected: pl.InOut[pl.Tensor[[capacity, 4096], pl.BF16]],
        head_projected: pl.InOut[pl.Tensor[[capacity, 32], pl.BF16]],
        rotated: pl.InOut[pl.Tensor[[capacity, 4096], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, 32, 128], pl.INT8]],
        query_scale: pl.InOut[pl.Tensor[[capacity, 32], pl.FP16]],
        weights: pl.InOut[pl.Tensor[[capacity, 32], pl.FP16]],
        key: pl.InOut[pl.Tensor[[key_capacity, 128], pl.INT8]],
        key_scale: pl.InOut[pl.Tensor[[key_capacity], pl.FP16]],
        scores: pl.Out[pl.Tensor[[capacity, key_capacity], pl.FP32]],
        indices: pl.Out[pl.Tensor[[capacity, 512], pl.INT32]],
        request: pl.Scalar[pl.INT32],
        length: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        project_query(latent, query_weight, channel_scale, projected, active)
        project_weights(hidden, head_weight, head_projected, active)
        rotate(projected, cosine, sine, rotated, active)
        query_heads = pl.reshape(rotated, [capacity, 32, 128])
        quantize_query(query_heads, query, query_scale, active)
        scale_weights(head_projected, weights, active)
        gather(cache, cache_scale, table, key, key_scale, request, length)
        select(query, query_scale, key, key_scale, weights, positions, scores, indices, active)
        return projected, head_projected, rotated, query, query_scale, weights, key, key_scale, scores, indices

    return prefill_indexer_chain


def golden_index_key_gather(args):
    """Pack one request's visible cache rows and zero the unused key arena."""
    import torch

    args["key"].zero_()
    args["key_scale"].zero_()
    length, request = int(args["length"]), int(args["request"])
    rows = args["cache"].shape[1]
    logical = torch.arange(length)
    pages = args["table"][request, logical // rows].long()
    args["key"][:length] = args["cache"][pages, logical % rows, 0]
    args["key_scale"][:length] = args["scales"][pages, logical % rows, 0, 0]


def build_sparse_controls_specs(capacity=16, active=16, *, compress_ratio=2, selection_ready=False):
    """Build paged, causal and inactive sparse-control fixtures."""
    from golden import ScalarSpec, TensorSpec

    if not 0 <= active <= capacity:
        raise ValueError("Active rows must fit the sparse-control capacity")
    positions = torch.arange(capacity, dtype=torch.int64) * 37 % 1024
    positions[:min(capacity, 4)] = torch.tensor([0, 1, 127, 1023])[:min(capacity, 4)]
    requests = torch.arange(capacity, dtype=torch.int32) % 2
    ids = torch.arange(INDEX_TOPK, dtype=torch.int32).flip(0).repeat(capacity, 1)
    ids[:, ::7] = -1
    if selection_ready:
        visible = (positions + 1) // compress_ratio
        ids.masked_fill_(ids.ge(visible[:, None]), -1)
        ids = ids.sort(stable=True).values.flip(1).contiguous()
    long_pages, window_pages = 16, 8
    long_table = torch.stack((torch.arange(long_pages), torch.arange(long_pages).flip(0))).int()
    window_table = torch.stack((torch.arange(window_pages).flip(0), torch.arange(window_pages))).int()
    return [
        TensorSpec("raw_selected", [capacity, INDEX_TOPK], torch.int32, init_value=ids),
        TensorSpec("positions", [capacity], torch.int64, init_value=positions),
        TensorSpec("request_ids", [capacity], torch.int32, init_value=requests),
        TensorSpec("long_block_table", [2, long_pages], torch.int32, init_value=long_table),
        TensorSpec("window_block_table", [2, window_pages], torch.int32, init_value=window_table),
        TensorSpec("selected", [capacity, INDEX_TOPK], torch.int32, init_value=-71),
        TensorSpec("selected_lengths", [capacity, 1], torch.int32, init_value=-73),
        TensorSpec("compressed_indices", [capacity, INDEX_TOPK], torch.int32, init_value=-79),
        TensorSpec("window_indices", [capacity, WINDOW_SIZE], torch.int32, init_value=-83),
        ScalarSpec("active", torch.int32, active, compile_runtime=True),
    ]


def validate_key_gather(*, device=0, sim=False, compile_only=False, cpu_only=False):
    """Check both compressed page geometries, permuted pages and partial final tiles."""
    import torch
    from golden import ScalarSpec, TensorSpec, run

    if cpu_only and compile_only:
        raise ValueError("CPU golden checks and kernel compilation are separate modes")
    records = []
    for rows in (64, 128):
        pages, blocks = KEY_CAPACITY // rows, KEY_CAPACITY // rows + 2
        physical = torch.arange(blocks * rows)
        codes = ((physical[:, None] + torch.arange(INDEX_WIDTH)[None, :]) % 127).to(torch.int8)
        codes = codes.reshape(blocks, rows, 1, INDEX_WIDTH)
        scales = ((physical.float() + 1) / 4096).half().reshape(blocks, rows, 1, 1)
        table = torch.stack((torch.arange(1, pages + 1), torch.arange(pages, 0, -1))).int()
        for length in (0, 1, 511, 512, 513, 2048, 4096, 4097, 8192, 8193, 16383, KEY_CAPACITY):
            specs = [
                TensorSpec("cache", [blocks, rows, 1, INDEX_WIDTH], torch.int8, init_value=codes),
                TensorSpec("scales", [blocks, rows, 1, 1], torch.float16, init_value=scales),
                TensorSpec("table", [2, pages], torch.int32, init_value=table),
                TensorSpec("key", [KEY_CAPACITY, INDEX_WIDTH], torch.int8, init_value=-53),
                TensorSpec("key_scale", [KEY_CAPACITY], torch.float16, init_value=-59),
                ScalarSpec("request", torch.int32, 1, compile_runtime=True),
                ScalarSpec("length", torch.int32, length, compile_runtime=True),
            ]
            if cpu_only:
                values = {spec.name: spec.create_tensor() if hasattr(spec, "create_tensor") else spec.to_python()
                          for spec in specs}
                golden_index_key_gather(values)
                expected_rows = [int(table[1, row // rows]) * rows + row % rows for row in range(length)]
                if (not torch.equal(values["key"][:length], codes.reshape(-1, INDEX_WIDTH)[expected_rows])
                        or (values["key"][length:] != 0).any() or (values["key_scale"][length:] != 0).any()):
                    raise RuntimeError("Paged key gather mixed requests, physical pages or padded rows")
            else:
                result = run(
                    fn=make_index_key_gather(blocks, 2, pages, page_rows=rows),
                    specs=specs,
                    golden_fn=golden_index_key_gather,
                    config=dict(backend_type="a2a3", execution_mode=bool(sim), device_id=device),
                    compile_only=compile_only,
                    rtol=0,
                    atol=0,
                )
                if not result.passed:
                    raise RuntimeError(f"index-key-gather-rows{rows}-length{length}: {result.error}")
            records.append(dict(page_rows=rows, length=length, compiled=not cpu_only,
                                cpu_golden="passed" if cpu_only else "not_run",
                                kernel="not_run" if cpu_only or compile_only else "passed"))
    return records


def run_index_query_selection(state, q_input, latent, cosine, sine, operands,
                              execute, rank, case, capacity, requests, table_pages,
                              source_prefix, own_prefix, codes, scales):
    """Compute quantized index queries, paged-key scores and sparse controls with PyPTO."""
    original, total = state["original"], state["total"]
    start, end, _per_rank, _padded = original["cp_token_range"]
    query_rows = max(0, min(end, total) - min(start, total))
    metadata = original["cache_metadata"]["actual_metadata"]
    global_meta = {name: item["global_metadata"] for name, item in metadata.items()}

    def zeros(*shape, dtype=torch.bfloat16):
        return torch.zeros(shape, dtype=dtype)

    def pad(value, rows=capacity, fill=0):
        result = torch.full((rows, *value.shape[1:]), fill, dtype=value.dtype)
        copy_runtime_tensor(state.get("runtime", {}), result[:len(value)], value, rank)
        return result

    def table(name):
        value = global_meta[name]["block_table"]
        target = torch.full((requests, table_pages), -1, dtype=torch.int32)
        target[:value.shape[0], :value.shape[1]].copy_(value)
        return target

    positions = pad(original["positions"])
    actual_starts = original["metadata"]["query_start_loc"]
    request_ids = torch.full((capacity,), -1, dtype=torch.int32)
    for request, (begin, finish) in enumerate(zip(actual_starts.tolist(), actual_starts.tolist()[1:])):
        request_ids[begin:finish].fill_(request)
    index_qb, weights_proj = zeros(capacity, 4096), zeros(capacity, 32)
    execute("index_wqb", rank, case, [latent, operands["index_wqb"], operands["index_wqb_scale"], index_qb, query_rows],
            dict(projected=index_qb), "index_key")
    execute("index_weights_proj", rank, case, [q_input, operands["index_weights"], weights_proj, query_rows],
            dict(projected=weights_proj), "index_key")
    q_cos, q_sin = pad(cosine[start:start + query_rows]), pad(sine[start:start + query_rows])
    query_key = zeros(capacity, 4096)
    execute("index_query_rope", rank, case, [index_qb, q_cos, q_sin, query_key, query_rows],
            dict(query=query_key), "rope")
    query_codes = zeros(capacity, 32, 128, dtype=torch.int8)
    query_scales, weights = zeros(capacity, 32, dtype=torch.float16), zeros(capacity, 32, dtype=torch.float16)
    execute("index_query_quant", rank, case, [query_key.reshape(capacity, 32, 128), query_codes, query_scales, query_rows],
            dict(codes=query_codes, scales=query_scales), "index_quant")
    execute("index_weights_scale", rank, case, [weights_proj, weights, query_rows], dict(weights=weights), "index_key")
    candidate_scores = zeros(capacity, KEY_CAPACITY, dtype=torch.float32)
    raw_selected = torch.full((capacity, 512), -1, dtype=torch.int32)
    q_positions, q_requests = pad(positions[start:start + query_rows]), pad(request_ids[start:start + query_rows], fill=-1)
    long_table, window_table = table(source_prefix + "long_kv_cache"), table(own_prefix + "swa_cache")
    cache_lengths = global_meta[source_prefix + "indexer.k_cache"]["cache_seq_lens"].tolist()
    if any(length < 0 or length > KEY_CAPACITY for length in cache_lengths):
        raise ValueError("Physical QLI history exceeds the supported index-key capacity")
    for request, (begin, finish) in enumerate(zip(actual_starts.tolist(), actual_starts.tolist()[1:])):
        left, right = max(begin, start), min(finish, start + query_rows)
        if left >= right:
            continue
        offset, count = left - start, right - left
        packed_key = zeros(KEY_CAPACITY, 128, dtype=torch.int8)
        packed_scale = zeros(KEY_CAPACITY, dtype=torch.float16)
        execute("index_key_gather", rank, case + f"-request-{request}",
                [codes, scales, long_table, packed_key, packed_scale, request, cache_lengths[request]],
                dict(key=packed_key, scales=packed_scale), "index_select")
        scores = zeros(capacity, KEY_CAPACITY, dtype=torch.float32)
        picked = torch.full((capacity, 512), -1, dtype=torch.int32)
        selection_args = [pad(query_codes[offset:offset + count]), pad(query_scales[offset:offset + count]),
                          packed_key, packed_scale, pad(weights[offset:offset + count]),
                          pad(q_positions[offset:offset + count]).to(torch.int32), scores, picked, count]
        execute("index_select", rank, case + f"-request-{request}", selection_args,
                dict(scores=scores, selected=picked), "index_select")
        copy_runtime_tensor(state.get("runtime", {}), raw_selected[offset:offset + count], picked[:count], rank)
        copy_runtime_tensor(state.get("runtime", {}), candidate_scores[offset:offset + count], scores[:count], rank)
    if not query_rows:
        # Empty CP ranks retain a recorded index-selection boundary.
        execute("index_weights_scale", rank, case + "-empty-selection", [weights_proj, weights, 0],
                dict(weights=weights), "index_select")
    selected = torch.full((capacity, 512), -1, dtype=torch.int32)
    lengths = zeros(capacity, 1, dtype=torch.int32)
    compressed_indices, window_indices = selected.clone(), torch.full((capacity, 128), -1, dtype=torch.int32)
    controls_args = [raw_selected, q_positions, q_requests, long_table, window_table,
                     selected, lengths, compressed_indices, window_indices, query_rows]
    execute("sparse_controls", rank, case, controls_args,
            dict(selected=selected, lengths=lengths, compressed=compressed_indices, window=window_indices), "index_select")
    visible_lengths = zeros(capacity, dtype=torch.int32)
    ratio = state["role"]["ratio"]
    visible = [(position + 1) // ratio for position in q_positions[:query_rows].tolist()]
    visible_lengths[:query_rows].copy_(torch.tensor(visible, dtype=torch.int32))
    return dict(selected=selected, lengths=lengths, compressed=compressed_indices,
                window=window_indices, scores=candidate_scores, visible_lengths=visible_lengths)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_key_gather(device=args.device, sim=args.sim,
        compile_only=args.compile_only, cpu_only=args.cpu_only)))
