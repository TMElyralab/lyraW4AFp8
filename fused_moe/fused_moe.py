# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/fused_moe.py

"""Fused MoE kernel."""

import functools
import json
import logging
import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl
from lyra_w4afp8 import machete_mm, ScalarType, scalar_types

from lyra_w4afp8 import moe_align_block_size as sgl_moe_align_block_size


logger = logging.getLogger(__name__)


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    cumsum_buffer = torch.empty(
        (num_experts + 1,), dtype=torch.int32, device=topk_ids.device
    )
    token_cnts_buffer = torch.empty(
        (num_experts + 1) * num_experts,
        dtype=torch.int32,
        device=topk_ids.device,
    )

    # Threshold based on benchmark results
    fuse_sorted_ids_padding = sorted_ids.shape[0] <= 4096
    if not fuse_sorted_ids_padding:
        sorted_ids.fill_(topk_ids.numel())

    sgl_moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        token_cnts_buffer,
        cumsum_buffer,
        fuse_sorted_ids_padding,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad



# _moe_sum_reduce_kernel kernel modified from https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/moe_sum_reduce.py
@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    token_start = token_block_id * BLOCK_M
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)

    dim_start = dim_block_id * BLOCK_DIM
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)

    for token_index in range(token_start, token_end):
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            tmp = tl.load(
                input_t_ptr + i * input_stride_1, mask=offs_dim < dim_end, other=0.0
            )
            accumulator += tmp
        accumulator = accumulator * routed_scaling_factor
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(
            store_t_ptr,
            accumulator.to(input_ptr.dtype.element_ty),
            mask=offs_dim < dim_end,
        )


@triton.jit
def _moe_sum_reduce_with_reorder_topk_weight_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    output_ptr,
    output_stride_0,
    output_stride_1,
    reorder_token_pos_ptr,
    topk_weight_ptr,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    token_start = token_block_id * BLOCK_M
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)

    dim_start = dim_block_id * BLOCK_DIM
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)

    for token_index in range(token_start, token_end):
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        input_t_ptr = input_ptr + offs_dim
        token_offset_0 = token_index * topk_num
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            token_offset = token_offset_0 + i
            i_real = tl.load(reorder_token_pos_ptr + token_offset)
            token_weight = tl.load(topk_weight_ptr + token_offset)

            tmp = tl.load(
                input_t_ptr + i_real * input_stride_0,
                mask=offs_dim < dim_end,
                other=0.0,
            )
            accumulator += tmp * token_weight
        accumulator = accumulator * routed_scaling_factor
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(
            store_t_ptr,
            accumulator.to(input_ptr.dtype.element_ty),
            mask=offs_dim < dim_end,
        )


def moe_sum_reduce_triton(
    input: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: float,
    reorder_token_pos: torch.Tensor = None,
    topk_weights: torch.Tensor = None,
):
    assert input.is_contiguous()
    assert output.is_contiguous()

    if reorder_token_pos is None and topk_weights is None:
        token_num, topk_num, hidden_dim = input.shape
        assert output.shape[0] == token_num and output.shape[1] == hidden_dim

        BLOCK_M = 1
        BLOCK_DIM = 2048
        NUM_STAGE = 1
        num_warps = 8

        grid = (
            triton.cdiv(token_num, BLOCK_M),
            triton.cdiv(hidden_dim, BLOCK_DIM),
        )

        _moe_sum_reduce_kernel[grid](
            input,
            *input.stride(),
            output,
            *output.stride(),
            token_num=token_num,
            topk_num=topk_num,
            hidden_dim=hidden_dim,
            routed_scaling_factor=routed_scaling_factor,
            BLOCK_M=BLOCK_M,
            BLOCK_DIM=BLOCK_DIM,
            NUM_STAGE=NUM_STAGE,
            num_warps=num_warps,
        )
    else:
        token_num, topk_num, hidden_dim = (
            reorder_token_pos.shape[0],
            reorder_token_pos.shape[1],
            input.shape[1],
        )
        BLOCK_M = 1
        BLOCK_DIM = 2048
        NUM_STAGE = 1
        num_warps = 8

        grid = (
            triton.cdiv(token_num, BLOCK_M),
            triton.cdiv(hidden_dim, BLOCK_DIM),
        )

        _moe_sum_reduce_with_reorder_topk_weight_kernel[grid](
            input,
            *input.stride(),
            output,
            *output.stride(),
            reorder_token_pos_ptr=reorder_token_pos,
            topk_weight_ptr=topk_weights,
            token_num=token_num,
            topk_num=topk_num,
            hidden_dim=hidden_dim,
            routed_scaling_factor=routed_scaling_factor,
            BLOCK_M=BLOCK_M,
            BLOCK_DIM=BLOCK_DIM,
            NUM_STAGE=NUM_STAGE,
            num_warps=num_warps,
        )


@triton.jit
def fill_moe_hidden_fp8_kernel(
    x_fp8_ptr,
    reorder_token_pos_ptr,
    x_ptr,
    sorted_ids_ptr,
    sorted_ids_len,
    BLOCK_M: tl.constexpr,
    invalid_id: tl.constexpr,
    num_topk: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    if tl.load(sorted_ids_ptr + pid * BLOCK_M) == invalid_id:
        return
    token_id = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_token = token_id < tl.load(sorted_ids_len)

    ids = tl.load(sorted_ids_ptr + token_id, mask=mask_token, other=invalid_id)
    ids = ids.to(tl.int64)

    valid_mask = ids < invalid_id

    for k_idx in range(0, hidden_dim, BLOCK_K):
        k_range = k_idx + tl.arange(0, BLOCK_K)
        k_mask = k_range < hidden_dim

        a_ptr = x_ptr + (ids[:, None] // num_topk * hidden_dim + k_range[None, :])

        a = tl.load(a_ptr, mask=valid_mask[:, None] & k_mask[None, :], other=0.0)

        row_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        dst_ptr = x_fp8_ptr + (row_idx[:, None] * hidden_dim + k_range[None, :])
        # dst_ptr = x_fp8_ptr + (tl.arange(0, BLOCK_M)[:, None] * hidden_dim + k_range[None, :])

        tl.store(dst_ptr, a, mask=mask_token[:, None] & k_mask[None, :])

    # for reorder_token_pos
    # ids: [0,1,invalid,invalid,...,invalid]
    reorder_pos = reorder_token_pos_ptr + ids
    tl.store(reorder_pos, token_id, mask=valid_mask)


def fill_moe_hidden_fp8(
    x_fp8,
    reorder_token_pos,
    x,
    sorted_ids,
    valid_sorted_ids_len,
    BLOCK_M,
    invalid_id,
    num_topk,
):
    BLOCK_M = 1
    hidden_dim = x.shape[1]
    # sorted_ids_len = sorted_ids.numel()  # 使用元素总数
    sorted_ids_len = valid_sorted_ids_len

    grid = ((sorted_ids.shape[0] + BLOCK_M - 1) // BLOCK_M,)

    BLOCK_K = 512
    fill_moe_hidden_fp8_kernel[grid](
        x_fp8,
        reorder_token_pos,
        x,
        sorted_ids,
        sorted_ids_len,
        BLOCK_M,
        invalid_id,
        num_topk,
        hidden_dim,
        BLOCK_K,
    )


@triton.jit
def silu_and_mul_with_mask_kernel(
    x_ptr,
    mask_ptr,
    output_ptr,
    row_num,
    dim,
    stride_x_row,
    stride_x_col,
    stride_mask,
    stride_output_row,
    stride_output_col,
    BLOCK_SIZE_DIM: tl.constexpr,
    INVALID_ID: tl.constexpr,
):
    row_idx = tl.program_id(0)

    if row_idx >= row_num:
        return

    mask_val = tl.load(mask_ptr + row_idx * stride_mask)

    if mask_val == INVALID_ID:
        return

    for dim_offset in range(0, dim, BLOCK_SIZE_DIM):
        col_idx = dim_offset + tl.arange(0, BLOCK_SIZE_DIM)
        col_mask = col_idx < 1000000

        offs_x1 = row_idx * stride_x_row + col_idx * stride_x_col

        x1_ptr = x_ptr + offs_x1
        x1 = tl.load(x1_ptr, mask=col_mask, other=0.0).to(tl.float32)

        x2_ptr = x1_ptr + dim
        x2 = tl.load(x2_ptr, mask=col_mask, other=0.0).to(tl.float32)

        silu_val = x1 * tl.sigmoid(x1)  # SiLU(x) = x * sigmoid(x)
        result_val = silu_val.to(x2.dtype) * x2

        offs_out = row_idx * stride_output_row + col_idx * stride_output_col
        output_ptr_pos = output_ptr + offs_out
        tl.store(output_ptr_pos, result_val, mask=col_mask)


def silu_and_mul_with_mask(
    output: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    invalid_id: int,
    BLOCK_SIZE_DIM: int = 256,
) -> torch.Tensor:
    """
    compute SiLU(x[:, :dim]) * x[:, dim:] with mask

    Args:
        x: [row_num, 2*dim]
        mask: [row_num]
        invalid_id: invalid id in mask
        BLOCK_SIZE_DIM: BLOCK_M size

    Returns: [row_num, dim]
    """
    assert x.dim() == 2, "x.dim() is not 2"
    row_num, two_dim = x.shape
    assert two_dim % 2 == 0, "x.dim[-1] % 2 != 0"
    dim = two_dim // 2

    grid = (row_num,)

    stride_x_row, stride_x_col = x.stride()
    stride_mask = mask.stride()[0] if mask.dim() > 1 else 0
    stride_output_row, stride_output_col = output.stride()

    BLOCK_SIZE_DIM = 256
    silu_and_mul_with_mask_kernel[grid](
        x,
        mask,
        output,
        row_num,
        dim,
        stride_x_row,
        stride_x_col,
        stride_mask,
        stride_output_row,
        stride_output_col,
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
        INVALID_ID=invalid_id,
    )

    return output


@dataclass
class MacheteTypes:
    act_type: torch.dtype
    weight_type: ScalarType
    output_type: Optional[torch.dtype]
    group_scale_type: Optional[torch.dtype]
    group_zero_type: Optional[torch.dtype]
    channel_scale_type: Optional[torch.dtype]
    token_scale_type: Optional[torch.dtype]


def fused_experts_machete_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    no_combine: bool = False,
    has_zp: bool = True,
    routed_scaling_factor: Optional[float] = None,
    schedule: str = None,
):
    a_type = hidden_states.dtype
    types = MacheteTypes(
        act_type=torch.float8_e4m3fn,
        weight_type=scalar_types.uint4b8,
        output_type=a_type,
        group_scale_type=a_type,
        group_zero_type=None,
        channel_scale_type=None,
        token_scale_type=None,
    )

    num_tokens, _ = hidden_states.shape
    E, _, N = w1.shape  # N = 2 * moe_intermediate_size
    group_size = 128

    finfo = torch.finfo(torch.float8_e4m3fn)

    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk_ids.shape[1], w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    # TODO: get_best_scheduler by tuning
    schedule_gateup = "256x16_2x1x1_TmaMI__TmaCoop_PersistentScheduler"
    schedule_down = schedule_gateup
    if schedule:
        schedule_gateup = schedule
        schedule_down = schedule
    BLOCK_SIZE_M = int(schedule_gateup.split("_")[0].split("x")[1])
    BLOCK_SIZE_M2 = int(schedule_down.split("_")[0].split("x")[1])
    if BLOCK_SIZE_M >= BLOCK_SIZE_M2:
        block_ratio_1 = 1
        block_ratio_2 = BLOCK_SIZE_M // BLOCK_SIZE_M2
    else:
        block_ratio_1 = BLOCK_SIZE_M2 // BLOCK_SIZE_M
        block_ratio_2 = 1

    def div_ceil(a, b):
        return (a + b - 1) // b * b

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, BLOCK_SIZE_M, E
    )
    valid_sorted_token_len = num_tokens_post_padded
    # hidden_states_fp8 = torch.empty([div_ceil(valid_sorted_token_len, BLOCK_SIZE_M), hidden_states.shape[1]], dtype=torch.float8_e4m3fn, device=hidden_states.device)
    hidden_states_fp8 = torch.empty(
        [div_ceil(sorted_token_ids.shape[0], BLOCK_SIZE_M), hidden_states.shape[1]],
        dtype=torch.float8_e4m3fn,
        device=hidden_states.device,
    )
    invalid_id = topk_ids.numel()
    num_topk = topk_ids.shape[1]
    reorder_token_pos = torch.empty(
        [num_tokens, num_topk], dtype=torch.int32, device=hidden_states.device
    )

    fill_moe_hidden_fp8(
        hidden_states_fp8,
        reorder_token_pos,
        hidden_states,
        sorted_token_ids,
        valid_sorted_token_len,
        BLOCK_SIZE_M,
        invalid_id,
        num_topk,
    )
    group_layout = expert_ids

    intermediate_cache1 = machete_mm(
        a=hidden_states_fp8,
        b_q=w1,
        b_type=types.weight_type,
        b_group_scales=w1_scale,
        b_group_zeros=w1_zp,
        b_group_size=group_size,
        b_channel_scales=None,
        a_token_scales=None,
        out_type=types.output_type,
        schedule=schedule_gateup,
        group_layout=group_layout,
        group_stride=block_ratio_1,
        valid_len=valid_sorted_token_len,
    )

    # mul_silu
    is_cache2_cast_fp8 = True
    dtype_cache2 = (
        intermediate_cache1.dtype if not is_cache2_cast_fp8 else torch.float8_e4m3fn
    )
    intermediate_cache2 = torch.empty(
        [intermediate_cache1.shape[0], N // 2],
        dtype=dtype_cache2,
        device=intermediate_cache1.device,
    )

    silu_and_mul_with_mask(
        intermediate_cache2,
        intermediate_cache1.view(-1, N),
        sorted_token_ids,
        invalid_id,
    )

    # down
    if not is_cache2_cast_fp8:
        intermediate_cache2_fp8 = torch.clamp(
            intermediate_cache2, finfo.min, finfo.max
        ).to(torch.float8_e4m3fn)
    else:
        intermediate_cache2_fp8 = intermediate_cache2
    intermediate_cache3 = machete_mm(
        a=intermediate_cache2_fp8,
        b_q=w2,
        b_type=types.weight_type,
        b_group_scales=w2_scale,
        b_group_zeros=w2_zp,
        b_group_size=group_size,
        b_channel_scales=None,
        a_token_scales=None,
        out_type=types.output_type,
        schedule=schedule_down,
        group_layout=group_layout,
        group_stride=block_ratio_2,
        valid_len=valid_sorted_token_len,
    )

    # reduce
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    # moe_reduce
    moe_sum_reduce_triton(
        intermediate_cache3.view(*intermediate_cache3.shape),
        out_hidden_states,
        routed_scaling_factor,
        reorder_token_pos=reorder_token_pos,
        topk_weights=topk_weights,
    )
    return out_hidden_states
