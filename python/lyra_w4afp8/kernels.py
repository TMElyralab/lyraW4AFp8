from typing import TYPE_CHECKING, Optional

import torch
from lyra_w4afp8.scalar_type import ScalarType


# machete
def machete_supported_schedules(
    a_type: torch.dtype,
    b_type: ScalarType,
    group_scales_type: Optional[torch.dtype],
    group_zeros_type: Optional[torch.dtype] = None,
    channel_scales_type: Optional[torch.dtype] = None,
    token_scales_type: Optional[torch.dtype] = None,
    out_type: Optional[torch.dtype] = None,
) -> list[str]:

    return torch.ops.lyra_w4afp8.machete_supported_schedules.default(
        a_type,
        b_type.id,
        group_scales_type,
        group_zeros_type,
        channel_scales_type,
        token_scales_type,
        out_type,
    )


def machete_mm(
    a: torch.Tensor,
    # b_q Should be the tensor returned by machete_prepack_B
    b_q: torch.Tensor,
    b_type: ScalarType,
    out_type: Optional[torch.dtype] = None,
    b_group_scales: Optional[torch.Tensor] = None,
    b_group_zeros: Optional[torch.Tensor] = None,
    b_group_size: Optional[int] = None,
    b_channel_scales: Optional[torch.Tensor] = None,
    a_token_scales: Optional[torch.Tensor] = None,
    schedule: Optional[str] = None,
    group_layout: Optional[torch.Tensor] = None,
    group_stride: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
    valid_len: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if output is None:
        M, K = a.shape[0], a.shape[1]
        N = b_q.shape[1]
        G = 1
        if len(b_q.shape) == 3:
            G, N = b_q.shape[0], b_q.shape[2]
        if out_type is None:
            out_type = torch.float16
        output = torch.empty([M, N], dtype=out_type, device=a.device)
    return torch.ops.lyra_w4afp8.machete_mm.default(
        a,
        b_q,
        output,
        b_type.id,
        out_type,
        b_group_scales,
        b_group_zeros,
        b_group_size,
        b_channel_scales,
        a_token_scales,
        schedule,
        group_layout,
        valid_len,
        group_stride,
    )


def machete_prepack_B(
    b_q_weight: torch.Tensor,
    a_type: torch.dtype,
    b_type: ScalarType,
    group_scales_type: Optional[torch.dtype],
) -> torch.Tensor:
    return torch.ops.lyra_w4afp8.machete_prepack_B.default(
        b_q_weight, a_type, b_type.id, group_scales_type
    )

def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    token_cnts_buffer,
    cumsum_buffer,
    pad_sorted_token_ids=False,
):
    torch.ops.lyra_w4afp8.moe_align_block_size.default(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        token_cnts_buffer,
        cumsum_buffer,
        pad_sorted_token_ids,
    )
