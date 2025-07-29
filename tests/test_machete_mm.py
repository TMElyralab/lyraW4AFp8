# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the machete kernel.

Run `pytest tests/test_machete_mm.py`.
"""

import os
import math
from dataclasses import dataclass, fields
from typing import Optional

import pytest
import torch

from common.utils import machete_quantize_and_pack, rand_data, maybe_convert_zeropoints, group_size_valid
from common.data_utils import create_gemm_data, TypeConfig, Tensors

from lyra_w4afp8 import ScalarType, scalar_types
from lyra_w4afp8 import machete_supported_schedules, machete_mm


MNK_SHAPES = [
    (1, 128, 128),
    (1, 7168, 2048),
    (64, 4096, 4096),
    (64, 8192, 28672),
    (1024, 4096, 8192),
    (1024, 8192, 4096),
]

GROUP_SIZES_TO_TEST: list[Optional[int]] = [128]

TEST_TYPES = [
    *(TypeConfig(act_type=torch.float8_e4m3fn,
                 weight_type=scalar_types.uint4b8,
                 output_type=a_type,
                 group_scale_type=a_type,
                 group_zero_type=None,
                 channel_scale_type=s_type,
                 token_scale_type=s_type)
      for s_type in [None, torch.float]
      for a_type in [torch.float16, torch.bfloat16])
]

# None stype means scales use the same dtype as a
def machete_mm_test_helper(types: TypeConfig,
                           tensors: Tensors,
                           group_size: Optional[int] = None,
                           schedule: Optional[str] = None):
    output_ref = torch.matmul(tensors.a_ref, tensors.w_ref)
    output_ref_type = output_ref.dtype

    if tensors.w_ch_s is not None:
        output_ref = (output_ref.to(tensors.w_ch_s.dtype) *
                      tensors.w_ch_s.unsqueeze(0)).to(output_ref_type)
    if tensors.w_tok_s is not None:
        output_ref = (output_ref.to(tensors.w_tok_s.dtype) *
                      tensors.w_tok_s.unsqueeze(1)).to(output_ref_type)

    group_layout = torch.zeros([tensors.a.shape[0]], dtype=torch.int32).to(tensors.a.device)
    valid_len = torch.tensor([63], dtype=torch.int32, device=tensors.a.device)

    output = machete_mm(
        a=tensors.a,
        b_q=tensors.w_q,
        b_type=types.weight_type,
        b_group_scales=tensors.w_g_s,
        b_group_zeros=tensors.w_g_zp,
        b_group_size=group_size,
        b_channel_scales=tensors.w_ch_s,
        a_token_scales=tensors.w_tok_s,
        out_type=types.output_type,
        schedule=schedule,
        group_layout=None
    )

    # Relax atol as our reduction dim becomes larger (more rounding error)
    # Relax atol when we have zeropoints since the way machete applies
    #  zeropoints (after scales) causes noise around 0
    atol = 1 if tensors.w_g_zp is not None else min(5e-2 * math.sqrt(tensors.a.shape[1]), 1)
    rtol = 1e-1 if tensors.a.element_size() >= 2 else 2e-1
    torch.testing.assert_close(output,output_ref.to(output.dtype),rtol=rtol,atol=atol)

@pytest.mark.parametrize("shape", MNK_SHAPES, ids=lambda x: "x".join(str(v) for v in x))
@pytest.mark.parametrize("types", TEST_TYPES)
def test_machete_all_schedules(shape, types: TypeConfig):

    group_sizes: list[Optional[int]] = []
    if types.group_scale_type is None:
        group_sizes = [None]
    else:
        group_sizes = GROUP_SIZES_TO_TEST

    for group_size in group_sizes:
        if not group_size_valid(shape, group_size):
            continue

        tensors = create_gemm_data(shape, types, group_size, mm_group_cnt=1)

        machete_mm_test_helper(types, tensors, group_size)

        # for schedule in machete_supported_schedules(
        #         types.act_type,
        #         types.weight_type,
        #         group_scales_type=types.group_scale_type,
        #         group_zeros_type=types.group_zero_type,
        #         out_type=types.output_type):

        #     print(f"="*100)
        #     print(f"MNK = {shape}")
        #     print(f"Testing schedule {schedule}")
        #     machete_mm_test_helper(types, tensors, group_size, schedule)

