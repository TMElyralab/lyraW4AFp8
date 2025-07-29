import ctypes
import os
import platform

import torch

SYSTEM_ARCH = platform.machine()

cuda_path = f"/usr/local/cuda/targets/{SYSTEM_ARCH}-linux/lib/libcudart.so.12"
if os.path.exists(cuda_path):
    ctypes.CDLL(cuda_path, mode=ctypes.RTLD_GLOBAL)

from lyra_w4afp8 import common_ops

from lyra_w4afp8.kernels import (
    machete_mm,
    machete_prepack_B,
    machete_supported_schedules,
    moe_align_block_size
)
from lyra_w4afp8.scalar_type import ScalarType, scalar_types
