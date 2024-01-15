import torch
import torch._custom_ops
from torch import Tensor

import fast_hadamard_transform_cuda
import quiptools_cuda

my_lib = torch.library.Library("quip_lib", "DEF")

@torch._custom_ops.custom_op("quip_lib::hadamard")
def hadamard(x: Tensor, scale: float) -> Tensor:
    raise NotImplementedError()

@torch._custom_ops.impl_abstract("quip_lib::hadamard")
def hadamard_abstract(x: Tensor, scale: float) -> Tensor:
    return x

@torch._custom_ops.impl("quip_lib::hadamard", device_types="cuda")
def hadamard_cuda(x: Tensor, scale: float) -> Tensor:
    return fast_hadamard_transform_cuda.fast_hadamard_transform(x, scale)

@torch._custom_ops.custom_op("quip_lib::e8p_mm_origorder")
def e8p_mm_origorder(x: Tensor, Qidxs: Tensor, grid: Tensor) -> Tensor:
    raise NotImplementedError()

@torch._custom_ops.impl_abstract("quip_lib::e8p_mm_origorder")
def e8p_mm_origorder_abstract(x: Tensor, Qidxs: Tensor, grid: Tensor) -> Tensor:
    assert x.dim() == 2
    assert Qidxs.dim() == 2
    assert x.shape[1] == Qidxs.shape[1] * 8
    assert x.device == Qidxs.device
    assert x.device == grid.device
    result = x.new_empty((x.shape[0], Qidxs.shape[0]), dtype=x.dtype)
    return result

@torch._custom_ops.impl("quip_lib::e8p_mm_origorder", device_types="cuda")
def e8p_mm_origorder_cuda(x: Tensor, Qidxs: Tensor, grid: Tensor) -> Tensor:
    return quiptools_cuda.e8p_mm_origorder(x, Qidxs, grid)

@torch._custom_ops.custom_op("quip_lib::e8prvq3_mm_origorder")
def e8prvq3_mm_origorder(x: Tensor, Qidxs: Tensor, grid: Tensor, grid2: Tensor, scale: float) -> Tensor:
    raise NotImplementedError()

@torch._custom_ops.impl_abstract("quip_lib::e8prvq3_mm_origorder")
def e8prvq3_mm_origorder_abstract(x: Tensor, Qidxs: Tensor, grid: Tensor, grid2: Tensor, scale: float) -> Tensor:
    assert x.dim() == 2
    assert Qidxs.dim() == 2
    assert x.device == Qidxs.device
    assert x.device == grid.device
    assert x.device == grid2.device
    result = x.new_empty((x.shape[0], Qidxs.shape[0]), dtype=x.dtype)
    return result

@torch._custom_ops.impl("quip_lib::e8prvq3_mm_origorder", device_types="cuda")
def e8prvq3_mm_origorder_cuda(x: Tensor, Qidxs: Tensor, grid: Tensor, grid2: Tensor, scale: float) -> Tensor:
    return quiptools_cuda.e8prvq3_mm_origorder(x, Qidxs, grid, grid2, scale)

@torch._custom_ops.custom_op("quip_lib::e8prvq4_mm_origorder")
def e8prvq4_mm_origorder(x: Tensor, Qidxs: Tensor, grid: Tensor, scale: float) -> Tensor:
    raise NotImplementedError()

@torch._custom_ops.impl_abstract("quip_lib::e8prvq4_mm_origorder")
def e8p_mm_origorder_abstract(x: Tensor, Qidxs: Tensor, grid: Tensor, scale: float) -> Tensor:
    assert x.dim() == 2
    assert Qidxs.dim() == 2
    assert x.device == Qidxs.device
    assert x.device == grid.device
    result = x.new_empty((x.shape[0], Qidxs.shape[0]), dtype=x.dtype)
    return result

@torch._custom_ops.impl("quip_lib::e8prvq4_mm_origorder", device_types="cuda")
def e8p_mm_origorder_cuda(x: Tensor, Qidxs: Tensor, grid: Tensor, scale: float) -> Tensor:
    return quiptools_cuda.e8prvq4_mm_origorder(x, Qidxs, grid, scale)
