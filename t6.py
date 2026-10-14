import torch
import triton
import triton.language as tl


@triton.jit
def _mul_relu_block_back_kernel(x_ptr, y_ptr, dz_ptr, dx_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    offs0 = pid0 * B0 + tl.arange(0, B0)
    offs1 = pid1 * B1 + tl.arange(0, B1)
    off = offs1[:, None] * N0 + offs0[None, :]

    mask0 = offs0 < N0
    mask1 = offs1 < N1

    mask = mask1[:, None] & mask0[None, :]

    x = tl.load(x_ptr + offs0, mask=mask0, other=0)
    y = tl.load(y_ptr + offs1, mask=mask1, other=0)
    dz = tl.load(dz_ptr + off, mask=mask, other=0)

    u = y[:, None] * x[None, :]
    relu_mask = u > 0
    g = tl.where(relu_mask, y[:, None] * dz, 0.0)

    prod = tl.maximum(y[:, None] * x[None, :], 0)

    partial = tl.sum(g, axis=0)
    tl.atomic_add(dx_ptr + offs0, partial, mask=mask0)


def mul_relu_block_back_triton(x: torch.Tensor, y: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, 'Тензор x должен быть на GPU (CUDA).'
    assert y.is_cuda, 'Тензор y должен быть на GPU (CUDA).'
    assert dz.is_cuda, 'Тензор dz должен быть на GPU (CUDA).'
    N0 = x.numel()
    N1 = y.numel()
    assert dz.shape == (N1, N0)
    dz = dz.contiguous()

    B0, B1 = 32, 32
    dx = torch.zeros_like(x)

    grid = (triton.cdiv(N0, B0), triton.cdiv(N1, B1))
    _mul_relu_block_back_kernel[grid](x, y, dz, dx, N0, N1, B0, B1)
    return dx