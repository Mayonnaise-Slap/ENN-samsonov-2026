import torch
import triton
import triton.language as tl

@triton.jit
def _mul_relu_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
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
    prod = tl.maximum(y[:, None] * x[None, :], 0)

    tl.store(z_ptr + off, prod, mask=mask)



def mul_relu_block_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, 'Тензор x должен быть на GPU (CUDA).'
    assert y.is_cuda, 'Тензор y должен быть на GPU (CUDA).'
    Nx = x.numel()
    Ny = y.numel()

    B = 32

    grid = (triton.cdiv(Nx, B), triton.cdiv(Ny, B))
    z = torch.empty((Ny, Nx), device=x.device, dtype=x.dtype)
    _mul_relu_block_kernel[grid](x, y, z, Nx, Ny, B, B)
    return z