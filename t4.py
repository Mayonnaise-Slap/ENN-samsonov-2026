import torch
import triton
import triton.language as tl

@triton.jit
def _outer_add_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    off_col = pid_0 * B0 + tl.arange(0, B0)
    off_row = pid_1 * B1 + tl.arange(0, B1)

    off = off_row[:, None] * N0 + off_col[None, :]

    mask_col = off_col < N0
    mask_row = off_row < N1

    mask = mask_row[:, None] & mask_col[None, :]

    x = tl.load(x_ptr + off_col, mask=mask_col, other=0)
    y = tl.load(y_ptr + off_row, mask=mask_row, other=0)

    tl.store(z_ptr + off, x[None, :] + y[:, None], mask=mask)


def add_vec_block_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, 'Тензор x должен быть на GPU (CUDA).'
    assert y.is_cuda, 'Тензор y должен быть на GPU (CUDA).'
    Nx = x.numel()
    Ny = y.numel()

    B = 32

    grid = (triton.cdiv(Nx, B), triton.cdiv(Ny, B))
    z = torch.empty((Ny, Nx), device=x.device, dtype=x.dtype)
    _outer_add_block_kernel[grid](x, y, z, Nx, Ny, B, B)
    return z