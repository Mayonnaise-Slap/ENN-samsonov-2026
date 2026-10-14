import torch
import triton
import triton.language as tl

@triton.jit
def _outer_add_kernel(x_ptr, y_ptr, z_ptr, N0: tl.constexpr, N1: tl.constexpr):
    xOffsets = tl.arange(0, N0)
    yOffsets = tl.arange(0, N1)

    x = tl.load(x_ptr + xOffsets)
    y = tl.load(y_ptr + yOffsets)

    output_offsets = yOffsets[:, None] * N0 + xOffsets[None, :]

    tl.store(z_ptr + output_offsets, y[:, None] + x[None, :])



def outer_vector_add_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, 'Тензор x должен быть на GPU (CUDA).'
    assert y.is_cuda, 'Тензор y должен быть на GPU (CUDA).'
    Nx = x.numel()
    Ny = y.numel()

    grid = (1,)
    z = torch.empty((Ny, Nx), device=x.device, dtype=x.dtype)
    _outer_add_kernel[grid](x, y, z, Nx, Ny)
    return z