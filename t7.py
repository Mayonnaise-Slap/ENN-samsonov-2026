import torch
import triton
import triton.language as tl


@triton.jit
def _long_sum_kernel(x_ptr, out_ptr, N0, T, B1: tl.constexpr):
    row = tl.program_id(0)

    acc = tl.zeros([B1], dtype=tl.float32)
    for start in range(0, T, B1):
        offs = start + tl.arange(0, B1)
        mask = offs < T
        x = tl.load(x_ptr + row * T + offs, mask=mask, other=0)
        acc += x

    tl.store(out_ptr + row, tl.sum(acc, axis=0))


def sum_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, 'Тензор x должен быть на GPU (CUDA).'
    assert x.dim() == 2
    x = x.contiguous()
    N0, T = x.shape

    B1 = 32
    out = torch.empty(N0, dtype=x.dtype, device=x.device)

    grid = (N0,)
    _long_sum_kernel[grid](x, out, N0, T, B1)
    return out
