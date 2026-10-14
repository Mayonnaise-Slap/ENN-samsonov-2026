import torch
import triton
import triton.language as tl

@triton.jit
def _long_softmax_kernel(x_ptr, out_ptr, N0, T, B1: tl.constexpr, e: tl.constexpr):
    pid0 = tl.program_id(0)

    offs_start = T * pid0
    total_max = float('-inf')
    exp_sum = 0.0

    # row max
    for block in tl.range(tl.cdiv(T, B1)):
        b_off = offs_start + block * B1 + tl.arange(0, B1)
        b_mask = block * B1 + tl.arange(0, B1) < T

        b_x = tl.load(x_ptr + b_off, b_mask, other=float("-inf"))

        b_max = tl.max(b_x)
        total_max = tl.maximum(total_max, b_max)


    # exp sum
    for block in tl.range(tl.cdiv(T, B1)):
        b_off = offs_start + block * B1 + tl.arange(0, B1)
        b_mask = block * B1 + tl.arange(0, B1) < T

        b_x = tl.load(x_ptr + b_off, b_mask, other=float('-inf'))

        exp_sum += tl.sum(
            tl.exp2((b_x - total_max) * tl.log2(e))
        )

    # out write
    for block in tl.range(tl.cdiv(T, B1)):
        b_off = offs_start + block * B1 + tl.arange(0, B1)
        b_mask = block * B1 + tl.arange(0, B1) < T

        b_x = tl.load(x_ptr + b_off, b_mask, other=0)

        exp = tl.exp2((b_x - total_max) * tl.log2(e)) / exp_sum
        tl.store(out_ptr + b_off, exp, mask=b_mask)



def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, 'Тензор x должен быть на GPU (CUDA).'
    N0, T = x.shape

    B = 256
    out = torch.zeros_like(x)

    grid = (N0,)

    euler = torch.e

    _long_softmax_kernel[grid](x, out, N0, T, B, euler)
    return out