import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _quantize_global(x_ptr, absmax_inv_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    gid0 = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE) + gid0 * BLOCK_SIZE
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0)
    absmax_inv = tl.load(absmax_inv_ptr)

    Q = tl.cast(127.0 * x * absmax_inv, dtype=tl.int8)
    tl.store(output_ptr + offs, Q, mask=mask)


def quantize_global(x: torch.Tensor):
    assert x.is_cuda, 'x on cuda plz'
    N = x.numel()

    B = 256

    absmax = x.abs().max()
    absmax_inv = 1 / absmax

    out = torch.zeros_like(x, device='cuda', dtype=torch.int8)

    grid = (triton.cdiv(N, B),)

    _quantize_global[grid](x, absmax_inv, out, N)
    return out, torch.unsqueeze(absmax, dim=0)