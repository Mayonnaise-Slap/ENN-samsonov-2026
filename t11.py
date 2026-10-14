import torch, math
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _quantize_rowwise(x_ptr, output_ptr, output_maxs, n_elements,
                      BLOCK_SIZE: tl.constexpr, P2: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, P2)
    mask = cols < BLOCK_SIZE
    offs = pid * BLOCK_SIZE + cols

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.max(tl.abs(x), axis=0)

    q = tl.cast(127.0 * (x / absmax), dtype=tl.int8)
    tl.store(output_ptr + offs, q.to(tl.int8), mask=mask)
    tl.store(output_maxs + pid, absmax.to(tl.float16))


def quantize_rowwise(x: torch.Tensor):
    assert x.is_cuda and x.is_contiguous() and x.ndim == 2

    n_rows, n_cols = x.shape
    out = torch.empty_like(x, dtype=torch.int8)
    out_absmax = torch.empty(n_rows, device=x.device, dtype=torch.float16)

    _quantize_rowwise[(n_rows,)](
        x, out, out_absmax, n_cols,
        BLOCK_SIZE=n_cols, P2=triton.next_power_of_2(n_cols),
    )
    return out, out_absmax
