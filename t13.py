import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "GROUP_M": 8}, num_warps=4, num_stages=2)
    ],
    key=["M", "N"],
)
@triton.jit
def _quantize_global_transpose(A, absmax_inv_ptr, B,
                              stride_am, stride_an,
                              stride_bn, stride_bm,
                              M, N,
                              BLOCK_M: tl.constexpr,
                              BLOCK_N: tl.constexpr,
                              GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm < M)[:, None] & (rn < N)[None, :]

    a = tl.load(A + rm[:, None] * stride_am + rn[None, :] * stride_an,
                mask=mask, other=0.0).to(tl.float32)
    absmax_inv = tl.load(absmax_inv_ptr)

    q = tl.cast(127.0 * a * absmax_inv, dtype=tl.int8)
    tl.store(B + rn[None, :] * stride_bn + rm[:, None] * stride_bm, q, mask=mask)


def quantize_global_transpose(x: torch.Tensor):
    """Return (q_T:int8[N,M], absmax:fp32[1])."""
    assert x.is_cuda and x.ndim == 2
    assert x.stride(0) == 1 or x.stride(1) == 1

    M, N = x.shape
    absmax = x.abs().max().to(torch.float32).reshape(1)
    absmax_inv = 1.0 / torch.clamp(absmax, min=127 * 1e-8)

    out = torch.empty(N, M, device=x.device, dtype=torch.int8)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _quantize_global_transpose[grid](x, absmax_inv, out,
                                     x.stride(0), x.stride(1),
                                     out.stride(0), out.stride(1),
                                     M, N)
    return out, absmax
