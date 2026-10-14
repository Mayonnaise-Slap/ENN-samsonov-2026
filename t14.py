import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=8, num_stages=2)
    ],
    key=["B", "IN", "OUT"],
)
@triton.jit
def _backward_dx_fused_kernel(dy_ptr, w_q_ptr, w_scale_ptr, dx_ptr,
                             B, IN, OUT,
                             BLOCK_M: tl.constexpr,
                             BLOCK_N: tl.constexpr,
                             BLOCK_K: tl.constexpr,
                             PER_CHANNEL: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    rm = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    mask_m = rm < B
    mask_n = rn < IN

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, OUT, BLOCK_K):
        kk = k + rk
        mask_k = kk < OUT
        dy = tl.load(dy_ptr + rm[:, None] * OUT + kk[None, :],
                     mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_q_ptr + rn[None, :] * OUT + kk[:, None],
                    mask=mask_k[:, None] & mask_n[None, :], other=0)
        if PER_CHANNEL:
            w_scale = tl.load(w_scale_ptr + kk, mask=mask_k, other=0.0)
            w = w.to(tl.float32) * w_scale[:, None]
        acc += tl.dot(dy.to(tl.float16), w.to(tl.float16))

    if not PER_CHANNEL:
        acc = acc * tl.load(w_scale_ptr)

    tl.store(dx_ptr + rm[:, None] * IN + rn[None, :], acc.to(tl.float16),
             mask=mask_m[:, None] & mask_n[None, :])


def backward_dx_int8_fused(dy: torch.Tensor,
                     w_q: torch.Tensor,
                     w_scale: torch.Tensor,
                     *, per_channel: bool = True) -> torch.Tensor:
    """Вернуть dX = dY @ dequant(W_q)^T, dtype fp16, shape (B, IN)."""
    assert dy.is_cuda and w_q.is_cuda and w_scale.is_cuda
    assert dy.dtype == torch.float16 and w_q.dtype == torch.int8

    B, OUT = dy.shape
    IN, OUT_w = w_q.shape
    assert OUT == OUT_w, "inner dims mismatch"

    dy = dy.contiguous()
    w_q = w_q.contiguous()
    w_scale = w_scale.reshape(-1).to(torch.float32).contiguous()
    assert w_scale.numel() == (OUT if per_channel else 1)

    dx = torch.empty(B, IN, device=dy.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(B, meta["BLOCK_M"]), triton.cdiv(IN, meta["BLOCK_N"]))
    _backward_dx_fused_kernel[grid](dy, w_q, w_scale, dx,
                                   B, IN, OUT, PER_CHANNEL=per_channel)
    return dx
