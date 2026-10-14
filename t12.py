import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4, num_stages=2)
    ],
    key=["B", "IN", "OUT"],
)
@triton.jit
def _forward_int8_fused_kernel(x_q_ptr, x_scale_ptr,
                               w_q_ptr, w_scale_ptr,
                               b_ptr, y_ptr,
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
    mask_n = rn < OUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, IN, BLOCK_K):
        kk = k + rk
        mask_k = kk < IN
        x = tl.load(x_q_ptr + rm[:, None] * IN + kk[None, :],
                    mask=mask_m[:, None] & mask_k[None, :], other=0)
        w = tl.load(w_q_ptr + kk[:, None] * OUT + rn[None, :],
                    mask=mask_k[:, None] & mask_n[None, :], other=0)
        acc += tl.dot(x.to(tl.float16), w.to(tl.float16))

    x_scale = tl.load(x_scale_ptr + rm, mask=mask_m, other=0.0).to(tl.float32)
    if PER_CHANNEL:
        w_scale = tl.load(w_scale_ptr + rn, mask=mask_n, other=0.0).to(tl.float32)
    else:
        w_scale = tl.load(w_scale_ptr).to(tl.float32)
    bias = tl.load(b_ptr + rn, mask=mask_n, other=0.0).to(tl.float32)

    y = acc * x_scale[:, None]
    if PER_CHANNEL:
        y = y * w_scale[None, :]
    else:
        y = y * w_scale
    y = y + bias[None, :]

    tl.store(y_ptr + rm[:, None] * OUT + rn[None, :], y.to(tl.float16),
             mask=mask_m[:, None] & mask_n[None, :])


def matmul_int8_fused(x_q: torch.Tensor,
                      x_scale: torch.Tensor,
                      w_q: torch.Tensor,
                      w_scale: torch.Tensor,
                      bias: torch.Tensor | None = None,
                      *, per_channel: bool = True) -> torch.Tensor:
    """Вернуть Y = dequant(X_q) @ dequant(W_q) + bias, dtype fp16, shape (B, OUT)."""
    assert x_q.is_cuda and w_q.is_cuda and x_scale.is_cuda and w_scale.is_cuda
    assert x_q.dtype == torch.int8 and w_q.dtype == torch.int8

    B, IN = x_q.shape
    IN_w, OUT = w_q.shape
    assert IN == IN_w, "inner dims mismatch"

    x_q = x_q.contiguous()
    w_q = w_q.contiguous()
    x_scale = x_scale.reshape(-1).expand(B).contiguous()
    w_scale = w_scale.reshape(-1).contiguous()
    assert w_scale.numel() == (OUT if per_channel else 1)

    if bias is None:
        bias = torch.zeros(OUT, device=x_q.device, dtype=torch.float16)
    assert bias.is_cuda and bias.numel() == OUT
    bias = bias.contiguous()

    y = torch.empty(B, OUT, device=x_q.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(B, meta["BLOCK_M"]), triton.cdiv(OUT, meta["BLOCK_N"]))
    _forward_int8_fused_kernel[grid](x_q, x_scale, w_q, w_scale, bias, y,
                                     B, IN, OUT, PER_CHANNEL=per_channel)
    return y
