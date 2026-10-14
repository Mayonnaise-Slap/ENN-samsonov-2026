import torch
import triton
import triton.language as tl


@triton.jit
def _flashatt_kernel(q_ptr, k_ptr, v_ptr, out_ptr, T, B1: tl.constexpr, E: tl.constexpr):
    gid0 = tl.program_id(0)
    q = tl.load(q_ptr + gid0)
    LOGE = tl.log2(E)

    m = float('-inf')
    l = 0.0
    acc = 0.0

    for block in tl.range(0, tl.cdiv(T, B1)):
        b_off = block * B1 + tl.arange(0, B1)
        b_mask = block * B1 + tl.arange(0, B1) < T

        b_k = tl.load(k_ptr + b_off, mask=b_mask, other=0)
        b_v = tl.load(v_ptr + b_off, mask=b_mask, other=0)

        S_b = q * b_k
        S_b = tl.where(b_mask, S_b, float('-inf'))

        m_new = tl.maximum(tl.max(S_b), m)

        alpha = tl.exp2((m - m_new) * LOGE)
        p = tl.exp2((S_b - m_new) * LOGE)

        l = l * alpha + tl.sum(p)
        acc = acc * alpha + tl.sum(p * b_v)
        m = m_new

    tl.store(out_ptr + gid0, acc / l)


def flashatt_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    assert q.is_cuda, 'Тензор q должен быть на GPU (CUDA).'
    assert k.is_cuda, 'Тензор k должен быть на GPU (CUDA).'
    assert v.is_cuda, 'Тензор v должен быть на GPU (CUDA).'
    T0 = q.numel()
    T1 = k.numel()
    T2 = v.numel()

    assert T0 == T1
    assert T1 == T2

    B = 32
    out = torch.zeros_like(q)

    grid = (T0,)

    _flashatt_kernel[grid](q, k, v, out, T0, B, torch.e)
    return out