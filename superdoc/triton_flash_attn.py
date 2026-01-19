"""
Triton Flash Attention with Bias support.

Based on Flash-Attention-with-Bias-Triton (https://github.com/Dao-AILab/flash-attention-with-bias-triton)
Adapted for SuperDoc's attention bias requirements.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q, K, V, Bias, Out,
    softmax_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_bb, stride_bh, stride_bm, stride_bn,
    stride_ob, stride_oh, stride_om, stride_ok,
    seqlen_q, seqlen_k, headdim,
    HAVE_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_HEADDIM: tl.constexpr,
):
    """Flash attention forward kernel with bias support."""
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // stride_qh if stride_qh > 0 else off_hb
    off_h = off_hb % stride_qh if stride_qh > 0 else 0

    # Recompute off_b and off_h properly
    off_b = tl.program_id(1) // tl.load(K + 0) if False else tl.program_id(1) // stride_bh if stride_bh > 0 else 0
    # Simple version: assume off_hb encodes batch * num_heads linearly

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_hb * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + off_hb * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + off_hb * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    if HAVE_BIAS:
        bias_ptrs = Bias + off_hb * stride_bh + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn

    # Initialize accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # Load Q block - stays in SRAM throughout
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    q = (q * softmax_scale).to(q.dtype)

    # Loop over K, V blocks
    for start_n in range(0, seqlen_k, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K block
        k = tl.load(k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)

        # Compute QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), qk)

        # Add bias if present
        if HAVE_BIAS:
            bias = tl.load(bias_ptrs + start_n * stride_bn,
                          mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
                          other=0.0)
            qk = qk + bias

        # Mask out-of-bounds
        qk = tl.where(
            (offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
            qk, float("-inf")
        )

        # Online softmax: compute new max
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # Compute attention weights with numerical stability
        p = tl.exp(qk - m_new[:, None])
        l_ij = tl.sum(p, axis=1)

        # Update accumulator scaling
        alpha = tl.exp(m_i - m_new)
        l_new = alpha * l_i + l_ij

        # Scale previous accumulator and add new contribution
        acc = acc * alpha[:, None]

        # Load V block
        v = tl.load(v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)

        # Accumulate
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

        # Update running stats
        m_i = m_new
        l_i = l_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output
    out_ptrs = Out + off_hb * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)


@triton.jit
def _bwd_preprocess(
    Out, DO, Delta,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_db, stride_dh, stride_dm,
    seqlen_q, headdim,
    BLOCK_M: tl.constexpr, BLOCK_HEADDIM: tl.constexpr,
):
    """Compute Delta = rowsum(Out * dO) for backward pass."""
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hb = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    o = tl.load(Out + off_hb * stride_oh + off_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
                mask=(off_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
    do = tl.load(DO + off_hb * stride_doh + off_m[:, None] * stride_dom + offs_d[None, :] * stride_dok,
                 mask=(off_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)

    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hb * stride_dh + off_m * stride_dm, delta, mask=off_m < seqlen_q)


@triton.jit
def _bwd_kernel(
    Q, K, V, Bias, DO, DQ, DK, DV, DBias, Delta,
    softmax_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_bb, stride_bh, stride_bm, stride_bn,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_dvb, stride_dvh, stride_dvn, stride_dvk,
    stride_dbb, stride_dbh, stride_dbm, stride_dbn,
    stride_deltab, stride_deltah, stride_deltam,
    seqlen_q, seqlen_k, headdim,
    HAVE_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_HEADDIM: tl.constexpr,
):
    """Flash attention backward kernel."""
    start_n = tl.program_id(0)
    off_hb = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Load K, V for this block
    k_ptrs = K + off_hb * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + off_hb * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)

    # Initialize dK, dV accumulators
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

    # Loop over Q blocks
    for start_m in range(0, seqlen_q, BLOCK_M):
        offs_m_curr = start_m + offs_m

        # Load Q, dO, Delta for this block
        q_ptrs = Q + off_hb * stride_qh + offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptrs = DO + off_hb * stride_doh + offs_m_curr[:, None] * stride_dom + offs_d[None, :] * stride_dok
        delta_ptrs = Delta + off_hb * stride_deltah + offs_m_curr * stride_deltam

        q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
        do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
        delta = tl.load(delta_ptrs, mask=offs_m_curr < seqlen_q, other=0.0)

        # Recompute attention: QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q * softmax_scale, tl.trans(k), qk)

        # Add bias if present
        if HAVE_BIAS:
            bias_ptrs = Bias + off_hb * stride_bh + offs_m_curr[:, None] * stride_bm + offs_n[None, :] * stride_bn
            bias = tl.load(bias_ptrs,
                          mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                          other=0.0)
            qk = qk + bias

        # Mask and compute softmax
        qk = tl.where(
            (offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
            qk, float("-inf")
        )
        p = tl.exp(qk - tl.max(qk, axis=1)[:, None])
        p = p / tl.sum(p, axis=1)[:, None]
        p = tl.where(
            (offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
            p, 0.0
        )

        # Compute dV
        p_f16 = p.to(do.dtype)
        dv = dv + tl.dot(tl.trans(p_f16), do)

        # Compute dP
        dp = tl.dot(do, tl.trans(v))

        # Compute dS = P * (dP - Delta)
        ds = p * (dp - delta[:, None])
        ds = ds * softmax_scale

        # Compute dK
        ds_f16 = ds.to(q.dtype)
        dk = dk + tl.dot(tl.trans(ds_f16), q)

        # Compute dQ
        dq = tl.dot(ds_f16, k)
        dq_ptrs = DQ + off_hb * stride_dqh + offs_m_curr[:, None] * stride_dqm + offs_d[None, :] * stride_dqk
        tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)

        # Compute dBias if needed
        if HAVE_BIAS:
            dbias_ptrs = DBias + off_hb * stride_dbh + offs_m_curr[:, None] * stride_dbm + offs_n[None, :] * stride_dbn
            tl.atomic_add(dbias_ptrs, ds,
                         mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k))

    # Store dK, dV
    dk_ptrs = DK + off_hb * stride_dkh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk
    dv_ptrs = DV + off_hb * stride_dvh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvk
    tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=offs_n[:, None] < seqlen_k)
    tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=offs_n[:, None] < seqlen_k)


class FlashAttnWithBiasFunc(torch.autograd.Function):
    """Autograd function for flash attention with bias."""

    @staticmethod
    def forward(ctx, q, k, v, bias, softmax_scale):
        """
        Forward pass.

        Args:
            q: (batch, nheads, seqlen_q, headdim)
            k: (batch, nheads, seqlen_k, headdim)
            v: (batch, nheads, seqlen_k, headdim)
            bias: (batch, nheads, seqlen_q, seqlen_k) or None
            softmax_scale: float

        Returns:
            out: (batch, nheads, seqlen_q, headdim)
        """
        batch, nheads, seqlen_q, headdim = q.shape
        _, _, seqlen_k, _ = k.shape

        assert headdim in {16, 32, 64, 128}, f"headdim={headdim} not supported"

        # Ensure contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        if bias is not None:
            bias = bias.contiguous()

        out = torch.empty_like(q)

        # Block sizes
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_HEADDIM = headdim

        # Grid
        grid = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)

        _fwd_kernel[grid](
            q, k, v, bias, out,
            softmax_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            bias.stride(0) if bias is not None else 0,
            bias.stride(1) if bias is not None else 0,
            bias.stride(2) if bias is not None else 0,
            bias.stride(3) if bias is not None else 0,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            seqlen_q, seqlen_k, headdim,
            HAVE_BIAS=bias is not None,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_HEADDIM=BLOCK_HEADDIM,
        )

        ctx.save_for_backward(q, k, v, bias, out)
        ctx.softmax_scale = softmax_scale
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_HEADDIM = BLOCK_HEADDIM

        return out

    @staticmethod
    def backward(ctx, do):
        """Backward pass."""
        q, k, v, bias, out = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        BLOCK_M = ctx.BLOCK_M
        BLOCK_N = ctx.BLOCK_N
        BLOCK_HEADDIM = ctx.BLOCK_HEADDIM

        batch, nheads, seqlen_q, headdim = q.shape
        _, _, seqlen_k, _ = k.shape

        do = do.contiguous()

        # Allocate gradients
        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbias = torch.zeros_like(bias) if bias is not None else None

        # Compute Delta = rowsum(O * dO)
        delta = torch.empty(batch, nheads, seqlen_q, device=q.device, dtype=torch.float32)

        grid_preprocess = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)
        _bwd_preprocess[grid_preprocess](
            out, do, delta,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2),
            seqlen_q, headdim,
            BLOCK_M=BLOCK_M, BLOCK_HEADDIM=BLOCK_HEADDIM,
        )

        # Backward kernel
        grid_bwd = (triton.cdiv(seqlen_k, BLOCK_N), batch * nheads)
        _bwd_kernel[grid_bwd](
            q, k, v, bias, do, dq, dk, dv, dbias, delta,
            softmax_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            bias.stride(0) if bias is not None else 0,
            bias.stride(1) if bias is not None else 0,
            bias.stride(2) if bias is not None else 0,
            bias.stride(3) if bias is not None else 0,
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            dbias.stride(0) if dbias is not None else 0,
            dbias.stride(1) if dbias is not None else 0,
            dbias.stride(2) if dbias is not None else 0,
            dbias.stride(3) if dbias is not None else 0,
            delta.stride(0), delta.stride(1), delta.stride(2),
            seqlen_q, seqlen_k, headdim,
            HAVE_BIAS=bias is not None,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_HEADDIM=BLOCK_HEADDIM,
        )

        return dq, dk, dv, dbias, None


def flash_attention_with_bias(query, key, value, bias=None, softmax_scale=None):
    """
    Flash attention with optional bias support.

    Args:
        query: (batch, nheads, seqlen_q, headdim)
        key: (batch, nheads, seqlen_k, headdim)
        value: (batch, nheads, seqlen_k, headdim)
        bias: (batch, nheads, seqlen_q, seqlen_k) or None - attention bias (includes both
              relative position bias and padding mask)
        softmax_scale: scaling factor, defaults to 1/sqrt(headdim)

    Returns:
        output: (batch, nheads, seqlen_q, headdim)
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / (query.shape[-1] ** 0.5)

    return FlashAttnWithBiasFunc.apply(query, key, value, bias, softmax_scale)
