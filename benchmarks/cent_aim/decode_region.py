# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CENT's decode block as one Allo region plus per-case host programs.

Every stage is an ``@allo.work`` kernel over the region's buffers; a case is a
host program that launches a subset of those kernels with its host transfers.
The compiler derives every AiM command from the matched kernels, the chosen
placements and these host programs.

Annotations here are evaluated eagerly (no ``from __future__ import
annotations``) because the buffer shapes come from the ``DecodeSpec``.
"""

import inspect
from typing import NamedTuple

import allo
from allo.dataflow import region
from allo.ir.types import bfloat16

# Buffers loaded once before the measured decode step (CENT --only-trace).
RESIDENT = ("g_x", "g_sa", "cs_q", "cs_k", "wq", "wk", "wv", "wo", "w1", "w3", "w2")


class CentCase(NamedTuple):
    """One CENT case: the shared decode region and the host program."""

    region: object
    host_program: object


def build_decode_region(spec, banks_per_channel: int = 16):
    """The full decode block for ``spec``.

    ``Q`` is the workload's RMS tree-reduction factor: each partial covers
    ``D / (channels_per_replica * banks / 2)`` elements, one per bank pair.
    Region arguments are declared in the order their bank rows are allocated.
    """
    D, H, Dh, F, L = spec.D, spec.H, spec.Dh, spec.F, spec.L
    M = spec.max_seq_len
    Q = D // (spec.channels_per_replica * (banks_per_channel // 2))
    P = D // Q
    R = spec.replicas
    HM = H * M
    HL = H * L
    D2 = 2 * D

    @region()
    def decode_block(
        x_x: bfloat16[D],
        s_x: bfloat16[D],
        t_x: bfloat16[D],
        g_x: bfloat16[D],
        o_x: bfloat16[D],
        x_sa: bfloat16[D],
        s_sa: bfloat16[D],
        t_sa: bfloat16[D],
        g_sa: bfloat16[D],
        o_sa: bfloat16[D],
        cs_q: bfloat16[D2],
        m_q: bfloat16[D2],
        r_q: bfloat16[D2],
        cs_k: bfloat16[D2],
        m_k: bfloat16[D2],
        r_k: bfloat16[D2],
        sm_a: bfloat16[HM],
        sm_b: bfloat16[HM],
        sm_c: bfloat16[HM],
        x1: bfloat16[F],
        sig: bfloat16[F],
        u: bfloat16[F],
        x3: bfloat16[F],
        z: bfloat16[F],
        k_cache: bfloat16[H, M, Dh],
        v_cache: bfloat16[H, Dh, M],
        wq: bfloat16[D, D],
        wk: bfloat16[D, D],
        wv: bfloat16[D, D],
        wo: bfloat16[D, D],
        w1: bfloat16[F, D],
        w3: bfloat16[F, D],
        w2: bfloat16[D, F],
        part_x: bfloat16[P],
        part_sa: bfloat16[P],
        h_in: bfloat16[D],
        q_out: bfloat16[D],
        k_out: bfloat16[D],
        v_out: bfloat16[D],
        att: bfloat16[D],
        o_out: bfloat16[D],
        ffn_in: bfloat16[D],
        w2_in: bfloat16[F],
        ffn_out: bfloat16[D],
        q_heads: bfloat16[H, Dh],
        scores: bfloat16[H, L],
        probs: bfloat16[H, L],
        heads_out: bfloat16[H, Dh],
        ra_a: bfloat16[D],
        ra_b: bfloat16[D],
        ra_y: bfloat16[D],
        rf_a: bfloat16[D],
        rf_b: bfloat16[D],
        rf_y: bfloat16[D],
    ):
        # ---------------- RMSNorm (x and sa) ---------------- #
        @allo.work(mapping=[R], args=[x_x, part_x])
        def rms_partial_x(xs: bfloat16[D], part: bfloat16[P]):
            for p in range(P):
                acc: bfloat16 = 0
                for k in range(Q):
                    acc += xs[p * Q + k] * xs[p * Q + k]
                part[p] = acc

        @allo.work(mapping=[R], args=[s_x, x_x, t_x])
        def rms_scale_x(sc: bfloat16[D], xs: bfloat16[D], ts: bfloat16[D]):
            for i in range(D):
                ts[i] = sc[i] * xs[i]

        @allo.work(mapping=[R], args=[g_x, t_x, o_x])
        def rms_norm_x(gs: bfloat16[D], ts: bfloat16[D], os: bfloat16[D]):
            for i in range(D):
                os[i] = gs[i] * ts[i]

        @allo.work(mapping=[R], args=[x_sa, part_sa])
        def rms_partial_sa(xs: bfloat16[D], part: bfloat16[P]):
            for p in range(P):
                acc: bfloat16 = 0
                for k in range(Q):
                    acc += xs[p * Q + k] * xs[p * Q + k]
                part[p] = acc

        @allo.work(mapping=[R], args=[s_sa, x_sa, t_sa])
        def rms_scale_sa(sc: bfloat16[D], xs: bfloat16[D], ts: bfloat16[D]):
            for i in range(D):
                ts[i] = sc[i] * xs[i]

        @allo.work(mapping=[R], args=[g_sa, t_sa, o_sa])
        def rms_norm_sa(gs: bfloat16[D], ts: bfloat16[D], os: bfloat16[D]):
            for i in range(D):
                os[i] = gs[i] * ts[i]

        # ---------------- resident-weight projections ---------------- #
        @allo.work(mapping=[R], args=[wq, h_in, q_out])
        def q_projection(W: bfloat16[D, D], v: bfloat16[D], y: bfloat16[D]):
            for i in range(D):
                acc: bfloat16 = 0
                for k in range(D):
                    acc += W[i, k] * v[k]
                y[i] = acc

        @allo.work(mapping=[R], args=[wk, h_in, k_out])
        def k_projection(W: bfloat16[D, D], v: bfloat16[D], y: bfloat16[D]):
            for i in range(D):
                acc: bfloat16 = 0
                for k in range(D):
                    acc += W[i, k] * v[k]
                y[i] = acc

        @allo.work(mapping=[R], args=[wv, h_in, v_out])
        def v_projection(W: bfloat16[D, D], v: bfloat16[D], y: bfloat16[D]):
            for i in range(D):
                acc: bfloat16 = 0
                for k in range(D):
                    acc += W[i, k] * v[k]
                y[i] = acc

        @allo.work(mapping=[R], args=[wo, att, o_out])
        def o_projection(W: bfloat16[D, D], v: bfloat16[D], y: bfloat16[D]):
            for i in range(D):
                acc: bfloat16 = 0
                for k in range(D):
                    acc += W[i, k] * v[k]
                y[i] = acc

        # W1 stores the pre-activation (RD_MAC) and its sigmoid (RD_AF).
        @allo.work(mapping=[R], args=[w1, ffn_in, x1, sig])
        def w1_projection_af(
            W: bfloat16[F, D], v: bfloat16[D], y: bfloat16[F], sg: bfloat16[F]
        ):
            for i in range(F):
                acc: bfloat16 = 0
                for k in range(D):
                    acc += W[i, k] * v[k]
                y[i] = acc
                sg[i] = 0.5 * allo.tanh(0.5 * acc) + 0.5

        @allo.work(mapping=[R], args=[w3, ffn_in, x3])
        def w3_projection(W: bfloat16[F, D], v: bfloat16[D], y: bfloat16[F]):
            for i in range(F):
                acc: bfloat16 = 0
                for k in range(D):
                    acc += W[i, k] * v[k]
                y[i] = acc

        @allo.work(mapping=[R], args=[w2, w2_in, ffn_out])
        def w2_projection(W: bfloat16[D, F], v: bfloat16[F], y: bfloat16[D]):
            for i in range(D):
                acc: bfloat16 = 0
                for k in range(F):
                    acc += W[i, k] * v[k]
                y[i] = acc

        # ---------------- RoPE device multiply ---------------- #
        @allo.work(mapping=[R], args=[cs_q, m_q, r_q])
        def rope_q(cs: bfloat16[D2], ms: bfloat16[D2], rs: bfloat16[D2]):
            for i in range(D):
                rs[i] = cs[i] * ms[i]

        @allo.work(mapping=[R], args=[cs_k, m_k, r_k])
        def rope_k(cs: bfloat16[D2], ms: bfloat16[D2], rs: bfloat16[D2]):
            for i in range(D):
                rs[i] = cs[i] * ms[i]

        # ---------------- attention ---------------- #
        @allo.work(mapping=[R], args=[q_heads, k_cache, scores])
        def attention_qk(
            q: bfloat16[H, Dh], kc: bfloat16[H, M, Dh], sc: bfloat16[H, L]
        ):
            for h in range(H):
                for s in range(L):
                    acc: bfloat16 = 0
                    for d in range(Dh):
                        acc += q[h, d] * kc[h, s, d]
                    sc[h, s] = acc

        @allo.work(mapping=[R], args=[sm_a, sm_b, sm_c])
        def softmax_mul(a: bfloat16[HM], b: bfloat16[HM], c: bfloat16[HM]):
            for i in range(HL):
                c[i] = a[i] * b[i]

        @allo.work(mapping=[R], args=[probs, v_cache, heads_out])
        def attention_sv(
            pr: bfloat16[H, L], vc: bfloat16[H, Dh, M], o: bfloat16[H, Dh]
        ):
            for h in range(H):
                for d in range(Dh):
                    acc: bfloat16 = 0
                    for s in range(L):
                        acc += pr[h, s] * vc[h, d, s]
                    o[h, d] = acc

        # ---------------- residuals (counted once, as in the vendor trace) --- #
        @allo.work(mapping=[1], args=[ra_a, ra_b, ra_y])
        def attention_residual(a: bfloat16[D], b: bfloat16[D], y: bfloat16[D]):
            for i in range(D):
                y[i] = a[i] + b[i]

        @allo.work(mapping=[1], args=[rf_a, rf_b, rf_y])
        def ffn_residual(a: bfloat16[D], b: bfloat16[D], y: bfloat16[D]):
            for i in range(D):
                y[i] = a[i] + b[i]

        # ---------------- FFN SiLU and gate ---------------- #
        @allo.work(mapping=[R], args=[x1, sig, u])
        def ffn_silu(xa: bfloat16[F], sa: bfloat16[F], ua: bfloat16[F]):
            for i in range(F):
                ua[i] = xa[i] * sa[i]

        @allo.work(mapping=[R], args=[x3, u, z])
        def ffn_gate(xb: bfloat16[F], ub: bfloat16[F], zb: bfloat16[F]):
            for i in range(F):
                zb[i] = xb[i] * ub[i]

    return decode_block


# --------------------------------------------------------------------- #
# Host programs: one ordered list of host steps per stage.
# --------------------------------------------------------------------- #


def _rms_steps(b, prefix):
    x, s, t, g, o = (b[f"{name}_{prefix}"] for name in ("x", "s", "t", "g", "o"))
    hx = allo.host_xfer
    hx.scatter(x, hx.banks)
    allo.launch(f"rms_partial_{prefix}", x, b[f"part_{prefix}"])
    hx.scatter(s, hx.banks)
    hx.scatter(x, hx.banks)
    allo.launch(f"rms_scale_{prefix}", s, x, t)
    allo.launch(f"rms_norm_{prefix}", g, t, o)
    hx.gather(o, hx.banks)


def _rope_steps(b):
    hx = allo.host_xfer
    hx.scatter(b["m_q"], hx.banks)
    hx.scatter(b["m_k"], hx.banks)
    allo.launch("rope_q", b["cs_q"], b["m_q"], b["r_q"])
    allo.launch("rope_k", b["cs_k"], b["m_k"], b["r_k"])
    hx.scatter(b["r_q"], hx.banks)
    hx.scatter(b["r_k"], hx.banks)


def _cache_append_steps(b, spec):
    hx = allo.host_xfer
    position = spec.L - 1
    hx.scatter(b["k_cache"][:, position, :], hx.banks)
    hx.broadcast(b["v_cache"][:, :, position], hx.banks)


def _softmax_steps(b, spec):
    hx = allo.host_xfer
    extent = spec.H * spec.L
    for _phase in ("scale", "normalize_exp"):
        hx.scatter(b["sm_a"][0:extent], hx.banks)
        hx.scatter(b["sm_b"][0:extent], hx.banks)
        allo.launch("softmax_mul", b["sm_a"], b["sm_b"], b["sm_c"])
        hx.gather(b["sm_c"][0:extent], hx.banks)


def _ffn_activation_steps(b):
    hx = allo.host_xfer
    hx.scatter(b["x1"], hx.banks)
    hx.scatter(b["sig"], hx.banks)
    allo.launch("ffn_silu", b["x1"], b["sig"], b["u"])
    hx.scatter(b["x3"], hx.banks)
    allo.launch("ffn_gate", b["x3"], b["u"], b["z"])
    hx.gather(b["z"], hx.banks)


def _launch(kernel, *names):
    def steps(b, _spec):
        allo.launch(kernel, *(b[name] for name in names))

    return steps


_STAGE_STEPS = {
    "rms_x": lambda b, _s: _rms_steps(b, "x"),
    "q_projection": _launch("q_projection", "wq", "h_in", "q_out"),
    "k_projection": _launch("k_projection", "wk", "h_in", "k_out"),
    "v_projection": _launch("v_projection", "wv", "h_in", "v_out"),
    "rope": lambda b, _s: _rope_steps(b),
    "attention_cache_append": _cache_append_steps,
    "attention_qk": _launch("attention_qk", "q_heads", "k_cache", "scores"),
    "softmax_host_split": _softmax_steps,
    "attention_sv": _launch("attention_sv", "probs", "v_cache", "heads_out"),
    "o_projection": _launch("o_projection", "wo", "att", "o_out"),
    "attention_residual": _launch("attention_residual", "ra_a", "ra_b", "ra_y"),
    "rms_sa": lambda b, _s: _rms_steps(b, "sa"),
    "w1_projection_af": _launch("w1_projection_af", "w1", "ffn_in", "x1", "sig"),
    "w3_projection": _launch("w3_projection", "w3", "ffn_in", "x3"),
    "ffn_activation": lambda b, _s: _ffn_activation_steps(b),
    "w2_projection": _launch("w2_projection", "w2", "w2_in", "ffn_out"),
    "ffn_residual": _launch("ffn_residual", "rf_a", "rf_b", "rf_y"),
}


def build_host_program(decode_region, stages, spec):
    """Host program launching ``stages`` of ``decode_region`` in order."""

    def host(**buffers):
        for stage in stages:
            try:
                steps = _STAGE_STEPS[stage]
            except KeyError as error:
                raise KeyError(f"unknown CENT AiM stage {stage!r}") from error
            steps(buffers, spec)

    host.__signature__ = inspect.signature(decode_region)
    return allo.host_program(decode_region, subset=True, resident=RESIDENT)(host)


__all__ = [
    "RESIDENT",
    "CentCase",
    "build_decode_region",
    "build_host_program",
]
