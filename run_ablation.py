# ============================================================
# run_ablation.py — MEMORY-SAFE ABLATION RUNNER
#
# All N×N kernel matrices replaced with chunked O(CHUNK×N) work.
# Gradient checkpointing reduces backward-pass memory from
# O(num_chunks × chunk_intermediates) to O(1 × chunk_intermediates).
# Each model wrapped in try/except so one failure never kills the run.
# Same mathematical results as the original.
# ============================================================


import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()  # Colab / Jupyter: __file__ is undefined
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PRICES_CACHE = DATA_DIR / "crypto_prices.npz"
LOG_PATH = RESULTS_DIR / "ablation_log.json"

# ----------------------------------------------------------------------
# MEMORY CONTROL
# CHUNK: process this many target events per chunk.
# Gradient checkpointing ensures backward memory is O(1 chunk) not O(all).
# ----------------------------------------------------------------------
CHUNK = 256

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DEFAULT_CONFIG = dict(
    TAUS=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    DIMS=[2, 4, 8, 16],
    BETA_SETS={
        "fast":    (0.35, 1.0),
        "default": (0.35, 1.0, 3.0, 8.0),
        "wide":    (0.1, 0.35, 1.0, 3.0, 8.0, 15.0),
    },
    SEEDS=[0, 1, 2],
    EPOCHS=400,
    LR=0.03,
    HID=16,
    SPLIT=0.70,
    VOL_WIN=60,
    RUN_LIT_BASELINES=True,
    LIT_EPOCHS=400,
    CLASSIC_EPOCHS=500,
    CLASSIC_LR=0.05,
    CANON_TAU=2.5,
    CANON_DIM=8,
    CANON_BETA="default",
)

CRYPTO_UNIVERSE = [
    ("BTC", "Store-of-Value"),    ("ETH", "Smart-Contract-L1"),
    ("BNB", "Exchange"),          ("XRP", "Payments"),
    ("ADA", "Smart-Contract-L1"), ("SOL", "Smart-Contract-L1"),
    ("DOGE", "Meme"),             ("DOT", "Infrastructure"),
    ("LINK", "Infrastructure"),   ("LTC", "Payments"),
    ("BCH", "Payments"),          ("AVAX", "Smart-Contract-L1"),
    ("MATIC", "Smart-Contract-L2"), ("UNI", "DeFi"),
    ("ATOM", "Infrastructure"),   ("XLM", "Payments"),
    ("ETC", "Smart-Contract-L1"), ("FIL", "Infrastructure"),
    ("TRX", "Smart-Contract-L1"), ("EOS", "Smart-Contract-L1"),
    ("AAVE", "DeFi"),             ("MKR", "DeFi"),
    ("ALGO", "Smart-Contract-L1"), ("XTZ", "Smart-Contract-L1"),
    ("LUNA", "Smart-Contract-L1"), ("DASH", "Payments"),
    ("ZEC", "Payments"),          ("COMP", "DeFi"),
    ("SNX", "DeFi"),              ("SUSHI", "DeFi"),
]


# ========================================================================
# 1. DATA LAYER
# ========================================================================
def get_returns_panel(rebuild=False):
    if PRICES_CACHE.exists() and not rebuild:
        d = np.load(PRICES_CACHE, allow_pickle=True)
        labels_list = list(d["labels"])
        cat_map = dict(CRYPTO_UNIVERSE)
        sectors = [cat_map.get(lbl, "Other") for lbl in labels_list]
        return d["returns"], list(d["dates"]), labels_list, sectors
    raise FileNotFoundError(
        f"{PRICES_CACHE} not found. Run Step 1 (Binance data download) first "
        f"to create it, then re-run this script.")


def events_from_returns(returns, tau, vol_win):
    df = pd.DataFrame(returns)
    z = df / df.rolling(vol_win).std()
    z = z.iloc[vol_win:].reset_index(drop=True)
    ev_day, ev_mark = [], []
    for j in range(df.shape[1]):
        hits = np.where(np.abs(z[j].values) > tau)[0]
        ev_day.extend(hits.tolist())
        ev_mark.extend([j] * len(hits))
    order = np.argsort(ev_day, kind="mergesort")
    times = np.array(ev_day, dtype=float)[order]
    marks = np.array(ev_mark, dtype=int)[order]
    T = float(len(z))
    return times, marks, T


def get_events(tau, vol_win):
    cache_path = DATA_DIR / f"events_tau_{tau}.npz"
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        return (d["times"], d["marks"].astype(int), float(d["T"]), int(d["D"]),
                list(d["labels"]), list(d["sectors"]))
    returns, dates, labels, sectors = get_returns_panel()
    times, marks, T = events_from_returns(returns, tau, vol_win)
    D = returns.shape[1]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, times=times, marks=marks, T=T, D=D,
             labels=np.array(labels), sectors=np.array(sectors))
    print(f"[data] tau={tau}: D={D} events={len(times)} T={int(T)}")
    return times, marks, T, D, labels, sectors


# ========================================================================
# CHUNKED KERNEL HELPERS
#
# Replace every N×N matrix with O(CHUNK×N) work.
# Gradient checkpointing: during training (grad enabled), each chunk's
# intermediate tensors are NOT stored — they are recomputed on backward.
# This cuts backward memory from O(all_chunks) to O(one_chunk).
# ========================================================================
def _chunk_exc_fn(tt, marks, W, beta, s, e):
    """Excitation for target events [s,e) from source events [0,e).
    Pure function — safe for gradient checkpointing."""
    dt = tt[s:e].unsqueeze(1) - tt[:e].unsqueeze(0)        # (cs, e)
    ri = torch.arange(s, e, dtype=torch.long).unsqueeze(1)
    ci = torch.arange(e, dtype=torch.long).unsqueeze(0)
    mask = (ci < ri).float()                                # strict past
    K = torch.exp(-beta * dt.clamp(min=0)) * mask           # (cs, e)
    return (K * W[marks[s:e]][:, marks[:e]]).sum(1)         # (cs,)


BLOCK = 128   # block size of the exact exponential-state recursion


def _state_matrix_raw(tt, marks, D, beta):
    """S[i,d] = Σ_{j<i, m_j=d} exp(-β(t_i − t_j))   (index-causal, ties allowed).
    EXACT block recursion of the exponential kernel: O(N·BLOCK + N·D) time,
    O(N·D) memory, differentiable w.r.t. beta.  All exponents are <= 0."""
    N = tt.shape[0]
    onehot = F.one_hot(marks, D).to(tt.dtype)                 # (N, D)
    carry = tt.new_zeros(D)          # Σ_{j<s} e^{-β(t_s−t_j)} e_{m_j}
    parts = []
    for s in range(0, N, BLOCK):
        e = min(s + BLOCK, N)
        tb = tt[s:e]
        b = e - s
        dt = (tb.unsqueeze(1) - tb.unsqueeze(0)).clamp(min=0)  # (b, b)
        mask = torch.tril(torch.ones(b, b, dtype=tt.dtype), diagonal=-1)
        K = torch.exp(-beta * dt) * mask                       # strict past
        S_blk = K @ onehot[s:e] \
            + torch.exp(-beta * (tb - tb[0])).unsqueeze(1) * carry.unsqueeze(0)
        parts.append(S_blk)
        if e < N:
            carry = torch.exp(-beta * (tt[e] - tb[0])) * carry \
                + torch.exp(-beta * (tt[e] - tb)) @ onehot[s:e]
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def _state_matrix(tt, marks, D, beta):
    """Cached when beta is a constant (no grad) and tt carries a cache dict."""
    beta = torch.as_tensor(beta, dtype=torch.float32)
    cache = getattr(tt, "_ex_cache", None)
    if cache is not None and not beta.requires_grad:
        key = (float(beta), D)
        S = cache.get(key)
        if S is None:
            with torch.no_grad():
                S = _state_matrix_raw(tt, marks, D, beta)
            cache[key] = S
        return S
    return _state_matrix_raw(tt, marks, D, beta)


def _chunked_excitation(tt, marks, W, beta, N):
    """Σ_{j<i} exp(-β(tᵢ−tⱼ)) · W[mᵢ,mⱼ]  for every target i  (exact).
    Computed as  Σ_d W[mᵢ,d]·S[i,d]  — identical value to the N×N form."""
    S = _state_matrix(tt, marks, W.shape[0], beta)
    return (W[marks] * S).sum(1)


def _chunked_excitation_ref(tt, marks, W, beta, N):
    """Reference (slow) chunked N×N implementation, kept for verification."""
    use_ckpt = torch.is_grad_enabled() and N > CHUNK
    parts = []
    for s in range(0, N, CHUNK):
        e = min(s + CHUNK, N)
        if use_ckpt:
            parts.append(ckpt(_chunk_exc_fn, tt, marks, W, beta, s, e,
                              use_reentrant=False))
        else:
            parts.append(_chunk_exc_fn(tt, marks, W, beta, s, e))
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def _chunk_exc_peredge_fn(tt, marks, AP_B, Bm, s, e):
    """Per-edge-beta excitation for one chunk. Pure function."""
    dt = tt[s:e].unsqueeze(1) - tt[:e].unsqueeze(0)
    ri = torch.arange(s, e, dtype=torch.long).unsqueeze(1)
    ci = torch.arange(e, dtype=torch.long).unsqueeze(0)
    mask = (ci < ri).float()
    beta_ev = Bm[marks[s:e]][:, marks[:e]]                 # (cs, e)
    K = torch.exp(-beta_ev * dt.clamp(min=0)) * mask
    return (K * AP_B[marks[s:e]][:, marks[:e]]).sum(1)


def _chunked_excitation_peredge(tt, marks, AP_B, Bm, N):
    """Σ_{j<i} AP_B[mᵢ,mⱼ]·exp(-Bm[mᵢ,mⱼ](tᵢ−tⱼ))  (exact block recursion).
    Carry C[a,d] = Σ_{j<s, mⱼ=d} exp(-Bm[a,d](t_s−tⱼ)) is passed between
    blocks, so cost is O(N·BLOCK + N·D) instead of O(N²)."""
    D = Bm.shape[0]
    onehot = F.one_hot(marks, D).to(tt.dtype)
    C = tt.new_zeros(D, D)
    outs = []
    for s in range(0, N, BLOCK):
        e = min(s + BLOCK, N)
        tb, mb, b = tt[s:e], marks[s:e], e - s
        dt = (tb.unsqueeze(1) - tb.unsqueeze(0)).clamp(min=0)
        mask = torch.tril(torch.ones(b, b, dtype=tt.dtype), diagonal=-1)
        K = torch.exp(-Bm[mb][:, mb] * dt) * mask
        inblk = (K * AP_B[mb][:, mb]).sum(1)
        carry_term = (AP_B[mb] * torch.exp(-Bm[mb] * (tb - tb[0]).unsqueeze(1))
                      * C[mb]).sum(1)
        outs.append(inblk + carry_term)
        if e < N:
            C = torch.exp(-Bm * (tt[e] - tb[0])) * C \
                + torch.exp(-Bm[:, mb] * (tt[e] - tb).unsqueeze(0)) @ onehot[s:e]
    return torch.cat(outs) if len(outs) > 1 else outs[0]


def _chunked_excitation_peredge_ref(tt, marks, AP_B, Bm, N):
    """Reference (slow) per-edge beta excitation with checkpointing."""
    use_ckpt = torch.is_grad_enabled() and N > CHUNK
    parts = []
    for s in range(0, N, CHUNK):
        e = min(s + CHUNK, N)
        if use_ckpt:
            parts.append(ckpt(_chunk_exc_peredge_fn, tt, marks, AP_B, Bm, s, e,
                              use_reentrant=False))
        else:
            parts.append(_chunk_exc_peredge_fn(tt, marks, AP_B, Bm, s, e))
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def _compute_G(tt, marks, D, Tw, beta):
    """G[d] = Σ_{k: mₖ=d} (1 − exp(−β(Tw − tₖ)))   — O(N) memory."""
    surv = 1 - torch.exp(-beta * (Tw - tt))
    return torch.zeros(D).index_add(0, marks, surv)


def _compute_colmass(tt, marks, D, a, b, beta):
    """Column-mass for the compensator on [a,b]:
    colmass[d] = Σ_{k: mₖ=d, tₖ<b} [exp(−β·max(a−tₖ,0)) − exp(−β(b−tₖ))]."""
    relu_a = torch.clamp(a - tt, min=0.0)
    term = torch.where(tt < b,
                       torch.exp(-beta * relu_a) - torch.exp(-beta * (b - tt)),
                       torch.zeros_like(tt))
    return torch.zeros(D).index_add(0, marks, term)


# ========================================================================
# 2. MODELS  (neg_loglik uses chunked helpers — same math)
# ========================================================================
class Emb(nn.Module):
    def __init__(self, D, dd):
        super().__init__()
        self.log_mu = nn.Parameter(torch.log(torch.rand(D) * 0.05 + 1e-3))
        self.src = nn.Parameter(torch.randn(D, dd) * 0.1)
        self.tgt = nn.Parameter(torch.randn(D, dd) * 0.1)
        self.gain = nn.Parameter(torch.tensor(0.0))

    def alpha(self):
        return F.softplus(self.tgt @ self.src.t() + self.gain)

    def mu(self):
        return torch.exp(self.log_mu)


def prep_hawkes(times, marks, D, Tw):
    """Shared lightweight prep for ALL Hawkes models.  Returns tensors only."""
    tt = torch.tensor(times, dtype=torch.float32)
    mm = torch.tensor(marks, dtype=torch.long)
    tt._ex_cache = {}      # constant-beta kernel states are computed once
    return tt, mm, D, float(Tw)


class SingleExpHawkes(nn.Module):
    """R1 -- shared single-exponential kernel."""
    def __init__(self, D, dd, beta):
        super().__init__()
        self.emb = Emb(D, dd)
        self.beta = beta

    def alpha(self): return self.emb.alpha()
    def mu(self): return self.emb.mu()

    def branching_ratio(self):
        with torch.no_grad():
            return float(torch.linalg.eigvals(self.alpha() / self.beta).abs().max())

    def neg_loglik(self, prep):
        tt, marks, D, Tw = prep
        N = len(tt)
        mu, A = self.mu(), self.alpha()
        G = _compute_G(tt, marks, D, Tw, self.beta)
        comp = Tw * mu.sum() + (A * G.unsqueeze(0)).sum() / self.beta
        exc = _chunked_excitation(tt, marks, A, self.beta, N)
        lam = mu[marks] + exc
        return comp - torch.log(lam.clamp(min=1e-12)).sum()


class MixtureHawkes(nn.Module):
    """R2 -- shared mixture-of-exponentials."""
    def __init__(self, D, dd, betas):
        super().__init__()
        self.emb = Emb(D, dd)
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.log_w = nn.Parameter(torch.zeros(len(betas)))

    def weights(self): return F.softplus(self.log_w)
    def alpha(self): return self.emb.alpha()
    def mu(self): return self.emb.mu()

    def branching_ratio(self):
        with torch.no_grad():
            r = torch.linalg.eigvals(self.alpha()).abs().max()
            return float(r * (self.weights() / self.betas).sum())

    def neg_loglik(self, prep):
        tt, marks, D, Tw = prep
        N = len(tt)
        mu, A, w = self.mu(), self.alpha(), self.weights()
        comp = Tw * mu.sum()
        lam = mu[marks].clone()
        for m in range(len(self.betas)):
            bm = self.betas[m]
            G = _compute_G(tt, marks, D, Tw, bm)
            comp = comp + (w[m] / bm) * (A * G.unsqueeze(0)).sum()
            exc = _chunked_excitation(tt, marks, w[m] * A, bm, N)
            lam = lam + exc
        return comp - torch.log(lam.clamp(min=1e-12)).sum()


class ConditionalKernelHawkes(nn.Module):
    def __init__(self, D, betas, dd, hid=16):
        super().__init__()
        self.D, self.dd = D, dd
        self.emb = Emb(D, dd)
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.M = len(betas)
        self.hyper = nn.Sequential(nn.Linear(2 * dd, hid), nn.Tanh(),
                                   nn.Linear(hid, self.M))

    def alpha(self): return self.emb.alpha()
    def mu(self): return self.emb.mu()

    def pi(self):
        t, s = self.emb.tgt, self.emb.src
        T_ = t.unsqueeze(1).expand(self.D, self.D, self.dd)
        S_ = s.unsqueeze(0).expand(self.D, self.D, self.dd)
        return torch.softmax(self.hyper(torch.cat([T_, S_], dim=-1)), dim=-1)

    def branching_ratio(self):
        with torch.no_grad():
            return float(torch.linalg.eigvals(self.alpha()).abs().max())

    def neg_loglik(self, prep):
        tt, marks, D, Tw = prep
        N = len(tt)
        mu, A, pi = self.mu(), self.alpha(), self.pi()
        comp = Tw * mu.sum()
        lam = mu[marks].clone()
        for m in range(self.M):
            bm = self.betas[m]
            Bm = A * pi[:, :, m] * bm                          # (D,D)
            G = _compute_G(tt, marks, D, Tw, bm)
            comp = comp + ((A * pi[:, :, m]) * G.unsqueeze(0)).sum()
            exc = _chunked_excitation(tt, marks, Bm, bm, N)
            lam = lam + exc
        return comp - torch.log(lam.clamp(min=1e-12)).sum()


# ========================================================================
# 2a-ext. LEARNABLE-BETA EXTENSION
# ========================================================================
class LearnableBetaHawkes(nn.Module):
    def __init__(self, D, M, dd, hid=16, ablation="A3",
                 lam_div=0.1, lam_emb=0.01, tau_div=0.5,
                 fixed_betas=None, beta_min=0.1, beta_max=15.0):
        super().__init__()
        self.D, self.dd, self.M = D, dd, M
        self.ablation = ablation
        self.lam_div, self.lam_emb, self.tau_div = lam_div, lam_emb, tau_div
        self.beta_min, self.beta_max = beta_min, beta_max
        self.emb = Emb(D, dd)
        self.hyper = nn.Sequential(nn.Linear(2 * dd, hid), nn.Tanh(),
                                   nn.Linear(hid, M))

        if ablation == "A4" and fixed_betas is not None:
            self.register_buffer("_fixed_betas",
                                 torch.tensor(fixed_betas, dtype=torch.float32))
            self._learnable = False
        else:
            self.rho = nn.Parameter(torch.zeros(M))
            self._learnable = True

    def betas(self):
        if not self._learnable:
            return self._fixed_betas
        ext = torch.cat([self.rho, self.rho.new_zeros(1)])
        c = torch.cumsum(torch.softmax(ext, 0), 0)[:self.M]
        return self.beta_min * (self.beta_max / self.beta_min) ** c

    def alpha(self): return self.emb.alpha()
    def mu(self): return self.emb.mu()

    def pi(self):
        t, s = self.emb.tgt, self.emb.src
        T_ = t.unsqueeze(1).expand(self.D, self.D, self.dd)
        S_ = s.unsqueeze(0).expand(self.D, self.D, self.dd)
        return torch.softmax(self.hyper(torch.cat([T_, S_], dim=-1)), dim=-1)

    def half_lives(self):
        with torch.no_grad():
            return (np.log(2) / self.betas()).numpy()

    def branching_ratio(self):
        with torch.no_grad():
            return float(torch.linalg.eigvals(self.alpha()).abs().max())

    def diversity_loss(self):
        b = self.betas()
        logb = torch.log(b)
        loss = torch.tensor(0.0)
        for i in range(self.M):
            for j in range(i + 1, self.M):
                gap = (logb[j] - logb[i]).abs()
                loss = loss + torch.clamp(self.tau_div - gap, min=0.0) ** 2
        return loss

    def emb_loss(self):
        return 0.5 * (self.emb.src.pow(2).mean() + self.emb.tgt.pow(2).mean())

    def penalty(self):
        n = getattr(self, "_n", 1)
        p = 0.0
        if self.ablation in ("A1", "A3"):
            p = p + self.lam_div * self.diversity_loss()
        if self.ablation in ("A2", "A3"):
            p = p + self.lam_emb * self.emb_loss()
        return n * p

    def neg_loglik(self, prep):
        tt, marks, D, Tw = prep
        N = len(tt)
        mu, A, pi = self.mu(), self.alpha(), self.pi()
        betas = self.betas()
        comp = Tw * mu.sum()
        lam = mu[marks].clone()
        for m in range(self.M):
            bm = betas[m]
            Bm = A * pi[:, :, m] * bm
            G = _compute_G(tt, marks, D, Tw, bm)
            comp = comp + ((A * pi[:, :, m]) * G.unsqueeze(0)).sum()
            exc = _chunked_excitation(tt, marks, Bm, bm, N)
            lam = lam + exc
        return comp - torch.log(lam.clamp(min=1e-12)).sum()


def window_ll_learnable(model, times_all, marks_all, D, a, b):
    with torch.no_grad():
        mu, A, pi = model.mu(), model.alpha(), model.pi()
        betas = model.betas()
        tt = torch.tensor(times_all, dtype=torch.float32)
        mm = torch.tensor(marks_all, dtype=torch.long)
        N = len(tt)
        lam = mu[mm].clone()
        comp = mu.sum() * (b - a)
        for m in range(model.M):
            bm = betas[m]
            Bm = A * pi[:, :, m] * bm
            exc = _chunked_excitation(tt, mm, Bm, bm, N)
            lam = lam + exc
            colmass = _compute_colmass(tt, mm, D, a, b, bm)
            comp = comp + ((A * pi[:, :, m]) * colmass.unsqueeze(0)).sum()
        inw = (tt > a) & (tt <= b)
        n_in = int(inw.sum())
        if n_in == 0:
            return 0.0
        return float((torch.log(lam[inw].clamp(min=1e-12)).sum() - comp).item()
                     / n_in)


# ========================================================================
# 2a-ext2. PER-EDGE NEURAL-BETA
# ========================================================================
class PerEdgeBetaHawkes(nn.Module):
    def __init__(self, D, M, dd, ablation="A3",
                 lam_div=0.1, lam_emb=0.01, tau_div=0.5,
                 beta_min=0.1, beta_max=15.0):
        super().__init__()
        self.D, self.dd, self.M = D, dd, M
        self.ablation = ablation
        self.lam_div, self.lam_emb, self.tau_div = lam_div, lam_emb, tau_div
        self.beta_min, self.beta_max = beta_min, beta_max
        self.emb = Emb(D, dd)
        self.hyper = nn.Sequential(
            nn.Linear(2 * dd, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 2 * M))

    def alpha(self): return self.emb.alpha()
    def mu(self): return self.emb.mu()

    def _hyper_out(self):
        t, s = self.emb.tgt, self.emb.src
        T_ = t.unsqueeze(1).expand(self.D, self.D, self.dd)
        S_ = s.unsqueeze(0).expand(self.D, self.D, self.dd)
        out = self.hyper(torch.cat([T_, S_], dim=-1))
        pi = torch.softmax(out[..., :self.M], dim=-1)
        beta = self.beta_min * (self.beta_max / self.beta_min) ** torch.sigmoid(
            out[..., self.M:])
        return pi, beta

    def mean_betas(self):
        with torch.no_grad():
            _, beta = self._hyper_out()
            return beta.mean(dim=(0, 1)).numpy()

    def half_lives(self):
        return np.log(2) / self.mean_betas()

    def branching_ratio(self):
        with torch.no_grad():
            return float(torch.linalg.eigvals(self.alpha()).abs().max())

    def diversity_loss(self):
        _, beta = self._hyper_out()
        logb = torch.log(beta)
        loss = torch.tensor(0.0)
        for i in range(self.M):
            for j in range(i + 1, self.M):
                gap = (logb[..., i] - logb[..., j]).abs()
                loss = loss + torch.clamp(self.tau_div - gap, min=0.0).pow(2).mean()
        return loss

    def emb_loss(self):
        return 0.5 * (self.emb.src.pow(2).mean() + self.emb.tgt.pow(2).mean())

    def penalty(self):
        n = getattr(self, "_n", 1)
        p = 0.0
        if self.ablation in ("A1", "A3"):
            p = p + self.lam_div * self.diversity_loss()
        if self.ablation in ("A2", "A3"):
            p = p + self.lam_emb * self.emb_loss()
        return n * p

    def neg_loglik(self, prep):
        tt, marks, D, Tw = prep
        N = len(tt)
        mu, A = self.mu(), self.alpha()
        pi, beta = self._hyper_out()
        comp = Tw * mu.sum()
        lam = mu[marks].clone()
        for m in range(self.M):
            AP = A * pi[:, :, m]
            Bm = beta[:, :, m]
            AP_B = AP * Bm                                      # (D,D)
            exc = _chunked_excitation_peredge(tt, marks, AP_B, Bm, N)
            lam = lam + exc
            # compensator: vectorised over target marks d ∈ [0,D)
            bvec_all = Bm[:, marks]                             # (D, N)
            ap_all = AP[:, marks]                               # (D, N)
            surv_all = 1 - torch.exp(-bvec_all * (Tw - tt).unsqueeze(0))
            comp = comp + (ap_all * surv_all).sum()
        return comp - torch.log(lam.clamp(min=1e-12)).sum()


def window_ll_peredge(model, times_all, marks_all, D, a, b):
    with torch.no_grad():
        mu, A = model.mu(), model.alpha()
        pi, beta = model._hyper_out()
        tt = torch.tensor(times_all, dtype=torch.float32)
        mm = torch.tensor(marks_all, dtype=torch.long)
        N = len(tt)
        lam = mu[mm].clone()
        comp = mu.sum() * (b - a)
        relu_a = torch.clamp(a - tt, min=0.0)
        active = (tt < b).float()
        for m in range(model.M):
            AP = A * pi[:, :, m]
            Bm = beta[:, :, m]
            AP_B = AP * Bm
            exc = _chunked_excitation_peredge(tt, mm, AP_B, Bm, N)
            lam = lam + exc
            # compensator vectorised
            bvec_all = Bm[:, mm]                                # (D, N)
            ap_all = AP[:, mm]                                  # (D, N)
            term_all = ((torch.exp(-bvec_all * relu_a.unsqueeze(0)) -
                         torch.exp(-bvec_all * (b - tt).unsqueeze(0))) *
                        active.unsqueeze(0))
            comp = comp + (ap_all * term_all).sum()
        inw = (tt > a) & (tt <= b)
        n_in = int(inw.sum())
        if n_in == 0:
            return 0.0
        return float((torch.log(lam[inw].clamp(min=1e-12)).sum() - comp).item()
                     / n_in)


# ========================================================================
# 2b. NEURAL-TPP BASELINES
# ========================================================================
def tpp_prep(times_all, marks_all):
    N = len(times_all)
    tt = torch.tensor(times_all, dtype=torch.float32)
    mm = torch.tensor(marks_all, dtype=torch.long)
    dur = torch.zeros(N)
    dur[1:] = tt[1:] - tt[:-1]
    return tt, mm, dur, N


def tpp_window_ll(tt, mm, T, rate, a, b):
    """Window log-likelihood — tt, mm are pre-built tensors."""
    tn = torch.cat([tt[1:], torch.tensor([T])])
    lam = rate.sum(1)
    ov = (torch.clamp(tn, max=b) - torch.clamp(tt, min=a)).clamp(min=0)
    comp = (lam * ov).sum()
    inw = (tn > a) & (tn <= b)
    inw[-1] = False
    mn = torch.cat([mm[1:], mm[-1:]])
    logl = torch.log(rate.gather(1, mn.view(-1, 1)).squeeze(1).clamp(min=1e-12))
    n_in = int(inw.sum())
    if n_in == 0:
        return torch.tensor(0.0)
    return (logl[inw].sum() - comp) / n_in


class GRUTPP(nn.Module):
    def __init__(self, D, me=16, hid=48):
        super().__init__()
        self.emb = nn.Embedding(D, me)
        self.gru = nn.GRU(me + 1, hid, batch_first=True)
        self.head = nn.Linear(hid, D)

    def rate(self, mm, dur, tt=None, T=None):
        x = torch.cat([self.emb(mm), dur.view(-1, 1)], 1).unsqueeze(0)
        h, _ = self.gru(x)
        return F.softplus(self.head(h.squeeze(0))) + 1e-6


class NodeTPP(nn.Module):
    def __init__(self, D, me=16, hid=48):
        super().__init__()
        self.emb = nn.Embedding(D, me)
        self.gru = nn.GRU(me + 1, hid, batch_first=True)
        self.dec = nn.Parameter(torch.tensor(0.3))
        self.head = nn.Linear(hid, D)

    def rate(self, mm, dur, tt=None, T=None):
        x = torch.cat([self.emb(mm), dur.view(-1, 1)], 1).unsqueeze(0)
        h, _ = self.gru(x)
        h = h.squeeze(0)
        g = F.softplus(self.dec) + 1e-2
        hd = h * torch.exp(-g * dur).unsqueeze(1)
        return F.softplus(self.head(hd)) + 1e-6


def _attn_chunk_fn(Q, K_full, V_full, scale, s, e):
    """One chunk of causal attention. Pure function for checkpointing."""
    att = Q[s:e] @ K_full[:e].t() / scale                    # (cs, e)
    ri = torch.arange(s, e, dtype=torch.long).unsqueeze(1)
    ci = torch.arange(e, dtype=torch.long).unsqueeze(0)
    cmask = (ci <= ri).float()                                # causal-inclusive
    att = att.masked_fill(cmask == 0, -1e9)
    return torch.softmax(att, -1) @ V_full[:e]                # (cs, dm)


class AttnTPP(nn.Module):
    """Transformer-Hawkes-style baseline.  Attention computed in chunks
    with gradient checkpointing."""
    def __init__(self, D, dm=32):
        super().__init__()
        self.emb = nn.Embedding(D, dm)
        self.dm = dm
        self.q = nn.Linear(dm, dm)
        self.k = nn.Linear(dm, dm)
        self.v = nn.Linear(dm, dm)
        self.ff = nn.Sequential(nn.Linear(dm, dm), nn.ReLU(), nn.Linear(dm, dm))
        self.head = nn.Linear(dm, D)

    def rate(self, mm, dur=None, tt=None, T=None):
        N = mm.shape[0]
        pos = (tt / T).unsqueeze(1)
        dim = torch.arange(self.dm).float()
        ang = pos / (10000 ** (2 * (dim // 2) / self.dm))
        pe = torch.zeros(N, self.dm)
        pe[:, 0::2] = torch.sin(ang[:, 0::2])
        pe[:, 1::2] = torch.cos(ang[:, 1::2])
        x = self.emb(mm) + pe
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scale = math.sqrt(self.dm)
        # --- chunked causal attention with gradient checkpointing ---
        use_ckpt = torch.is_grad_enabled() and N > CHUNK
        parts = []
        for s in range(0, N, CHUNK):
            e = min(s + CHUNK, N)
            if use_ckpt:
                parts.append(ckpt(_attn_chunk_fn, Q, K, V, scale, s, e,
                                  use_reentrant=False))
            else:
                parts.append(_attn_chunk_fn(Q, K, V, scale, s, e))
        h = torch.cat(parts) if len(parts) > 1 else parts[0]
        h = self.ff(h) + h
        return F.softplus(self.head(h)) + 1e-6


class CNTPP(nn.Module):
    def __init__(self, D, me=16, hid=48, K=4):
        super().__init__()
        self.emb = nn.Embedding(D, me)
        self.gru = nn.GRU(me + 1, hid, batch_first=True)
        self.gate = nn.Linear(hid, K)
        self.heads = nn.ModuleList([nn.Linear(hid, D) for _ in range(K)])
        self.K = K

    def rate(self, mm, dur, tt=None, T=None):
        x = torch.cat([self.emb(mm), dur.view(-1, 1)], 1).unsqueeze(0)
        h, _ = self.gru(x); h = h.squeeze(0)
        g = torch.softmax(self.gate(h), -1)
        r = sum(g[:, k:k + 1] * F.softplus(self.heads[k](h))
                for k in range(self.K))
        return r + 1e-6


class HyperHawkesTPP(nn.Module):
    def __init__(self, D, me=16, hid=48, lat=32):
        super().__init__()
        self.emb = nn.Embedding(D, me)
        self.gru = nn.GRU(me + 1, hid, batch_first=True)
        self.expand = nn.Linear(hid, lat)
        self.hyper = nn.Sequential(nn.Linear(lat, 64), nn.Tanh(),
                                   nn.Linear(64, D))

    def rate(self, mm, dur, tt=None, T=None):
        x = torch.cat([self.emb(mm), dur.view(-1, 1)], 1).unsqueeze(0)
        h, _ = self.gru(x); h = h.squeeze(0)
        z = torch.tanh(self.expand(h))
        return F.softplus(self.hyper(z)) + 1e-6


def train_tpp(model, times_all, marks_all, T, t_cut, epochs, lr=0.01):
    """Train a neural TPP with pre-allocated tensors and NaN protection."""
    _check_time()
    tt, mm, dur, N = tpp_prep(times_all, marks_all)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        rate = model.rate(mm, dur, tt, T)
        loss = -tpp_window_ll(tt, mm, T, rate, 0.0, t_cut)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    [warn] NaN/Inf loss at epoch {ep}, stopping early")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if ep % 100 == 99:
            gc.collect()
    with torch.no_grad():
        rate = model.rate(mm, dur, tt, T)
        return float(tpp_window_ll(tt, mm, T, rate, t_cut, T))


# ========================================================================
# 2b-classical. CLASSICAL HAWKES BASELINES
# ========================================================================
class DenseHawkes(nn.Module):
    def __init__(self, D, beta=0.35, lam_l1=0.0):
        super().__init__()
        self.log_mu = nn.Parameter(torch.log(torch.rand(D) * 0.05 + 1e-3))
        self.log_a = nn.Parameter(torch.randn(D, D) * 0.1 - 4.0)
        self.beta = beta
        self.lam_l1 = lam_l1

    def mu(self): return torch.exp(self.log_mu)
    def alpha(self): return F.softplus(self.log_a)
    def penalty(self):
        return (self.lam_l1 * self.alpha().abs().sum() if self.lam_l1 > 0
                else 0.0 * self.log_mu.sum())

    def branching_ratio(self):
        with torch.no_grad():
            return float(torch.linalg.eigvals(self.alpha() / self.beta).abs().max())

    def neg_loglik(self, prep):
        tt, marks, D, Tw = prep
        N = len(tt)
        mu, A = self.mu(), self.alpha()
        G = _compute_G(tt, marks, D, Tw, self.beta)
        comp = Tw * mu.sum() + (A * G.unsqueeze(0)).sum() / self.beta
        exc = _chunked_excitation(tt, marks, A, self.beta, N)
        lam = mu[marks] + exc
        return comp - torch.log(lam.clamp(min=1e-12)).sum()


class DiagonalHawkes(nn.Module):
    def __init__(self, D, beta=0.35):
        super().__init__()
        self.log_mu = nn.Parameter(torch.log(torch.rand(D) * 0.05 + 1e-3))
        self.log_d = nn.Parameter(torch.randn(D) * 0.1 - 4.0)
        self.beta = beta

    def mu(self): return torch.exp(self.log_mu)
    def alpha(self): return torch.diag(F.softplus(self.log_d))

    def branching_ratio(self):
        with torch.no_grad():
            return float((F.softplus(self.log_d) / self.beta).max())

    def neg_loglik(self, prep):
        tt, marks, D, Tw = prep
        N = len(tt)
        mu, A = self.mu(), self.alpha()
        G = _compute_G(tt, marks, D, Tw, self.beta)
        comp = Tw * mu.sum() + (A * G.unsqueeze(0)).sum() / self.beta
        exc = _chunked_excitation(tt, marks, A, self.beta, N)
        lam = mu[marks] + exc
        return comp - torch.log(lam.clamp(min=1e-12)).sum()


def cojump_correlation(times, marks, D, n_days):
    counts = np.zeros((int(n_days) + 1, D))
    for t, m in zip(times, marks):
        counts[int(t), int(m)] += 1.0
    C = np.corrcoef(counts.T)
    C = np.nan_to_num(C)
    np.fill_diagonal(C, 0.0)
    return np.clip(C, 0.0, None)


# ========================================================================
# 2c. POISSON NULL BASELINES
# ========================================================================
def fit_poisson_rates(times, marks, D, t_cut, per_mark=True):
    tr = times < t_cut
    if per_mark:
        counts = np.bincount(marks[tr].astype(int), minlength=D).astype(float)
        mu = counts / max(float(t_cut), 1e-9)
    else:
        mu = np.full(D, int(tr.sum()) / (D * max(float(t_cut), 1e-9)))
    return mu


def window_ll_poisson(mu, times_all, marks_all, a, b):
    tt = np.asarray(times_all, dtype=float)
    mm = np.asarray(marks_all).astype(int)
    inw = (tt > a) & (tt <= b)
    n = int(inw.sum())
    if n == 0:
        return 0.0
    loglam = np.log(np.clip(mu[mm[inw]], 1e-12, None)).sum()
    comp = (b - a) * float(mu.sum())
    return float((loglam - comp) / max(n, 1))


# ========================================================================
# 3. TRAIN / EVAL  (chunked held-out log-likelihood functions)
# ========================================================================
def train(model, prep, epochs, lr, wd=0.0):
    """Train a Hawkes model with NaN protection and periodic GC."""
    _check_time()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for ep in range(epochs):
        opt.zero_grad()
        loss = model.neg_loglik(prep)
        if hasattr(model, "penalty"):
            loss = loss + model.penalty()
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    [warn] NaN/Inf loss at epoch {ep}, stopping early")
            break
        loss.backward()
        opt.step()
        # periodic GC to prevent memory creep from autograd graph fragments
        if ep % 100 == 99:
            gc.collect()
    return model


def window_ll_single(mu, A, beta, times_all, marks_all, D, a, b):
    with torch.no_grad():
        tt = torch.tensor(times_all, dtype=torch.float32)
        mm = torch.tensor(marks_all, dtype=torch.long)
        N = len(tt)
        exc = _chunked_excitation(tt, mm, A, beta, N)
        lam = mu[mm] + exc
        colmass = _compute_colmass(tt, mm, D, a, b, beta)
        comp = mu.sum() * (b - a) + (A * colmass.unsqueeze(0)).sum() / beta
        inw = (tt > a) & (tt <= b)
        n_in = int(inw.sum())
        if n_in == 0:
            return 0.0
        return float((torch.log(lam[inw].clamp(min=1e-12)).sum() - comp).item()
                     / n_in)


def window_ll_mixture(mu, A, w, betas, times_all, marks_all, D, a, b):
    with torch.no_grad():
        tt = torch.tensor(times_all, dtype=torch.float32)
        mm = torch.tensor(marks_all, dtype=torch.long)
        N = len(tt)
        lam = mu[mm].clone()
        comp = mu.sum() * (b - a)
        for m, bm in enumerate(betas):
            exc = _chunked_excitation(tt, mm, w[m] * A, bm, N)
            lam = lam + exc
            colmass = _compute_colmass(tt, mm, D, a, b, bm)
            comp = comp + (w[m] / bm) * (A * colmass.unsqueeze(0)).sum()
        inw = (tt > a) & (tt <= b)
        n_in = int(inw.sum())
        if n_in == 0:
            return 0.0
        return float((torch.log(lam[inw].clamp(min=1e-12)).sum() - comp).item()
                     / n_in)


def window_ll_conditional(model, times_all, marks_all, D, a, b):
    with torch.no_grad():
        mu, A, pi = model.mu(), model.alpha(), model.pi()
        tt = torch.tensor(times_all, dtype=torch.float32)
        mm = torch.tensor(marks_all, dtype=torch.long)
        N = len(tt)
        lam = mu[mm].clone()
        comp = mu.sum() * (b - a)
        for m in range(model.M):
            bm = model.betas[m]
            Bm = A * pi[:, :, m] * bm
            exc = _chunked_excitation(tt, mm, Bm, bm, N)
            lam = lam + exc
            colmass = _compute_colmass(tt, mm, D, a, b, bm)
            comp = comp + ((A * pi[:, :, m]) * colmass.unsqueeze(0)).sum()
        inw = (tt > a) & (tt <= b)
        n_in = int(inw.sum())
        if n_in == 0:
            return 0.0
        return float((torch.log(lam[inw].clamp(min=1e-12)).sum() - comp).item()
                     / n_in)


# ========================================================================
# 4. SWEEP RUNNER (checkpointed / resumable)
# ========================================================================
class OutOfTime(BaseException):
    """Raised (outside `except Exception`) when --max-hours is exhausted."""


DEADLINE = None


def _check_time():
    if DEADLINE is not None and time.time() > DEADLINE:
        raise OutOfTime()


def set_dirs(data_dir=None, results_dir=None):
    """Re-point data/results folders (e.g. to Google Drive) before running."""
    global DATA_DIR, RESULTS_DIR, PRICES_CACHE, LOG_PATH
    if data_dir:
        DATA_DIR = Path(data_dir)
        PRICES_CACHE = DATA_DIR / "crypto_prices.npz"
    if results_dir:
        RESULTS_DIR = Path(results_dir)
        LOG_PATH = RESULTS_DIR / "ablation_log.json"


def load_log():
    bak = LOG_PATH.with_suffix(".json.bak")
    for p in (LOG_PATH, bak):
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                print(f"[warn] {p.name} unreadable, trying backup")
    return []


def save_log(log):
    """Atomic write: a crash/disconnect mid-save can never corrupt the log."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LOG_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(log, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    if LOG_PATH.exists():
        try:
            import shutil
            shutil.copyfile(LOG_PATH, LOG_PATH.with_suffix(".json.bak"))
        except OSError:
            pass
    os.replace(tmp, LOG_PATH)


def run_key(tag, tau, dim, beta_name, seed):
    return f"{tag}|tau={tau}|dim={dim}|beta={beta_name}|seed={seed}"


def _free():
    """Force garbage collection between model trainings."""
    gc.collect()


def _mem_mb():
    """Rough RSS in MB (Linux / Colab)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1


def run_comparison_baselines(cfg, log, done, tau):
    dim = cfg["CANON_DIM"]; beta_name = cfg["CANON_BETA"]
    betas = cfg["BETA_SETS"].get(beta_name,
                                 tuple(cfg["BETA_SETS"].values())[0])
    times, marks, T, D, labels, sectors = get_events(tau, cfg["VOL_WIN"])
    t_cut = cfg["SPLIT"] * T
    tr = times < t_cut
    print(f"\n=== comparison baselines @ canonical (tau={tau}, dim={dim}, "
          f"beta={beta_name})"
          f" | D={D} events={len(times)} train={int(tr.sum())} "
          f"test={len(times)-int(tr.sum())} | RSS={_mem_mb():.0f}MB ===")

    def _log(rec):
        log.append(rec); save_log(log); done.add(rec["key"])
        print("  ", rec)

    # -- Poisson null baselines (closed-form) --
    for name, per_mark in [("P0-homogeneous-poisson", False),
                           ("P1-permark-poisson", True)]:
        key = run_key(name, tau, dim, beta_name, 0)
        if key in done:
            continue
        try:
            mu_p = fit_poisson_rates(times, marks, D, t_cut,
                                     per_mark=per_mark)
            ll = window_ll_poisson(mu_p, times, marks, t_cut, T)
            _log(dict(key=key, model=name, tau=tau, dim=dim,
                      beta_set=beta_name, seed=0,
                      held_out_ll=round(ll, 4), branching=0.0,
                      n_events=len(times)))
        except Exception as exc:
            print(f"  [ERROR] {name}: {exc}")
            traceback.print_exc()

    # -- R1 single-exponential Hawkes --
    key = run_key("R1-single-exp", tau, dim, beta_name, 0)
    if key not in done:
        try:
            torch.manual_seed(0); beta_ref = 0.35
            prep = prep_hawkes(times[tr], marks[tr], D, t_cut)
            m = train(SingleExpHawkes(D, dim, beta_ref), prep,
                      cfg["EPOCHS"], cfg["LR"])
            with torch.no_grad():
                ll = window_ll_single(m.mu(), m.alpha(), beta_ref,
                                      times, marks, D, t_cut, T)
            _log(dict(key=key, model="R1-single-exp", tau=tau, dim=dim,
                      beta_set=beta_name, seed=0,
                      held_out_ll=round(ll, 4),
                      branching=round(m.branching_ratio(), 3),
                      n_events=len(times)))
            del m, prep; _free()
        except Exception as exc:
            print(f"  [ERROR] R1-single-exp: {exc}")
            traceback.print_exc(); _free()

    # -- R2 shared mixture-of-exponentials Hawkes --
    key = run_key("R2-shared-mixture", tau, dim, beta_name, 0)
    if key not in done:
        try:
            torch.manual_seed(0)
            prep = prep_hawkes(times[tr], marks[tr], D, t_cut)
            m = train(MixtureHawkes(D, dim, betas), prep,
                      cfg["EPOCHS"], cfg["LR"])
            with torch.no_grad():
                ll = window_ll_mixture(m.mu(), m.alpha(), m.weights(),
                                       betas, times, marks, D, t_cut, T)
            _log(dict(key=key, model="R2-shared-mixture", tau=tau, dim=dim,
                      beta_set=beta_name, seed=0,
                      held_out_ll=round(ll, 4),
                      branching=round(m.branching_ratio(), 3),
                      n_events=len(times)))
            del m, prep; _free()
        except Exception as exc:
            print(f"  [ERROR] R2-shared-mixture: {exc}")
            traceback.print_exc(); _free()

    # -- classical Hawkes ladder --
    beta_ref = 0.35
    c_ep = cfg.get("CLASSIC_EPOCHS", 500)
    c_lr = cfg.get("CLASSIC_LR", 0.05)

    # Dense Hawkes
    key = run_key("Dense-Hawkes", tau, dim, beta_name, 0)
    if key not in done:
        try:
            torch.manual_seed(0)
            prep = prep_hawkes(times[tr], marks[tr], D, t_cut)
            m = train(DenseHawkes(D, beta_ref, lam_l1=0.0), prep, c_ep, c_lr,
                      wd=1e-4)
            with torch.no_grad():
                ll = window_ll_single(m.mu(), m.alpha(), beta_ref,
                                      times, marks, D, t_cut, T)
            _log(dict(key=key, model="Dense-Hawkes", tau=tau, dim=dim,
                      beta_set=beta_name, seed=0,
                      held_out_ll=round(ll, 4),
                      branching=round(m.branching_ratio(), 3),
                      n_events=len(times)))
            del m, prep; _free()
        except Exception as exc:
            print(f"  [ERROR] Dense-Hawkes: {exc}")
            traceback.print_exc(); _free()

    # L1-Sparse Hawkes
    key = run_key("L1-Sparse-Hawkes", tau, dim, beta_name, 0)
    if key not in done:
        try:
            torch.manual_seed(0)
            prep = prep_hawkes(times[tr], marks[tr], D, t_cut)
            m = train(DenseHawkes(D, beta_ref, lam_l1=0.02), prep, c_ep, c_lr)
            with torch.no_grad():
                ll = window_ll_single(m.mu(), m.alpha(), beta_ref,
                                      times, marks, D, t_cut, T)
            _log(dict(key=key, model="L1-Sparse-Hawkes", tau=tau, dim=dim,
                      beta_set=beta_name, seed=0,
                      held_out_ll=round(ll, 4),
                      branching=round(m.branching_ratio(), 3),
                      n_events=len(times)))
            del m, prep; _free()
        except Exception as exc:
            print(f"  [ERROR] L1-Sparse-Hawkes: {exc}")
            traceback.print_exc(); _free()

    # Diagonal Hawkes
    key = run_key("Diagonal-Hawkes", tau, dim, beta_name, 0)
    if key not in done:
        try:
            torch.manual_seed(0)
            prep = prep_hawkes(times[tr], marks[tr], D, t_cut)
            m = train(DiagonalHawkes(D, beta_ref), prep, c_ep, c_lr)
            with torch.no_grad():
                ll = window_ll_single(m.mu(), m.alpha(), beta_ref,
                                      times, marks, D, t_cut, T)
            _log(dict(key=key, model="Diagonal-Hawkes", tau=tau, dim=dim,
                      beta_set=beta_name, seed=0,
                      held_out_ll=round(ll, 4),
                      branching=round(m.branching_ratio(), 3),
                      n_events=len(times)))
            del m, prep; _free()
        except Exception as exc:
            print(f"  [ERROR] Diagonal-Hawkes: {exc}")
            traceback.print_exc(); _free()

    # Correlation network
    key = run_key("Correlation-network", tau, dim, beta_name, 0)
    if key not in done:
        try:
            mu_p = torch.tensor(
                fit_poisson_rates(times, marks, D, t_cut, per_mark=True),
                dtype=torch.float32)
            C = torch.tensor(
                cojump_correlation(times[tr], marks[tr], D, t_cut),
                dtype=torch.float32)
            best_c, best = 0.0, -1e18
            for c in np.linspace(0, 2, 41):
                ll_tr = window_ll_single(mu_p, c * C, beta_ref,
                                         times, marks, D, 0.0, t_cut)
                if ll_tr > best:
                    best, best_c = ll_tr, c
            ll = window_ll_single(mu_p, best_c * C, beta_ref,
                                  times, marks, D, t_cut, T)
            br = float(
                torch.linalg.eigvals(best_c * C / beta_ref).abs().max())
            _log(dict(key=key, model="Correlation-network", tau=tau, dim=dim,
                      beta_set=beta_name, seed=0,
                      held_out_ll=round(ll, 4),
                      branching=round(br, 3),
                      n_events=len(times)))
            del mu_p, C; _free()
        except Exception as exc:
            print(f"  [ERROR] Correlation-network: {exc}")
            traceback.print_exc(); _free()

    # Neural-kernel Hawkes
    key = run_key("Neural-kernel-Hawkes", tau, dim, beta_name, 0)
    if key not in done:
        try:
            fine = tuple(float(x) for x in np.geomspace(0.1, 15.0, 10))
            torch.manual_seed(0)
            prep = prep_hawkes(times[tr], marks[tr], D, t_cut)
            m = train(MixtureHawkes(D, dim, fine), prep,
                      cfg["EPOCHS"], cfg["LR"])
            with torch.no_grad():
                ll = window_ll_mixture(m.mu(), m.alpha(), m.weights(),
                                       fine, times, marks, D, t_cut, T)
            _log(dict(key=key, model="Neural-kernel-Hawkes", tau=tau, dim=dim,
                      beta_set=beta_name, seed=0,
                      held_out_ll=round(ll, 4),
                      branching=round(m.branching_ratio(), 3),
                      n_events=len(times)))
            del m, prep; _free()
        except Exception as exc:
            print(f"  [ERROR] Neural-kernel-Hawkes: {exc}")
            traceback.print_exc(); _free()

    # -- neural-TPP baselines --
    if cfg.get("RUN_LIT_BASELINES", True):
        neural = [("RMTPP", GRUTPP), ("Transformer-Hawkes", AttnTPP),
                  ("Neural-ODE-TPP", NodeTPP),
                  ("C-NTPP", CNTPP), ("Hyper-Hawkes", HyperHawkesTPP)]
        for name, ModelCls in neural:
            key = run_key(name, tau, dim, beta_name, 0)
            if key in done:
                continue
            try:
                torch.manual_seed(0)
                model = ModelCls(D)
                ll = train_tpp(model, times, marks, T, t_cut,
                               cfg["LIT_EPOCHS"])
                _log(dict(key=key, model=name, tau=tau, dim=dim,
                          beta_set=beta_name, seed=0,
                          held_out_ll=round(ll, 4), branching=None,
                          n_events=len(times)))
                del model; _free()
            except Exception as exc:
                print(f"  [ERROR] {name}: {exc}")
                traceback.print_exc(); _free()


def run_sweep(cfg):
    log = load_log()
    done = {rec["key"] for rec in log}
    t0 = time.time()

    for tau in cfg["TAUS"]:
        run_comparison_baselines(cfg, log, done, tau)
        done = {rec["key"] for rec in log}

    for tau in cfg["TAUS"]:
        times, marks, T, D, labels, sectors = get_events(tau, cfg["VOL_WIN"])
        t_cut = cfg["SPLIT"] * T
        tr = times < t_cut
        n_grid = (len(cfg["DIMS"]) * len(cfg["BETA_SETS"])
                  * len(cfg["SEEDS"]))
        print(f"\n=== [proposed grid] tau={tau}  D={D}  events={len(times)}  "
              f"train={int(tr.sum())} test={len(times)-int(tr.sum())}  "
              f"({n_grid} R3 runs) | RSS={_mem_mb():.0f}MB ===")
        for dim in cfg["DIMS"]:
            for beta_name, betas in cfg["BETA_SETS"].items():
                for seed in cfg["SEEDS"]:
                    key = run_key("R3-conditional", tau, dim,
                                  beta_name, seed)
                    if key in done:
                        continue
                    try:
                        torch.manual_seed(seed)
                        prep = prep_hawkes(times[tr], marks[tr], D, t_cut)
                        m = train(
                            ConditionalKernelHawkes(D, betas, dim,
                                                   cfg["HID"]),
                            prep, cfg["EPOCHS"], cfg["LR"])
                        ll = window_ll_conditional(m, times, marks, D,
                                                   t_cut, T)
                        rec = dict(
                            key=key, model="R3-conditional", tau=tau,
                            dim=dim, beta_set=beta_name, seed=seed,
                            held_out_ll=round(ll, 4),
                            branching=round(m.branching_ratio(), 3),
                            n_events=len(times))
                        log.append(rec); save_log(log)
                        done.add(key)
                        print(f"  [{time.time()-t0:6.0f}s] {rec}")
                        del m, prep; _free()
                    except Exception as exc:
                        print(f"  [ERROR] R3 tau={tau} dim={dim} "
                              f"beta={beta_name} s={seed}: {exc}")
                        traceback.print_exc(); _free()

    return log


# ========================================================================
# 4b. LEARNABLE-BETA SWEEP
# ========================================================================
LEARNABLE_BETA_CONFIG = dict(
    TAU=2.5,
    DIMS=[8],
    MS=[2, 4, 6, 8],
    ABLATIONS=["A4", "A0", "A1", "A2", "A3", "PE"],
    SEEDS=[0, 1, 2],
    EPOCHS=500,
    LR=0.03,
    HID=16,
    LAM_DIV=0.1,
    LAM_EMB=0.01,
    TAU_DIV=0.5,
    BETA_MIN=0.1,
    BETA_MAX=15.0,
    FIXED_BETAS=(0.35, 1.0, 3.0, 8.0),
)


def run_learnable_sweep(lb_cfg=None, base_cfg=None):
    if lb_cfg is None:
        lb_cfg = LEARNABLE_BETA_CONFIG
    if base_cfg is None:
        base_cfg = DEFAULT_CONFIG
    log = load_log()
    done = {rec["key"] for rec in log}
    t0 = time.time()

    tau = lb_cfg["TAU"]
    times, marks, T, D, labels, sectors = get_events(tau, base_cfg["VOL_WIN"])
    t_cut = base_cfg["SPLIT"] * T
    tr = times < t_cut

    n_runs = (len(lb_cfg["DIMS"]) * len(lb_cfg["MS"])
              * len(lb_cfg["ABLATIONS"]) * len(lb_cfg["SEEDS"]))
    print(f"\n=== LEARNABLE-BETA SWEEP @ tau={tau}  D={D}  "
          f"events={len(times)}  "
          f"train={int(tr.sum())} test={len(times)-int(tr.sum())}  "
          f"({n_runs} runs) | RSS={_mem_mb():.0f}MB ===")

    prep = prep_hawkes(times[tr], marks[tr], D, t_cut)

    for dim in lb_cfg["DIMS"]:
        for M in lb_cfg["MS"]:
            for abl in lb_cfg["ABLATIONS"]:
                for seed in lb_cfg["SEEDS"]:
                    tag = f"LB-{abl}-M{M}"
                    key = run_key(tag, tau, dim, f"M={M}", seed)
                    if key in done:
                        continue
                    try:
                        torch.manual_seed(seed)
                        n_train = int(tr.sum())
                        if abl == "A4":
                            fb = lb_cfg["FIXED_BETAS"]
                            if M <= len(fb):
                                fb_use = fb[:M]
                            else:
                                fb_use = tuple(fb) + tuple(
                                    np.geomspace(fb[-1], fb[-1] * 2,
                                                 M - len(fb) + 1)[1:])
                            model = LearnableBetaHawkes(
                                D, M, dim, lb_cfg["HID"], ablation="A4",
                                fixed_betas=fb_use)
                            model._n = n_train
                            m = train(model, prep, lb_cfg["EPOCHS"],
                                      lb_cfg["LR"])
                            ll = window_ll_learnable(m, times, marks, D,
                                                     t_cut, T)
                            betas_arr = m.betas().detach().numpy()
                        elif abl == "PE":
                            model = PerEdgeBetaHawkes(
                                D, M, dim, ablation="A3",
                                lam_div=lb_cfg["LAM_DIV"],
                                lam_emb=lb_cfg["LAM_EMB"],
                                tau_div=lb_cfg["TAU_DIV"],
                                beta_min=lb_cfg["BETA_MIN"],
                                beta_max=lb_cfg["BETA_MAX"])
                            model._n = n_train
                            m = train(model, prep, lb_cfg["EPOCHS"],
                                      lb_cfg["LR"])
                            ll = window_ll_peredge(m, times, marks, D,
                                                   t_cut, T)
                            betas_arr = m.mean_betas()
                        else:
                            model = LearnableBetaHawkes(
                                D, M, dim, lb_cfg["HID"], ablation=abl,
                                lam_div=lb_cfg["LAM_DIV"],
                                lam_emb=lb_cfg["LAM_EMB"],
                                tau_div=lb_cfg["TAU_DIV"],
                                beta_min=lb_cfg["BETA_MIN"],
                                beta_max=lb_cfg["BETA_MAX"])
                            model._n = n_train
                            m = train(model, prep, lb_cfg["EPOCHS"],
                                      lb_cfg["LR"])
                            ll = window_ll_learnable(m, times, marks, D,
                                                     t_cut, T)
                            betas_arr = m.betas().detach().numpy()

                        hlives_arr = np.log(2) / betas_arr
                        betas_str = ",".join(f"{b:.3f}" for b in betas_arr)
                        hlives_str = ",".join(f"{h:.1f}" for h in hlives_arr)
                        note = ("per-edge betas: mean over edges"
                                if abl == "PE" else "")
                        rec = dict(
                            key=key, model=tag, tau=tau, dim=dim,
                            beta_set=f"M={M}", seed=seed, ablation=abl,
                            M=M, held_out_ll=round(ll, 4),
                            branching=round(m.branching_ratio(), 3),
                            learned_betas=betas_str,
                            half_lives=hlives_str,
                            note=note,
                            n_events=len(times))
                        log.append(rec); save_log(log); done.add(key)
                        print(f"  [{time.time()-t0:5.0f}s] {abl} M={M} "
                              f"s={seed}  LL={ll:.4f}  "
                              f"betas=[{betas_str}]  t½=[{hlives_str}]")
                        del m, model; _free()
                    except Exception as exc:
                        print(f"  [ERROR] {tag} s={seed}: {exc}")
                        traceback.print_exc(); _free()

    # ---- compile learnable-beta results ----
    log = load_log()   # reload in case something changed
    df = pd.DataFrame([r for r in log if r.get("model", "").startswith("LB-")])
    if df.empty:
        print("No learnable-beta results yet.")
        return log

    agg = (df.groupby(["ablation", "M"])["held_out_ll"]
             .agg(["mean", "std", "count"]).reset_index()
             .sort_values("mean", ascending=False))
    agg.to_csv(RESULTS_DIR / "learnable_beta_ablation.csv", index=False)
    print("\n=== LEARNABLE-BETA ABLATION (mean over seeds) ===")
    print(agg.to_string(index=False))

    best_idx = df.groupby(["ablation", "M"])["held_out_ll"].idxmax()
    best = df.loc[best_idx][["ablation", "M", "held_out_ll",
                              "learned_betas", "half_lives"]]
    best.to_csv(RESULTS_DIR / "learned_timescales.csv", index=False)
    print("\n=== BEST LEARNED TIMESCALES ===")
    print(best.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"A0": "#DD8452", "A1": "#4C72B0", "A2": "#55A868",
              "A3": "#C44E52", "A4": "#999999"}
    ax = axes[0]
    for abl in sorted(agg.ablation.unique()):
        sub = agg[agg.ablation == abl].sort_values("M")
        ax.errorbar(sub.M, sub["mean"], yerr=sub["std"].fillna(0), marker="o",
                    label=abl, color=colors.get(abl, "k"), capsize=3, lw=1.5)
    ax.set_xlabel("M (number of timescale components)")
    ax.set_ylabel("held-out LL/event"); ax.legend()
    ax.set_title("Learnable-beta ablation: LL vs bank size M",
                 fontweight="bold")

    ax = axes[1]
    a3 = best[best.ablation == "A3"]
    if not a3.empty:
        for _, row in a3.iterrows():
            hls = [float(x) for x in row["half_lives"].split(",")]
            ax.barh([f"M={int(row['M'])} β{i+1}" for i in range(len(hls))],
                    hls, height=0.6, color="#C44E52", alpha=0.7)
        ax.set_xlabel("half-life (days)")
        ax.set_title("Learned half-lives (A3, best seed)", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "no A3 runs", transform=ax.transAxes, ha="center")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "learnable_beta_results.png", dpi=130)
    plt.close(fig)
    print("\nSaved learnable_beta_ablation.csv, learned_timescales.csv, "
          "learnable_beta_results.png")

    return log


# ========================================================================
# 5. COMPILE RESULTS
# ========================================================================
def compile_results(cfg=None):
    if cfg is None:
        cfg = DEFAULT_CONFIG
    log = load_log()
    if not log:
        print("No results in results/ablation_log.json yet — "
              "run the sweep first.")
        return
    df = pd.DataFrame(log)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)

    r3 = df[df.model == "R3-conditional"]
    if r3.empty:
        print(f"Saved {len(df)} rows to results/ablation_results.csv "
              f"(no R3 runs yet).")
        return

    agg = (r3.groupby(["tau", "dim", "beta_set"])["held_out_ll"]
              .agg(["mean", "std", "count"]).reset_index()
              .sort_values("mean", ascending=False))
    agg.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)
    print("\n=== TOP configs, proposed model (held-out LL/event, "
          "avg over seeds) ===")
    print(agg.head(10).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_beta = agg.iloc[0]["beta_set"]

    piv = agg[agg.beta_set == best_beta].pivot(
        index="dim", columns="tau", values="mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("tau (event threshold, std devs)")
    ax.set_ylabel("embedding dim")
    ax.set_title(f"Held-out LL/event  (beta-bank = {best_beta})",
                 fontweight="bold")
    fig.colorbar(im, ax=ax, label="LL/event")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "heatmap_tau_dim.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for beta_name in sorted(agg.beta_set.unique()):
        for tau in sorted(agg.tau.unique()):
            sub = agg[(agg.beta_set == beta_name) &
                      (agg.tau == tau)].sort_values("dim")
            if len(sub):
                ax.errorbar(sub.dim, sub["mean"],
                            yerr=sub["std"].fillna(0.0), marker="o",
                            label=f"{beta_name}  (tau={tau})",
                            capsize=3, lw=1.4)
    ax.set_xlabel("embedding dim")
    ax.set_ylabel("held-out LL/event")
    ax.set_title("Ablation: beta-bank x dim x tau", fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "beta_dim_ablation.png", dpi=130)
    plt.close(fig)

    canon_dim = cfg["CANON_DIM"]
    canon_beta = cfg["CANON_BETA"]
    disp = {
        "R3-conditional": "Proposed conditional kernel",
        "R2-shared-mixture": "Shared mixture (R2)",
        "R1-single-exp": "Embedding single-exp (R1)",
        "P0-homogeneous-poisson": "Homogeneous Poisson",
        "P1-permark-poisson": "Per-mark Poisson",
        "Dense-Hawkes": "Dense Hawkes",
        "L1-Sparse-Hawkes": "L1-Sparse Hawkes",
        "Diagonal-Hawkes": "Diagonal Hawkes",
        "Correlation-network": "Correlation network",
        "Neural-kernel-Hawkes": "Neural-kernel Hawkes",
        "RMTPP": "RMTPP",
        "Transformer-Hawkes": "Transformer Hawkes",
        "Neural-ODE-TPP": "Neural-ODE TPP",
        "C-NTPP": "C-NTPP",
        "Hyper-Hawkes": "Hyper-Hawkes",
    }
    papers = {
        "R3-conditional": (
            "(proposed) edge-conditioned Hawkes kernel", "--"),
        "R2-shared-mixture": (
            "(reference) shared mixture-of-exponentials Hawkes", "--"),
        "R1-single-exp": (
            "(reference) embedding single-exponential Hawkes", "--"),
        "P0-homogeneous-poisson": (
            "An Introduction to the Theory of Point Processes", "r_dvj"),
        "P1-permark-poisson": (
            "An Introduction to the Theory of Point Processes", "r_dvj"),
        "Dense-Hawkes": (
            "Maximum likelihood estimation of Hawkes' self-exciting "
            "point processes", "r_ozaki"),
        "L1-Sparse-Hawkes": (
            "Sparse and low-rank multivariate Hawkes processes",
            "bacry2020lowrank"),
        "Diagonal-Hawkes": (
            "Spectra of some self-exciting and mutually exciting "
            "point processes", "hawkes1971"),
        "Correlation-network": (
            "Hierarchical structure in financial markets", "mantegna1999"),
        "Neural-kernel-Hawkes": (
            "Hawkes Process with Flexible Triggering Kernels", "isik2023"),
        "RMTPP": (
            "Recurrent Marked Temporal Point Processes", "r_du"),
        "Transformer-Hawkes": (
            "Transformer Hawkes Process", "zuo2020thp"),
        "Neural-ODE-TPP": (
            "Decoupled Marked Temporal Point Process using Neural ODEs",
            "song2024"),
        "C-NTPP": (
            "c-NTPP: Learning Cluster-Aware Neural Temporal Point Process",
            "ding2023cntpp"),
        "Hyper-Hawkes": (
            "Hyper Hawkes Processes: Interpretable Models of Marked TPPs",
            "boyd2025hyper"),
    }
    baseline_order = [
        "R1-single-exp", "R2-shared-mixture",
        "Dense-Hawkes", "L1-Sparse-Hawkes",
        "Diagonal-Hawkes", "Correlation-network",
        "Neural-kernel-Hawkes",
        "P0-homogeneous-poisson", "P1-permark-poisson",
        "RMTPP", "Transformer-Hawkes",
        "Neural-ODE-TPP", "C-NTPP", "Hyper-Hawkes",
    ]

    # --- per-tau comparison tables ---
    all_taus = sorted(df["tau"].unique())
    all_comp_frames = []
    for tau_val in all_taus:
        rows = []
        r3c = agg[(agg.tau == tau_val) & (agg.dim == canon_dim) &
                   (agg.beta_set == canon_beta)]
        if len(r3c):
            rows.append(("Proposed conditional kernel",
                         float(r3c.iloc[0]["mean"])))
        for mdl in baseline_order:
            sub = df[(df.model == mdl) & (df.tau == tau_val) &
                     (df.dim == canon_dim) & (df.beta_set == canon_beta)]
            if len(sub):
                rows.append((disp.get(mdl, mdl),
                             float(sub.iloc[0]["held_out_ll"])))
        if not rows:
            continue
        comp = (pd.DataFrame(rows, columns=["Model", "LL/event"])
                  .sort_values("LL/event", ascending=False)
                  .reset_index(drop=True))
        comp.insert(0, "Rank", comp.index + 1)
        comp.to_csv(RESULTS_DIR / f"comparison_table_tau{tau_val}.csv",
                    index=False)
        all_comp_frames.append((tau_val, comp))
        print(f"\n=== COMPARISON TABLE @ tau={tau_val}, dim={canon_dim}, "
              f"beta={canon_beta} ===")
        print(comp.to_string(index=False))

        fig, ax = plt.subplots(figsize=(8, 0.5 * len(comp) + 1))
        colors_bar = ["#C44E52" if m.startswith("Proposed") else "#4C72B0"
                      for m in comp["Model"]]
        ax.barh(comp["Model"][::-1], comp["LL/event"][::-1],
                color=colors_bar[::-1])
        ax.set_xlabel("held-out LL/event  (higher = better)")
        ax.set_title(f"Method comparison @ tau={tau_val}, dim={canon_dim}, "
                     f"beta={canon_beta}", fontweight="bold")
        for i, v in enumerate(comp["LL/event"][::-1]):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"comparison_ranking_tau{tau_val}.png",
                    dpi=130)
        plt.close(fig)

    # also save a combined table for backward compat
    if all_comp_frames:
        _, last_comp = all_comp_frames[-1]
        last_comp.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)

    mp_rows = ([("R3-conditional", "Proposed conditional kernel")]
               + [(m, disp[m]) for m in baseline_order])
    mp = pd.DataFrame(
        [(dn, papers[code][0], papers[code][1]) for code, dn in mp_rows],
        columns=["Method", "Paper implemented", "Citation key"])
    mp.to_csv(RESULTS_DIR / "methods_and_papers.csv", index=False)
    print("\n=== METHODS AND THE PAPER EACH IMPLEMENTS ===")
    print(mp.to_string(index=False))

    print(f"\nSaved to {RESULTS_DIR}/:")
    tau_list_str = ", ".join(f"comparison_table_tau{t}.csv"
                             for t, _ in all_comp_frames)
    print(f"  ablation_results.csv, leaderboard.csv, {tau_list_str}, "
          f"methods_and_papers.csv")
    print(f"  heatmap_tau_dim.png, beta_dim_ablation.png")
    print("  " + ", ".join(f"comparison_ranking_tau{t}.png"
                           for t, _ in all_comp_frames))


# ========================================================================
# 6. ENTRYPOINT
# ========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Ablation runner for the edge-conditioned Hawkes kernel.")
    ap.add_argument("--quick", action="store_true",
                    help="tiny grid + few epochs, for a smoke test")
    ap.add_argument("--compile-only", action="store_true",
                    help="skip the sweep, just re-summarize "
                         "results/ablation_log.json")
    ap.add_argument("--taus", type=float, nargs="+", default=None)
    ap.add_argument("--dims", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-lit-baselines", action="store_true",
                    help="skip ALL neural-TPP baselines")
    ap.add_argument("--learnable-beta", action="store_true",
                    help="ALSO run the learnable-beta sweep")
    ap.add_argument("--only-learnable-beta", action="store_true",
                    help="run ONLY the learnable-beta sweep")
    ap.add_argument("--lb-quick", action="store_true",
                    help="learnable-beta quick smoke test")
    ap.add_argument("--lb-ms", type=int, nargs="+", default=None)
    ap.add_argument("--lb-ablations", nargs="+", default=None)
    ap.add_argument("--lb-seeds", type=int, nargs="+", default=None)
    ap.add_argument("--lb-epochs", type=int, default=None)
    ap.add_argument("--results-dir", default=os.environ.get("ABL_RESULTS_DIR"),
                    help="folder for log/CSV/PNG (use a Google Drive path on "
                         "Colab so results survive a disconnect)")
    ap.add_argument("--data-dir", default=os.environ.get("ABL_DATA_DIR"),
                    help="folder holding data/crypto_prices.npz")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly after this many hours, then compile")
    args, _unknown = ap.parse_known_args()

    global DEADLINE
    set_dirs(args.data_dir, args.results_dir)
    if args.max_hours:
        DEADLINE = time.time() + args.max_hours * 3600

    # --- pre-flight data check ---
    if not args.compile_only:
        if not PRICES_CACHE.exists():
            print(f"ERROR: {PRICES_CACHE} not found.\n"
                  f"Run Step 1 (Binance data download) first to create it.")
            sys.exit(1)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = {k: (dict(v) if isinstance(v, dict)
               else list(v) if isinstance(v, list)
               else v)
           for k, v in DEFAULT_CONFIG.items()}

    if args.quick:
        cfg.update(
            TAUS=[2.5], DIMS=[4, 8],
            BETA_SETS={"fast": (0.35, 1.0),
                       "default": (0.35, 1.0, 3.0, 8.0)},
            SEEDS=[0], EPOCHS=80, LIT_EPOCHS=80, CLASSIC_EPOCHS=300,
            CANON_TAU=2.5, CANON_DIM=8, CANON_BETA="default")
    if args.taus:
        cfg["TAUS"] = args.taus
    if args.dims:
        cfg["DIMS"] = args.dims
    if args.seeds:
        cfg["SEEDS"] = args.seeds
    if args.epochs:
        cfg["EPOCHS"] = args.epochs
    if args.no_lit_baselines:
        cfg["RUN_LIT_BASELINES"] = False

    run_lb = (args.learnable_beta or args.only_learnable_beta
              or args.lb_quick)

    print(f"[env] PyTorch {torch.__version__}  |  CPU  |  "
          f"CHUNK={CHUNK}  |  RSS={_mem_mb():.0f}MB")

    if not args.only_learnable_beta:
        print("Config:", {k: v for k, v in cfg.items()
                          if k not in ("BETA_SETS",)})
        print("Beta banks:", cfg["BETA_SETS"])
        if not args.compile_only:
            try:
                run_sweep(cfg)
            except OutOfTime:
                print("\n[time budget reached] stopped cleanly; all finished "
                      "runs are saved. Re-run the same command to resume.")
        compile_results(cfg)

    if run_lb:
        lb_cfg = dict(LEARNABLE_BETA_CONFIG)
        if args.lb_quick:
            lb_cfg.update(MS=[2, 4], ABLATIONS=["A3", "A4", "PE"],
                          SEEDS=[0], EPOCHS=120)
        if args.lb_ms:
            lb_cfg["MS"] = args.lb_ms
        if args.lb_ablations:
            lb_cfg["ABLATIONS"] = args.lb_ablations
        if args.lb_seeds:
            lb_cfg["SEEDS"] = args.lb_seeds
        if args.lb_epochs:
            lb_cfg["EPOCHS"] = args.lb_epochs
        try:
            run_learnable_sweep(lb_cfg, cfg)
        except OutOfTime:
            print("\n[time budget reached] learnable-beta sweep stopped "
                  "cleanly; compiling what is finished. Re-run to resume.")
            DEADLINE = None
            run_learnable_sweep(dict(lb_cfg, ABLATIONS=[]), cfg)

    print(f"\n[done] total RSS={_mem_mb():.0f}MB")


if __name__ == "__main__":
    main()
