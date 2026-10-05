"""
Pareto Frontier SSL — MultiBench Benchmark (MOSI / UR-FUNNY)
==============================================================
Three encoder approaches:

  1  MeanPool   : (B,T,p) → mean over T → Linear(p→enc_dim) → 4 static ProjectionHeads
  2  Transformer: CoMM 5h/5L/40d Transformer jointly trained with 4 static ProjectionHeads
  3  DualFiLM   : CoMM Transformer + DualFiLMProjectionHead — z(λ) = f(h, λ)
  4  LoRA       : CoMM Transformer + LoRADualProjectionHead — z(λ) = f(h, λ), quadratic subspace

Approaches 1/2 film_mode: none | token | full   (controls encoder λ-conditioning)
Approaches 3/4 film_mode: proj | token | full   (proj=λ-blind enc, token/full=λ-conditioned enc)
  Approach 3: DualFiLMProjectionHead — separate γ/β branches, shared W (weak, collapses)
  Approach 4: LoRADualProjectionHead — separate low-rank ΔW branches, shared W base
    L_R calls proj.forward_r(h) → updates only A_r, B_r (+ shared W)
    L_U calls proj.forward_u(h) → updates only A_u, B_u (+ shared W)
    Inference: z(λ) = proj.forward(h,λ) via W + (λA_r+(1-λ)A_u)(λB_r+(1-λ)B_u)

Two methods:
  simclr_single_per_batch  : L = 2λ·InfoNCE(v,t) + 2(1-λ)·(SimCLR(v)+SimCLR(t))/2
  factorcl_warmup          : same + CLUB (approaches 1/2 only) + phased λ curriculum

Three datasets: mosi | humor (UR-FUNNY) | mustard (MUsTARD). Modalities: vision + text only.

Output layout (under --out_dir):
  train_meta.json, enc_v.pth, enc_t.pth
  approaches 1/2: proj_{v,t}_{r,u}.pth
  approaches 3/4: proj_v.pth, proj_t.pth (DualFiLMProjectionHead / LoRADualProjectionHead)

Usage:
  cd <repo root>
  python pareto_ssl/multibench/benchmark_multibench.py \\
      --dataset humor --approach 2 --method factorcl_warmup \\
      --out_dir pareto_ssl/multibench/results/humor/approach2/factorcl_warmup_seed42

  python pareto_ssl/multibench/benchmark_multibench.py \\
      --dataset humor --approach 3 --method simclr_single_per_batch --film_mode proj \\
      --out_dir pareto_ssl/multibench/results/humor/approach3/simclr_filmproj_seed42
"""

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "multibench_config.yaml"

def _load_config(path=None):
    p = Path(path) if path else _DEFAULT_CONFIG
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f)
    return {}

def _cfg(config, *keys, default=None):
    """Traverse nested config dict with dot-path keys."""
    v = config
    for k in keys:
        if not isinstance(v, dict):
            return default
        v = v.get(k, default)
    return v

# Path setup (file is at Program/pareto_ssl/multibench/)
_PROG = Path(__file__).resolve().parent.parent.parent   # → Program/
sys.path.insert(0, str(_PROG))
sys.path.insert(0, str(_PROG / "CoMM"))

from einops import rearrange as _rearrange

from pareto_ssl.multibench.mb_transformer import Transformer as _CoMMTransformer
from pareto_ssl.multibench.affect_data import (
    AffectDataset, AffectSSLDataset, collate_affect, collate_affect_ssl,
    data_path as _affect_path, data_root as _data_root,
)


def _comm_fusion_transformer():
    """FusionTransformer, needed only by the comm_based architecture.

    That one is CoMM's own fusion block and is not reimplemented here, so the
    import stays lazy: the STEER / factorcl_based path never reaches it.
    """
    try:
        from models.mmfusion import FusionTransformer
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "arch='comm_based' needs CoMM's FusionTransformer, a separate project "
            "that is not redistributed here.\n"
            "    git clone https://github.com/Duplums/CoMM\n"
            "so that <repo root>/CoMM/models/ exists, then re-run.\n"
            "arch='factorcl_based' (the reported STEER configuration) does not need it."
        ) from e
    return FusionTransformer

from pareto_ssl.losses import infonce_cross, nt_xent, gmc_loss, comm_loss, comm_loss_dual, cross_self_loss, clip_loss, factorcl as factorcl_loss
from pareto_ssl.factorcl import CLUBInfoNCECritic, InfoNCECritic
from pareto_ssl.networks import (ProjectionHead, DualFiLMProjectionHead,
                                 DualFiLMProjectionHeadPreNorm,
                                 LoRADualLayer, LoRATriLayer,
                                 LoRADualProjectionHead, LoRATriProjectionHead,
                                 LoRATransformerEncoder)
from pareto_ssl.benchmark import (
    _sample_lam, sample_lambda_warmup, LAM_CLUB,
    DEFAULT_LAM_DIST, DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_BATCH_SIZE,
    DEFAULT_PROJ_DIM, DEFAULT_TRAIN_SEED,
    set_seed,
)
# All multi-label datasets (mosei_multitask, chsims) route through the registry, which
# dispatches on the dataset NAME -- the loaders take the name as their first argument.
from pareto_ssl.multibench.multitask_registry import (
    is_multitask_dataset, ssl_loader as mt_ssl_loader, probe_loader as mt_probe_loader,
)
from pareto_ssl.multibench.image_backends import (
    IMAGE_DATASETS, is_image_dataset, make_image_encoders,
    image_ssl_loader, image_probe_loader,
)

# Ablation 1: balanced-start warmup schedule
# Original warmup: Beta(5,1) → starts at λ≈1 (all shared), starves L_U early.
# New schedule   : Beta(5,5) → starts balanced at λ≈0.5, then free exploration,
#                  then bimodal extremes at the end to carve out λ=0 and λ=1.
_WARMUP_V2_PHASES = [
    {"frac": 0.20, "dist": "beta", "alpha": 5.0, "beta": 5.0},  # balanced start
    {"frac": 0.60, "dist": "uniform"},                            # free exploration
    {"frac": 0.20, "dist": "beta", "alpha": 0.2, "beta": 0.2},  # bimodal extremes
]

def _sample_lam_mb(lam_dist: str, dirichlet_alpha: float, epoch: int, epochs: int) -> float:
    if lam_dist == "warmup_v2":
        return sample_lambda_warmup(epoch, epochs, phases=_WARMUP_V2_PHASES)
    return _sample_lam(lam_dist, dirichlet_alpha, epoch, epochs)

warnings.filterwarnings("ignore")

# Dataset constants
# Confirmed shapes (2026-06-26):
#   mosi    : T=50, vision=20,  text=300 — train/valid/test: 1284/229/686
#   humor   : T=20, vision=371, text=300 — train/valid/test: 8074/1034/1058
#   mustard : T=50, vision=371, text=300 — train/valid/test: 414/138/138  (MUsTARD)
#   mosei   : T=50, vision=35,  text=300 — standard MultiBench mosei_senti_data.pkl
#             (vision+text subset; sentiment binarized via task="classification",
#              same protocol as mosi). PID (Liang+ 2023, Table 3): R=0.30, U1=0.70.
FEAT_DIM = {
    "mosi":    {"vision": 20,  "text": 300},
    "humor":   {"vision": 371, "text": 300},
    "mustard": {"vision": 371, "text": 300},
    "mosei":   {"vision": 35,  "text": 300},
    # mosei_multitask: SAME features as mosei, but 7 labels instead of 1. It is a
    # SEPARATE dataset — `mosei` is untouched.
    "mosei_multitask": {"vision": 35, "text": 300},
    # CH-SIMS (MMSA unaligned_39.pkl): OpenFace vision, BERT text. chsims.check_file
    # verifies these against the pickle and raises on any mismatch.
    "chsims": {"vision": 709, "text": 768},
}
MODALITIES    = ("vision", "text")
# Transformer width. NOT free: CoMM's Transformer hardcodes nhead=5 (so dim must
# be a multiple of 5) and build_1d_sincos_posemb rejects odd dims — hence multiples
# of 10 only: {40, 30, 20, 10}. Overridable with --enc_width for the capacity
# sweep; 40 is the CoMM default.
#
# Why this matters: this is the SHARED bottleneck. Whatever redundancy and
# uniqueness both need, they must both get it from these dimensions. proj_dim sits
# downstream and can only reformat what already survived, which is why squeezing
# proj_dim alone is unlikely to create objective conflict.
ENC_DIM_XFMR = 40   # CoMM default


# FiLM conditioning modes
# none  : λ only in loss weights, encoder is λ-blind (baseline)
# token : λ encoded as an extra token fed into the transformer / injected at input
# full  : λ modulates every layer via scale+shift (FiLM at every depth)

FILM_MODES = ("none", "token", "full", "proj", "lora", "palora_proj", "palora_enc",
              "simplex", "simplex4h", "simplex4h_norm", "simplex6h", "duo_txt",
              "simplex_enc_decomp_R", "simplex_proj_decomp_R", "simplex_both_decomp_R")

# The two 4-head simplex variants. Unlike `simplex` (2 shared-base LoRA heads
# interpolated in WEIGHT space via forward_mix) these keep four independent
# ProjectionHeads and mix their OUTPUTS, which is what palora_enc does — extended
# from a 1-D lambda to the 2-simplex so vision and text get independent unique
# coefficients.
#   simplex4h      : normalize(w_R * zhat_R + w_U * zhat_U)   -- heads normalise first
#   simplex4h_norm : normalize(w_R * z_R    + w_U * z_U   )   -- mix RAW, normalise once
# The pair is the experiment: palora_enc_4h's accuracy curve sags in the interior
# (67.08 -> 64.72 -> 67.61) and its middle preferences are representationally
# degenerate (CKA 0.949-0.976). The suspected cause is blending two already-unit
# vectors, which lands in neither head's space. simplex4h reproduces that; the
# _norm variant removes the double normalisation and isolates whether it is the
# cause.
SIMPLEX4H_MODES = ("simplex4h", "simplex4h_norm", "duo_txt")

# duo_txt — H5: match the controller dimension to the number of LIVE information
# directions. Encoder sufficiency on MOSEI found only one of the three simplex
# coordinates carries anything:
#     vision only 63.85 | text 74.57 | concat 74.56  -> vision adds nothing
#     vision perp text  54.03 (chance)               -> no vision-unique info
#     corr(global lambda-profile, lam_U_vision) = +0.118   dead
#     corr(global lambda-profile, lam_U_text)   = -0.794   live
# So drop the vision-uniqueness axis entirely and run a SCALAR lambda over
# (redundancy, text-uniqueness):
#     preference (lam, 0, 1-lam)
#     loss       lam*L_R + (1-lam)*L_U_text        -- L_U_vision never enters
#     vision     pure redundancy head (lam-independent)
#     text       normalize(lam*z_r + (1-lam)*z_u)  -- mixed RAW, like simplex4h_norm
# Structurally identical to simplex4h_norm (same four heads, same output mixing,
# same save/load), so the ONLY difference is the preference set. That makes
# simplex4h_norm the matched 3-coordinate control: same everything, 3 axes vs 1.
DUO_TXT = "duo_txt"

# simplex6h — a TRUE 3-way blend. simplex4h* give each modality only two of the
# three coordinates (vision never sees lam_U_text), mirroring the shared-base
# `simplex`. Here each modality carries THREE heads and blends all three
# coefficients:
#     z_v = normalize(w_R*A_v(h) + w_U1*B_v(h) + w_U2*C_v(h))
#     z_t = normalize(w_R*A_t(h) + w_U1*B_t(h) + w_U2*C_t(h))
# Every coordinate reaches every output, and because the simplex weights sum to
# 1 there is no all-zero vertex -- the degeneracy that simplex4h* has to guard.
# The heads are held in a ModuleDict per modality so the existing single-handle
# save/load path (proj_v.pth / proj_t.pth) works unchanged.
SIMPLEX6H = "simplex6h"

# simplex_enc_decomp_R — the objectives, not just the mixing.
#
# THE PROBLEM. Every arm above uses L_U = NT-Xent on one modality, which asks only
# for augmentation-invariant content and contains NO term referencing the other
# modality. So L_R captures SHARED and L_U captures SHARED + UNIQUE: the two are
# NESTED, not competing, and the lambda axis between them is only as long as the
# unique part is large. On MOSEI (vision perp text = 54.03, chance) it is zero.
#
# THE FIX. Add a CLUB penalty that pushes the U head away from the R head, within
# modality -- exactly FactorCL's formulation, club_v(z_vua, z_vr.detach()).
#
# STRUCTURE. lambda-dependence lives in the ENCODER (PaLoRA weight space, so the
# chain lambda -> W(lambda) -> z(lambda) -> L(lambda) holds and a symmetric grid
# cannot collapse to equal weighting); cleanliness lives in the HEADS (four plain
# heads, one objective each, no head carrying mixture weight without the matching
# gradient -- the simplex6h defect). Readout is a component-wise CONCAT with
# per-block scaling, not a weighted sum, so at (0,1,0) the R and U2 blocks are
# exactly zero and no text reaches the probe.
#
#
DECOMP_R = "simplex_enc_decomp_R"

# simplex_proj_decomp_R — the same decomposition (4 heads, CLUB pushing U off R,
# sqrt-lambda concat readout) with the PaLoRA adapters moved OUT of the encoder
# and INTO the heads: lambda -> W_head(lambda) -> z(lambda) -> L(lambda).
#
# Why the adapters cannot simply be dropped: with a lambda-blind encoder AND plain
# heads, zvr/zvu/ztr/ztu are identical at every preference, so
#     mean_l [ l_R*L_R + l_U1*L_U1 + l_U2*L_U2 ]  ==  lbar_R*L_R + ... ,
# i.e. training at the MEAN preference. That is the collapse PaLoRA warns about,
# and it would make this arm silently identical to a fixed-uniform-lambda control
# rather than a test of it. Putting LoRA in the heads keeps L(lambda) genuinely
# preference-dependent while the transformer runs ONCE per batch instead of 15x
# (60 encoder forwards -> 4), which is the entire point of the arm.
DECOMP_R_PROJ = "simplex_proj_decomp_R"

# Either decomposition variant: same heads, same CLUB, same readout, differing
# only in WHERE the preference acts.
# Both sites at once: LoRA in the encoder AND in the four heads. lambda is then applied
# TWICE -- W(lambda) then head(lambda) -- so its effect is multiplicative and the vertices
# separate further than their simplex coordinates imply. Tests whether DECOMP_R_PROJ's
# accuracy advantage and DECOMP_R's working preference axis combine.
DECOMP_R_BOTH = "simplex_both_decomp_R"
DECOMP_MODES = (DECOMP_R, DECOMP_R_PROJ, DECOMP_R_BOTH)

def _lin_cka_t(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two batch representations, as a float.

    CLUB reports its own estimate of I(u;r), but that estimate is only meaningful
    when the critic is well fit. This measures the EFFECT the penalty is supposed
    to have -- how aligned u and r actually are -- independently of CLUB. If
    lam_club is working, this must FALL over training.
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xy = (X.T @ Y).pow(2).sum()
    xx = (X.T @ X).pow(2).sum().sqrt()
    yy = (Y.T @ Y).pow(2).sum().sqrt()
    return float(xy / (xx * yy)) if float(xx * yy) > 0 else 0.0


def _eff_rank_t(Z: torch.Tensor) -> float:
    """Participation ratio of the covariance eigenvalues.

    Guards the opposite failure: a CLUB penalty that is TOO strong drives z_u to a
    constant, which trivially decorrelates it from z_r while destroying the
    representation. If this collapses toward 1, lam_club is too large.
    """
    Z = Z - Z.mean(0, keepdim=True)
    ev = torch.linalg.eigvalsh(torch.cov(Z.T.float()))
    ev = torch.clamp(ev, min=0.0)
    s1, s2 = ev.sum(), (ev ** 2).sum()
    return float(s1 * s1 / s2) if float(s2) > 0 else 0.0


def _lam_t(x: torch.Tensor, lam: float) -> torch.Tensor:
    """Scalar λ → (B,1) tensor on same device/dtype as x."""
    return torch.full((x.size(0), 1), lam, device=x.device, dtype=x.dtype)


# Encoders

class MeanPoolEncoder(nn.Module):
    """
    Approach 1 encoder — three FiLM modes:
      none  : mean(T) → Linear → L2-norm
      token : λ injected at input feature level before Linear
      full  : λ injected at input AND output (FiLM scale+shift after Linear)
    """
    def __init__(self, n_features: int, enc_dim: int, film_mode: str = "none"):
        super().__init__()
        self.film_mode = film_mode
        self.proj = nn.Linear(n_features, enc_dim)
        if film_mode in ("token", "full"):
            self.lam_in = nn.Linear(1, n_features)
        if film_mode == "full":
            self.film_gamma = nn.Linear(1, enc_dim)
            self.film_beta  = nn.Linear(1, enc_dim)

    def forward(self, x: torch.Tensor, lam: float = None) -> torch.Tensor:
        x_mean = x.mean(dim=1)
        if self.film_mode in ("token", "full"):
            l = _lam_t(x, lam)
            x_mean = x_mean + self.lam_in(l)
        h = self.proj(x_mean)
        if self.film_mode == "full":
            l = _lam_t(x, lam)
            h = self.film_gamma(l) * h + self.film_beta(l)
        return F.normalize(h, dim=-1)


class TransformerEncoder(nn.Module):
    """
    Approach 2/3 encoder — three FiLM modes:
      none  : CoMM Transformer as-is, λ-blind
      token : λ encoded as the last token; every attention layer attends to it;
              last token (= λ token) is used as the output
      full  : λ conditions every transformer sublayer via FiLM (scale+shift)
              after each of the 5 TransformerEncoderLayer calls
    """
    def __init__(self, n_features: int, film_mode: str = "none"):
        super().__init__()
        self.film_mode = film_mode
        self._xfmr = _CoMMTransformer(
            n_features=n_features, dim=ENC_DIM_XFMR,
            max_seq_length=50, return_seq=(film_mode != "none"),
            positional_encoding=False,
        )
        if film_mode == "token":
            self.lam_embed = nn.Linear(1, ENC_DIM_XFMR)
        if film_mode == "full":
            self.film_gamma = nn.ModuleList([nn.Linear(1, ENC_DIM_XFMR) for _ in range(5)])
            self.film_beta  = nn.ModuleList([nn.Linear(1, ENC_DIM_XFMR) for _ in range(5)])

    def forward(self, x: torch.Tensor, lam: float = None) -> torch.Tensor:
        if self.film_mode == "none":
            return F.normalize(self._xfmr(x), dim=-1)

        # Manual forward: Conv1d projection → conditioning → Transformer layers
        h = _rearrange(self._xfmr.conv(_rearrange(x, 'b l n -> b n l')), 'b n l -> b l n')
        l = _lam_t(x, lam)

        if self.film_mode == "token":
            lam_tok = self.lam_embed(l).unsqueeze(1)       # (B,1,40)
            h = torch.cat([h, lam_tok], dim=1)             # (B,T+1,40)
            h = self._xfmr.transformer(h)                  # attend over [seq + λ]
            return F.normalize(h[:, -1], dim=-1)           # last token = λ token

        if self.film_mode == "full":
            for i, layer in enumerate(self._xfmr.transformer.layers):
                h = layer(h)                               # standard attention + FFN
                gamma = self.film_gamma[i](l).unsqueeze(1) # (B,1,40) → broadcast (B,T,40)
                beta  = self.film_beta[i](l).unsqueeze(1)
                h = gamma * h + beta                       # FiLM at every layer
            return F.normalize(h[:, -1], dim=-1)


class FusionEncoder(nn.Module):
    """Joint encoder for gmc/comm: concat(z_v, z_t) → head → L2-norm.

    Default (hidden=None) is the original single Linear(2*in_dim, out_dim) — kept
    bit-identical so every existing fusion.pth still loads.

    HEAD-CAPACITY CONTROL. At evaluation each method is probed through its own
    trained head, and those heads are very unequal:

        simplex6h (ours)   6x ProjectionHead(40->64)   25,584 params   128-d out
        FactorCL           4x ProjectionHead(40->40)   13,120          80-d
        CoMM / GMC         1x Linear(80->40)            3,240          40-d

    so CoMM competes with ~1/8 the evaluation-time head parameters of our largest
    arm. Parameter count does NOT order the results (FactorCL has 4x CoMM's head
    budget and scores a point BELOW it), but "we disclosed it" is a weaker rebuttal
    than "we controlled it". Passing hidden gives CoMM/GMC an MLP head mirroring
    ProjectionHead's shape (Linear→ReLU→Linear), so head capacity becomes a
    controlled variable rather than a confound.

    This can only make the baseline STRONGER, so it is a conservative test: if CoMM
    still loses at matched capacity, the gap is not a head-budget artifact.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: int = None):
        super().__init__()
        self.hidden = hidden
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(2 * in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        else:
            self.fc = nn.Linear(2 * in_dim, out_dim)

    def forward(self, z_v: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_v, z_t], dim=-1)
        return F.normalize(self.net(x) if self.hidden else self.fc(x), dim=-1)


# comm_based architecture

class TransformerSeqEncoder(nn.Module):
    """comm_based: CoMM Transformer returning full token sequence (B, T, 40)."""
    def __init__(self, n_features: int):
        super().__init__()
        self._xfmr = _CoMMTransformer(
            n_features=n_features, dim=ENC_DIM_XFMR,
            max_seq_length=50, return_seq=True, positional_encoding=False,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._xfmr(x)  # (B, T, 40)


class CommMLPHead(nn.Module):
    """3-layer MLP with BN + L2 norm — real CoMM projection head (out_dim=256)."""
    def __init__(self, in_dim: int = ENC_DIM_XFMR, mlp_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_dim), nn.BatchNorm1d(mlp_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, mlp_dim), nn.BatchNorm1d(mlp_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class LoRACommMLPHead(nn.Module):
    """
    CommMLPHead with LoRA on the last N linear layers — no BN.

    lora_layers : how many of the 3 MLP layers carry LoRA, counting from the output end.
      lora_layers=3 (default) : all 3 layers have LoRA  [most expressive]
      lora_layers=2           : layers 2+3 have LoRA, layer 1 is plain Linear
      lora_layers=1           : only layer 3 has LoRA   [lightest]
    Plain layers produce the same output for forward_r, forward_u, and forward(lam)
    — differentiation only happens in the LoRA layers.

    2-branch (branches=2): R+U linear blend via LoRADualLayer.
    3-branch (branches=3): R+U+M quadratic curve via LoRATriLayer.
    Output is always L2-normalised.
    """
    def __init__(self, in_dim: int = ENC_DIM_XFMR, mlp_dim: int = 512, out_dim: int = 256,
                 rank: int = 4, rank_m: int = None, branches: int = 2, lora_layers: int = 3):
        super().__init__()
        if lora_layers not in (1, 2, 3):
            raise ValueError(f"lora_layers must be 1, 2, or 3, got {lora_layers}")
        rank_m = rank_m if rank_m is not None else rank
        LoraCls = LoRATriLayer if branches == 3 else LoRADualLayer
        lora_kw = dict(rank=rank, rank_m=rank_m) if branches == 3 else dict(rank=rank)
        self.layer1 = LoraCls(in_dim,   mlp_dim, **lora_kw) if lora_layers >= 3 else nn.Linear(in_dim,   mlp_dim)
        self.layer2 = LoraCls(mlp_dim,  mlp_dim, **lora_kw) if lora_layers >= 2 else nn.Linear(mlp_dim,  mlp_dim)
        self.layer3 = LoraCls(mlp_dim,  out_dim, **lora_kw)   # always LoRA

    @staticmethod
    def _fwd(layer, h, mode, lam=None):
        """Apply one layer in the requested mode (r / u / lam); plain Linear ignores mode."""
        if isinstance(layer, (LoRADualLayer, LoRATriLayer)):
            if mode == "r":   return layer.forward_r(h)
            if mode == "u":   return layer.forward_u(h)
            return layer(h, lam)
        return layer(h)   # plain nn.Linear

    def forward_r(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self._fwd(self.layer1, x,  "r"))
        h = F.relu(self._fwd(self.layer2, h,  "r"))
        return F.normalize(self._fwd(self.layer3, h, "r"), dim=-1)

    def forward_u(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self._fwd(self.layer1, x,  "u"))
        h = F.relu(self._fwd(self.layer2, h,  "u"))
        return F.normalize(self._fwd(self.layer3, h, "u"), dim=-1)

    def forward(self, x: torch.Tensor, lam: float) -> torch.Tensor:
        h = F.relu(self._fwd(self.layer1, x,  "lam", lam))
        h = F.relu(self._fwd(self.layer2, h,  "lam", lam))
        return F.normalize(self._fwd(self.layer3, h, "lam", lam), dim=-1)


def _make_fusion_xfmr(device: str) -> nn.Module:
    """Shared FusionTransformer for comm_based: CLS-pooled, 1-layer, 8-head."""
    return _comm_fusion_transformer()(
        width=ENC_DIM_XFMR, n_heads=8, n_layers=1,
        fusion="concat", pool="cls", batch_first=True,
    ).to(device)


def _image_root(dataset: str) -> str:
    """Data root for an image dataset (avmnist/enrico), from our own catalog."""
    return _data_root(dataset)


def enc_film_mode_for(film_mode: str, approach: int) -> str:
    """Which film_mode the ENCODER is built with, given the run's film_mode.

    Several modes put all λ-conditioning in the projection head, leaving the
    encoder λ-blind. Approach-4 simplex is one of them: the 3-vector preference
    is applied by proj.forward_mix, and the encoder never sees it.

    This MUST be identical in the trainer and the probe. It previously was not —
    probe.py omitted the approach-4 simplex case, so it rebuilt the encoder as
    film_mode="simplex". TransformerEncoder has no branch for that value, so its
    forward fell off the end and returned None. The state dict is byte-identical
    between "none" and "simplex" (neither adds parameters), so loading succeeded
    and the failure only surfaced as a None deref during feature extraction.
    Both sides now call this function.
    """
    if film_mode in ("proj", "lora", "palora_proj"):
        return "none"
    if film_mode == "simplex" and approach == 4:
        return "none"
    if film_mode in SIMPLEX4H_MODES or film_mode == SIMPLEX6H:
        return "none"          # all preference handling lives in the heads
    if film_mode == DECOMP_R_PROJ:
        return "none"          # preference lives in the LoRA heads; encoder is blind
    if film_mode in (DECOMP_R, DECOMP_R_BOTH):
        # DECOMP_R: the ONLY place the preference acts is the encoder (plain heads).
        # DECOMP_R_BOTH: the encoder is conditioned AS WELL AS the heads, so it needs
        # the same LoRA encoder; the heads add their own conditioning on top.
        return "simplex"
    return film_mode


def _make_encoders(approach: int, dataset: str, enc_dim: int, device: str,
                   film_mode: str = "none",
                   lora_branches: int = 2, lora_rank: int = 4, lora_rank_m: int = None,
                   freeze_vgg: bool = True, enc_alpha: float = None,
                   plain_encoder: bool = False):
    """plain_encoder forces a NON-LoRA encoder even on approach 5.

    Needed by simplex_proj_decomp_R, whose whole premise is that the encoder holds
    no preference-conditioned parameters at all — every adapter lives in the heads.
    It cannot be inferred from film_mode: enc_film_mode_for maps both "proj" and
    simplex_proj_decomp_R to "none", but approach-5 "proj" deliberately KEEPS the
    LoRA encoder, so the two are indistinguishable downstream. Hence an explicit flag.
    """
    if is_image_dataset(dataset):
        # Image datasets (avmnist/enrico) use CNN encoders that output an
        # ENC_DIM_XFMR-d vector — same interface as TransformerEncoder — so all
        # downstream projection-head / LoRA / SimCLR machinery is unchanged.
        # LoRA (approach 4/5) lives entirely in the projection head for these,
        # so the encoder is identical across approaches. freeze_vgg controls
        # ENRICO's VGG feature fine-tuning (ignored by avmnist's LeNet).
        # film_mode here is the ENCODER mode from enc_film_mode_for(). "simplex" is
        # the only value that asks for a preference-conditioned encoder (approach-5
        # simplex and simplex_enc_decomp_R both map to it), so it is also the only
        # case that needs LoRA on the image adapter. Every other mode keeps the
        # plain adapter, so existing avmnist/enrico checkpoints stay loadable.
        return make_image_encoders(dataset, device, out_dim=ENC_DIM_XFMR,
                                   freeze_vgg=freeze_vgg,
                                   lora=(film_mode == "simplex" and not plain_encoder),
                                   lora_rank=lora_rank, lora_alpha=enc_alpha)
    fv, ft = FEAT_DIM[dataset]["vision"], FEAT_DIM[dataset]["text"]
    if approach == 5 and not plain_encoder:
        return (LoRATransformerEncoder(fv, branches=lora_branches, rank=lora_rank,
                                       rank_m=lora_rank_m, alpha=enc_alpha).to(device),
                LoRATransformerEncoder(ft, branches=lora_branches, rank=lora_rank,
                                       rank_m=lora_rank_m, alpha=enc_alpha).to(device))
    if approach == 1:
        return (MeanPoolEncoder(fv, enc_dim, film_mode).to(device),
                MeanPoolEncoder(ft, enc_dim, film_mode).to(device))
    return (TransformerEncoder(fv, film_mode).to(device),
            TransformerEncoder(ft, film_mode).to(device))


def _actual_enc_dim(approach: int, enc_dim: int) -> int:
    return enc_dim if approach == 1 else ENC_DIM_XFMR


def _params(*modules):
    return [p for m in modules for p in m.parameters() if p.requires_grad]


# Data loaders

def _ssl_loader(dataset: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    if is_multitask_dataset(dataset):
        return mt_ssl_loader(dataset, _image_root(dataset), batch_size, num_workers,
                             modalities=MODALITIES)
    if is_image_dataset(dataset):
        return image_ssl_loader(dataset, _image_root(dataset), batch_size, num_workers)
    ds = AffectSSLDataset(_affect_path(dataset), dataset, split="train",
                          modalities=MODALITIES, task="classification",
                          augmentations="drop+noise")
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=True, drop_last=True, collate_fn=collate_affect_ssl)


def _probe_loader(dataset: str, split: str, batch_size: int, num_workers: int = 4,
                  task: str = None) -> DataLoader:
    if is_multitask_dataset(dataset):
        return mt_probe_loader(dataset, _image_root(dataset), split, batch_size, task=task,
                               num_workers=num_workers, modalities=MODALITIES)
    if is_image_dataset(dataset):
        return image_probe_loader(dataset, _image_root(dataset), split, batch_size, num_workers)
    ds = AffectDataset(_affect_path(dataset), dataset, split=split,
                       modalities=MODALITIES, task="classification")
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=True, collate_fn=collate_affect)


# Training

def _lora_rhos(*heads):
    """rho_R / rho_U = ||(alpha/r)·dW|| / ||W0||, averaged over every LoRA layer.

    Measures how loud each preference-specific adapter is relative to the shared
    base. Small rho => W0 dominates the effective weights at every preference.
    """
    rr, ru = [], []
    for head in heads:
        for mod in head.modules():
            if not (hasattr(mod, "A_r") and hasattr(mod, "linear")):
                continue
            sc = getattr(mod, "scale", 1.0)
            n0 = torch.linalg.norm(mod.linear.weight.detach())
            if float(n0) == 0.0:
                continue
            rr.append(float(sc * torch.linalg.norm(mod.A_r.detach() @ mod.B_r.detach()) / n0))
            ru.append(float(sc * torch.linalg.norm(mod.A_u.detach() @ mod.B_u.detach()) / n0))
    return (sum(rr) / len(rr) if rr else float("nan"),
            sum(ru) / len(ru) if ru else float("nan"))


def _cka_of(X, Y):
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    xy = float(torch.linalg.norm(X.T @ Y) ** 2)
    xx = float(torch.linalg.norm(X.T @ X)); yy = float(torch.linalg.norm(Y.T @ Y))
    return xy / (xx * yy) if xx > 0 and yy > 0 else float("nan")


@torch.no_grad()
def _endpoint_cka_mix(zfun):
    """Endpoint CKA for the OUTPUT-MIXING variants (simplex4h*, simplex6h).

    _endpoint_cka below calls proj(h, lam) — the shared-base LoRA signature. The
    four/six-head variants have no such forward (proj_v is None, or a ModuleDict
    with no forward at all), so they logged NaN and lost the diagnostic entirely.
    This takes a closure z(pref) instead, so any head layout can be measured.

    Endpoints match the probe's: pure-unique (0, .5, .5) vs pure-shared (1, 0, 0).
    Note (0,1,0) would be wrong — it hands the text head (0,0).
    """
    try:
        return _cka_of(zfun((0.0, 0.5, 0.5)).float(), zfun((1.0, 0.0, 0.0)).float())
    except Exception:
        return float("nan")


@torch.no_grad()
def _endpoint_cka(proj_v, proj_t, h_v, h_t):
    """Linear CKA between the two ENDPOINT representations z(λ=[1,0]) and z(λ=[0,1]).

    ~1 => the preference endpoints are the same representation (no specialisation).
    """
    def _z(lam):
        return torch.cat([proj_v(h_v, lam), proj_t(h_t, lam)], dim=-1).float()
    X, Y = _z(1.0), _z(0.0)
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    xy = float(torch.linalg.norm(X.T @ Y) ** 2)
    xx = float(torch.linalg.norm(X.T @ X)); yy = float(torch.linalg.norm(Y.T @ Y))
    return xy / (xx * yy) if xx > 0 and yy > 0 else float("nan")


def preference_rank_penalty(losses, weights, margin: float = 0.02):
    """Fix 9 — pairwise ranking penalty making a preference MEAN something.

    THE PROBLEM. Training only requires each preference's weighted loss to be low.
    A single lambda-invariant representation satisfies all of them at once, so
    nothing prevents collapse: measured adjacent-preference CKA is 0.94-0.97 and
    the accuracy-vs-lambda curves are flat plateaus whose argmax is seed noise.

    THE FIX. If preference m weights objective o more heavily than preference m'
    does, then z_m should actually BE BETTER at o than z_m' is. That is what a
    Pareto front means, and it is currently unenforced. Hinge-penalise violations:

        for w_o^m > w_o^m' :   penalty += max(0, L_o(z_m) - L_o(z_m') + margin)

    Deliberately expressed in LOSS space, not representation space. A CKA-based
    repulsion term would be circular — CKA is our evaluation metric, and training
    against it would make it meaningless as evidence. L_R/L_U are already computed
    for every preference, so this costs only scalar comparisons: no extra forwards.

    losses  : list over preferences of (L_R, L_U1, L_U2) 0-dim tensors
    weights : list over preferences of (w_R, w_U1, w_U2) floats
    margin  : separation demanded between the EXTREME preferences, in nats;
              intermediate pairs are required to separate in proportion to their
              weight difference (margin * (w_o^m - w_o^m')). InfoNCE losses here
              run ~2-3 nats, so 0.02 is ~1% and is satisfied at init — use 0.1-0.3.
    """
    M = len(losses)
    if M < 2:
        return losses[0][0] * 0.0 if M else None
    terms, n = None, 0
    for m in range(M):
        for mp in range(M):
            if m == mp:
                continue
            for o in range(len(weights[m])):
                if weights[m][o] <= weights[mp][o]:
                    continue                      # only ordered pairs constrain us
                # Scale the required gap by HOW MUCH more preference m weights
                # objective o. A flat margin is wrong: with 15 preferences the
                # neighbouring ones differ by 0.25 in weight and should need almost
                # no separation, while the extremes differ by 1.0 and should need
                # the full margin. A flat 0.02 was met at initialisation (observed
                # penalty 0.009-0.018 before any training), so the term did nothing.
                gap = margin * (weights[m][o] - weights[mp][o])
                v = torch.relu(losses[m][o] - losses[mp][o] + gap)
                terms = v if terms is None else terms + v
                n += 1
    if terms is None:
        return losses[0][0] * 0.0
    return terms / n


def simplex_grid(side: int = 5):
    """Deterministic triangular grid on the 2-simplex (λ_R, λ_U1, λ_U2), Σ=1.
    side=5 -> 15 preferences, including the three vertices."""
    n = side - 1
    return [(i / n, j / n, (n - i - j) / n)
            for i in range(n + 1) for j in range(n + 1 - i)]


def duo_grid(side: int = 5):
    """1-D controller on the (redundancy, text-uniqueness) edge: (lam, 0, 1-lam).

    Returns the SAME number of points as simplex_grid(side) -- side*(side+1)/2,
    i.e. 15 for side=5 -- so training cost and preference density are matched and
    the only difference from the 2-simplex is the dimension of the controller.
    """
    n = side * (side + 1) // 2
    return [(i / (n - 1), 0.0, 1.0 - i / (n - 1)) for i in range(n)]


def anneal_simplex(prefs, tau: float, q: float, mode: str = "power"):
    """Anneal TARGET preferences to the EFFECTIVE ones used for this step.

    mode="power" (default, historical): lambda_i^gamma / sum_j lambda_j^gamma.
    mode="linear": lambda_eff = (1 - eta) * centroid + eta * lambda_target.

    Why "linear" exists. The power map leaves the VERTICES untouched for every
    gamma > 0 -- (1,0,0) is already (1,0,0) at gamma = 0.25 -- while interior points
    crawl outward, so it anneals exactly the preferences that need it least. Linear
    interpolation gives the intended progression for a vertex target:
        eta 0 -> (.333,.333,.333), .25 -> (.5,.25,.25), .5 -> (.667,.167,.167), 1 -> (1,0,0)
    Default stays "power" so every existing arm reproduces bit-for-bit.
    """
    g = tau / q
    if mode == "linear":
        eta = min(1.0, max(0.0, g))
        c = 1.0 / 3.0
        return [tuple((1.0 - eta) * c + eta * max(x, 0.0) for x in p) for p in prefs]
    if g <= 0.0:
        return [(1 / 3, 1 / 3, 1 / 3)] * len(prefs)
    out = []
    for p in prefs:
        pe = [max(c, 0.0) ** g for c in p]
        t = sum(pe) or 1e-12
        out.append(tuple(c / t for c in pe))
    return out


class PrefCycle:
    """Balanced subset scheduler: shuffle the grid, hand out groups of M, reshuffle.

    Every target preference is visited once per cycle, so over training each receives
    the same number of updates up to one partial group -- unlike i.i.d. sampling, which
    leaves some preferences under-trained by chance. M >= len(grid) yields the full grid
    every call, i.e. the historical behaviour, with no shuffling.
    """

    def __init__(self, n, m, seed=0):
        self.n, self.m = n, max(1, int(m))
        self.rng = np.random.default_rng(seed)
        self.order, self.pos = [], 0
        self.counts = np.zeros(n, dtype=int)

    def next(self):
        if self.m >= self.n:
            self.counts += 1
            return list(range(self.n))
        out = []
        while len(out) < self.m:
            if self.pos >= len(self.order):
                self.order = list(self.rng.permutation(self.n)); self.pos = 0
            out.append(int(self.order[self.pos])); self.pos += 1
        self.counts[out] += 1
        return out


def _anneal_preferences(m: int, tau: float, q: float, device) -> torch.Tensor:
    """PaLoRA deterministic preference annealing (Dimitriadis et al.), T=2 objectives.

        p            = torch.linspace(0.0, 1.0, M)
        base_prefs   = torch.stack([p, 1.0 - p], dim=1)          # (M, 2), rows sum to 1
        gamma        = tau / Q
        lambda_m     = base_prefs[m].pow(gamma)
        lambda_m     = lambda_m / lambda_m.sum()

    tau == 0 is handled explicitly (every preference -> [0.5, 0.5]) to avoid 0**0.
    For every tau > 0 the exact formula is preserved, so the base endpoints [0,1]
    and [1,0] REMAIN endpoints (they are fixed points of the map) while only the
    interior preferences anneal outward. This is the intended PaLoRA behaviour.

    Returns (M, 2): column 0 weights L_R / dW_R, column 1 weights L_U / dW_U.
    """
    p = torch.linspace(0.0, 1.0, m, device=device)
    base_preferences = torch.stack([p, 1.0 - p], dim=1)          # (M, 2)
    gamma = tau / q
    if gamma <= 0.0:                                              # tau == 0
        return torch.full((m, 2), 0.5, device=device)
    lam = base_preferences.pow(gamma)                             # 0**gamma = 0 (gamma>0)
    return lam / lam.sum(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def _club_diagnostics(critic, x_samples, y_samples):
    """Open up a CLUBInfoNCECritic so a flat scalar loss can be interpreted.

    The critic's TRAINING loss is the negative InfoNCE lower bound
        L_critic = -[ E(T0) - (LSE(T1) - log B) ]
    while the quantity the MAIN objective consumes is
        I_CLUB   = E(T0) - E(T1)
    These are different functions of the same scores and only coincide at a
    constant critic, so logging L_critic alone cannot distinguish "critic never
    learned" from "critic correctly reports near-independence".

    T0 scores MATCHED pairs (the diagonal); T1 scores the full B x B grid. Note
    I_CLUB is already E[T0] - E[T1_all], so t_diag_minus_off below differs from it
    only by the diagonal's O(1/B) share -- the two track each other by construction
    and the independent evidence is grad_norm / param_delta.
    """
    B = y_samples.shape[0]
    x_tile = x_samples.unsqueeze(0).repeat((B, 1, 1))
    y_tile = y_samples.unsqueeze(1).repeat((1, B, 1))
    T0 = critic._f(torch.cat([y_samples, x_samples], dim=-1))
    T1 = critic._f(torch.cat([y_tile, x_tile], dim=-1))
    eye = torch.eye(B, dtype=torch.bool, device=T1.device)
    t_off = T1.squeeze(-1)[~eye].mean()
    return {
        "L_critic":  float(-(T0.mean() - (T1.squeeze(-1).logsumexp(dim=1).mean()
                                          - math.log(B)))),
        "I_club":    float(T0.mean() - T1.mean()),
        "t_diag":    float(T0.mean()),
        "t_off":     float(t_off),
        "t_diag_minus_off": float(T0.mean() - t_off),
        "T1_std":    float(T1.std()),
    }


def train(
    approach: int, method: str, dataset: str, out_dir: str,
    enc_dim: int = 128, proj_dim: int = DEFAULT_PROJ_DIM,
    epochs: int = DEFAULT_EPOCHS, lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE, lam_dist: str = DEFAULT_LAM_DIST,
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
    prefs_per_batch: int = 0,
    anneal_mode: str = "power",
    overwrite_ok: bool = False,
    temperature: float = 0.1,
    lam_club: float = 0.5,
    club_hidden_dim: int = 512,
    club_layers: int = 1,
    ssl_scale: float = 1.0,
    loss_w=None,
    clip_logit_scale: bool = False,
    proj_dim_explicit: bool = True,
    club_iters: int = 1,
    critic_lr_mult: float = 1.0,
    factorcl_official: bool = False,
    weight_decay: float = 1e-4,
    device: str = "cuda", seed: int = DEFAULT_TRAIN_SEED,
    enc_ckpt_v: str = None, enc_ckpt_t: str = None,
    film_mode: str = "none",
    # Architecture key
    # factorcl_based : pooled TransformerEncoder + ProjectionHead (original)
    # comm_based     : TransformerSeqEncoder + FusionTransformer + CommMLPHead
    arch: str = "factorcl_based",
    # Ablation 2: separate LR — projection heads trained at lr * head_lr_mult
    head_lr_mult: float = 1.0,
    # Approach 4 LoRA ablation keys
    lora_branches: int = 3,        # 3 = 3-branch (R+U+M curvature), 2 = linear ablation (R+U only)
    lora_rank: int = 4,            # rank for R and U branches
    lora_rank_m: int = None,       # rank for M branch (defaults to lora_rank)
    lora_lr_mult: float = 10.0,    # LR multiplier for R/U branches: lr * head_lr_mult * lora_lr_mult
    lora_lr_mult_m: float = 3.0,   # additional LR multiplier for M branch on top of lora_lr_mult
    lora_lr_mult_t: float = 1.0,   # fix2: text LoRA LR relative to vision LoRA (< 1 = slower text)
    proj_prenorm: bool = False,    # approach 3 variant: FiLM applied before L2 norm (prenorm head)
    warmup_epochs: int = 0,        # fix3: endpoint training for first N epochs, then curve
    var_reg_weight: float = 0.0,   # fix1: variance regularization weight (0 = disabled)
    var_reg_gamma: float = 0.05,   # fix1: minimum std per embedding dimension
    curve_reg_weight: float = 0.0, # fix6: curve smoothness regularization weight (0 = disabled)
    orth_reg_weight: float = 0.0,  # fix7: LoRA R/U subspace orthogonality weight (0 = disabled)
    lora_layers: int = 3,          # comm_based ap4: how many MLP layers carry LoRA (1/2/3, from output end)
    resume: bool = False,          # resume ap3/4/5 training from <out_dir>/resume.pth if it exists
    # Fix 8: PaLoRA multi-preference training (approach 4/5, film_mode=proj)
    num_preferences: int = 1,             # M preferences per minibatch (1 = single-λ baseline)
    preference_schedule: str = "single",  # single | fixed | annealed
    annealing_temperature: float = 1.0,   # Q in the center-to-edge schedule (>0)
    lora_alpha: float = None,             # PaLoRA α; θ scale = α/r. Applied only when annealed
    rank_penalty: float = 0.0,            # Fix 9: β on the preference-ranking penalty (0 = off)
    rank_margin: float = 0.20,            # Fix 9: separation demanded between extreme prefs
    enrico_unfreeze: bool = False,        # ENRICO: fine-tune the VGG conv features (default frozen)
    simplex_side: int = 5,                # film_mode=simplex: grid points per edge
    fusion_hidden: int = None,            # gmc/comm: MLP fusion head width (None = original Linear)
    fusion_out: int = None,               # gmc/comm: fusion output dim (None = enc_dim)
    freeze_W0: bool = False,              # PaLoRA Pareto-expansion ablation: freeze the
                                          # shared base W0 and train only ΔW_R / ΔW_U.
                                          # Default False = PaLoRA from-scratch setting.
):
    if film_mode not in FILM_MODES:
        raise ValueError(f"film_mode must be one of {FILM_MODES}, got {film_mode!r}")
    if approach in (3, 4) and film_mode == "none":
        raise ValueError(f"Approach {approach} requires film_mode in (proj, token, full) — 'none' is not valid.")
    if approach not in (3, 4, 5) and film_mode == "proj":
        raise ValueError("film_mode='proj' is only valid for approaches 3/4.")
    # palora_proj: approach 1/2 four-head architecture (separate, NO shared base)
    # trained PaLoRA-style — the output-space mixture z(λ)=norm(λ·z_R+(1-λ)·z_U) is
    # optimised at M annealed preferences instead of being a probe-time-only interpolation.
    if film_mode == "palora_proj" and approach not in (1, 2):
        raise ValueError("film_mode='palora_proj' is only valid for approaches 1/2.")
    if approach == 5 and film_mode not in ("lora", "proj", "palora_enc", "simplex",
                                           DECOMP_R, DECOMP_R_PROJ, DECOMP_R_BOTH):
        raise ValueError(f"Approach 5 requires film_mode in "
                         f"(lora, proj, palora_enc, simplex, {DECOMP_R}, "
                         f"{DECOMP_R_PROJ}, {DECOMP_R_BOTH}), got {film_mode!r}.")
    # palora_enc: approach 5 PaLoRA — BOTH the LoRA encoder and the LoRA head are
    # conditioned on the same annealed preference λ_m, so h itself varies with λ.
    if film_mode == "palora_enc" and approach != 5:
        raise ValueError("film_mode='palora_enc' is only valid for approach 5.")
    # simplex: 3 objectives (R, U_vision, U_text) on the 2-simplex. Each modality
    # head sees (λ_R, λ_U{own}); the vision head never sees λ_U_text and vice versa.
    # simplex on approach 4 = heads only ("simplex proj"); on approach 5 the LoRA
    # ENCODER is preference-conditioned as well ("simplex enc").
    if film_mode == DECOMP_R and approach != 5:
        raise ValueError(
            f"film_mode={DECOMP_R} requires --approach 5.\n"
            f"  The whole point of this mode is that the PREFERENCE acts on the ENCODER "
            f"(lambda -> W(lambda) -> z(lambda) -> L(lambda)), which needs a "
            f"LoRATransformerEncoder with forward_mix.\n"
            f"  _make_encoders only builds one for approach 5; approach {approach} gets a "
            f"plain TransformerEncoder and the training loop would die with "
            f"AttributeError: 'TransformerEncoder' object has no attribute 'forward_mix'.")
    if film_mode == DECOMP_R_BOTH and approach != 5:
        raise ValueError(f"film_mode={DECOMP_R_BOTH} requires --approach 5: it needs the "
                         f"LoRA encoder (forward_mix) that only approach 5 builds, on top "
                         f"of the LoRA heads.")
    if film_mode == DECOMP_R_PROJ and approach != 5:
        raise ValueError(
            f"film_mode={DECOMP_R_PROJ} requires --approach 5 — not because the "
            f"encoder needs forward_mix (it is deliberately lambda-blind here), but "
            f"so the arm is capacity- and protocol-matched to {DECOMP_R}, which is "
            f"the only comparison it exists to make.")
    if film_mode == "simplex" and approach not in (4, 5):
        raise ValueError("film_mode='simplex' is only valid for approaches 4/5.")
    # Multi-preference flags — must be defined BEFORE the optimizer/head construction.
    _palora = (film_mode == "palora_proj")
    _palora_enc = (film_mode == "palora_enc")
    _simplex = (film_mode == "simplex")
    _simplex4h = (film_mode in SIMPLEX4H_MODES)
    _decomp_r_proj = (film_mode == DECOMP_R_PROJ)
    _decomp_r  = (film_mode in DECOMP_MODES)   # shared heads/CLUB/readout path
    # WHERE the preference acts, as two INDEPENDENT sites (one flag each, not either/or):
    #   DECOMP_R       enc only   -- LoRA encoder, plain heads      (reported STEER)
    #   DECOMP_R_PROJ  proj only  -- lambda-blind encoder, LoRA heads (lambda-inert arm)
    #   DECOMP_R_BOTH  enc + proj -- LoRA at both, lambda multiplicative
    _enc_cond  = _decomp_r and film_mode != DECOMP_R_PROJ
    _head_cond = _decomp_r and film_mode in (DECOMP_R_PROJ, DECOMP_R_BOTH)
    _simplex6h = (film_mode == SIMPLEX6H)
    # duo_txt mixes raw like simplex4h_norm, so simplex4h_norm is the matched
    # 3-coordinate control and mixing style is not a confound.
    _s4h_premix = (film_mode in ("simplex4h_norm", DUO_TXT))  # mix BEFORE normalising
    _duo = (film_mode == DUO_TXT)
    if _palora_enc:
        print(f"  palora_enc: M={num_preferences} | schedule={preference_schedule} "
              f"| Q={annealing_temperature} | alpha={lora_alpha} "
              f"| λ-conditioned LoRA encoder + 4 separate heads (encoder rerun per preference)")

    set_seed(seed)
    # do not silently overwrite a DIFFERENT completed run
    # Run names encode most settings, but not all: a rerun with a new
    # prefs_per_batch / anneal_mode / lam_club under an unchanged name would replace
    # finished weights that results already depend on, and the loss would be silent
    # (the old train_meta.json is rewritten too). Fail fast instead; the cost of a
    # false positive is one flag, the cost of a false negative is a lost experiment.
    _meta_path = os.path.join(out_dir, "train_meta.json")
    if os.path.exists(_meta_path) and not overwrite_ok:
        try:
            _old = json.load(open(_meta_path))
        except Exception:
            _old = {}
        _now = dict(film_mode=film_mode, prefs_per_batch=prefs_per_batch,
                    anneal_mode=anneal_mode, lam_club=lam_club, epochs=epochs,
                    seed=seed, proj_dim=proj_dim, method=method, dataset=dataset)
        _diff = {k: (_old.get(k), v) for k, v in _now.items()
                 if k in _old and _old.get(k) != v}
        if _diff:
            raise SystemExit(
                f"REFUSING to overwrite {out_dir}\n"
                f"  it holds a completed run with different settings: "
                + ", ".join(f"{k}: on disk {a!r} vs requested {b!r}"
                            for k, (a, b) in _diff.items())
                + "\n  give the new run its own --out_dir (the launcher adds _m<M> / "
                  "_ann<mode> tags for exactly this), or pass --overwrite_ok.")
    os.makedirs(out_dir, exist_ok=True)
    adim = _actual_enc_dim(approach, enc_dim)

    # Encoder creation (arch-dependent)
    if arch == "comm_based":
        fv, ft = FEAT_DIM[dataset]["vision"], FEAT_DIM[dataset]["text"]
        enc_v = TransformerSeqEncoder(fv).to(device)
        enc_t = TransformerSeqEncoder(ft).to(device)
    else:
        enc_film_mode = enc_film_mode_for(film_mode, approach)
        enc_v, enc_t = _make_encoders(approach, dataset, enc_dim, device, enc_film_mode,
                                      lora_branches=lora_branches, lora_rank=lora_rank,
                                      lora_rank_m=lora_rank_m, freeze_vgg=not enrico_unfreeze,
                                      enc_alpha=(lora_alpha if film_mode in ("palora_enc", "simplex") else None),
                                      plain_encoder=(film_mode == DECOMP_R_PROJ))
        print(f"  film_mode={film_mode}")

    # comm_based: GMC / CoMM with FusionTransformer + MLP head
    # Architecture: TransformerSeqEncoder → FusionTransformer (CLS) → CommMLPHead
    # Vision-only, text-only, and joint representations all pass through the shared
    # FusionTransformer, matching the real CoMM paper (Dufumier et al., ICLR 2025).
    if arch == "comm_based" and method in ("gmc", "comm"):
        COMM_PROJ_DIM = 256
        fusion_xfmr = _make_fusion_xfmr(device)
        mlp_head    = CommMLPHead(adim, 512, COMM_PROJ_DIM).to(device)
        # CoMM-exact optimizer: weight_decay=0 for 1-D params (bias, BN, LN)
        def _wd_groups(modules_lr):
            groups = []
            for mods, lr_val in modules_lr:
                p_wd, p_no_wd = [], []
                for mod in mods:
                    for n, p in mod.named_parameters():
                        if not p.requires_grad:
                            continue
                        if p.ndim < 2 or "bias" in n or "bn" in n or "ln" in n:
                            p_no_wd.append(p)
                        else:
                            p_wd.append(p)
                groups += [{"params": p_wd,    "lr": lr_val, "weight_decay": weight_decay},
                           {"params": p_no_wd, "lr": lr_val, "weight_decay": 0.0}]
            return groups
        opt_enc = optim.AdamW(
            _wd_groups([([enc_v, enc_t], lr), ([fusion_xfmr, mlp_head], lr * head_lr_mult)]),
        )
        # CoMM uses constant LR (no scheduler); GMC keeps cosine decay
        sch_enc = (optim.lr_scheduler.CosineAnnealingLR(opt_enc, T_max=epochs)
                   if method == "gmc" else None)
        writer  = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
        loader  = _ssl_loader(dataset, batch_size)
        all_params = _params(enc_v, enc_t, fusion_xfmr, mlp_head)
        print(f"  method={method} | {dataset} | arch=comm_based | epochs={epochs} | no lambda")
        for epoch in range(1, epochs + 1):
            total = 0.0
            for aug1, aug2 in loader:
                opt_enc.zero_grad()
                if method == "gmc":
                    v_a = aug1[0].float().to(device)
                    t_a = aug1[1].float().to(device)
                    tv  = enc_v(v_a); tt = enc_t(t_a)
                    z_v = mlp_head(fusion_xfmr([tv]))
                    z_t = mlp_head(fusion_xfmr([tt]))
                    z_j = mlp_head(fusion_xfmr([tv, tt]))
                    loss = gmc_loss([z_v, z_t], z_j, temperature)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                else:
                    # CoMM-exact: dual-augmentation, each mod vs other-aug joint
                    v_a = aug1[0].float().to(device); v_b = aug2[0].float().to(device)
                    t_a = aug1[1].float().to(device); t_b = aug2[1].float().to(device)
                    tv_a = enc_v(v_a); tt_a = enc_t(t_a)
                    tv_b = enc_v(v_b); tt_b = enc_t(t_b)
                    z1 = [mlp_head(fusion_xfmr([tv_a])),
                          mlp_head(fusion_xfmr([tt_a])),
                          mlp_head(fusion_xfmr([tv_a, tt_a]))]
                    z2 = [mlp_head(fusion_xfmr([tv_b])),
                          mlp_head(fusion_xfmr([tt_b])),
                          mlp_head(fusion_xfmr([tv_b, tt_b]))]
                    loss = comm_loss_dual(z1, z2, temperature)
                    loss.backward()  # no grad clipping — matches CoMM
                opt_enc.step()
                total += loss.item()
            if sch_enc is not None:
                sch_enc.step()
            avg = total / len(loader)
            writer.add_scalar("loss/total", avg, epoch)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}/{epochs}  loss={avg:.4f}")
        writer.close()
        torch.save(enc_v.state_dict(),       os.path.join(out_dir, "enc_v.pth"))
        torch.save(enc_t.state_dict(),       os.path.join(out_dir, "enc_t.pth"))
        torch.save(fusion_xfmr.state_dict(), os.path.join(out_dir, "fusion_xfmr.pth"))
        torch.save(mlp_head.state_dict(),    os.path.join(out_dir, "mlp_head.pth"))
        with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
            json.dump(dict(approach=approach, method=method, dataset=dataset,
                           enc_dim=adim, proj_dim=COMM_PROJ_DIM, epochs=epochs,
                           seed=seed, film_mode=film_mode, arch=arch), f, indent=2)
        print(f"  Saved → {out_dir}")
        return

    # comm_based: vanilla FactorCL (CondNCE, no CLUB)
    # Encoder: FusionTransformer (comm_based backbone).
    # Heads: ProjectionHead (no BN) — BN + CondNCE causes catastrophic instability at init.
    if arch == "comm_based" and method == "factorcl":
        fusion_xfmr = _make_fusion_xfmr(device)
        proj_v_r = ProjectionHead(adim, proj_dim).to(device)
        proj_t_r = ProjectionHead(adim, proj_dim).to(device)
        proj_v_u = ProjectionHead(adim, proj_dim).to(device)
        proj_t_u = ProjectionHead(adim, proj_dim).to(device)
        opt_enc = optim.AdamW([
            {"params": _params(enc_v, enc_t, fusion_xfmr),                  "lr": lr},
            {"params": _params(proj_v_r, proj_t_r, proj_v_u, proj_t_u),     "lr": lr * head_lr_mult},
        ], weight_decay=weight_decay)
        sch_enc = optim.lr_scheduler.CosineAnnealingLR(opt_enc, T_max=epochs)
        writer  = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
        loader  = _ssl_loader(dataset, batch_size)
        trainable = _params(enc_v, enc_t, fusion_xfmr, proj_v_r, proj_t_r, proj_v_u, proj_t_u)
        print(f"  method=factorcl | {dataset} | arch=comm_based | epochs={epochs} | CondNCE, no CLUB")
        for epoch in range(1, epochs + 1):
            total = 0.0
            for aug1, aug2 in loader:
                v_a = aug1[0].float().to(device); v_b = aug2[0].float().to(device)
                t_a = aug1[1].float().to(device); t_b = aug2[1].float().to(device)
                h_va = fusion_xfmr([enc_v(v_a)]); h_vb = fusion_xfmr([enc_v(v_b)])
                h_ta = fusion_xfmr([enc_t(t_a)]); h_tb = fusion_xfmr([enc_t(t_b)])
                z_vr  = proj_v_r(h_va); z_tr  = proj_t_r(h_ta)
                z_vua = proj_v_u(h_va); z_vub = proj_v_u(h_vb)
                z_tua = proj_t_u(h_ta); z_tub = proj_t_u(h_tb)
                opt_enc.zero_grad()
                loss = factorcl_loss(
                    z_vua, z_vub, z_tua, z_tub,
                    cond1=z_tr.detach(), cond2=z_vr.detach(),
                    z1_r=z_vr, z2_r=z_tr,
                    tau_r=temperature, tau_u=temperature,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt_enc.step()
                total += loss.item()
            sch_enc.step()
            avg = total / len(loader)
            writer.add_scalar("loss/total", avg, epoch)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}/{epochs}  loss={avg:.4f}")
        writer.close()
        torch.save(enc_v.state_dict(),       os.path.join(out_dir, "enc_v.pth"))
        torch.save(enc_t.state_dict(),       os.path.join(out_dir, "enc_t.pth"))
        torch.save(fusion_xfmr.state_dict(), os.path.join(out_dir, "fusion_xfmr.pth"))
        torch.save(proj_v_r.state_dict(),    os.path.join(out_dir, "proj_v_r.pth"))
        torch.save(proj_t_r.state_dict(),    os.path.join(out_dir, "proj_t_r.pth"))
        torch.save(proj_v_u.state_dict(),    os.path.join(out_dir, "proj_v_u.pth"))
        torch.save(proj_t_u.state_dict(),    os.path.join(out_dir, "proj_t_u.pth"))
        with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
            json.dump(dict(approach=approach, method=method, dataset=dataset,
                           enc_dim=adim, proj_dim=proj_dim, epochs=epochs,
                           seed=seed, film_mode=film_mode, arch=arch,
                           head_lr_mult=head_lr_mult), f, indent=2)
        print(f"  Saved → {out_dir}")
        return

    # comm_based: LoRA projection heads (approach 3/4)
    # Architecture: TransformerSeqEncoder → FusionTransformer (CLS, λ-blind) → LoRACommMLPHead
    # film_mode='proj' means the encoder does not see λ; LoRA branches carry all λ information.
    # No BN in heads — avoids BN double-update (forward_r + forward_u in same step) and the
    # BN+CondNCE init instability documented for factorcl comm_based.
    if arch == "comm_based" and approach in (3, 4) and method in ("simclr_single_per_batch", "factorcl_warmup"):
        fusion_xfmr = _make_fusion_xfmr(device)
        HEAD_DIM    = 256
        rm          = lora_rank_m if lora_rank_m is not None else lora_rank
        proj_v = LoRACommMLPHead(adim, 512, HEAD_DIM, rank=lora_rank, rank_m=rm, branches=lora_branches, lora_layers=lora_layers).to(device)
        proj_t = LoRACommMLPHead(adim, 512, HEAD_DIM, rank=lora_rank, rank_m=rm, branches=lora_branches, lora_layers=lora_layers).to(device)

        _ru_keys = ("A_r", "B_r", "A_u", "B_u")
        _m_keys  = ("A_m", "B_m")
        lora_ru_v_params = [p for n, p in proj_v.named_parameters() if any(k in n for k in _ru_keys)]
        lora_ru_t_params = [p for n, p in proj_t.named_parameters() if any(k in n for k in _ru_keys)]
        lora_m_params    = [p for m in (proj_v, proj_t)
                            for n, p in m.named_parameters() if any(k in n for k in _m_keys)]
        base_head_params = [p for m in (proj_v, proj_t)
                            for n, p in m.named_parameters()
                            if not any(k in n for k in _ru_keys + _m_keys)]

        lora_ru_v_lr = lr * head_lr_mult * lora_lr_mult
        lora_ru_t_lr = lr * head_lr_mult * lora_lr_mult * lora_lr_mult_t
        lora_m_lr    = lr * head_lr_mult * lora_lr_mult * lora_lr_mult_m

        if freeze_W0:
            # PaLoRA Pareto-expansion ablation: shared base W0 frozen, only the
            # preference-specific LoRA branches train.
            for _p in base_head_params:
                _p.requires_grad_(False)
            print("  freeze_W0: shared base W0 FROZEN — training only ΔW_R / ΔW_U")
        param_groups = [
            {"params": _params(enc_v, enc_t, fusion_xfmr), "lr": lr},
        ] + ([] if freeze_W0 else [
            {"params": base_head_params, "lr": lr * head_lr_mult}]) + [
            {"params": lora_ru_v_params, "lr": lora_ru_v_lr},
            {"params": lora_ru_t_params, "lr": lora_ru_t_lr},
        ]
        if lora_m_params:
            param_groups.append({"params": lora_m_params, "lr": lora_m_lr})

        trainable = (_params(enc_v, enc_t, fusion_xfmr) + base_head_params +
                     lora_ru_v_params + lora_ru_t_params + lora_m_params)

        opt_enc  = optim.AdamW(param_groups, weight_decay=weight_decay)
        sch_enc  = optim.lr_scheduler.CosineAnnealingLR(opt_enc, T_max=epochs)
        writer   = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
        loader   = _ssl_loader(dataset, batch_size)
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_every = max(1, epochs // 10)

        rm_tag = rm if lora_branches == 3 else lora_rank
        print(f"  arch=comm_based | ap={approach} | {method} | {dataset} | epochs={epochs}"
              f" | lam={lam_dist} | branches={lora_branches} | rank={lora_rank}/rank_m={rm_tag}"
              f" | head_lr_mult={head_lr_mult} | lora_lr_mult={lora_lr_mult}")
        print(f"  Checkpoints every {ckpt_every} epochs → {ckpt_dir}")

        for epoch in range(1, epochs + 1):
            enc_total = LR_total = LU_total = 0.0
            for aug1, aug2 in loader:
                v_a = aug1[0].float().to(device);  v_b = aug2[0].float().to(device)
                t_a = aug1[1].float().to(device);  t_b = aug2[1].float().to(device)
                lam = _sample_lam_mb(lam_dist, dirichlet_alpha, epoch, epochs)
                h_va = fusion_xfmr([enc_v(v_a)]);  h_vb = fusion_xfmr([enc_v(v_b)])
                h_ta = fusion_xfmr([enc_t(t_a)]);  h_tb = fusion_xfmr([enc_t(t_b)])
                if warmup_epochs > 0 and epoch <= warmup_epochs:
                    z_va_r = proj_v.forward_r(h_va);  z_ta_r = proj_t.forward_r(h_ta)
                    z_va_u = proj_v.forward_u(h_va);  z_vb_u = proj_v.forward_u(h_vb)
                    z_ta_u = proj_t.forward_u(h_ta);  z_tb_u = proj_t.forward_u(h_tb)
                    opt_enc.zero_grad()
                    L_R  = infonce_cross(z_va_r, z_ta_r, temperature=temperature)
                    L_Uv = nt_xent(z_va_u, z_vb_u, temperature=temperature)
                    L_Ut = nt_xent(z_ta_u, z_tb_u, temperature=temperature)
                else:
                    z_va = proj_v(h_va, lam);  z_vb = proj_v(h_vb, lam)
                    z_ta = proj_t(h_ta, lam);  z_tb = proj_t(h_tb, lam)
                    opt_enc.zero_grad()
                    L_R  = infonce_cross(z_va, z_ta, temperature=temperature)
                    L_Uv = nt_xent(z_va, z_vb, temperature=temperature)
                    L_Ut = nt_xent(z_ta, z_tb, temperature=temperature)
                    if var_reg_weight > 0:
                        L_var = (F.relu(var_reg_gamma - z_va.std(dim=0)).mean()
                               + F.relu(var_reg_gamma - z_ta.std(dim=0)).mean()) / 2
                        L_Uv = L_Uv + var_reg_weight * L_var
                L_U  = (L_Uv + L_Ut) / 2
                loss = 2 * lam * L_R + 2 * (1.0 - lam) * L_U
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt_enc.step()
                enc_total += loss.item();  LR_total += L_R.item();  LU_total += L_U.item()

            sch_enc.step()
            n = len(loader)
            writer.add_scalar("loss/total", enc_total / n, epoch)
            writer.add_scalar("loss/L_R",   LR_total  / n, epoch)
            writer.add_scalar("loss/L_U",   LU_total  / n, epoch)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}/{epochs}  total={enc_total/n:.4f}  L_R={LR_total/n:.4f}  L_U={LU_total/n:.4f}")

            if epoch % ckpt_every == 0 or epoch == epochs:
                ep_dir = os.path.join(ckpt_dir, f"ep{epoch:04d}")
                os.makedirs(ep_dir, exist_ok=True)
                torch.save(enc_v.state_dict(),       os.path.join(ep_dir, "enc_v.pth"))
                torch.save(enc_t.state_dict(),       os.path.join(ep_dir, "enc_t.pth"))
                torch.save(fusion_xfmr.state_dict(), os.path.join(ep_dir, "fusion_xfmr.pth"))
                torch.save(proj_v.state_dict(),      os.path.join(ep_dir, "proj_v.pth"))
                torch.save(proj_t.state_dict(),      os.path.join(ep_dir, "proj_t.pth"))

        writer.close()
        torch.save(enc_v.state_dict(),       os.path.join(out_dir, "enc_v.pth"))
        torch.save(enc_t.state_dict(),       os.path.join(out_dir, "enc_t.pth"))
        torch.save(fusion_xfmr.state_dict(), os.path.join(out_dir, "fusion_xfmr.pth"))
        torch.save(proj_v.state_dict(),      os.path.join(out_dir, "proj_v.pth"))
        torch.save(proj_t.state_dict(),      os.path.join(out_dir, "proj_t.pth"))
        with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
            json.dump(dict(approach=approach, method=method, dataset=dataset,
                           enc_dim=adim, proj_dim=HEAD_DIM, epochs=epochs,
                           lam_dist=lam_dist, seed=seed, film_mode=film_mode, arch=arch,
                           head_lr_mult=head_lr_mult, lora_branches=lora_branches,
                           lora_rank=lora_rank, lora_rank_m=rm,
                           lora_lr_mult=lora_lr_mult, lora_lr_mult_m=lora_lr_mult_m,
                           lora_lr_mult_t=lora_lr_mult_t,
                           var_reg_weight=var_reg_weight, var_reg_gamma=var_reg_gamma,
                           warmup_epochs=warmup_epochs, lora_layers=lora_layers,
                           simplex_side=simplex_side, lora_alpha=lora_alpha,
                           enc_width=ENC_DIM_XFMR,
                           rank_penalty=rank_penalty, rank_margin=rank_margin), f, indent=2)
        print(f"  Saved → {out_dir}")
        return

    # comm_based: SimCLR / FactorCL-warmup (lambda-weighted, same FusionTransformer)
    # SimCLR uses InfoNCE → CommMLPHead (BN) is fine.
    # factorcl_warmup uses CondNCE → must use ProjectionHead (no BN) to avoid init instability.
    if arch == "comm_based" and method in ("simclr_single_per_batch", "factorcl_warmup"):
        fusion_xfmr = _make_fusion_xfmr(device)
        use_club = (method == "factorcl_warmup")
        if use_club:
            # CondNCE heads: no BN
            HEAD_DIM = proj_dim
            proj_v_r = ProjectionHead(adim, HEAD_DIM).to(device)
            proj_t_r = ProjectionHead(adim, HEAD_DIM).to(device)
            proj_v_u = ProjectionHead(adim, HEAD_DIM).to(device)
            proj_t_u = ProjectionHead(adim, HEAD_DIM).to(device)
            club_v = CLUBInfoNCECritic(HEAD_DIM, HEAD_DIM,
                                       hidden_dim=club_hidden_dim, layers=club_layers,
                                       activation="relu").to(device)
            club_t = CLUBInfoNCECritic(HEAD_DIM, HEAD_DIM,
                                       hidden_dim=club_hidden_dim, layers=club_layers,
                                       activation="relu").to(device)
        else:
            # SimCLR: InfoNCE → CommMLPHead (BN) is standard and safe
            HEAD_DIM = 256
            proj_v_r = CommMLPHead(adim, 512, HEAD_DIM).to(device)
            proj_t_r = CommMLPHead(adim, 512, HEAD_DIM).to(device)
            proj_v_u = CommMLPHead(adim, 512, HEAD_DIM).to(device)
            proj_t_u = CommMLPHead(adim, 512, HEAD_DIM).to(device)
            opt_critic = optim.AdamW(_params(club_v, club_t), lr=lr, weight_decay=weight_decay)
            sch_critic = optim.lr_scheduler.CosineAnnealingLR(opt_critic, T_max=epochs)
        opt_enc = optim.AdamW([
            {"params": _params(enc_v, enc_t, fusion_xfmr),              "lr": lr},
            {"params": _params(proj_v_r, proj_t_r, proj_v_u, proj_t_u), "lr": lr * head_lr_mult},
        ], weight_decay=weight_decay)
        sch_enc   = optim.lr_scheduler.CosineAnnealingLR(opt_enc, T_max=epochs)
        writer    = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
        loader    = _ssl_loader(dataset, batch_size)
        trainable = _params(enc_v, enc_t, fusion_xfmr, proj_v_r, proj_t_r, proj_v_u, proj_t_u)
        print(f"  arch=comm_based | {method} | {dataset} | epochs={epochs} | lam={lam_dist} | head_lr_mult={head_lr_mult}")
        for epoch in range(1, epochs + 1):
            enc_total = crit_total = LR_total = LU_total = 0.0
            for aug1, aug2 in loader:
                v_a = aug1[0].float().to(device); v_b = aug2[0].float().to(device)
                t_a = aug1[1].float().to(device); t_b = aug2[1].float().to(device)
                lam = _sample_lam_mb(lam_dist, dirichlet_alpha, epoch, epochs)
                h_va = fusion_xfmr([enc_v(v_a)]); h_vb = fusion_xfmr([enc_v(v_b)])
                h_ta = fusion_xfmr([enc_t(t_a)]); h_tb = fusion_xfmr([enc_t(t_b)])
                z_vr  = proj_v_r(h_va); z_tr  = proj_t_r(h_ta)
                z_vua = proj_v_u(h_va); z_vub = proj_v_u(h_vb)
                z_tua = proj_t_u(h_ta); z_tub = proj_t_u(h_tb)
                if use_club:
                    opt_critic.zero_grad()
                    crit_loss = (
                        club_v.learning_loss(z_vua.detach(), z_vr.detach()) +
                        club_t.learning_loss(z_tua.detach(), z_tr.detach())
                    )
                    crit_loss.backward()
                    torch.nn.utils.clip_grad_norm_(_params(club_v, club_t), 1.0)
                    opt_critic.step()
                    crit_total += crit_loss.item()
                opt_enc.zero_grad()
                L_R  = infonce_cross(z_vr, z_tr, temperature=temperature)
                L_Uv = nt_xent(z_vua, z_vub, temperature=temperature)
                L_Ut = nt_xent(z_tua, z_tub, temperature=temperature)
                if use_club:
                    L_Uv = L_Uv + lam_club * torch.clamp(club_v(z_vua, z_vr.detach()), min=0.0)
                    L_Ut = L_Ut + lam_club * torch.clamp(club_t(z_tua, z_tr.detach()), min=0.0)
                L_U  = (L_Uv + L_Ut) / 2
                loss = 2 * lam * L_R + 2 * (1.0 - lam) * L_U
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt_enc.step()
                enc_total += loss.item(); LR_total += L_R.item(); LU_total += L_U.item()
            sch_enc.step()
            if use_club:
                sch_critic.step()
            n = len(loader)
            writer.add_scalar("loss/total", enc_total / n, epoch)
            writer.add_scalar("loss/L_R",   LR_total  / n, epoch)
            writer.add_scalar("loss/L_U",   LU_total  / n, epoch)
            if use_club:
                writer.add_scalar("loss/critic", crit_total / n, epoch)
            if epoch % 10 == 0 or epoch == 1:
                cstr = f"  crit={crit_total/n:.4f}" if use_club else ""
                print(f"  epoch {epoch:3d}/{epochs}  total={enc_total/n:.4f}  L_R={LR_total/n:.4f}  L_U={LU_total/n:.4f}{cstr}")
        writer.close()
        torch.save(enc_v.state_dict(),       os.path.join(out_dir, "enc_v.pth"))
        torch.save(enc_t.state_dict(),       os.path.join(out_dir, "enc_t.pth"))
        torch.save(fusion_xfmr.state_dict(), os.path.join(out_dir, "fusion_xfmr.pth"))
        for name, mod in [("proj_v_r", proj_v_r), ("proj_t_r", proj_t_r),
                          ("proj_v_u", proj_v_u), ("proj_t_u", proj_t_u)]:
            torch.save(mod.state_dict(), os.path.join(out_dir, f"{name}.pth"))
        with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
            json.dump(dict(approach=approach, method=method, dataset=dataset,
                           enc_dim=adim, proj_dim=HEAD_DIM, epochs=epochs,
                           lam_dist=lam_dist, seed=seed, film_mode=film_mode, arch=arch,
                           head_lr_mult=head_lr_mult), f, indent=2)
        print(f"  Saved → {out_dir}")
        return

    # GMC / CoMM baselines: joint encoder, no lambda, no projection heads
    is_joint = method in ("gmc", "comm")
    if is_joint:
        # fusion_hidden / fusion_out let CoMM+GMC be given a head budget matching
        # the lambda-arms; both default to the original architecture.
        _f_out = fusion_out or adim
        fusion = FusionEncoder(adim, _f_out, hidden=fusion_hidden).to(device)
        # gmc_loss / comm_loss contrast each UNIMODAL embedding against the JOINT
        # one, so both must live in the same space. --fusion_out widened only the
        # joint side, which raised
        #   "Expected size 64 but got size 40 for tensor number 1".
        # GMC (Poklukar et al.) maps every modality through a SHARED projection into
        # the common space before the contrastive term; replicate that. It also
        # removes a standing asymmetry -- STEER has ProjectionHead(40->64) and
        # FactorCL ProjectionHead(40->D_out), while gmc/comm alone read straight off
        # the encoder. None when the widths already agree, so every existing
        # fusion.pth reproduces bit-identically.
        mod_proj = nn.Linear(adim, _f_out).to(device) if _f_out != adim else None
        print(f"  fusion head: in={2*adim} hidden={fusion_hidden} out={_f_out} "
              f"params={sum(p.numel() for p in fusion.parameters()):,}"
              + (f" | shared mod_proj {adim}->{_f_out}" if mod_proj is not None
                 else " | mod_proj: none (widths agree)"))
        _head_mods = [fusion] if mod_proj is None else [fusion, mod_proj]
        opt_enc = optim.AdamW([
            {"params": _params(enc_v, enc_t),  "lr": lr},
            {"params": _params(*_head_mods),   "lr": lr * head_lr_mult},
        ], weight_decay=weight_decay)
        sch_enc = optim.lr_scheduler.CosineAnnealingLR(opt_enc, T_max=epochs)
        writer  = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
        loader  = _ssl_loader(dataset, batch_size)
        if loss_w is not None and len(loss_w) != 2:
            raise ValueError(f"gmc/comm --loss_w needs 2 weights (vision,text), got {loss_w}")
        print(f"  method={method} | {dataset} | epochs={epochs} | no lambda"
              + (f" | loss_w(v,t)={loss_w}" if loss_w else ""))
        for epoch in range(1, epochs + 1):
            total = 0.0
            for aug1, aug2 in loader:
                v_a = aug1[0].float().to(device)
                t_a = aug1[1].float().to(device)
                z_v = enc_v(v_a); z_t = enc_t(t_a)
                # fusion still consumes the RAW encoder outputs (in_dim=adim); only
                # the unimodal terms of the contrastive loss are lifted to _f_out.
                z_j = fusion(z_v, z_t)
                if mod_proj is not None:
                    z_v = F.normalize(mod_proj(z_v), dim=-1)
                    z_t = F.normalize(mod_proj(z_t), dim=-1)
                opt_enc.zero_grad()
                if method == "gmc":
                    loss = gmc_loss([z_v, z_t], z_j, temperature, weights=loss_w)
                else:
                    loss = comm_loss([z_v, z_t], z_j, temperature, weights=loss_w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_params(enc_v, enc_t, *_head_mods), 1.0)
                opt_enc.step()
                total += loss.item()
            sch_enc.step()
            avg = total / len(loader)
            writer.add_scalar("loss/total", avg, epoch)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}/{epochs}  loss={avg:.4f}")
        writer.close()
        torch.save(enc_v.state_dict(),  os.path.join(out_dir, "enc_v.pth"))
        torch.save(enc_t.state_dict(),  os.path.join(out_dir, "enc_t.pth"))
        torch.save(fusion.state_dict(), os.path.join(out_dir, "fusion.pth"))
        if mod_proj is not None:   # training-only; the probe reads the joint head
            torch.save(mod_proj.state_dict(), os.path.join(out_dir, "mod_proj.pth"))
        with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
            json.dump(dict(approach=approach, method=method, dataset=dataset,
                           enc_dim=adim, proj_dim=proj_dim, epochs=epochs,
                           seed=seed, film_mode=film_mode,
                           fusion_hidden=fusion_hidden, fusion_out=_f_out,
                           loss_w=loss_w,
                           mod_proj=(None if mod_proj is None else _f_out)), f, indent=2)
        print(f"  Saved → {out_dir}")
        return

    # Vanilla FactorCL (Liang et al., NeurIPS 2023)
    # 6-loss formulation matching the original FactorCLSSL code:
    #   InfoNCE(v, t)                       — cross-modal shared alignment
    #   CLUB(v, t)                          — minimise cross-modal MI (ensure uniqueness)
    #   InfoNCE(v, v_aug)                   — within-vision unique alignment
    #   InfoNCE(t, t_aug)                   — within-text unique alignment
    #   InfoNCE([v,v_aug], [t,t_aug])       — conditional cross-modal alignment
    #   CLUB([v,v_aug], [t,t_aug])          — conditional CLUB
    # Each term uses a learned neural critic (concat MLP); CLUB critics trained
    # alternately from encoder/head updates.
    # CLIP / Cross+Self baselines (factorcl_based arch)
    # CLIP       : symmetric cross-modal InfoNCE only  -- FactorCL's "SimCLR" baseline
    # cross_self : + a unimodal (Xi, Xi') InfoNCE per modality, weighted by ssl_scale
    # Both read out as concat(proj_v(h_v), proj_t(h_t)) = 2*proj_dim, so --proj_dim 32
    # gives the 64-dim width every other arm is held to.
    if method in ("clip", "cross_self"):
        proj_v = ProjectionHead(adim, proj_dim).to(device)
        proj_t = ProjectionHead(adim, proj_dim).to(device)
        # CLIP's learned temperature. Reference init is log(1/0.07) and CLIP excludes
        # it from weight decay, so it gets its own zero-decay parameter group.
        _use_ls = bool(clip_logit_scale) and method == "clip"
        logit_scale = (nn.Parameter(torch.tensor(math.log(1.0 / 0.07), device=device))
                       if _use_ls else None)
        _groups = [
            {"params": _params(enc_v, enc_t),   "lr": lr},
            {"params": _params(proj_v, proj_t), "lr": lr * head_lr_mult},
        ]
        if logit_scale is not None:
            _groups.append({"params": [logit_scale], "lr": lr, "weight_decay": 0.0})
        opt_enc = optim.AdamW(_groups, weight_decay=weight_decay)
        sch_enc = optim.lr_scheduler.CosineAnnealingLR(opt_enc, T_max=epochs)
        writer  = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
        loader  = _ssl_loader(dataset, batch_size)
        trainable = _params(enc_v, enc_t, proj_v, proj_t) + (
            [logit_scale] if logit_scale is not None else [])
        print(f"  method={method} | {dataset} | epochs={epochs} | readout 2*{proj_dim}"
              + (f" | ssl_scale={ssl_scale}" if method == "cross_self" else "")
              + (" | LEARNED logit_scale (init 1/0.07)" if logit_scale is not None
                 else f" | fixed temperature {temperature}"))
        for epoch in range(1, epochs + 1):
            total = 0.0
            for aug1, aug2 in loader:
                v_a = aug1[0].float().to(device); v_b = aug2[0].float().to(device)
                t_a = aug1[1].float().to(device); t_b = aug2[1].float().to(device)
                z_va = proj_v(enc_v(v_a)); z_ta = proj_t(enc_t(t_a))
                opt_enc.zero_grad()
                if method == "clip":
                    loss = (clip_loss(z_va, z_ta, logit_scale) if logit_scale is not None
                            else infonce_cross(z_va, z_ta, temperature))
                else:
                    z_vb = proj_v(enc_v(v_b)); z_tb = proj_t(enc_t(t_b))
                    loss = cross_self_loss(z_va, z_vb, z_ta, z_tb, temperature, ssl_scale)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt_enc.step()
                total += loss.item()
            sch_enc.step()
            avg = total / len(loader)
            writer.add_scalar("loss/total", avg, epoch)
            if logit_scale is not None:
                writer.add_scalar("clip/logit_scale",
                                  float(logit_scale.exp().clamp(max=100.0)), epoch)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}/{epochs}  loss={avg:.4f}"
                      + (f"  logit_scale={float(logit_scale.exp().clamp(max=100.0)):.2f}"
                         if logit_scale is not None else ""))
        writer.close()
        torch.save(enc_v.state_dict(),  os.path.join(out_dir, "enc_v.pth"))
        torch.save(enc_t.state_dict(),  os.path.join(out_dir, "enc_t.pth"))
        torch.save(proj_v.state_dict(), os.path.join(out_dir, "proj_v.pth"))
        torch.save(proj_t.state_dict(), os.path.join(out_dir, "proj_t.pth"))
        with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
            json.dump(dict(approach=approach, method=method, dataset=dataset,
                           enc_dim=adim, proj_dim=proj_dim, epochs=epochs, seed=seed,
                           film_mode=film_mode, arch=arch,
                           embed_dim=2 * proj_dim,
                           ssl_scale=(ssl_scale if method == "cross_self" else None),
                           clip_logit_scale=(float(logit_scale.exp().clamp(max=100.0))
                                             if logit_scale is not None else None)),
                      f, indent=2)
        print(f"  Saved → {out_dir}")
        return

    if method == "factorcl":
        D = adim  # head INPUT width = encoder width (40)
        # Head OUTPUT width. Historically hardcoded to adim, which made FactorCL's
        # probe-time z = concat(proj_v_r, proj_t_r) 2*40 = 80 dims -- 2x the 40-dim
        # GMC/CoMM fusion output and 1.25x the 64-dim lambda-arms. --proj_dim was
        # plumbed through the launcher but silently ignored here. Pass --proj_dim 32
        # for a 64-dim width-matched FactorCL. Omitting it reproduces the old runs.
        D_out = adim if proj_dim in (None, 0, DEFAULT_PROJ_DIM) else int(proj_dim)
        # Official get_embedding() concatenates 5 heads per modality -> 10*D_out.
        # --fusion_out adds a projection head compressing that to the common
        # comparison width, trained by an SSL objective exactly like every other
        # method's head (GMC/CoMM's fusion, STEER's ProjectionHeads) rather than
        # by a post-hoc PCA. It is an ADDITION to FactorCL, not part of the paper:
        # the six published losses act on individual head outputs and give a
        # post-concatenation head no gradient, so it needs its own term.
        # --factorcl_official controls the TRAINING RECIPE (Adam, lr 1e-4, no decay,
        # no clip, no scheduler, main-then-critic, unnormalised heads). It does NOT
        # pin the width: the official default is mlp_head(d, d) with d = adim = 40,
        # giving a 400-dim get_embedding(), but --proj_dim 7 keeps that recipe while
        # matching the 64-70 dim budget every other method is held to. Conflating the
        # two would make the faithful arm and the width-matched arm mutually exclusive.
        _OFF = bool(factorcl_official)
        if _OFF and not proj_dim_explicit:
            # Official default is mlp_head(d, d) with d = encoder width.
            D_out = adim
        _PH = lambda i, o: ProjectionHead(i, o, normalize=not _OFF).to(device)
        _emb_dim = 10 * D_out
        head_out = (_PH(_emb_dim, int(fusion_out)) if fusion_out else None)
        print(f"  factorcl heads: in={D} out={D_out} | get_embedding width = 10*{D_out} = {_emb_dim}"
              + (f" | compression head {_emb_dim}->{int(fusion_out)} (NT-Xent)"
                 if head_out is not None else ""))
        # Projection heads: one 2-layer MLP per critic per modality (no BN)
        head_r_v   = _PH(D, D_out)   # cross-modal InfoNCE
        head_r_t   = _PH(D, D_out)
        head_cl_v  = _PH(D, D_out)   # cross-modal CLUB
        head_cl_t  = _PH(D, D_out)
        head_u_v   = _PH(D, D_out)   # within-vision InfoNCE
        head_u_t   = _PH(D, D_out)   # within-text InfoNCE
        head_cr_v  = _PH(D, D_out)   # conditional InfoNCE
        head_cr_t  = _PH(D, D_out)
        head_ccl_v = _PH(D, D_out)   # conditional CLUB
        head_ccl_t = _PH(D, D_out)
        # Neural critics (concat MLP, input = A_dim + B_dim → scalar)
        critic_r    = InfoNCECritic(D_out,   D_out,   club_hidden_dim, club_layers, "relu").to(device)
        critic_cl   = CLUBInfoNCECritic(D_out,   D_out,   club_hidden_dim, club_layers, "relu").to(device)
        critic_u_v  = InfoNCECritic(D_out,   D_out,   club_hidden_dim, club_layers, "relu").to(device)
        critic_u_t  = InfoNCECritic(D_out,   D_out,   club_hidden_dim, club_layers, "relu").to(device)
        critic_cond = InfoNCECritic(D_out*2, D_out*2, club_hidden_dim, club_layers, "relu").to(device)
        critic_ccl  = CLUBInfoNCECritic(D_out*2, D_out*2, club_hidden_dim, club_layers, "relu").to(device)
        # Separate optimisers: CLUB critics alternate from encoder + InfoNCE critics
        non_club_ps = _params(enc_v, enc_t,
                              head_r_v, head_r_t, head_cl_v, head_cl_t,
                              head_u_v, head_u_t, head_cr_v, head_cr_t, head_ccl_v, head_ccl_t,
                              critic_r, critic_u_v, critic_u_t, critic_cond,
                              *( [head_out] if head_out is not None else [] ))
        club_ps     = _params(critic_cl, critic_ccl)
        # Official: plain Adam, lr 1e-4, NO weight decay, NO scheduler, NO grad clip.
        # AdamW's decay pulls every critic weight toward 0, and the constant critic is
        # exactly the degenerate solution we observe -- so the decay is a prime suspect
        # and must be off in the faithful arm. A single Adam over a group is
        # mathematically identical to the official per-module Adam list (Adam state is
        # per-parameter), so that difference is cosmetic and not reproduced.
        _opt = optim.Adam if _OFF else optim.AdamW
        _lr  = 1e-4 if _OFF else lr
        _wd  = 0.0  if _OFF else weight_decay
        opt_enc    = _opt(non_club_ps, lr=_lr, weight_decay=_wd)
        opt_critic = _opt(club_ps, lr=_lr * critic_lr_mult, weight_decay=_wd)
        if loss_w is not None and len(loss_w) != 3:
            raise ValueError(f"factorcl --loss_w needs 3 weights (R,U_v,U_t), got {loss_w}")
        if loss_w is not None:
            print(f"  loss_w(R,U_v,U_t)={loss_w}  -> r/ccl x{loss_w[0]}, u_v x{loss_w[1]}, "
                  f"u_t x{loss_w[2]}, cl/cond x{0.5*(loss_w[1]+loss_w[2]):.3f}")
        if _OFF:
            print(f"  [OFFICIAL] Adam lr={_lr} wd=0 | no scheduler | no grad clip | "
                  f"main-then-critic | heads {adim}->{D_out} unnormalised "
                  f"| get_embedding {_emb_dim}d")
        if club_iters != 1 or critic_lr_mult != 1.0:
            print(f"  [DIAGNOSTIC] club_iters={club_iters} critic_lr_mult={critic_lr_mult} "
                  f"-- NOT the official FactorCL setup (1 iter, same LR)")
        sch_enc    = None if _OFF else optim.lr_scheduler.CosineAnnealingLR(opt_enc,    T_max=epochs)
        sch_critic = None if _OFF else optim.lr_scheduler.CosineAnnealingLR(opt_critic, T_max=epochs)
        writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
        loader = _ssl_loader(dataset, batch_size)
        print(f"  method=factorcl | {dataset} | epochs={epochs} | 6-loss InfoNCE+CLUB (original)")
        _epoch_diag = None

        def _fwd(v_a, v_b, t_a, t_b):
            """One full forward: all ten head outputs plus the two conditional pairs."""
            h_va, h_vb = enc_v(v_a), enc_v(v_b)
            h_ta, h_tb = enc_t(t_a), enc_t(t_b)
            z = dict(
                r_v=head_r_v(h_va),   r_t=head_r_t(h_ta),
                cl_v=head_cl_v(h_va), cl_t=head_cl_t(h_ta),
                u_va=head_u_v(h_va),  u_vb=head_u_v(h_vb),
                u_ta=head_u_t(h_ta),  u_tb=head_u_t(h_tb),
                cr_va=head_cr_v(h_va),  cr_vb=head_cr_v(h_vb),
                cr_ta=head_cr_t(h_ta),  cr_tb=head_cr_t(h_tb),
                cc_va=head_ccl_v(h_va), cc_vb=head_ccl_v(h_vb),
                cc_ta=head_ccl_t(h_ta), cc_tb=head_ccl_t(h_tb),
            )
            # Conditional terms condition by concatenation: (B, D*2)
            z["cr_v"]  = torch.cat([z["cr_va"], z["cr_vb"]], dim=-1)
            z["cr_t"]  = torch.cat([z["cr_ta"], z["cr_tb"]], dim=-1)
            z["ccl_v"] = torch.cat([z["cc_va"], z["cc_vb"]], dim=-1)
            z["ccl_t"] = torch.cat([z["cc_ta"], z["cc_tb"]], dim=-1)
            return z

        def _main_step(z):
            opt_enc.zero_grad()
            # Eq.8  S  = I_NCE(x1;x2) - I_NCE-CLUB(x1;x2|x')      -> r, ccl
            # Eq.9  Ui = I_NCE(xi;xi') - I_NCE-CLUB(x1;x2)
            #                          + I_NCE(x1;x2|x')            -> u_i, and cl/cr
            # cl and cr appear in BOTH U terms, so they carry (w_U1 + w_U2)/2.
            w_r, w_u1, w_u2 = (1.0, 1.0, 1.0) if loss_w is None else loss_w
            w_sh = 0.5 * (w_u1 + w_u2) if loss_w is not None else 1.0
            loss = (
                w_r  * critic_r(z["r_v"], z["r_t"])       # InfoNCE(v, t)          -> R
                + w_sh * critic_cl(z["cl_v"], z["cl_t"])  # CLUB(v, t)             -> U1,U2
                + w_u1 * critic_u_v(z["u_va"], z["u_vb"])  # InfoNCE(v, v_aug)     -> U1
                + w_u2 * critic_u_t(z["u_ta"], z["u_tb"])  # InfoNCE(t, t_aug)     -> U2
                + w_sh * critic_cond(z["cr_v"], z["cr_t"])  # cond InfoNCE         -> U1,U2
                + w_r  * critic_ccl(z["ccl_v"], z["ccl_t"])  # cond CLUB           -> R
            )
            if head_out is not None:
                _za = torch.cat([z["r_v"], z["cl_v"], z["u_va"], z["cr_va"], z["cc_va"],
                                 z["r_t"], z["cl_t"], z["u_ta"], z["cr_ta"], z["cc_ta"]], dim=-1)
                _zb = torch.cat([z["r_v"], z["cl_v"], z["u_vb"], z["cr_vb"], z["cc_vb"],
                                 z["r_t"], z["cl_t"], z["u_tb"], z["cr_tb"], z["cc_tb"]], dim=-1)
                loss = loss + nt_xent(head_out(_za), head_out(_zb), temperature)
            loss.backward()
            if not _OFF:                      # official clips nothing
                torch.nn.utils.clip_grad_norm_(non_club_ps, 1.0)
            opt_enc.step()
            return float(loss.item())

        def _critic_step(z):
            """CLUB critics, inputs detached. Returns (summed learn_loss, grad norm)."""
            _c, _g = 0.0, 0.0
            for _ci in range(club_iters):
                opt_critic.zero_grad()
                learn_loss = (
                    critic_cl.learning_loss(z["cl_v"].detach(), z["cl_t"].detach()) +
                    critic_ccl.learning_loss(z["ccl_v"].detach(), z["ccl_t"].detach())
                )
                learn_loss.backward()
                if _OFF:
                    _g = float(torch.sqrt(sum((q.grad ** 2).sum()
                               for q in club_ps if q.grad is not None)))
                else:
                    # clip_grad_norm_ returns the PRE-clip norm, so the diagnostic is free
                    _g = float(torch.nn.utils.clip_grad_norm_(club_ps, 1.0))
                opt_critic.step()
                _c += learn_loss.item() / club_iters
            return _c, _g

        for epoch in range(1, epochs + 1):
            total = crit_total = 0.0
            for batch_idx, (aug1, aug2) in enumerate(loader):
                v_a = aug1[0].float().to(device);  v_b = aug2[0].float().to(device)
                t_a = aug1[1].float().to(device);  t_b = aug2[1].float().to(device)
                _p_before = ([q.detach().clone() for q in club_ps]
                             if batch_idx == 0 else None)
                if _OFF:
                    # Official ordering: main loss first, then the CLUB critics on
                    # FRESHLY recomputed embeddings -- model.learning_loss() re-runs the
                    # backbones, so the critic sees the POST-update encoder, not the
                    # pre-update one our original ordering handed it.
                    total += _main_step(_fwd(v_a, v_b, t_a, t_b))
                    _z = _fwd(v_a, v_b, t_a, t_b)
                    _c, _gnorm = _critic_step(_z)
                    crit_total += _c
                else:
                    _z = _fwd(v_a, v_b, t_a, t_b)
                    _c, _gnorm = _critic_step(_z)
                    crit_total += _c
                    total += _main_step(_z)
                if _p_before is not None:
                    _dp = float(torch.sqrt(sum(((q - r) ** 2).sum()
                                for q, r in zip(club_ps, _p_before))))
                    _cd = _club_diagnostics(critic_cl,  _z["cl_v"].detach(),  _z["cl_t"].detach())
                    _cc = _club_diagnostics(critic_ccl, _z["ccl_v"].detach(), _z["ccl_t"].detach())
                    _epoch_diag = dict(grad_norm=_gnorm, param_delta=_dp, cl=_cd, ccl=_cc)
            if sch_enc is not None:
                sch_enc.step(); sch_critic.step()
            avg = total / len(loader)
            writer.add_scalar("loss/total",  avg,                      epoch)
            writer.add_scalar("loss/critic", crit_total / len(loader), epoch)
            if _epoch_diag is not None:
                writer.add_scalar("club/grad_norm",   _epoch_diag["grad_norm"],   epoch)
                writer.add_scalar("club/param_delta", _epoch_diag["param_delta"], epoch)
                for _nm in ("cl", "ccl"):
                    for _k, _v in _epoch_diag[_nm].items():
                        writer.add_scalar(f"club_{_nm}/{_k}", _v, epoch)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}/{epochs}  loss={avg:.4f}  crit={crit_total/len(loader):.4f}")
        writer.close()
        torch.save(enc_v.state_dict(),      os.path.join(out_dir, "enc_v.pth"))
        torch.save(enc_t.state_dict(),      os.path.join(out_dir, "enc_t.pth"))
        torch.save(head_r_v.state_dict(),   os.path.join(out_dir, "proj_v_r.pth"))
        torch.save(head_r_t.state_dict(),   os.path.join(out_dir, "proj_t_r.pth"))
        torch.save(head_u_v.state_dict(),   os.path.join(out_dir, "proj_v_u.pth"))
        torch.save(head_u_t.state_dict(),   os.path.join(out_dir, "proj_t_u.pth"))
        # The official FactorCLSSL.get_embedding() concatenates FIVE heads per
        # modality -- infonce_x1x2, club_x1x2, infonce_x{1,2}y, infonce_x1x2_cond,
        # club_x1x2_cond -- not just the shared pair. Persisting only 4 of the 10
        # trained heads meant the probe could evaluate the SHARED representation
        # only, discarding the unique and conditional components that are the
        # paper's actual contribution. Save all ten.
        torch.save(head_cl_v.state_dict(),  os.path.join(out_dir, "proj_v_cl.pth"))
        torch.save(head_cl_t.state_dict(),  os.path.join(out_dir, "proj_t_cl.pth"))
        torch.save(head_cr_v.state_dict(),  os.path.join(out_dir, "proj_v_cr.pth"))
        torch.save(head_cr_t.state_dict(),  os.path.join(out_dir, "proj_t_cr.pth"))
        torch.save(head_ccl_v.state_dict(), os.path.join(out_dir, "proj_v_ccl.pth"))
        torch.save(head_ccl_t.state_dict(), os.path.join(out_dir, "proj_t_ccl.pth"))
        if head_out is not None:
            torch.save(head_out.state_dict(), os.path.join(out_dir, "proj_out.pth"))
        with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
            json.dump(dict(approach=approach, method=method, dataset=dataset,
                           enc_dim=adim, proj_dim=D_out, epochs=epochs,
                           seed=seed, film_mode=film_mode,
                           factorcl_heads=10, embed_dim=_emb_dim,
                           factorcl_out=(int(fusion_out) if head_out else None),
                           factorcl_official=_OFF, head_no_norm=_OFF,
                           loss_w=loss_w,
                           optimizer=("adam" if _OFF else "adamw"),
                           lr=_lr, weight_decay=_wd,
                           grad_clip=(None if _OFF else 1.0),
                           lr_scheduler=(None if _OFF else "cosine"),
                           step_order=("main_then_critic" if _OFF else "critic_then_main")),
                          f, indent=2)
        print(f"  Saved → {out_dir}")
        return

    # approaches 3/4: one dual projection head per modality with structural branch separation
    # approach 1/2: 4 static heads — CLUB only for factorcl_warmup
    # Head handles — initialised so downstream logging can test `is not None`
    # regardless of which branch below actually builds them.
    proj_v = proj_t = None
    proj_v_r = proj_t_r = proj_v_u = proj_t_u = None

    if approach == 3:
        use_club = False
        _film_head_cls = DualFiLMProjectionHeadPreNorm if proj_prenorm else DualFiLMProjectionHead
        proj_v = _film_head_cls(adim, proj_dim).to(device)
        proj_t = _film_head_cls(adim, proj_dim).to(device)
        head_params = _params(proj_v, proj_t)
    elif approach == 5 and film_mode == "palora_enc":
        # palora_enc: λ-conditioned LoRA ENCODER + FOUR SEPARATE projection heads.
        # Removes BOTH shared bottlenecks at once — h itself is λ-specific, and the
        # R/U heads share no base weights. λ mixes the head OUTPUTS (as in ap2).
        use_club = False
        proj_v_r = ProjectionHead(adim, proj_dim).to(device)
        proj_t_r = ProjectionHead(adim, proj_dim).to(device)
        proj_v_u = ProjectionHead(adim, proj_dim).to(device)
        proj_t_u = ProjectionHead(adim, proj_dim).to(device)
        proj_v = proj_t = None
        head_params = _params(proj_v_r, proj_t_r, proj_v_u, proj_t_u)
    elif approach == 5 and film_mode in DECOMP_MODES:
        # FOUR heads, one objective each.
        #   DECOMP_R      : plain heads -- all preference-dependence is in the encoder.
        #   DECOMP_R_PROJ : LoRA heads  -- the encoder is lambda-blind and the
        #                   preference acts here instead, via forward_mix(h, w_r, w_u).
        #                   Each modality's heads see only their own unique weight:
        #                   vision gets (w_R, w_U_vision), text gets (w_R, w_U_text).
        use_club = True
        if _head_cond:
            def _mkhead():
                return LoRADualProjectionHead(adim, proj_dim, rank=lora_rank,
                                              alpha=lora_alpha).to(device)
            proj_v_r, proj_t_r = _mkhead(), _mkhead()
            proj_v_u, proj_t_u = _mkhead(), _mkhead()
        else:
            proj_v_r = ProjectionHead(adim, proj_dim).to(device)
            proj_t_r = ProjectionHead(adim, proj_dim).to(device)
            proj_v_u = ProjectionHead(adim, proj_dim).to(device)
            proj_t_u = ProjectionHead(adim, proj_dim).to(device)
        proj_v = proj_t = None
        head_params = _params(proj_v_r, proj_t_r, proj_v_u, proj_t_u)
        # one critic per modality, shared across the sampled preferences
        club_v = CLUBInfoNCECritic(proj_dim, proj_dim, club_hidden_dim,
                                   club_layers, "relu").to(device)
        club_t = CLUBInfoNCECritic(proj_dim, proj_dim, club_hidden_dim,
                                   club_layers, "relu").to(device)
        opt_critic = torch.optim.Adam(list(club_v.parameters()) +
                                      list(club_t.parameters()), lr=lr)
        sch_critic = torch.optim.lr_scheduler.CosineAnnealingLR(opt_critic, T_max=epochs)
        if film_mode == DECOMP_R_PROJ:
            print(f"  {DECOMP_R_PROJ}: 4 LoRA heads (r={lora_rank}, alpha={lora_alpha}) "
                  f"+ 2 CLUB critics | lam_club={lam_club} "
                  f"| readout=concat(3 x {proj_dim}) | encoder is LAMBDA-BLIND, "
                  f"preference acts in the heads (4 encoder forwards/batch, not 60)")
        else:
            print(f"  {DECOMP_R}: 4 plain heads + 2 CLUB critics | lam_club={lam_club} "
                  f"| readout=concat(3 x {proj_dim}) | encoder is preference-conditioned")
    elif approach in (4, 5) and film_mode == SIMPLEX6H:
        use_club = False
        proj_v = nn.ModuleDict({k: ProjectionHead(adim, proj_dim) for k in ("r", "u1", "u2")}).to(device)
        proj_t = nn.ModuleDict({k: ProjectionHead(adim, proj_dim) for k in ("r", "u1", "u2")}).to(device)
        head_params = _params(proj_v, proj_t)
    elif approach in (4, 5) and film_mode in SIMPLEX4H_MODES:
        # Four independent ProjectionHeads, no weight sharing and no LoRA: the
        # preference acts purely by mixing head OUTPUTS. Saved under the same
        # proj_{v,t}_{r,u}.pth names as palora_enc so the probe loads them the
        # same way.
        use_club = False
        proj_v_r = ProjectionHead(adim, proj_dim).to(device)
        proj_t_r = ProjectionHead(adim, proj_dim).to(device)
        proj_v_u = ProjectionHead(adim, proj_dim).to(device)
        proj_t_u = ProjectionHead(adim, proj_dim).to(device)
        proj_v = proj_t = None
        head_params = _params(proj_v_r, proj_t_r, proj_v_u, proj_t_u)
    elif approach in (4, 5):
        use_club = False
        if film_mode == "simplex" and lora_branches == 3:
            raise ValueError(
                "film_mode=simplex is incompatible with lora_branches=3 (the 'curve' "
                "variant).\n  LoRATriLayer's third branch is a CURVATURE term on a "
                "scalar lambda:\n      dW = a*A_r B_r + (1-a)*A_u B_u + a(1-a)*A_m B_m\n"
                "  That is a 1-D parameterisation and cannot take the two independent\n"
                "  coefficients the simplex routes to each head (vision gets\n"
                "  (lam_R, lam_U_vision), text gets (lam_R, lam_U_text)).\n"
                "  Use --branches linear (lora_branches=2, LoRADualProjectionHead).")
        if lora_branches == 3:
            HEAD = LoRATriProjectionHead
            head_kw = dict(rank=lora_rank, rank_m=lora_rank_m)
        else:
            HEAD = LoRADualProjectionHead
            head_kw = dict(rank=lora_rank)
        # PaLoRA α/r scaling — applied ONLY for Fix 8 (annealed); base runs keep α=None → scale 1.0
        if preference_schedule == "annealed":
            head_kw["alpha"] = lora_alpha
        proj_v = HEAD(adim, proj_dim, **head_kw).to(device)
        proj_t = HEAD(adim, proj_dim, **head_kw).to(device)
        head_params = _params(proj_v, proj_t)
    else:
        use_club = (method == "factorcl_warmup")
        proj_v_r = ProjectionHead(adim, proj_dim).to(device)
        proj_t_r = ProjectionHead(adim, proj_dim).to(device)
        proj_v_u = ProjectionHead(adim, proj_dim).to(device)
        proj_t_u = ProjectionHead(adim, proj_dim).to(device)
        if use_club:
            club_v = CLUBInfoNCECritic(proj_dim, proj_dim, hidden_dim=club_hidden_dim, layers=club_layers, activation="relu").to(device)
            club_t = CLUBInfoNCECritic(proj_dim, proj_dim, hidden_dim=club_hidden_dim, layers=club_layers, activation="relu").to(device)
            opt_critic = optim.AdamW(_params(club_v, club_t), lr=lr, weight_decay=weight_decay)
            sch_critic = optim.lr_scheduler.CosineAnnealingLR(opt_critic, T_max=epochs)
        head_params = _params(proj_v_r, proj_t_r, proj_v_u, proj_t_u)

    enc_params  = _params(enc_v, enc_t)
    # The LoRA parameter-group split below inspects proj_v/proj_t for A_r/B_r/... .
    # It only applies to the shared-base LoRA heads. The output-mixing variants have
    # no LoRA at all: simplex4h* set proj_v=None (four separate heads live in
    # proj_*_r/u), and simplex6h holds three plain heads in a ModuleDict. Both hit
    # this block and crashed — NoneType.named_parameters() and ModuleDict.layer1.
    # decomp_R also sets proj_v/proj_t = None -- its four plain heads live in
    # proj_{v,t}_{r,u}, same as simplex4h*. Same crash, same guard.
    _no_lora_heads = _simplex4h or _simplex6h or _decomp_r
    if approach in (4, 5) and not _palora_enc and not _no_lora_heads:
        _ru_keys = ("A_r", "B_r", "A_u", "B_u")
        _m_keys  = ("A_m", "B_m")
        # proj LoRA params (both approaches)
        lora_ru_v_params = [p for n, p in proj_v.named_parameters() if any(k in n for k in _ru_keys)]
        lora_ru_t_params = [p for n, p in proj_t.named_parameters() if any(k in n for k in _ru_keys)]
        lora_m_params    = [p for m in (proj_v, proj_t)
                            for n, p in m.named_parameters() if any(k in n for k in _m_keys)]
        base_head_params = [p for m in (proj_v, proj_t)
                            for n, p in m.named_parameters()
                            if not any(k in n for k in _ru_keys + _m_keys)]
        lora_ru_v_lr = lr * head_lr_mult * lora_lr_mult
        lora_ru_t_lr = lr * head_lr_mult * lora_lr_mult * lora_lr_mult_t
        lora_m_lr    = lr * head_lr_mult * lora_lr_mult * lora_lr_mult_m
        param_groups = [
            {"params": base_head_params,   "lr": lr * head_lr_mult},
            {"params": lora_ru_v_params,   "lr": lora_ru_v_lr},
            {"params": lora_ru_t_params,   "lr": lora_ru_t_lr},
        ]
        if lora_m_params:
            param_groups.append({"params": lora_m_params, "lr": lora_m_lr})
        if approach == 5:
            # Approach 5: encoder also has LoRA — separate enc LoRA params at enc lr
            enc_lora_ru_v = [p for n, p in enc_v.named_parameters() if any(k in n for k in _ru_keys)]
            enc_lora_ru_t = [p for n, p in enc_t.named_parameters() if any(k in n for k in _ru_keys)]
            enc_lora_m    = [p for m in (enc_v, enc_t)
                             for n, p in m.named_parameters() if any(k in n for k in _m_keys)]
            enc_base      = [p for m in (enc_v, enc_t)
                             for n, p in m.named_parameters()
                             if not any(k in n for k in _ru_keys + _m_keys)]
            param_groups = [
                {"params": enc_base,         "lr": lr},
                {"params": enc_lora_ru_v,    "lr": lr * lora_lr_mult},
                {"params": enc_lora_ru_t,    "lr": lr * lora_lr_mult * lora_lr_mult_t},
            ] + param_groups
            if enc_lora_m:
                param_groups.append({"params": enc_lora_m, "lr": lr * lora_lr_mult * lora_lr_mult_m})
            trainable = enc_base + enc_lora_ru_v + enc_lora_ru_t + enc_lora_m + \
                        lora_ru_v_params + lora_ru_t_params + lora_m_params + base_head_params
        else:
            param_groups.insert(0, {"params": enc_params, "lr": lr})
            trainable = enc_params + lora_ru_v_params + lora_ru_t_params + lora_m_params + base_head_params
        opt_enc = optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        trainable   = enc_params + head_params
        opt_enc = optim.AdamW([
            {"params": enc_params,  "lr": lr},
            {"params": head_params, "lr": lr * head_lr_mult},
        ], weight_decay=weight_decay)
    sch_enc   = optim.lr_scheduler.CosineAnnealingLR(opt_enc, T_max=epochs)
    writer    = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
    loader    = _ssl_loader(dataset, batch_size)
    ckpt_dir  = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_every = max(1, epochs // 10)   # save ~10 checkpoints evenly spaced

    if approach in (4, 5):
        rm = lora_rank_m if lora_rank_m is not None else lora_rank
        enc_tag = " | enc=LoRA" if approach == 5 else ""
        lora_info = (f" | branches={lora_branches} | rank={lora_rank}/rank_m={rm}"
                     f" | lora_lr_mult={lora_lr_mult} | lora_lr_mult_m={lora_lr_mult_m}"
                     f" | lora_lr_mult_t={lora_lr_mult_t}"
                     f" | var_reg_weight={var_reg_weight} | var_reg_gamma={var_reg_gamma}{enc_tag}")
    else:
        lora_info = ""
    print(f"  ap={approach} | {method} | {dataset} | epochs={epochs} | lam={lam_dist} | head_lr_mult={head_lr_mult}{lora_info}")
    print(f"  Checkpoints every {ckpt_every} epochs → {ckpt_dir}")

    enc_grad_params = _params(enc_v, enc_t)
    # The preference acts in the HEADS for every output-mixing / LoRA-head variant,
    # so encoder-only gradient conflict measures it in the wrong place. Track head
    # parameters too and log both series.
    _head_grad_params = [p for m in (proj_v, proj_t, proj_v_r, proj_t_r, proj_v_u, proj_t_u)
                         if m is not None for p in m.parameters() if p.requires_grad]

    # Resume support (ap3/4/5)
    # A resume.pth bundle (models + optimizer + scheduler + RNG + epoch) is written
    # alongside each periodic checkpoint. With --resume, pick up where it left off.
    resume_path = os.path.join(out_dir, "resume.pth")
    start_epoch = 1
    if resume and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        enc_v.load_state_dict(ck["enc_v"]);   enc_t.load_state_dict(ck["enc_t"])
        # Shared-base layout stores proj_v/proj_t; the four-head variants
        # (simplex4h*, duo_txt, decomp_R) store proj_{v,t}_{r,u} instead. Before
        # this, _save_resume was skipped entirely for those, so a cancelled or
        # preempted run had to restart from epoch 1.
        if ck.get("proj_v") is not None and proj_v is not None:
            proj_v.load_state_dict(ck["proj_v"]); proj_t.load_state_dict(ck["proj_t"])
        else:
            for _k, _m in (("proj_v_r", proj_v_r), ("proj_t_r", proj_t_r),
                           ("proj_v_u", proj_v_u), ("proj_t_u", proj_t_u)):
                if ck.get(_k) is not None and _m is not None:
                    _m.load_state_dict(ck[_k])
        # CLUB critics: without these a resume restarts with untrained critics on a
        # long-trained encoder, and the bound is meaningless until they catch up.
        if ck.get("club_v") is not None and use_club:
            club_v.load_state_dict(ck["club_v"]); club_t.load_state_dict(ck["club_t"])
            opt_critic.load_state_dict(ck["opt_critic"])
            sch_critic.load_state_dict(ck["sch_critic"])
        opt_enc.load_state_dict(ck["opt"]);   sch_enc.load_state_dict(ck["sch"])
        torch.set_rng_state(ck["rng_torch"])
        if ck.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        np.random.set_state(ck["rng_numpy"])
        start_epoch = ck["epoch"] + 1
        print(f"  RESUMED from {resume_path}: done through epoch {ck['epoch']} → continue at {start_epoch}")
    elif resume:
        print(f"  --resume set but no {resume_path} found → starting fresh at epoch 1")

    if start_epoch > epochs:
        print(f"  Checkpoint already at epoch {start_epoch - 1} >= target {epochs}; nothing to train.")

    def _save_resume(ep):
        _heads = ({"proj_v": proj_v.state_dict(), "proj_t": proj_t.state_dict()}
                  if proj_v is not None else
                  {k: m.state_dict() for k, m in
                   (("proj_v_r", proj_v_r), ("proj_t_r", proj_t_r),
                    ("proj_v_u", proj_v_u), ("proj_t_u", proj_t_u)) if m is not None})
        _critics = ({"club_v": club_v.state_dict(), "club_t": club_t.state_dict(),
                     "opt_critic": opt_critic.state_dict(),
                     "sch_critic": sch_critic.state_dict()} if use_club else {})
        torch.save({
            "epoch": ep,
            "enc_v": enc_v.state_dict(), "enc_t": enc_t.state_dict(),
            **_heads, **_critics,
            "opt": opt_enc.state_dict(), "sch": sch_enc.state_dict(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_numpy": np.random.get_state(),
        }, resume_path)

    # Fix 8: PaLoRA multi-preference setup
    steps_per_epoch = len(loader)
    total_steps = max(1, epochs * steps_per_epoch)
    if _simplex or _simplex4h or _simplex6h or _decomp_r:
        if _duo:
            simplex_prefs = duo_grid(simplex_side)
            print(f"  duo_txt: 1-D controller -> {len(simplex_prefs)} preferences "
                  f"(λ_R, 0, 1-λ_R) | vision-uniqueness axis DROPPED "
                  f"| schedule={preference_schedule} | Q={annealing_temperature}")
        else:
            simplex_prefs = simplex_grid(simplex_side)
            _pref_cycle = PrefCycle(len(simplex_prefs), prefs_per_batch or len(simplex_prefs),
                                    seed=seed + 991)
            _pref_loss_sum = np.zeros(len(simplex_prefs))   # per-TARGET mean loss
            print(f"  {'decomp_R' if _decomp_r else 'simplex'}: side={simplex_side} -> "
                  f"{len(simplex_prefs)} preferences "
                  f"(λ_R, λ_U_vision, λ_U_text) | schedule={preference_schedule} "
                  f"| Q={annealing_temperature} | alpha={lora_alpha}")

    # simplex_enc_decomp_R diagnostics, reset per epoch below
    club_raw_last = club_used_last = club_clampfrac_last = 0.0
    cka_ur_last = erank_u_last = cos_r_last = 0.0

    _pref_on = (approach in (4, 5) and preference_schedule != "single")
    # palora_proj (approach 1/2): PaLoRA-style multi-preference training of the
    # four-head output mixture. Always multi-preference — that is the whole point.
    if _palora:
        if num_preferences < 2:
            raise ValueError("palora_proj needs num_preferences >= 2 (default 5)")
        if annealing_temperature <= 0:
            raise ValueError("annealing_temperature (Q) must be > 0")
        print(f"  palora_proj: M={num_preferences} preferences | schedule={preference_schedule} "
              f"| Q={annealing_temperature} | 4 separate heads, output-space mixture")
    if _pref_on:
        if num_preferences < 1:
            raise ValueError("num_preferences (M) must be >= 1")
        if annealing_temperature <= 0:
            raise ValueError("annealing_temperature (Q) must be > 0")
        if film_mode != "proj":
            print(f"  WARNING: Fix 8 multi-preference assumes a λ-blind encoder "
                  f"(film_mode=proj); got film_mode={film_mode}.")
        print(f"  Fix 8: M={num_preferences} preferences | schedule={preference_schedule} "
              f"| Q={annealing_temperature}")

    for epoch in range(start_epoch, epochs + 1):
        enc_total = crit_total = LR_total = LU_total = Lcurve_total = Lorth_total = 0.0
        club_raw_total = cka_ur_total = clampfrac_total = erank_u_total = 0.0
        cos_r_total = 0.0
        cos_total = gR_norm_total = gU_norm_total = 0.0
        cos_head_total = 0.0; _n_cos_head = 0
        tau_last = 0.0; gamma_last = 0.0; prefs_last = None
        _n_cos = 0
        rank_last = 0.0   # Fix 9: last preference-ranking penalty value
        lam_min_e = lam_max_e = lam_mean_e = 0.5; _pref_seen = False

        for batch_idx, (aug1, aug2) in enumerate(loader):
            v_a = aug1[0].float().to(device);  v_b = aug2[0].float().to(device)
            t_a = aug1[1].float().to(device);  t_b = aug2[1].float().to(device)

            lam = _sample_lam_mb(lam_dist, dirichlet_alpha, epoch, epochs)
            # Graph-connected copies of the two objectives for the gradient-conflict
            # diagnostic. The preference branches overwrite L_R/L_U with DETACHED
            # scalars for logging, which is why cos_sim was never computed there —
            # leaving cos_sim and endpoint_cka on disjoint sets of runs and the
            # conflict hypothesis untestable from the logs.
            _LR_g = _LU_g = None
            # Multi-preference active only for ap4/5, non-warmup, schedule != single
            _multi = (_pref_on and not (warmup_epochs > 0 and epoch <= warmup_epochs))

            if approach == 3:
                # Approach 3 (DualFiLM): separate forward_r / forward_u paths, λ-blind training.
                if film_mode == "proj":
                    h_va = enc_v(v_a);  h_vb = enc_v(v_b)
                    h_ta = enc_t(t_a);  h_tb = enc_t(t_b)
                else:
                    h_va = enc_v(v_a, lam);  h_vb = enc_v(v_b, lam)
                    h_ta = enc_t(t_a, lam);  h_tb = enc_t(t_b, lam)
                z_va_r = proj_v.forward_r(h_va);  z_ta_r = proj_t.forward_r(h_ta)
                z_va_u = proj_v.forward_u(h_va);  z_vb_u = proj_v.forward_u(h_vb)
                z_ta_u = proj_t.forward_u(h_ta);  z_tb_u = proj_t.forward_u(h_tb)
                opt_enc.zero_grad()
                L_R  = infonce_cross(z_va_r, z_ta_r, temperature=temperature)
                L_Uv = nt_xent(z_va_u, z_vb_u, temperature=temperature)
                L_Ut = nt_xent(z_ta_u, z_tb_u, temperature=temperature)
            elif approach in (4, 5):
                if approach == 5:
                    # Approach 5: encoder is LoRA-conditioned
                    if warmup_epochs > 0 and epoch <= warmup_epochs:
                        h_va_r = enc_v.forward_r(v_a); h_ta_r = enc_t.forward_r(t_a)
                        h_vb_u = enc_v.forward_u(v_b); h_tb_u = enc_t.forward_u(t_b)
                        h_va_u = enc_v.forward_u(v_a); h_ta_u = enc_t.forward_u(t_a)
                    else:
                        h_va = enc_v(v_a, lam); h_vb = enc_v(v_b, lam)
                        h_ta = enc_t(t_a, lam); h_tb = enc_t(t_b, lam)
                else:
                    # Approach 4: encoder is λ-blind (proj) or FiLM-conditioned
                    if film_mode == "proj":
                        h_va = enc_v(v_a);  h_vb = enc_v(v_b)
                        h_ta = enc_t(t_a);  h_tb = enc_t(t_b)
                    else:
                        h_va = enc_v(v_a, lam);  h_vb = enc_v(v_b, lam)
                        h_ta = enc_t(t_a, lam);  h_tb = enc_t(t_b, lam)
                if warmup_epochs > 0 and epoch <= warmup_epochs:
                    if approach == 5:
                        z_va_r = proj_v.forward_r(h_va_r); z_ta_r = proj_t.forward_r(h_ta_r)
                        z_va_u = proj_v.forward_u(h_va_u); z_vb_u = proj_v.forward_u(h_vb_u)
                        z_ta_u = proj_t.forward_u(h_ta_u); z_tb_u = proj_t.forward_u(h_tb_u)
                    else:
                        z_va_r = proj_v.forward_r(h_va);  z_ta_r = proj_t.forward_r(h_ta)
                        z_va_u = proj_v.forward_u(h_va);  z_vb_u = proj_v.forward_u(h_vb)
                        z_ta_u = proj_t.forward_u(h_ta);  z_tb_u = proj_t.forward_u(h_tb)
                    opt_enc.zero_grad()
                    L_R  = infonce_cross(z_va_r, z_ta_r, temperature=temperature)
                    L_Uv = nt_xent(z_va_u, z_vb_u, temperature=temperature)
                    L_Ut = nt_xent(z_ta_u, z_tb_u, temperature=temperature)
                elif _decomp_r:
                    # simplex_enc_decomp_R
                    # lambda -> W_enc(lambda) -> h(lambda) -> plain heads -> losses.
                    # The encoder is preference-conditioned so L_t(W(l1)) != L_t(W(l2))
                    # and a symmetric grid cannot collapse to equal weighting.
                    tau = (((epoch - 1) * steps_per_epoch + batch_idx) / (total_steps - 1)
                           if total_steps > 1 else 1.0)
                    tau = min(1.0, max(0.0, tau))
                    # TARGETS for this step: M of the 15 grid points, balanced over
                    # training by PrefCycle (M >= 15 -> the full grid, unchanged).
                    _idx = _pref_cycle.next()
                    _targets = [simplex_prefs[i] for i in _idx]
                    # EFFECTIVE preferences: what actually conditions W(lambda) AND weights
                    # the losses. The target is only the operating point the schedule aims at.
                    prefs_l = (anneal_simplex(_targets, tau, annealing_temperature,
                                              mode=anneal_mode)
                               if preference_schedule == "annealed" else _targets)

                    # ---- 1. encoder forwards for the 'a' views, computed ONCE ----
                    # Both the critic step and the main step need these. Recomputing
                    # them cost 30 of 90 encoder forwards per batch (~33% of the step):
                    # the encoder is preference-conditioned so each of the 15
                    # preferences needs its own forward, and LoRATransformerEncoder
                    # dominates -- the plain MLP heads are cheap by comparison.
                    #
                    # Safe to share: the critic trains on .detach()ed head outputs, so
                    # no gradient reaches the encoder from crit_loss and the autograd
                    # graph built here survives intact for the main backward. The critic
                    # simply sees h from before its own update -- the standard
                    # alternating-optimisation ordering.
                    if not _enc_cond:
                        # lambda-blind encoder: ONE forward per view per modality for
                        # the whole batch (4 total), reused at every preference. The
                        # preference-dependence lives in the LoRA heads below, so
                        # L(lambda) still varies -- see DECOMP_R_PROJ.
                        _hv_a = enc_v(v_a); _ht_a = enc_t(t_a)
                        _hv_b = enc_v(v_b); _ht_b = enc_t(t_b)
                        _HV = [_hv_a] * len(prefs_l)
                        _HT = [_ht_a] * len(prefs_l)
                    else:
                        _HV = [enc_v.forward_mix(v_a, w_r, w_u1) for w_r, w_u1, _ in prefs_l]
                        _HT = [enc_t.forward_mix(t_a, w_r, w_u2) for w_r, _, w_u2 in prefs_l]

                    # ---- 2. critic step: fit q(r|u) on the DETACHED pooled batch ----
                    with torch.no_grad():
                        if _head_cond:
                            _cu_v = [proj_v_u.forward_mix(h, p[0], p[1])
                                     for h, p in zip(_HV, prefs_l)]
                            _cr_v = [proj_v_r.forward_mix(h, p[0], p[1])
                                     for h, p in zip(_HV, prefs_l)]
                            _cu_t = [proj_t_u.forward_mix(h, p[0], p[2])
                                     for h, p in zip(_HT, prefs_l)]
                            _cr_t = [proj_t_r.forward_mix(h, p[0], p[2])
                                     for h, p in zip(_HT, prefs_l)]
                        else:
                            _cu_v = [proj_v_u(h) for h in _HV]; _cr_v = [proj_v_r(h) for h in _HV]
                            _cu_t = [proj_t_u(h) for h in _HT]; _cr_t = [proj_t_r(h) for h in _HT]
                    opt_critic.zero_grad()
                    crit_loss = (club_v.learning_loss(torch.cat(_cu_v), torch.cat(_cr_v)) +
                                 club_t.learning_loss(torch.cat(_cu_t), torch.cat(_cr_t)))
                    crit_loss.backward()
                    opt_critic.step()
                    crit_total += float(crit_loss.item())

                    # ---- 3. encoder + head step ----
                    opt_enc.zero_grad()
                    loss = 0.0
                    LR_sum = LU_sum = 0.0
                    _club_raw_sum = 0.0; _club_used_sum = 0.0; _club_clamped = 0; _club_n = 0
                    _cka_ur_sum = 0.0; _erank_u_sum = 0.0; _cos_r_sum = 0.0
                    for _pi, (w_r, w_u1, w_u2) in enumerate(prefs_l):
                        hv_a = _HV[_pi]                       # reused, not recomputed
                        ht_a = _HT[_pi]                       # reused, not recomputed
                        if _enc_cond:
                            hv_b = enc_v.forward_mix(v_b, w_r, w_u1)
                            ht_b = enc_t.forward_mix(t_b, w_r, w_u2)
                        else:
                            hv_b, ht_b = _hv_b, _ht_b         # lambda-blind: also cached
                        if _head_cond:
                            zvr = proj_v_r.forward_mix(hv_a, w_r, w_u1)
                            zvu = proj_v_u.forward_mix(hv_a, w_r, w_u1)
                            ztr = proj_t_r.forward_mix(ht_a, w_r, w_u2)
                            ztu = proj_t_u.forward_mix(ht_a, w_r, w_u2)
                            zvu_b = proj_v_u.forward_mix(hv_b, w_r, w_u1)
                            ztu_b = proj_t_u.forward_mix(ht_b, w_r, w_u2)
                        else:
                            zvr, zvu = proj_v_r(hv_a), proj_v_u(hv_a)
                            ztr, ztu = proj_t_r(ht_a), proj_t_u(ht_a)
                            zvu_b, ztu_b = proj_v_u(hv_b), proj_t_u(ht_b)

                        L_R_m  = infonce_cross(zvr, ztr, temperature=temperature)
                        L_U1_m = nt_xent(zvu, zvu_b, temperature=temperature)
                        L_U2_m = nt_xent(ztu, ztu_b, temperature=temperature)

                        # CLUB: push U away from R, WITHIN modality (FactorCL's form).
                        # r is detached so the penalty moves u, not r.
                        _rawv = club_v(zvu, zvr.detach())
                        _rawt = club_t(ztu, ztr.detach())
                        _usev = torch.clamp(_rawv, min=0.0)
                        _uset = torch.clamp(_rawt, min=0.0)
                        L_U1_m = L_U1_m + lam_club * _usev
                        L_U2_m = L_U2_m + lam_club * _uset
                        # -- diagnostics: is CLUB actually doing anything? --
                        _club_raw_sum  += float(_rawv.item() + _rawt.item()) / 2
                        _club_used_sum += float(_usev.item() + _uset.item()) / 2
                        _club_clamped  += int(_rawv.item() <= 0) + int(_rawt.item() <= 0)
                        _club_n += 2
                        with torch.no_grad():
                            # the EFFECT the penalty is supposed to have, measured
                            # independently of CLUB's own estimate
                            _cka_ur_sum += (_lin_cka_t(zvu, zvr) + _lin_cka_t(ztu, ztr)) / 2
                            _erank_u_sum += (_eff_rank_t(zvu) + _eff_rank_t(ztu)) / 2
                            # Have the two R heads converged? L_R aligns them, so this
                            # should rise toward 1. Two consequences if it does:
                            #  - averaging them for the R readout block is well-defined
                            #  - a single TIED R head would cost nothing, which answers
                            #    "why two R heads?" empirically instead of by argument
                            _cos_r_sum += float(F.cosine_similarity(zvr, ztr, dim=-1).mean())

                        _l_pref = w_r * L_R_m + w_u1 * L_U1_m + w_u2 * L_U2_m
                        loss = loss + _l_pref
                        _pref_loss_sum[_idx[_pi]] += float(_l_pref.item())
                        LR_sum += L_R_m.item(); LU_sum += (L_U1_m.item() + L_U2_m.item()) / 2
                        _LR_g = L_R_m if _LR_g is None else _LR_g + L_R_m
                        _LUm  = (L_U1_m + L_U2_m) / 2
                        _LU_g = _LUm if _LU_g is None else _LU_g + _LUm
                    loss = loss / len(prefs_l)
                    _np = len(prefs_l)
                    club_raw_last    = _club_raw_sum / _np
                    club_used_last   = _club_used_sum / _np
                    club_clampfrac_last = _club_clamped / max(1, _club_n)
                    cka_ur_last      = _cka_ur_sum / _np
                    erank_u_last     = _erank_u_sum / _np
                    cos_r_last       = _cos_r_sum / _np
                    cos_r_total     += cos_r_last
                    club_raw_total   += club_raw_last
                    cka_ur_total     += cka_ur_last
                    clampfrac_total  += club_clampfrac_last
                    erank_u_total    += erank_u_last
                    L_R = torch.as_tensor(LR_sum / _np)
                    L_U = torch.as_tensor(LU_sum / _np)
                    tau_last = tau; gamma_last = tau / annealing_temperature
                    _pref_seen = True; prefs_last = [(p[0], p[1]) for p in prefs_l]
                    _rcol = [p[0] for p in prefs_l]
                    lam_min_e = min(_rcol); lam_max_e = max(_rcol)
                    lam_mean_e = sum(_rcol) / len(_rcol)
                elif _simplex6h:
                    # True 3-way simplex blend over 3 heads per modality
                    tau = (((epoch - 1) * steps_per_epoch + batch_idx) / (total_steps - 1)
                           if total_steps > 1 else 1.0)
                    tau = min(1.0, max(0.0, tau))
                    prefs_l = (anneal_simplex(simplex_prefs, tau, annealing_temperature)
                               if preference_schedule == "annealed" else simplex_prefs)
                    opt_enc.zero_grad()
                    loss = 0.0
                    LR_sum = LU_sum = 0.0
                    _rank_losses, _rank_w = [], []

                    def _mix3(pd, h, w_r, w_u1, w_u2):
                        return F.normalize(w_r * pd["r"](h) + w_u1 * pd["u1"](h)
                                           + w_u2 * pd["u2"](h), dim=-1)

                    for w_r, w_u1, w_u2 in prefs_l:
                        z_va = _mix3(proj_v, h_va, w_r, w_u1, w_u2)
                        z_vb = _mix3(proj_v, h_vb, w_r, w_u1, w_u2)
                        z_ta = _mix3(proj_t, h_ta, w_r, w_u1, w_u2)
                        z_tb = _mix3(proj_t, h_tb, w_r, w_u1, w_u2)
                        L_R_m  = infonce_cross(z_va, z_ta, temperature=temperature)
                        L_U1_m = nt_xent(z_va, z_vb, temperature=temperature)
                        L_U2_m = nt_xent(z_ta, z_tb, temperature=temperature)
                        loss = loss + (w_r * L_R_m + w_u1 * L_U1_m + w_u2 * L_U2_m)
                        LR_sum += L_R_m.item(); LU_sum += (L_U1_m.item() + L_U2_m.item()) / 2
                        _LR_g = L_R_m if _LR_g is None else _LR_g + L_R_m
                        _LUm  = (L_U1_m + L_U2_m) / 2
                        _LU_g = _LUm if _LU_g is None else _LU_g + _LUm
                        if rank_penalty > 0.0:
                            _rank_losses.append((L_R_m, L_U1_m, L_U2_m))
                            _rank_w.append((w_r, w_u1, w_u2))
                    loss = loss / len(prefs_l)
                    if rank_penalty > 0.0:
                        _lrank = preference_rank_penalty(_rank_losses, _rank_w, margin=rank_margin)
                        loss = loss + rank_penalty * _lrank
                        rank_last = float(_lrank.item())
                    L_R = torch.as_tensor(LR_sum / len(prefs_l))
                    L_U = torch.as_tensor(LU_sum / len(prefs_l))
                    tau_last = tau; gamma_last = tau / annealing_temperature
                    _pref_seen = True; prefs_last = [(p[0], p[1]) for p in prefs_l]
                    _rcol = [p[0] for p in prefs_l]
                    lam_min_e = min(_rcol); lam_max_e = max(_rcol)
                    lam_mean_e = sum(_rcol) / len(_rcol)
                elif _simplex4h:
                    # 4-head simplex: mix head OUTPUTS, not weights
                    # Encoder and heads are preference-blind; the preference only
                    # sets how the R-head and U-head outputs are combined, per
                    # modality: vision uses (w_R, w_U_vision), text (w_R, w_U_text).
                    tau = (((epoch - 1) * steps_per_epoch + batch_idx) / (total_steps - 1)
                           if total_steps > 1 else 1.0)
                    tau = min(1.0, max(0.0, tau))
                    prefs_l = (anneal_simplex(simplex_prefs, tau, annealing_temperature)
                               if preference_schedule == "annealed" else simplex_prefs)
                    opt_enc.zero_grad()
                    loss = 0.0
                    LR_sum = LU_sum = 0.0
                    _rank_losses, _rank_w = [], []

                    def _mix(head_r, head_u, h, w_r, w_u):
                        """simplex4h      : blend the two heads' UNIT vectors.
                        simplex4h_norm : blend the raw pre-normalisation outputs and
                        normalise once. Blending two unit vectors lands in a
                        direction neither head was trained to produce, which is the
                        suspected cause of the interior sag; mixing raw keeps each
                        head's magnitude information and is the control for it."""
                        # Two of the 15 grid points hand one modality
                        # (w_R, w_U) = (0, 0) -- vertex (0,0,1) for vision and
                        # (0,1,0) for text. Output mixing would give the zero
                        # vector and normalize(0) = NaN. The shared-base simplex
                        # never hits this because the weights scale dW and (0,0)
                        # simply leaves W0. Fall back to an equal blend, which is
                        # the least-committal reading of "this modality is
                        # unweighted".
                        if w_r + w_u <= 0.0:
                            # duo_txt: vision has no uniqueness objective at all, so
                            # proj_v_u never receives gradient and is still at random
                            # init. Falling back to 0.5/0.5 here would inject that
                            # random head into the output at lam=0. Fall back to the
                            # pure redundancy head instead.
                            w_r, w_u = (1.0, 0.0) if _duo else (0.5, 0.5)
                        if _s4h_premix:
                            return F.normalize(w_r * head_r.net(h) + w_u * head_u.net(h), dim=-1)
                        return F.normalize(w_r * head_r(h) + w_u * head_u(h), dim=-1)

                    for w_r, w_u1, w_u2 in prefs_l:
                        z_va = _mix(proj_v_r, proj_v_u, h_va, w_r, w_u1)
                        z_vb = _mix(proj_v_r, proj_v_u, h_vb, w_r, w_u1)
                        z_ta = _mix(proj_t_r, proj_t_u, h_ta, w_r, w_u2)
                        z_tb = _mix(proj_t_r, proj_t_u, h_tb, w_r, w_u2)
                        L_R_m  = infonce_cross(z_va, z_ta, temperature=temperature)
                        L_U1_m = nt_xent(z_va, z_vb, temperature=temperature)
                        L_U2_m = nt_xent(z_ta, z_tb, temperature=temperature)
                        loss = loss + (w_r * L_R_m + w_u1 * L_U1_m + w_u2 * L_U2_m)
                        LR_sum += L_R_m.item(); LU_sum += (L_U1_m.item() + L_U2_m.item()) / 2
                        _LR_g = L_R_m if _LR_g is None else _LR_g + L_R_m
                        _LUm  = (L_U1_m + L_U2_m) / 2
                        _LU_g = _LUm if _LU_g is None else _LU_g + _LUm
                        if rank_penalty > 0.0:
                            _rank_losses.append((L_R_m, L_U1_m, L_U2_m))
                            _rank_w.append((w_r, w_u1, w_u2))
                    loss = loss / len(prefs_l)
                    if rank_penalty > 0.0:
                        _lrank = preference_rank_penalty(_rank_losses, _rank_w, margin=rank_margin)
                        loss = loss + rank_penalty * _lrank
                        rank_last = float(_lrank.item())
                    L_R = torch.as_tensor(LR_sum / len(prefs_l))
                    L_U = torch.as_tensor(LU_sum / len(prefs_l))
                    tau_last = tau; gamma_last = tau / annealing_temperature
                    _pref_seen = True; prefs_last = [(p[0], p[1]) for p in prefs_l]
                    _rcol = [p[0] for p in prefs_l]
                    lam_min_e = min(_rcol); lam_max_e = max(_rcol)
                    lam_mean_e = sum(_rcol) / len(_rcol)
                elif _simplex:
                    # Simplex step: 3 objectives (R, U_vision, U_text)
                    # Encoder is λ-blind; the vision head sees (λ_R, λ_U_vision) and
                    # the text head sees (λ_R, λ_U_text), so each modality's unique
                    # branch is weighted independently — impossible with a 1-D λ.
                    tau = (((epoch - 1) * steps_per_epoch + batch_idx) / (total_steps - 1)
                           if total_steps > 1 else 1.0)
                    tau = min(1.0, max(0.0, tau))
                    prefs_l = (anneal_simplex(simplex_prefs, tau, annealing_temperature)
                               if preference_schedule == "annealed" else simplex_prefs)
                    opt_enc.zero_grad()
                    loss = 0.0
                    LR_sum = LU_sum = 0.0
                    _rank_losses, _rank_w = [], []
                    for w_r, w_u1, w_u2 in prefs_l:
                        if approach == 5:
                            # "simplex enc": the LoRA encoder is preference-conditioned
                            # too, so h itself is recomputed for each preference. The
                            # vision side uses (λ_R, λ_U_vision), text (λ_R, λ_U_text).
                            hv_a = enc_v.forward_mix(v_a, w_r, w_u1)
                            hv_b = enc_v.forward_mix(v_b, w_r, w_u1)
                            ht_a = enc_t.forward_mix(t_a, w_r, w_u2)
                            ht_b = enc_t.forward_mix(t_b, w_r, w_u2)
                        else:
                            hv_a, hv_b, ht_a, ht_b = h_va, h_vb, h_ta, h_tb
                        z_va = proj_v.forward_mix(hv_a, w_r, w_u1)
                        z_vb = proj_v.forward_mix(hv_b, w_r, w_u1)
                        z_ta = proj_t.forward_mix(ht_a, w_r, w_u2)
                        z_tb = proj_t.forward_mix(ht_b, w_r, w_u2)
                        L_R_m  = infonce_cross(z_va, z_ta, temperature=temperature)
                        L_U1_m = nt_xent(z_va, z_vb, temperature=temperature)
                        L_U2_m = nt_xent(z_ta, z_tb, temperature=temperature)
                        loss = loss + (w_r * L_R_m + w_u1 * L_U1_m + w_u2 * L_U2_m)
                        LR_sum += L_R_m.item();  LU_sum += (L_U1_m.item() + L_U2_m.item()) / 2
                        _LR_g = L_R_m if _LR_g is None else _LR_g + L_R_m
                        _LUm  = (L_U1_m + L_U2_m) / 2
                        _LU_g = _LUm if _LU_g is None else _LU_g + _LUm
                        if rank_penalty > 0.0:
                            _rank_losses.append((L_R_m, L_U1_m, L_U2_m))
                            _rank_w.append((w_r, w_u1, w_u2))
                    loss = loss / len(prefs_l)
                    if rank_penalty > 0.0:
                        _lrank = preference_rank_penalty(_rank_losses, _rank_w,
                                                         margin=rank_margin)
                        loss = loss + rank_penalty * _lrank
                        rank_last = float(_lrank.item())
                    L_R = torch.as_tensor(LR_sum / len(prefs_l))
                    L_U = torch.as_tensor(LU_sum / len(prefs_l))
                    tau_last = tau; gamma_last = tau / annealing_temperature
                    _pref_seen = True; prefs_last = [(p[0], p[1]) for p in prefs_l]
                    _rcol = [p[0] for p in prefs_l]
                    lam_min_e = min(_rcol); lam_max_e = max(_rcol)
                    lam_mean_e = sum(_rcol) / len(_rcol)
                elif _multi:
                    # PaLoRA multi-preference step (approach 4/5, film_mode=proj)
                    # Encoder features h_* are λ-blind and were computed once above;
                    # only the projection heads are rerun for each of the M annealed
                    # preferences. M forwards -> ONE backward -> ONE optimizer step.
                    tau = (((epoch - 1) * steps_per_epoch + batch_idx) / (total_steps - 1)
                           if total_steps > 1 else 1.0)
                    tau = min(1.0, max(0.0, tau))
                    gamma = tau / annealing_temperature
                    if preference_schedule == "annealed":
                        prefs = _anneal_preferences(num_preferences, tau,
                                                    annealing_temperature, device)
                    else:  # "fixed": evenly spaced base preferences, no annealing
                        _p = torch.linspace(0.0, 1.0, num_preferences, device=device)
                        prefs = torch.stack([_p, 1.0 - _p], dim=1)
                    prefs_l = [(float(a), float(b)) for a, b in prefs]
                    opt_enc.zero_grad()
                    loss = 0.0
                    LR_sum = LU_sum = 0.0
                    for w_r, w_u in prefs_l:
                        # W(λ_m) = W0 + (α/r)(λ_m[0]·ΔW_R + λ_m[1]·ΔW_U) is applied
                        # inside the LoRA head; λ_m[1] == 1 - λ_m[0] by construction.
                        if _palora_enc:
                            # palora_enc: the LoRA ENCODER is conditioned on λ_m, so h
                            # itself is preference-specific and must be recomputed for
                            # every λ_m. The four separate heads then mix at the OUTPUT:
                            #   z(λ) = normalize(λ·z_R + (1-λ)·z_U)
                            h_va = enc_v(v_a, w_r);  h_vb = enc_v(v_b, w_r)
                            h_ta = enc_t(t_a, w_r);  h_tb = enc_t(t_b, w_r)
                            _mx = lambda r, u: F.normalize(w_r * r + w_u * u, dim=-1)
                            z_va = _mx(proj_v_r(h_va), proj_v_u(h_va))
                            z_vb = _mx(proj_v_r(h_vb), proj_v_u(h_vb))
                            z_ta = _mx(proj_t_r(h_ta), proj_t_u(h_ta))
                            z_tb = _mx(proj_t_r(h_tb), proj_t_u(h_tb))
                        else:
                            z_va = proj_v(h_va, w_r);  z_vb = proj_v(h_vb, w_r)
                            z_ta = proj_t(h_ta, w_r);  z_tb = proj_t(h_tb, w_r)
                        L_R_m  = infonce_cross(z_va, z_ta, temperature=temperature)
                        L_Uv_m = nt_xent(z_va, z_vb, temperature=temperature)
                        L_Ut_m = nt_xent(z_ta, z_tb, temperature=temperature)
                        L_U_m  = (L_Uv_m + L_Ut_m) / 2
                        _LR_g = L_R_m if _LR_g is None else _LR_g + L_R_m
                        _LU_g = L_U_m if _LU_g is None else _LU_g + L_U_m
                        # PaLoRA scalarisation: L_m = λ_m[0]·L_R + λ_m[1]·L_U
                        loss = loss + (w_r * L_R_m + w_u * L_U_m)
                        LR_sum += L_R_m.item();  LU_sum += L_U_m.item()
                    loss = loss / num_preferences                     # mean over M
                    L_R = torch.as_tensor(LR_sum / num_preferences)   # scalars for logging
                    L_U = torch.as_tensor(LU_sum / num_preferences)
                    tau_last = tau; gamma_last = gamma; _pref_seen = True
                    prefs_last = prefs_l
                    _r_col = [a for a, _b in prefs_l]
                    lam_min_e = min(_r_col); lam_max_e = max(_r_col)
                    lam_mean_e = sum(_r_col) / len(_r_col)
                else:
                    z_va = proj_v(h_va, lam);  z_vb = proj_v(h_vb, lam)
                    z_ta = proj_t(h_ta, lam);  z_tb = proj_t(h_tb, lam)
                    opt_enc.zero_grad()
                    L_R  = infonce_cross(z_va, z_ta, temperature=temperature)
                    L_Uv = nt_xent(z_va, z_vb, temperature=temperature)
                    L_Ut = nt_xent(z_ta, z_tb, temperature=temperature)
                    if var_reg_weight > 0:
                        L_var = (F.relu(var_reg_gamma - z_va.std(dim=0)).mean()
                               + F.relu(var_reg_gamma - z_ta.std(dim=0)).mean()) / 2
                        L_Uv = L_Uv + var_reg_weight * L_var
                    if curve_reg_weight > 0:
                        lam_b = _sample_lam_mb(lam_dist, dirichlet_alpha, epoch, epochs)
                        if film_mode == "proj":
                            z_va_b = proj_v(h_va, lam_b)
                            z_ta_b = proj_t(h_ta, lam_b)
                        else:
                            h_va_b = enc_v(v_a, lam_b)
                            h_ta_b = enc_t(t_a, lam_b)
                            z_va_b = proj_v(h_va_b, lam_b)
                            z_ta_b = proj_t(h_ta_b, lam_b)
                        cos_v = (z_va * z_va_b).sum(dim=-1).mean()
                        cos_t = (z_ta * z_ta_b).sum(dim=-1).mean()
                        L_curve = curve_reg_weight * abs(lam - lam_b) * (cos_v + cos_t) / 2
                        loss = loss + L_curve
                        Lcurve_total += L_curve.item()
            elif _palora:
                # palora_proj: PaLoRA-style training of the OUTPUT-SPACE mixture
                # Four separate heads (NO shared base, as in film=none), but the
                # mixture z(λ)=normalize(λ·z_R+(1-λ)·z_U) is now OPTIMISED at M
                # annealed preferences instead of being a probe-time-only
                # interpolation. Endpoints reduce exactly to the pure objectives
                # (λ=1 → L_R on z_R only; λ=0 → L_U on z_U only), so the heads keep
                # their specialisation pressure while intermediate λ become trained.
                # Encoder is λ-blind → h computed once; only the small heads rerun.
                h_va = enc_v(v_a);  h_vb = enc_v(v_b)
                h_ta = enc_t(t_a);  h_tb = enc_t(t_b)
                r_va, r_vb = proj_v_r(h_va), proj_v_r(h_vb)
                u_va, u_vb = proj_v_u(h_va), proj_v_u(h_vb)
                r_ta, r_tb = proj_t_r(h_ta), proj_t_r(h_tb)
                u_ta, u_tb = proj_t_u(h_ta), proj_t_u(h_tb)

                tau = (((epoch - 1) * steps_per_epoch + batch_idx) / (total_steps - 1)
                       if total_steps > 1 else 1.0)
                tau = min(1.0, max(0.0, tau))
                if preference_schedule == "annealed":
                    prefs = _anneal_preferences(num_preferences, tau, annealing_temperature, device)
                else:
                    _p = torch.linspace(0.0, 1.0, num_preferences, device=device)
                    prefs = torch.stack([_p, 1.0 - _p], dim=1)
                # _anneal_preferences returns (M, 2) — unpack each row into scalars
                prefs_l = [(float(a), float(b)) for a, b in prefs]
                lams = [a for a, _b in prefs_l]

                opt_enc.zero_grad()
                loss = 0.0
                LR_sum = LU_sum = 0.0
                for lam_m, lam_u in prefs_l:
                    zc = lambda r, u: F.normalize(lam_m * r + lam_u * u, dim=-1)
                    z_va, z_vb = zc(r_va, u_va), zc(r_vb, u_vb)
                    z_ta, z_tb = zc(r_ta, u_ta), zc(r_tb, u_tb)
                    L_R_m  = infonce_cross(z_va, z_ta, temperature=temperature)
                    L_Uv_m = nt_xent(z_va, z_vb, temperature=temperature)
                    L_Ut_m = nt_xent(z_ta, z_tb, temperature=temperature)
                    L_U_m  = (L_Uv_m + L_Ut_m) / 2
                    _LR_g = L_R_m if _LR_g is None else _LR_g + L_R_m
                    _LU_g = L_U_m if _LU_g is None else _LU_g + L_U_m
                    loss = loss + (lam_m * L_R_m + lam_u * L_U_m)
                    LR_sum += L_R_m.item();  LU_sum += L_U_m.item()
                loss = loss / num_preferences
                L_R = torch.as_tensor(LR_sum / num_preferences)
                L_U = torch.as_tensor(LU_sum / num_preferences)
                tau_last = tau; gamma_last = tau / annealing_temperature
                prefs_last = prefs_l; _pref_seen = True
                lam_min_e = min(lams); lam_max_e = max(lams)
                lam_mean_e = sum(lams) / len(lams)
            else:
                h_va = enc_v(v_a, lam);  h_vb = enc_v(v_b, lam)
                h_ta = enc_t(t_a, lam);  h_tb = enc_t(t_b, lam)
                z_vr  = proj_v_r(h_va);  z_tr  = proj_t_r(h_ta)
                z_vua = proj_v_u(h_va);  z_vub = proj_v_u(h_vb)
                z_tua = proj_t_u(h_ta);  z_tub = proj_t_u(h_tb)
                if use_club:
                    opt_critic.zero_grad()
                    crit_loss = (
                        club_v.learning_loss(z_vua.detach(), z_vr.detach()) +
                        club_t.learning_loss(z_tua.detach(), z_tr.detach())
                    )
                    crit_loss.backward()
                    torch.nn.utils.clip_grad_norm_(_params(club_v, club_t), 1.0)
                    opt_critic.step()
                    crit_total += crit_loss.item()
                opt_enc.zero_grad()
                L_R  = infonce_cross(z_vr, z_tr, temperature=temperature)
                L_Uv = nt_xent(z_vua, z_vub, temperature=temperature)
                L_Ut = nt_xent(z_tua, z_tub, temperature=temperature)
                if use_club:
                    L_Uv = L_Uv + lam_club * torch.clamp(club_v(z_vua, z_vr.detach()), min=0.0)
                    L_Ut = L_Ut + lam_club * torch.clamp(club_t(z_tua, z_tr.detach()), min=0.0)
            if not _multi and not _palora and not _simplex and not _simplex4h and not _simplex6h:
                # single-preference loss (multi-preference already set loss + L_R/L_U)
                L_U  = (L_Uv + L_Ut) / 2
                loss = 2 * lam * L_R + 2 * (1.0 - lam) * L_U
                _LR_g, _LU_g = L_R, L_U

            if (approach in (4, 5) and orth_reg_weight > 0 and proj_v is not None
                    and hasattr(proj_v, "orth_loss")):
                L_orth = orth_reg_weight * (proj_v.orth_loss() + proj_t.orth_loss())
                loss = loss + L_orth
                Lorth_total += L_orth.item()

            # Gradient conflict diagnostic
            # Compute g_R and g_U w.r.t. encoder params only, BEFORE the main
            # backward. Low/negative cosine similarity = objectives are fighting.
            # Skipped under multi-preference (L_R/L_U are logged scalars there).
            # Now runs for preference methods too, via the graph-connected handles.
            # Throttled to every 20th step: under multi-preference the retained graph
            # spans all M preferences, so an every-step diagnostic would dominate the
            # step cost. cos_total/_n_cos average only over the steps actually taken.
            _cos_due = (_LR_g is not None and _LU_g is not None
                        and (batch_idx % 20 == 0 or not (_multi or _palora or _simplex
                                                         or _simplex4h or _simplex6h)))
            if enc_grad_params and _cos_due:
                g_R = torch.autograd.grad(_LR_g, enc_grad_params, retain_graph=True, allow_unused=True)
                g_U = torch.autograd.grad(_LU_g, enc_grad_params, retain_graph=True, allow_unused=True)
                g_R_flat = torch.cat([g.flatten() for g in g_R if g is not None])
                g_U_flat = torch.cat([g.flatten() for g in g_U if g is not None])
                if g_R_flat.numel() > 0 and g_U_flat.numel() > 0:
                    cos_sim = F.cosine_similarity(g_R_flat.unsqueeze(0), g_U_flat.unsqueeze(0)).item()
                    cos_total      += cos_sim
                    gR_norm_total  += g_R_flat.norm().item()
                    gU_norm_total  += g_U_flat.norm().item()
                    _n_cos         += 1
                if _head_grad_params:
                    try:
                        hR = torch.autograd.grad(_LR_g, _head_grad_params,
                                                 retain_graph=True, allow_unused=True)
                        hU = torch.autograd.grad(_LU_g, _head_grad_params,
                                                 retain_graph=True, allow_unused=True)
                        a = torch.cat([g.flatten() for g in hR if g is not None])
                        b = torch.cat([g.flatten() for g in hU if g is not None])
                        if a.numel() > 0 and b.numel() > 0:
                            cos_head_total += F.cosine_similarity(a.unsqueeze(0),
                                                                  b.unsqueeze(0)).item()
                            _n_cos_head += 1
                    except Exception:
                        pass

            loss.backward()

            # LoRA branch gradient norms (logged every 20 steps)
            # proj_v is None for the four-head variants (palora_enc/palora_proj),
            # whose adapters live in the encoder or nowhere — no head branches to log.
            # simplex6h keeps three PLAIN heads in a ModuleDict: proj_v is not None
            # but has no .layer1/.A_r, so the `is not None` test alone is not enough.
            # hasattr is the check that actually covers every variant.
            if (approach in (4, 5) and proj_v is not None and (enc_total == 0.0)
                    and hasattr(proj_v, "layer1")):
                for mtag, head in [("v", proj_v), ("t", proj_t)]:
                    for li, layer in enumerate([head.layer1, head.layer2, head.layer3], 1):
                        branches = [("r", layer.A_r, layer.B_r),
                                    ("u", layer.A_u, layer.B_u)]
                        if hasattr(layer, "A_m"):
                            branches.append(("m", layer.A_m, layer.B_m))
                        for btag, A, B in branches:
                            if A.grad is not None and B.grad is not None:
                                gnorm = (A.grad.norm()**2 + B.grad.norm()**2).sqrt().item()
                                writer.add_scalar(f"lora_grad/{mtag}/L{li}/{btag}", gnorm, epoch)

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt_enc.step()
            enc_total  += loss.item()
            LR_total   += L_R.item()
            LU_total   += L_U.item()

        sch_enc.step()
        if use_club:
            sch_critic.step()

        n    = len(loader)
        avg  = enc_total / n
        avg_LR = LR_total / n
        avg_LU = LU_total / n
        writer.add_scalar("loss/total",    avg,    epoch)
        writer.add_scalar("loss/L_R",      avg_LR, epoch)
        writer.add_scalar("loss/L_U",      avg_LU, epoch)
        if curve_reg_weight > 0:
            writer.add_scalar("loss/L_curve", Lcurve_total / n, epoch)
        if orth_reg_weight > 0:
            writer.add_scalar("loss/L_orth", Lorth_total / n, epoch)
        if use_club:
            writer.add_scalar("loss/critic", crit_total / n, epoch)
        if enc_grad_params and _n_cos > 0:
            # divide by the number of steps the diagnostic ACTUALLY ran on, not by
            # len(loader) — it is throttled to every 20th step for preference methods
            writer.add_scalar("grad/cos_sim_LR_LU", cos_total     / _n_cos, epoch)
            writer.add_scalar("grad/norm_LR",       gR_norm_total / _n_cos, epoch)
            writer.add_scalar("grad/norm_LU",       gU_norm_total / _n_cos, epoch)
        if _n_cos_head > 0:
            # conflict measured where the preference actually acts
            writer.add_scalar("grad/cos_sim_heads", cos_head_total / _n_cos_head, epoch)

        if _pref_seen:   # PaLoRA preference / annealing state + adapter diagnostics
            writer.add_scalar("pref/tau",      tau_last,   epoch)
            if _decomp_r:
                # IS THE CLUB OBJECTIVE ACTUALLY WORKING?
                # Five signals, each catching a different silent failure:
                #
                #  club/raw        CLUB's own estimate of I(u;r), BEFORE clamping.
                #                  Should be > 0 and FALLING. Persistently <= 0
                #                  means the critic is undertrained and the term
                #                  contributes no gradient at all.
                #  club/clamp_frac fraction of (preference, modality) evaluations
                #                  clamped to zero. Near 1.0 -> the penalty is
                #                  inert; the run is equivalent to lam_club = 0.
                #  loss/critic     the critic's own learning loss. Must DECREASE;
                #                  a flat critic invalidates the CLUB bound.
                #  decomp/cka_u_r  linear CKA between z_u and z_r, measured
                #                  independently of CLUB. THIS is the effect the
                #                  penalty is supposed to have -- it must FALL.
                #                  If club/raw falls but this does not, CLUB is
                #                  optimising its own estimator, not the geometry.
                #  decomp/eff_rank_u participation ratio of z_u. Guards the
                #                  opposite failure: too large a lam_club drives
                #                  z_u to a constant, which decorrelates it
                #                  trivially while destroying the representation.
                #                  Collapse toward 1 means lam_club is too big.
                _n = max(1, len(loader))
                writer.add_scalar("club/raw",             club_raw_total / _n,  epoch)
                writer.add_scalar("club/clamp_frac",      clampfrac_total / _n, epoch)
                writer.add_scalar("decomp/cka_u_r",       cka_ur_total / _n,    epoch)
                writer.add_scalar("decomp/eff_rank_u",    erank_u_total / _n,   epoch)
                writer.add_scalar("club/lam_club",        lam_club,             epoch)
                #  decomp/cos_r1_r2  cosine between the two R heads. L_R aligns them so
                #                  this should RISE toward 1. If it plateaus near 1 the
                #                  heads are redundant and a single tied R head would do;
                #                  if it stays low, L_R is not converging and the averaged
                #                  R readout block is not well-defined.
                writer.add_scalar("decomp/cos_r1_r2",     cos_r_total / _n,     epoch)
            if rank_penalty > 0.0:
                # falls toward 0 as the preferences become genuinely ordered;
                # if it plateaus high, no Pareto front exists on this data
                writer.add_scalar("pref/rank_penalty", rank_last, epoch)
            writer.add_scalar("pref/gamma",    gamma_last, epoch)
            writer.add_scalar("pref/lam_min",  lam_min_e,  epoch)
            writer.add_scalar("pref/lam_max",  lam_max_e,  epoch)
            writer.add_scalar("pref/lam_mean", lam_mean_e, epoch)
            _scale = getattr(getattr(proj_v, "layer1", None), "scale", 1.0) if proj_v is not None else 1.0
            _four_head = proj_v is None and proj_v_r is not None
            writer.add_scalar("pref/alpha_over_r", float(_scale), epoch)
            if prefs_last is not None:
                for _i, (_a, _b) in enumerate(prefs_last):
                    writer.add_scalar(f"pref/lambda_{_i}_R", _a, epoch)
                    writer.add_scalar(f"pref/lambda_{_i}_U", _b, epoch)
            # Where the LoRA adapters live differs per variant:
            #   ap4/5 LoRA head -> heads;  palora_enc -> encoder;
            #   palora_proj -> nowhere (4 plain heads), so rho is undefined (NaN).
            if _palora_enc:
                _rho_R, _rho_U = _lora_rhos(enc_v, enc_t)
            elif proj_v is not None:
                _rho_R, _rho_U = _lora_rhos(proj_v, proj_t)
            else:
                _rho_R = _rho_U = float("nan")
            writer.add_scalar("adapter/rho_R", _rho_R, epoch)
            writer.add_scalar("adapter/rho_U", _rho_U, epoch)
            try:
                if _simplex4h or _simplex6h:
                    # Output-mixing variants: the simplex endpoints, through the
                    # same mixing rule training used. These previously fell to the
                    # shared-base path and logged NaN.
                    def _zmix(pref):
                        w_r, w_u1, w_u2 = pref
                        if _simplex6h:
                            m = lambda pd, h: F.normalize(w_r * pd["r"](h) + w_u1 * pd["u1"](h)
                                                          + w_u2 * pd["u2"](h), dim=-1)
                            return torch.cat([m(proj_v, h_va), m(proj_t, h_ta)], dim=-1)
                        def m4(pr, pu, h, wu):
                            # same duo_txt fallback as the training mix: vision's
                            # uniqueness head is untrained under duo_txt.
                            _fb = (1.0, 0.0) if _duo else (0.5, 0.5)
                            a, b = (w_r, wu) if (w_r + wu) > 0 else _fb
                            if _s4h_premix:
                                return F.normalize(a * pr.net(h) + b * pu.net(h), dim=-1)
                            return F.normalize(a * pr(h) + b * pu(h), dim=-1)
                        return torch.cat([m4(proj_v_r, proj_v_u, h_va, w_u1),
                                          m4(proj_t_r, proj_t_u, h_ta, w_u2)], dim=-1)
                    _ecka = _endpoint_cka_mix(_zmix)
                elif _four_head:
                    # z(λ)=normalize(λ·z_R+(1-λ)·z_U) through the four separate heads.
                    # palora_enc additionally re-runs the λ-conditioned encoder.
                    _hv = (lambda l: enc_v(v_a, l)) if _palora_enc else (lambda l: h_va)
                    _ht = (lambda l: enc_t(t_a, l)) if _palora_enc else (lambda l: h_ta)
                    _mk = lambda l: torch.cat([
                        F.normalize(l * proj_v_r(_hv(l)) + (1 - l) * proj_v_u(_hv(l)), dim=-1),
                        F.normalize(l * proj_t_r(_ht(l)) + (1 - l) * proj_t_u(_ht(l)), dim=-1)], dim=-1)
                    with torch.no_grad():
                        _X, _Y = _mk(1.0).float(), _mk(0.0).float()
                        _X = _X - _X.mean(0, keepdim=True); _Y = _Y - _Y.mean(0, keepdim=True)
                        _ecka = float(torch.linalg.norm(_X.T @ _Y) ** 2 /
                                      (torch.linalg.norm(_X.T @ _X) * torch.linalg.norm(_Y.T @ _Y)))
                else:
                    _ecka = _endpoint_cka(proj_v, proj_t, h_va, h_ta)
                writer.add_scalar("adapter/endpoint_cka", _ecka, epoch)
            except Exception:
                _ecka = float("nan")

        if epoch % 10 == 0 or epoch == 1:
            cstr  = f"  crit={crit_total/n:.4f}" if use_club else ""
            if _decomp_r:
                # Is the CLUB objective actually doing anything? club_raw is CLUB's own
                # estimate and is only meaningful once the critic is fit; cka(u,r) is the
                # EFFECT, measured independently, and must FALL. effrank_u guards the
                # opposite failure -- too large a lam_club collapses z_u to a constant.
                _cf = clampfrac_total / n
                _verdict = ("CLUB INERT (clamped)" if _cf > 0.8 else
                            "collapse risk" if erank_u_total / n < 2.0 else "binding")
                cstr += (f"  club_raw={club_raw_total/n:+.4f}  clamp={_cf:.2f}"
                         f"  cka(u,r)={cka_ur_total/n:.4f}"
                         f"  effrank_u={erank_u_total/n:.1f}"
                         f"  cos(r1,r2)={cos_r_total/n:+.3f}  [{_verdict}]")
            gstr  = (f"  cos(gR,gU)={cos_total/_n_cos:+.3f}"
                     if enc_grad_params and _n_cos > 0 else "")
            crstr  = f"  L_curve={Lcurve_total/n:.4f}" if curve_reg_weight > 0 else ""
            orstr  = f"  L_orth={Lorth_total/n:.4f}"   if orth_reg_weight  > 0 else ""
            # simplex_proj_decomp_R: the arm's premise is that the HEAD adapters can
            # express the preference. If dW/W stays near zero the heads are effectively
            # lambda-blind and the arm silently degenerates to fixed-uniform training --
            # which would look like "placement does not matter" when it actually means
            # "the adapters were too small to place anything". decompR's trained ENCODER
            # sits at dW/W ~ 0.20, so that is the number to compare against.
            hstr = ""
            if _head_cond:
                with torch.no_grad():
                    _rat = []
                    for _hd in (proj_v_r, proj_t_r, proj_v_u, proj_t_u):
                        for _ly in (_hd.layer1, _hd.layer2, _hd.layer3):
                            _w = _ly.linear.weight.norm()
                            if _w > 0:
                                _rat.append(max(float((_ly.A_r @ _ly.B_r).norm()),
                                                float((_ly.A_u @ _ly.B_u).norm())) / float(_w))
                    if _rat:
                        _rat.sort()
                        hstr = f"  head_dW/W={_rat[len(_rat)//2]:.4f}"
            if _pref_seen:
                _pl = " ".join(f"[{a:.2f},{b:.2f}]" for a, b in (prefs_last or []))
                pstr = (f"  τ={tau_last:.3f} γ={gamma_last:.3f} α/r={float(_scale):g}"
                        f"  ρ_R={_rho_R:.3f} ρ_U={_rho_U:.3f} eCKA={_ecka:.3f}  λ={_pl}")
            else:
                pstr = ""
            print(f"  epoch {epoch:3d}/{epochs}  total={avg:.4f}  L_R={avg_LR:.4f}  L_U={avg_LU:.4f}{cstr}{crstr}{orstr}{gstr}{hstr}{pstr}")

        # Periodic checkpoint
        if epoch % ckpt_every == 0 or epoch == epochs:
            ep_dir = os.path.join(ckpt_dir, f"ep{epoch:04d}")
            os.makedirs(ep_dir, exist_ok=True)
            torch.save(enc_v.state_dict(), os.path.join(ep_dir, "enc_v.pth"))
            torch.save(enc_t.state_dict(), os.path.join(ep_dir, "enc_t.pth"))
            if proj_v is not None:      # shared-base head (ap3/4/5 LoRA or FiLM)
                torch.save(proj_v.state_dict(), os.path.join(ep_dir, "proj_v.pth"))
                torch.save(proj_t.state_dict(), os.path.join(ep_dir, "proj_t.pth"))
            else:
                for name, mod in [("proj_v_r", proj_v_r), ("proj_t_r", proj_t_r),
                                   ("proj_v_u", proj_v_u), ("proj_t_u", proj_t_u)]:
                    torch.save(mod.state_dict(), os.path.join(ep_dir, f"{name}.pth"))
            # Resumable bundle: models + optimizer + scheduler + RNG + epoch, and
            # the CLUB critics when present. _save_resume now covers BOTH the
            # shared-base head and the four-head variants, so preference-conditioned
            # runs are resumable too -- previously they were not, and a cancelled
            # 5-hour run had to start over.
            if approach in (3, 4, 5):
                _save_resume(epoch)

    writer.close()

    torch.save(enc_v.state_dict(), os.path.join(out_dir, "enc_v.pth"))
    torch.save(enc_t.state_dict(), os.path.join(out_dir, "enc_t.pth"))
    if proj_v is not None:          # shared-base head (ap3/4/5 LoRA or FiLM)
        torch.save(proj_v.state_dict(), os.path.join(out_dir, "proj_v.pth"))
        torch.save(proj_t.state_dict(), os.path.join(out_dir, "proj_t.pth"))
    else:
        for name, mod in [("proj_v_r", proj_v_r), ("proj_t_r", proj_t_r),
                          ("proj_v_u", proj_v_u), ("proj_t_u", proj_t_u)]:
            torch.save(mod.state_dict(), os.path.join(out_dir, f"{name}.pth"))

    with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
        json.dump(dict(approach=approach, method=method, dataset=dataset,
                       enc_dim=adim, proj_dim=proj_dim, epochs=epochs,
                       lam_dist=lam_dist, seed=seed, film_mode=film_mode,
                       prefs_per_batch=prefs_per_batch, anneal_mode=anneal_mode,
                       pref_visits=(_pref_cycle.counts.tolist()
                                    if "_pref_cycle" in locals() else None),
                       pref_loss_mean=(( _pref_loss_sum /
                                         np.maximum(_pref_cycle.counts, 1)).tolist()
                                       if "_pref_cycle" in locals() else None),
                       head_lr_mult=head_lr_mult,
                       proj_prenorm=proj_prenorm, warmup_epochs=warmup_epochs,
                       lora_branches=lora_branches, lora_rank=lora_rank,
                       lora_rank_m=lora_rank_m if lora_rank_m is not None else lora_rank,
                       lora_lr_mult=lora_lr_mult, lora_lr_mult_m=lora_lr_mult_m,
                       lora_lr_mult_t=lora_lr_mult_t,
                       var_reg_weight=var_reg_weight, var_reg_gamma=var_reg_gamma,
                       curve_reg_weight=curve_reg_weight, orth_reg_weight=orth_reg_weight,
                       lora_alpha=lora_alpha, freeze_W0=freeze_W0,
                       num_preferences=num_preferences,
                       preference_schedule=preference_schedule,
                       annealing_temperature=annealing_temperature,
                       simplex_side=(simplex_side if film_mode in ("simplex", DUO_TXT) else None),
                       duo_axis=("text" if film_mode == DUO_TXT else None)), f, indent=2)
    print(f"  Saved → {out_dir}")


# CLI

def _set_enc_width(w):
    """Override the module-level Transformer width. Must run before any encoder is
    built, since ENC_DIM_XFMR is read at construction time."""
    global ENC_DIM_XFMR
    if w is None:
        return
    if w % 10 != 0:
        raise ValueError(f"--enc_width must be a multiple of 10 (nhead=5, even-dim "
                         f"sincos posemb); got {w}")
    ENC_DIM_XFMR = w
    # The approach-5 encoder is LoRATransformerEncoder, which carries its OWN
    # hardcoded width (_D = 40) in networks.py and never reads ENC_DIM_XFMR. Setting
    # only the module constant gave a 40-d encoder feeding 20-d heads:
    #   RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x40 and 20x20)
    LoRATransformerEncoder._D = w
    print(f"  ENC_DIM_XFMR overridden -> {w}  (LoRATransformerEncoder._D too)")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config",     default=None, help="Path to YAML config (default: multibench_config.yaml)")
    p.add_argument("--approach",   type=int, choices=[1, 2, 3, 4, 5], default=None)
    p.add_argument("--method",     choices=["simclr_single_per_batch", "factorcl_warmup", "gmc", "comm", "factorcl", "clip", "cross_self"], default=None)
    p.add_argument("--dataset",    choices=["mosi", "humor", "mustard", "mosei", "avmnist", "enrico", "mosei_multitask", "chsims"], default=None)
    p.add_argument("--out_dir",    default="pareto_ssl/multibench/results/run")
    p.add_argument("--enc_dim",    type=int, default=None)
    p.add_argument("--proj_dim",   type=int, default=None)
    p.add_argument("--enc_width",  type=int, default=None,
                   help="Transformer hidden dim (ap2-5). Must be a multiple of 10 "
                        "(nhead=5 and even-dim sincos). Default 40 = CoMM. Lowering "
                        "it squeezes the SHARED representation, which is the "
                        "intended way to force objective conflict.")
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lam_dist",   default=None)
    p.add_argument("--dirichlet_alpha", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--lam_club",   type=float, default=None)
    p.add_argument("--prefs_per_batch", type=int, default=0,
                   help="M target preferences per minibatch, drawn by a balanced shuffled "
                        "cycle over the 15-point grid (each preference visited once per "
                        "cycle). One forward per selected preference, ONE backward and ONE "
                        "optimizer step on their mean, exactly as with all 15. 0 or >=15 = "
                        "the full grid (default). Cost scales with M: every preference needs "
                        "its own encoder forward on the transformer backbones, which is why "
                        "MOSEI is ~50x a baseline at M=15.")
    p.add_argument("--overwrite_ok", action="store_true",
                   help="allow writing into an out_dir that holds a completed run with "
                        "DIFFERENT settings (refused by default, to protect finished work)")
    p.add_argument("--anneal_mode", default="power", choices=["power", "linear"],
                   help="target -> effective preference map. power (default, historical) "
                        "leaves vertices at the extremes for any gamma>0; linear is the true "
                        "centre-to-target interpolation (1-eta)*centroid + eta*target.")
    p.add_argument("--club_hidden_dim", type=int, default=None)
    p.add_argument("--club_layers", type=int, default=None)
    p.add_argument("--club_iters", type=int, default=1,
                   help="factorcl: CLUB-critic optimisation steps per batch. The official "
                        "code's default is 1; >1 is a DIAGNOSTIC for critic under-training, "
                        "not a reproduction of the published setup.")
    p.add_argument("--clip_logit_scale", action="store_true", default=False,
                   help="clip: use a LEARNED temperature (CLIP's logit_scale, init "
                        "log(1/0.07), clamped at 100) instead of the fixed --temperature. "
                        "This is the faithful CLIP objective; the fixed-temperature form "
                        "is a simplification.")
    p.add_argument("--loss_w", default=None,
                   help='Re-weight a baseline\'s loss terms, e.g. "0.5,0.25,0.25". This is '
                        'the control for "why not train N baselines with different loss '
                        'weights instead of one STEER?" -- it gives each baseline the same '
                        '15-point preference grid STEER sweeps, one TRAINED MODEL per grid '
                        'point. gmc/comm take 2 weights (vision, text: the only axis those '
                        'objectives have, since neither contains a uniqueness term). '
                        'factorcl takes 3 (R, U_v, U_t), grouped per Eqs. 8-9: R -> '
                        '{infonce_x1x2, club_cond}, U_i -> {infonce_xi_xi\'}, with '
                        '{club_x1x2, infonce_cond} shared by both U terms. Omit for the '
                        'published uniform weighting.')
    p.add_argument("--ssl_scale", type=float, default=1.0,
                   help="cross_self: weight on the two unimodal InfoNCE terms. 1.0 is the "
                        "reference default (CoMM repo losses/cross_self.py).")
    p.add_argument("--factorcl_official", action="store_true", default=False,
                   help="factorcl: reproduce the published setup EXACTLY -- Adam (not "
                        "AdamW), lr 1e-4, no weight decay, no LR scheduler, no gradient "
                        "clipping, main-loss-then-critic ordering with the critic seeing "
                        "recomputed post-update embeddings, and unnormalised mlp_head(d,d) "
                        "heads. Width is INDEPENDENT: default d=enc_dim (400-dim "
                        "get_embedding, faithful); add --proj_dim 7 for the same recipe "
                        "at a 70-dim width-matched embedding.")
    p.add_argument("--critic_lr_mult", type=float, default=1.0,
                   help="factorcl: multiplier on the CLUB critic LR. Official uses the same "
                        "LR as the rest of the model (1.0). Diagnostic only.")
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--enc_ckpt_v", default=None)
    p.add_argument("--enc_ckpt_t", default=None)
    p.add_argument("--film_mode",    choices=list(FILM_MODES), default=None,
                   help="λ-conditioning: none | token | full | proj (FiLM projection heads)")
    p.add_argument("--head_lr_mult", type=float, default=None,
                   help="Ablation 2: projection heads use lr * head_lr_mult")
    p.add_argument("--proj_prenorm", action="store_true", default=False,
                   help="Approach 3 variant: apply FiLM before L2 norm (DualFiLMProjectionHeadPreNorm)")
    p.add_argument("--lora_branches", type=int, choices=[2, 3], default=None,
                   help="Approach 4: 3=R+U+M curvature (default), 2=linear ablation R+U only")
    p.add_argument("--lora_rank",    type=int, default=None,
                   help="Approach 4: rank for R and U branches (default 4)")
    p.add_argument("--lora_rank_m",  type=int, default=None,
                   help="Approach 4: rank for M branch (default = lora_rank)")
    p.add_argument("--lora_lr_mult", type=float, default=None,
                   help="Approach 4: R/U branch LR = lr * head_lr_mult * lora_lr_mult (default 10)")
    p.add_argument("--lora_lr_mult_m", type=float, default=None,
                   help="Approach 4: M branch LR = lora_lr * lora_lr_mult_m (default 3)")
    p.add_argument("--lora_lr_mult_t", type=float, default=None,
                   help="Approach 4 fix2: text LoRA LR relative to vision LoRA (default 1.0 = same; use <1 to slow text)")
    p.add_argument("--warmup_epochs", type=int, default=None,
                   help="Approach 4 fix3: endpoint training for first N epochs then curve (default 0)")
    p.add_argument("--var_reg_weight", type=float, default=None,
                   help="Approach 4 fix1: variance regularization weight (default 0 = disabled)")
    p.add_argument("--var_reg_gamma",  type=float, default=None,
                   help="Approach 4 fix1: minimum std per embedding dimension (default 0.05)")
    p.add_argument("--curve_reg_weight", type=float, default=None,
                   help="Approach 6: curve smoothness regularization weight β (default 0 = disabled)")
    p.add_argument("--orth_reg_weight", type=float, default=None,
                   help="Approach 7: LoRA R/U subspace orthogonality weight γ (default 0 = disabled)")
    p.add_argument("--lora_layers",    type=int, choices=[1, 2, 3], default=None,
                   help="comm_based ap4: how many MLP layers carry LoRA, from output end (default 3 = all)")
    p.add_argument("--arch", choices=["factorcl_based", "comm_based"], default=None,
                   help="Encoder architecture: factorcl_based (original) or comm_based (real CoMM FusionTransformer+MLP)")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Resume ap3/4/5 training from <out_dir>/resume.pth if present (else start fresh)")
    # Fix 8: PaLoRA multi-preference training
    p.add_argument("--num_preferences", type=int, default=None,
                   help="M preferences per minibatch (Fix 8). 1 = single-λ baseline")
    p.add_argument("--preference_schedule", choices=["single", "fixed", "annealed"], default=None,
                   help="single = one sampled λ (baseline); fixed = M evenly-spaced λ; "
                        "annealed = PaLoRA center-to-edge schedule")
    p.add_argument("--annealing_temperature", type=float, default=None,
                   help="Q in the PaLoRA annealing schedule (>0; lower = faster separation)")
    p.add_argument("--rank_penalty", type=float, default=0.0,
                   help="Fix 9: weight beta on the preference-ranking penalty. "
                        "Forces a preference that up-weights an objective to be "
                        "genuinely better at it. 0 disables (default).")
    p.add_argument("--rank_margin", type=float, default=0.20,
                   help="Fix 9: loss separation demanded between the EXTREME "
                        "preferences, in nats; scaled by weight distance for "
                        "intermediate pairs (default 0.20)")
    p.add_argument("--lora_alpha", type=float, default=None,
                   help="PaLoRA α; effective LoRA scale is α/r. Applied only when "
                        "--preference_schedule annealed (default None → α=r, scale 1.0)")
    p.add_argument("--fusion_hidden", type=int, default=None,
                   help="gmc/comm only: give the fusion head an MLP of this width so its "
                        "evaluation-time parameter budget can be matched to the lambda-arms")
    p.add_argument("--fusion_out", type=int, default=None,
                   help="gmc/comm only: fusion output dim (default enc_dim)")
    p.add_argument("--simplex_side", type=int, default=5,
                   help="film_mode=simplex: points per edge of the (λ_R,λ_U1,λ_U2) grid "
                        "(5 -> 15 preferences)")
    p.add_argument("--freeze_W0", action="store_true", default=False,
                   help="PaLoRA Pareto-expansion ablation: freeze the shared base W0 and "
                        "train only the LoRA branches. Default OFF (from-scratch setting).")
    p.add_argument("--enrico_unfreeze", action="store_true", default=False,
                   help="ENRICO: fine-tune the VGG conv features instead of freezing them "
                        "(ImageNet features are out-of-domain for UI images)")
    args = p.parse_args()
    _set_enc_width(args.enc_width)   # before any encoder is constructed

    # Load config then override with any CLI args that were explicitly set
    cfg = _load_config(args.config)

    def _get(cli_val, *cfg_keys, default):
        return cli_val if cli_val is not None else _cfg(cfg, *cfg_keys, default=default)

    train(
        approach    = _get(args.approach,        default=2),
        method      = _get(args.method,          default="factorcl_warmup"),
        dataset     = _get(args.dataset,         default="humor"),
        out_dir     = args.out_dir,
        enc_dim     = _get(args.enc_dim,         "encoder",   "enc_dim",       default=128),
        proj_dim    = _get(args.proj_dim,        "projection","proj_dim",       default=64),
        # Did the CLI actually name a width? _get() resolves an unset --proj_dim to the
        # config default (64), so "unset" is indistinguishable downstream -- which made
        # --factorcl_official silently build 64-wide heads (640-dim get_embedding)
        # instead of the official mlp_head(d, d) at the encoder width.
        proj_dim_explicit = args.proj_dim is not None,
        epochs      = _get(args.epochs,          "training",  "epochs",        default=200),
        lr          = _get(args.lr,              "training",  "lr",            default=3e-4),
        batch_size  = _get(args.batch_size,      "training",  "batch_size",    default=128),
        weight_decay= _get(args.weight_decay,    "training",  "weight_decay",  default=1e-4),
        seed        = _get(args.seed,            "training",  "seed",          default=42),
        lam_dist    = _get(args.lam_dist,        "lambda_sampling", "dist",    default="uniform"),
        dirichlet_alpha = _get(args.dirichlet_alpha, "lambda_sampling", "dirichlet_alpha", default=1.0),
        temperature = _get(args.temperature,     "loss",      "temperature",   default=0.1),
        lam_club    = _get(args.lam_club,        "loss",      "lam_club",      default=0.5),
        prefs_per_batch = args.prefs_per_batch,
        anneal_mode = args.anneal_mode,
        overwrite_ok = args.overwrite_ok,
        club_hidden_dim = _get(args.club_hidden_dim, "club", "hidden_dim",     default=512),
        club_layers = _get(args.club_layers,     "club",      "layers",        default=1),
        device      = args.device,
        enc_ckpt_v  = args.enc_ckpt_v,
        enc_ckpt_t  = args.enc_ckpt_t,
        film_mode       = _get(args.film_mode,      "encoder",  "film_mode",      default="none"),
        head_lr_mult    = _get(args.head_lr_mult,   "training", "head_lr_mult",   default=1.0),
        lora_branches   = _get(args.lora_branches,  "lora",     "branches",       default=3),
        lora_rank       = _get(args.lora_rank,      "lora",     "rank",           default=4),
        lora_rank_m     = _get(args.lora_rank_m,    "lora",     "rank_m",         default=None),
        lora_lr_mult    = _get(args.lora_lr_mult,   "lora",     "lr_mult",        default=10.0),
        lora_lr_mult_m  = _get(args.lora_lr_mult_m, "lora",     "lr_mult_m",     default=3.0),
        proj_prenorm    = args.proj_prenorm,
        warmup_epochs   = _get(args.warmup_epochs, "lora", "warmup_epochs", default=0),
        lora_lr_mult_t  = _get(args.lora_lr_mult_t, "lora",     "lr_mult_t",     default=1.0),
        var_reg_weight   = _get(args.var_reg_weight,   "lora", "var_reg_weight",   default=0.0),
        var_reg_gamma    = _get(args.var_reg_gamma,    "lora", "var_reg_gamma",    default=0.05),
        curve_reg_weight = _get(args.curve_reg_weight, "lora", "curve_reg_weight", default=0.0),
        orth_reg_weight  = _get(args.orth_reg_weight,  "lora", "orth_reg_weight",  default=0.0),
        lora_layers     = _get(args.lora_layers,     "lora",     "lora_layers",    default=3),
        arch            = _get(args.arch,            "training", "arch",           default="factorcl_based"),
        resume          = args.resume,
        num_preferences       = _get(args.num_preferences,       "palora", "num_preferences",       default=1),
        preference_schedule   = _get(args.preference_schedule,   "palora", "preference_schedule",   default="single"),
        annealing_temperature = _get(args.annealing_temperature, "palora", "annealing_temperature", default=1.0),
        lora_alpha            = _get(args.lora_alpha,            "palora", "lora_alpha",            default=None),
        enrico_unfreeze       = args.enrico_unfreeze,
        freeze_W0             = args.freeze_W0,
        simplex_side          = args.simplex_side,
        fusion_hidden         = args.fusion_hidden,
        fusion_out            = args.fusion_out,
        club_iters            = args.club_iters,
        critic_lr_mult        = args.critic_lr_mult,
        factorcl_official     = args.factorcl_official,
        ssl_scale             = args.ssl_scale,
        loss_w                = ([float(x) for x in args.loss_w.split(",")]
                                 if args.loss_w else None),
        clip_logit_scale      = args.clip_logit_scale,
        rank_penalty          = args.rank_penalty,
        rank_margin           = args.rank_margin,
    )


if __name__ == "__main__":
    main()
