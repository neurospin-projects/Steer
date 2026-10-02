"""
Pareto Frontier SSL — Benchmark (AlexNet version)
==================================================
Dataset  : trifeatures_corr00, 2400 train / 200 test (matching CoMM notebook).
Probe    : pair-based K-shot (default 10 pairs/class = 100 total).
           K-shot forces the probe to use only the clearest structure in the
           representation — methods that blur unique info show lower scores.

Architectures
-------------
  simclr_both / clip / factorcl_heads / factorcl
    → standard AlexNetEncoder(latent_dim) — global avg-pool → flat vector
    → unimodal_split probe: enc_m1 for share/unique1, enc_m2 for unique2

  gmc / comm
    → CoMM's MMFusion: AlexNetEncoder(global_pool='') → 256×6×6 spatial maps
                       PatchedInputAdapter → 36 tokens × 512-d
                       FusionTransformer with CLS token → 512-d joint representation
    → mmfusion_joint probe: MMFusion([M1, M2]) CLS token

Usage
-----
cd <repo root>
python pareto_ssl/benchmark.py \\
    --data_dir data/trifeatures_corr00 \\
    --out_dir  pareto_ssl/results_alexnet \\
    --epochs   100 --probe_shots 10 --device cuda

# Re-plot without retraining:
python pareto_ssl/benchmark.py --out_dir pareto_ssl/results_alexnet --plot_only
"""

import argparse
import datetime
import os
import json
import shutil
import sys
import warnings
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CoMM"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# CoMM architecture -- needed ONLY by the gmc / comm baselines and the
# mmfusion_joint probe. It is imported lazily (see _import_comm below) so that
# STEER, CLIP and FactorCL run without a CoMM checkout present.
_COMM_HINT = (
    "The '{what}' path needs the CoMM reference implementation, which is a separate "
    "project and is not redistributed here.\n"
    "Clone it next to this repository:\n"
    "    git clone https://github.com/Duplums/CoMM\n"
    "so that <repo root>/CoMM/models/ exists, then re-run.\n"
    "STEER itself, CLIP and FactorCL do not need it."
)


def _import_comm(what="gmc / comm"):
    """Import CoMM's encoder/fusion modules on demand, with an actionable error."""
    try:
        from models.alexnet import AlexNetEncoder as CoMMEncoder
        from models.input_adapters import PatchedInputAdapter
        from models.mmfusion import MMFusion
    except ImportError as e:
        raise ImportError(_COMM_HINT.format(what=what)) from e
    return CoMMEncoder, PatchedInputAdapter, MMFusion

from pareto_ssl.datasets import (set_pair as set_modality_pair, pair_unique_keys,
                                 get_pair as _ds_pair, set_augment as _set_aug,
                                 get_augment as _get_aug, TrifeatureTrainDataset,
                                 PairProbeDataset, TASKS)
from pareto_ssl.networks import (AlexNetEncoder, ProjectionHead, CoMMProjectionHead,
                                  FiLMProjectionHead, HyperNetEncoder,
                                  LoRADualProjectionHead, LoRATriProjectionHead,
                                  LoRAAlexNetEncoder)
from pareto_ssl.losses import nt_xent, infonce_cross, comm_infonce, gmc_loss
from pareto_ssl.factorcl import CLUBInfoNCECritic
from torch.utils.tensorboard import SummaryWriter

# Load central config (pareto_config.yaml)
_CFG_PATH = Path(__file__).resolve().parent / "pareto_config.yaml"
with open(_CFG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

METHODS        = _CFG["methods"]
INFERENCE_MODE = _CFG["inference_mode"]
LAMBDA_METHODS = set(_CFG["lambda_methods"])
COMM_METHODS   = set(_CFG["comm_methods"])
LAMBDA_GRID    = _CFG["lambda_grid"]

_P = _CFG.get("paths", {})
DEFAULT_DATA_DIR = _P.get("data_dir", "data/trifeatures_corr00")
DEFAULT_OUT_DIR  = _P.get("out_dir",  "pareto_ssl/results_alexnet")

_T = _CFG["training"]
DEFAULT_EPOCHS      = _T["epochs"]
DEFAULT_LR          = _T["lr"]
DEFAULT_BATCH_SIZE  = _T["batch_size"]
DEFAULT_LATENT_DIM  = _T["latent_dim"]
DEFAULT_PROJ_DIM    = _T["proj_dim"]
DEFAULT_PROBE_SHOTS = _T["probe_shots"]
DEFAULT_TRAIN_SEED  = _T["train_seed"]
_cfg_device         = _T.get("device", "auto")
DEFAULT_DEVICE      = ("cuda" if torch.cuda.is_available() else "cpu") \
                      if _cfg_device == "auto" else _cfg_device

_L = _CFG["loss_weights"]
LAM_CLUB     = _L["lam_club"]
ALPHA_SMOOTH = _L["alpha_smooth"]
BETA_DIV     = _L["beta_div"]

_S = _CFG["lambda_sampling"]
DEFAULT_LAM_DIST        = _S["dist"]
DEFAULT_DIRICHLET_ALPHA = _S["dirichlet_alpha"]

_A = _CFG.get("architecture", {})
DEFAULT_ENCODER_PARAM = _A.get("encoder_param", "projection")

_R = _CFG.get("run", {})
DEFAULT_METHODS_LIST   = _R.get("default_methods", None)   # None → all METHODS keys
DEFAULT_SKIP_TRAINING  = bool(_R.get("skip_training", False))
DEFAULT_REPROBE_ALL    = bool(_R.get("reprobe_all",   False))
DEFAULT_PLOT_ONLY      = bool(_R.get("plot_only",     False))

DEFAULT_PARETO_SOLVER  = _CFG.get("pareto_solver", "ls")
_CHECKPOINT_ALIAS      = _CFG.get("checkpoint_alias", {})

_WU = _CFG.get("lambda_warmup", {})
LAMBDA_WARMUP_PHASES = _WU.get("phases", [
    {"frac": 0.20, "dist": "beta",    "alpha": 5.0, "beta": 1.0},
    {"frac": 0.60, "dist": "uniform"},
    {"frac": 0.20, "dist": "beta",    "alpha": 0.5, "beta": 0.5},
])
_frac_sum = round(sum(p["frac"] for p in LAMBDA_WARMUP_PHASES), 6)
if abs(_frac_sum - 1.0) > 1e-4:
    raise ValueError(
        f"lambda_warmup.phases fracs must sum to 1.0, got {_frac_sum:.4f}. "
        f"Fix pareto_config.yaml."
    )


def sample_lambda(dist: str = DEFAULT_LAM_DIST,
                  alpha: float = DEFAULT_DIRICHLET_ALPHA,
                  beta_param: float = None) -> float:
    """
    Draw λ ∈ (0,1) from the requested distribution.

    dist="uniform"   → λ ~ Uniform(0,1)
    dist="beta"      → λ ~ Beta(alpha, beta_param)   [asymmetric supported]
    dist="dirichlet" → λ ~ Beta(alpha, alpha)         [symmetric, legacy name]
    """
    if dist == "uniform":
        return torch.rand(1).item()
    elif dist in ("beta", "dirichlet"):
        b = beta_param if beta_param is not None else alpha
        return float(torch.distributions.Beta(
            torch.tensor(alpha), torch.tensor(b)
        ).sample().item())
    else:
        raise ValueError(f"Unknown lam_dist {dist!r}. Choose 'uniform', 'beta', or 'dirichlet'.")


def sample_lambda_warmup(epoch: int, total_epochs: int,
                         phases: list = LAMBDA_WARMUP_PHASES) -> float:
    """
    Phase-aware λ sampler for factorcl_warmup.

    Iterates phases in order; each phase covers `frac` of total_epochs.
    Within the active phase samples λ from the specified distribution.

    Motivation: CLUB needs a converged z_r before it can disentangle z_u.
    Starting with Beta(5,1) (λ≈1, shared-first) lets z_r stabilise before
    the bimodal late phase pushes toward the unique endpoint.
    """
    progress = (epoch - 1) / max(total_epochs - 1, 1)   # 0 → 1
    cumulative = 0.0
    for phase in phases:
        cumulative += phase["frac"]
        if progress <= cumulative + 1e-9:
            return sample_lambda(
                dist       = phase.get("dist",  "uniform"),
                alpha      = phase.get("alpha", 1.0),
                beta_param = phase.get("beta",  phase.get("alpha", 1.0)),
            )
    # fallback: last phase
    last = phases[-1]
    return sample_lambda(last.get("dist", "uniform"),
                         last.get("alpha", 1.0),
                         last.get("beta", last.get("alpha", 1.0)))

def _sample_lam(lam_dist: str, dirichlet_alpha: float,
                epoch: int = 1, epochs: int = 1) -> float:
    """Unified λ sampler — handles 'warmup' in addition to the base distributions."""
    if lam_dist == "warmup":
        return sample_lambda_warmup(epoch, epochs)
    return sample_lambda(lam_dist, dirichlet_alpha)


def _warmup_grid_weights(epoch: int, total_epochs: int,
                         phases: list, grid: list) -> list:
    """
    Per-grid-point weights based on the active warmup phase's Beta PDF.
    Used by hyper_lambda when lam_dist='warmup' to bias which lambda values
    contribute most to the batch loss (mirrors single-sample warmup behaviour).
    Returns a list of floats that sum to 1.0.
    """
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    cumulative = 0.0
    current_phase = phases[-1]
    for phase in phases:
        cumulative += phase["frac"]
        if progress <= cumulative + 1e-9:
            current_phase = phase
            break

    dist = current_phase.get("dist", "uniform")
    if dist == "uniform":
        return [1.0 / len(grid)] * len(grid)

    a = float(current_phase.get("alpha", 1.0))
    b = float(current_phase.get("beta", a))
    beta_d = torch.distributions.Beta(torch.tensor(a), torch.tensor(b))
    weights = [
        beta_d.log_prob(torch.tensor(max(min(float(lam), 1 - 1e-6), 1e-6))).exp().item()
        for lam in grid
    ]
    total = sum(weights) or 1.0
    return [w / total for w in weights]


def set_seed(seed: int):
    """Seed all RNGs so training is fully reproducible."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# CoMM MMFusion factory

def _make_mmfusion(device: str):
    """
    Exact architecture from CoMM trifeatures notebook:
      AlexNetEncoder(global_pool='') → (B, 256, 6, 6) spatial maps
      PatchedInputAdapter            → 36 tokens of 512-d
      FusionTransformer + CLS token  → 512-d joint representation
    """
    CoMMEncoder, PatchedInputAdapter, MMFusion = _import_comm()
    return MMFusion(
        encoders=[
            CoMMEncoder(latent_dim=512, global_pool='').to(device),
            CoMMEncoder(latent_dim=512, global_pool='').to(device),
        ],
        input_adapters=[
            PatchedInputAdapter(num_channels=256, stride_level=1,
                                patch_size_full=1, dim_tokens=512, image_size=6).to(device),
            PatchedInputAdapter(num_channels=256, stride_level=1,
                                patch_size_full=1, dim_tokens=512, image_size=6).to(device),
        ],
        embed_dim=512,
    ).to(device)

# Training

def _params(*modules):
    return [p for m in modules for p in m.parameters()]


def save_random_encoders(out_dir: str, device: str, latent_dim: int):
    """Save randomly initialised (untrained) encoders — sanity baseline."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(AlexNetEncoder(latent_dim).state_dict(), os.path.join(out_dir, "enc_0.pth"))
    torch.save(AlexNetEncoder(latent_dim).state_dict(), os.path.join(out_dir, "enc_1.pth"))
    print(f"  Saved random encoders to {out_dir}")


def _anneal_preferences(m: int, tau: float, q: float, device) -> torch.Tensor:
    """PaLoRA center-to-edge preference annealing (Dimitriadis et al.).

    Base λ̃_m evenly spaced in [0,1]; p̃_m = [λ̃_m, 1-λ̃_m]. With τ ∈ [0,1] and temp Q:
        exponent = τ/Q ;  p_m = p̃_m**exponent / Σ_j p̃_m,j**exponent ;  λ_m = p_m[0]
    τ=0 → all 0.5 (handled explicitly to avoid 0**0); τ=1 → the base λ̃.
    Identical to the multibench trainer's helper so both benchmarks anneal the same.
    """
    base = torch.linspace(0.0, 1.0, m, device=device)
    exponent = tau / q
    if exponent <= 0.0:
        return torch.full((m,), 0.5, device=device)
    p = torch.stack([base, 1.0 - base], dim=1).clamp_min(0.0)
    pe = p.pow(exponent)
    pe = pe / pe.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return pe[:, 0]


def simplex_grid(side: int = 5):
    """Deterministic triangular grid on the 2-simplex (λ_R, λ_U1, λ_U2), Σ=1.

    `side` = number of points per edge. Divisions n = side-1; the grid is every
    (i, j, k) with i+j+k = n, normalised by n → n(n+1)/2 + ... = C(n+2,2) points.
    side=5 → n=4 → 15 preferences, including the three vertices (pure R / U1 / U2)
    and the centroid-adjacent interior points. Same spirit as PaLoRA's
    torch.linspace(0,1,M) for two objectives, lifted to three.
    """
    n = side - 1
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            pts.append((i / n, j / n, k / n))
    return pts


def anneal_simplex(prefs, tau: float, q: float, mode: str = "power"):
    """PaLoRA center-to-edge annealing generalised to the 2-simplex.

        exponent = τ/Q ;  p(τ) = p̃**exponent / Σ_j p̃_j**exponent

    τ=0 → every preference collapses to the CENTROID (1/3, 1/3, 1/3);
    τ=1 → the original grid preferences. τ=0 handled explicitly (no 0**0), and the
    normaliser is clamped so an all-zero row can never produce NaNs.
    `prefs` is a list of 3-tuples; returns a list of 3-tuples.

    mode="linear" is the true centre-to-target interpolation:
        λ_eff = (1 - η) · (1/3, 1/3, 1/3) + η · λ_target,   η = τ/Q clipped to [0, 1]
    It exists because the power map leaves every VERTEX untouched at any exponent —
    (1,0,0) is already (1,0,0) at γ=0.25 — so it anneals precisely the preferences
    that need it least. Default stays "power" so every existing arm reproduces
    bit-for-bit.
    """
    if mode == "linear":
        eta = min(1.0, max(0.0, tau / q))
        c = 1.0 / 3.0
        return [tuple((1.0 - eta) * c + eta * max(x, 0.0) for x in p) for p in prefs]
    exponent = tau / q
    if exponent <= 0.0:
        return [(1.0 / 3, 1.0 / 3, 1.0 / 3)] * len(prefs)
    out = []
    for p in prefs:
        pe = [max(c, 0.0) ** exponent for c in p]
        s = sum(pe)
        s = s if s > 1e-12 else 1e-12
        out.append(tuple(c / s for c in pe))
    return out


class PrefCycle:
    """Balanced subset scheduler: shuffle the grid, hand out groups of M, reshuffle.

    Every target preference is visited once per cycle, so over training each receives
    the same number of updates up to one partial group -- unlike i.i.d. sampling, which
    leaves some preferences under-trained by chance. M >= len(grid) yields the full grid
    every call, i.e. the historical behaviour, with no shuffling, so M=15 on the side-5
    grid reproduces the reported runs exactly.
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


def train_encoders(method: str, data_dir: str, out_dir: str, device: str,
                   epochs: int, lr: float, batch_size: int,
                   latent_dim: int, proj_dim: int,
                   encoder_param: str = "projection",
                   lam_dist: str = DEFAULT_LAM_DIST,
                   dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
                   pareto_solver: str = DEFAULT_PARETO_SOLVER,
                   seed: int = DEFAULT_TRAIN_SEED,
                   modality: str = None,
                   num_preferences: int = 1,
                   preference_schedule: str = "single",
                   annealing_temperature: float = 1.0,
                   lora_alpha: float = None,
                   simplex_side: int = 5,
                   prefs_per_batch: int = 0,   # M preferences per step (0 = all, the reported config)
                   anneal_mode: str = "power",
                   lam_club: float = 1.0):   # simplex_enc_decomp_R: CLUB weight (0 = readout-only ablation)
    # modality: None = bimodal (default), 'm1' = train only M1 unique head,
    #           'm2' = train only M2 unique head.
    # Cross-modal L_R always uses both modalities during training regardless.
    _m1_only = (modality == 'm1')
    _m2_only = (modality == 'm2')
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    loader = DataLoader(
        TrifeatureTrainDataset(data_dir),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    # gmc / comm: CoMM MMFusion architecture
    if method in COMM_METHODS:
        MASKS = [[True, False], [False, True], [True, True]]  # M1-only, M2-only, joint
        mmfusion = _make_mmfusion(device)
        # CoMM's exact projection head: 3-layer MLP with BatchNorm (no pre-normalisation)
        proj = CoMMProjectionHead(in_dim=512, mlp_dim=512, out_dim=proj_dim).to(device)

        if method == "gmc":
            opt_modules = [mmfusion, proj]
            def step(views):
                m1 = views[0][0].to(device)
                m2 = views[1][0].to(device)
                z_list = mmfusion([m1, m2], mask_modalities=MASKS)
                # project then L2-normalise (matching CoMM's loss normalisation step)
                z_m1 = F.normalize(proj(z_list[0]), dim=-1)
                z_m2 = F.normalize(proj(z_list[1]), dim=-1)
                z_j  = F.normalize(proj(z_list[2]), dim=-1)
                return gmc_loss([z_m1, z_m2], z_j)

        else:  # comm
            opt_modules = [mmfusion, proj]
            def step(views):
                m1a, m2a = views[0][0].to(device), views[1][0].to(device)
                m1b, m2b = views[0][1].to(device), views[1][1].to(device)
                z_a = mmfusion([m1a, m2a], mask_modalities=MASKS)
                z_b = mmfusion([m1b, m2b], mask_modalities=MASKS)
                # project then L2-normalise — matches CoMM: head(z) then normalize in loss
                za = [F.normalize(proj(z), dim=-1) for z in z_a]
                zb = [F.normalize(proj(z), dim=-1) for z in z_b]
                z_ja, z_jb = za[-1], zb[-1]
                # CoMM loss: each modality (both views) vs the joint prototype
                loss = torch.tensor(0., device=device)
                for za_i, zb_i in zip(za, zb):
                    loss = loss + (comm_infonce(za_i, z_jb) + comm_infonce(zb_i, z_ja)) / 2
                return loss / len(za)

        optimizer = optim.AdamW(_params(*opt_modules), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

        for epoch in range(1, epochs + 1):
            total = 0.0
            for views in loader:
                optimizer.zero_grad()
                loss = step(views)
                loss.backward()
                optimizer.step()
                total += loss.item()
            scheduler.step()
            avg = total / len(loader)
            writer.add_scalar("loss/train", avg, epoch)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  [{method}] epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

        writer.close()
        torch.save(mmfusion.state_dict(), os.path.join(out_dir, "mmfusion.pth"))

    # simclr_both / clip / factorcl: standard AlexNet encoders
    else:
        enc_m1  = AlexNetEncoder(latent_dim).to(device)
        enc_m2  = AlexNetEncoder(latent_dim).to(device)
        proj_m1 = ProjectionHead(latent_dim, proj_dim).to(device)
        proj_m2 = ProjectionHead(latent_dim, proj_dim).to(device)

        def _z(enc, pr, x):
            return pr(enc(x.to(device)))

        if method == "simclr_both":
            opt_modules = [enc_m1, proj_m1, enc_m2, proj_m2]
            def step(views):
                l1 = nt_xent(_z(enc_m1, proj_m1, views[0][0]),
                             _z(enc_m1, proj_m1, views[0][1]))
                l2 = nt_xent(_z(enc_m2, proj_m2, views[1][0]),
                             _z(enc_m2, proj_m2, views[1][1]))
                return (l1 + l2) / 2

        elif method == "clip":
            opt_modules = [enc_m1, proj_m1, enc_m2, proj_m2]
            def step(views):
                return infonce_cross(_z(enc_m1, proj_m1, views[0][0]),
                                     _z(enc_m2, proj_m2, views[1][0]))

        elif method == "factorcl_heads":
            proj_m1_r = ProjectionHead(latent_dim, proj_dim).to(device)
            proj_m2_r = ProjectionHead(latent_dim, proj_dim).to(device)
            proj_m1_u = ProjectionHead(latent_dim, proj_dim).to(device)
            proj_m2_u = ProjectionHead(latent_dim, proj_dim).to(device)
            club_m1 = CLUBInfoNCECritic(proj_dim, proj_dim,
                                        hidden_dim=512, layers=1, activation='relu').to(device)
            club_m2 = CLUBInfoNCECritic(proj_dim, proj_dim,
                                        hidden_dim=512, layers=1, activation='relu').to(device)

            opt_enc = optim.AdamW(
                _params(enc_m1, enc_m2, proj_m1_r, proj_m2_r, proj_m1_u, proj_m2_u),
                lr=lr, weight_decay=1e-4)
            opt_critic = optim.AdamW(
                _params(club_m1, club_m2), lr=lr, weight_decay=1e-4)
            sch_enc    = optim.lr_scheduler.CosineAnnealingLR(opt_enc,    T_max=epochs)
            sch_critic = optim.lr_scheduler.CosineAnnealingLR(opt_critic, T_max=epochs)
            writer_fc  = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

            for epoch in range(1, epochs + 1):
                enc_total = 0.0
                crit_total = 0.0
                for views in loader:
                    m1a = views[0][0].to(device);  m1b = views[0][1].to(device)
                    m2a = views[1][0].to(device);  m2b = views[1][1].to(device)

                    h1a = enc_m1(m1a);  h1b = enc_m1(m1b)
                    h2a = enc_m2(m2a);  h2b = enc_m2(m2b)
                    z1_r  = proj_m1_r(h1a);  z2_r  = proj_m2_r(h2a)
                    z1_ua = proj_m1_u(h1a);  z1_ub = proj_m1_u(h1b)
                    z2_ua = proj_m2_u(h2a);  z2_ub = proj_m2_u(h2b)

                    # critic step — train CLUB critics with detached representations
                    opt_critic.zero_grad()
                    crit_loss = (club_m1.learning_loss(z1_ua.detach(), z1_r.detach()) +
                                 club_m2.learning_loss(z2_ua.detach(), z2_r.detach()))
                    crit_loss.backward()
                    torch.nn.utils.clip_grad_norm_(_params(club_m1, club_m2), max_norm=1.0)
                    opt_critic.step()
                    crit_total += crit_loss.item()

                    # encoder step — L_R (cross-modal) + L_U (within-modal - CLUB*0.5)
                    # lam_club=0.5: balance between disentanglement and within-modal alignment
                    opt_enc.zero_grad()
                    L_R  = infonce_cross(z1_r, z2_r)
                    L_U1 = nt_xent(z1_ua, z1_ub) + LAM_CLUB * torch.clamp(club_m1(z1_ua, z1_r.detach()), min=0.0)
                    L_U2 = nt_xent(z2_ua, z2_ub) + LAM_CLUB * torch.clamp(club_m2(z2_ua, z2_r.detach()), min=0.0)
                    enc_loss = L_R + L_U1 + L_U2
                    enc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        _params(enc_m1, enc_m2, proj_m1_r, proj_m2_r, proj_m1_u, proj_m2_u),
                        max_norm=1.0)
                    opt_enc.step()
                    enc_total += enc_loss.item()

                sch_enc.step()
                sch_critic.step()
                avg_enc = enc_total / len(loader)
                writer_fc.add_scalar("loss/encoder", avg_enc, epoch)
                writer_fc.add_scalar("loss/critic",  crit_total / len(loader), epoch)
                if epoch % 10 == 0 or epoch == 1:
                    print(f"  [factorcl_heads] epoch {epoch:3d}/{epochs}  loss={avg_enc:.4f}")

            writer_fc.close()
            torch.save(enc_m1.state_dict(),    os.path.join(out_dir, "enc_0.pth"))
            torch.save(enc_m2.state_dict(),    os.path.join(out_dir, "enc_1.pth"))
            torch.save(proj_m1_r.state_dict(), os.path.join(out_dir, "proj_r.pth"))
            torch.save(proj_m1_u.state_dict(), os.path.join(out_dir, "proj_u.pth"))
            torch.save(proj_m2_u.state_dict(), os.path.join(out_dir, "proj_m2_u.pth"))
            print(f"  Saved to {out_dir}")
            return

        elif method == "simclr_single_per_batch":
            # L(λ) = 2λ·InfoNCE(proj_r(M1), proj_r(M2))
            #      + 2(1−λ)·SimCLR(proj_u(M1_a), proj_u(M1_b))
            # 2× rescaling: E[2λ]=1, E[2(1-λ)]=1 → each head gets 100% expected gradient
            # (same as factorcl baseline) while lambda still steers the objective.
            # proj_r ← InfoNCE gradients only  → captures shared info (R)
            # proj_u ← SimCLR gradients only   → captures unique info (U)
            # Inference: z = normalize(λ·proj_r(h) + (1−λ)·proj_u(h))
            #
            # encoder_param="encoder"    → HyperNetEncoder: FiLM inside every conv block
            # encoder_param="projection" → plain AlexNetEncoder: only heads see λ
            use_hyp = (encoder_param == "encoder")
            if use_hyp:
                enc_m1 = HyperNetEncoder(latent_dim).to(device)
                enc_m2 = HyperNetEncoder(latent_dim).to(device)
            # else: enc_m1, enc_m2 already created above as AlexNetEncoder

            def _enc_lc(mod, x, lam):
                return mod(x, lam) if use_hyp else mod(x)

            proj_m1_r = ProjectionHead(latent_dim, proj_dim).to(device)
            proj_m2_r = ProjectionHead(latent_dim, proj_dim).to(device)
            proj_m1_u = None if _m2_only else ProjectionHead(latent_dim, proj_dim).to(device)
            proj_m2_u = None if _m1_only else ProjectionHead(latent_dim, proj_dim).to(device)

            _lc_heads = [p for p in [proj_m1_u, proj_m2_u] if p is not None]
            enc_params_lc = _params(enc_m1, enc_m2, proj_m1_r, proj_m2_r, *_lc_heads)
            opt_lc = optim.AdamW(enc_params_lc, lr=lr, weight_decay=1e-4)
            sch_lc = optim.lr_scheduler.CosineAnnealingLR(opt_lc, T_max=epochs)
            writer_lc = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

            if pareto_solver == "epo":
                from pareto_ssl.epo import EPOSolver as _EPO
                _slc = _EPO(n_tasks=2,
                            n_params=sum(p.numel() for p in enc_params_lc))

            for epoch in range(1, epochs + 1):
                total = 0.0
                for views in loader:
                    m1a, m1b = views[0][0].to(device), views[0][1].to(device)
                    m2a, m2b = views[1][0].to(device), views[1][1].to(device)

                    lam = _sample_lam(lam_dist, dirichlet_alpha, epoch, epochs)

                    h1a = _enc_lc(enc_m1, m1a, lam);  h1b = _enc_lc(enc_m1, m1b, lam)
                    h2a = _enc_lc(enc_m2, m2a, lam);  h2b = _enc_lc(enc_m2, m2b, lam)

                    z1_r = proj_m1_r(h1a);  z2_r = proj_m2_r(h2a)
                    L_shared = infonce_cross(z1_r, z2_r)

                    if _m1_only:
                        z1_ua = proj_m1_u(h1a);  z1_ub = proj_m1_u(h1b)
                        L_unique = nt_xent(z1_ua, z1_ub)
                    elif _m2_only:
                        z2_ua = proj_m2_u(h2a);  z2_ub = proj_m2_u(h2b)
                        L_unique = nt_xent(z2_ua, z2_ub)
                    else:
                        z1_ua = proj_m1_u(h1a);  z1_ub = proj_m1_u(h1b)
                        z2_ua = proj_m2_u(h2a);  z2_ub = proj_m2_u(h2b)
                        L_unique = (nt_xent(z1_ua, z1_ub) + nt_xent(z2_ua, z2_ub)) / 2

                    if pareto_solver == "epo":
                        ray  = torch.tensor([lam, 1.0 - lam],
                                            device=device, dtype=torch.float32)
                        loss = _slc(torch.stack([L_shared, L_unique]),
                                    ray, enc_params_lc)
                    else:
                        loss = 2 * lam * L_shared + 2 * (1.0 - lam) * L_unique

                    opt_lc.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(enc_params_lc, max_norm=1.0)
                    opt_lc.step()
                    total += loss.item()

                sch_lc.step()
                avg = total / len(loader)
                writer_lc.add_scalar("loss/train", avg, epoch)
                if epoch % 10 == 0 or epoch == 1:
                    print(f"  [simclr_single_per_batch/{encoder_param}] epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

            writer_lc.close()
            enc_prefix = "hyp" if use_hyp else "enc"
            torch.save(enc_m1.state_dict(),    os.path.join(out_dir, f"{enc_prefix}_0.pth"))
            torch.save(enc_m2.state_dict(),    os.path.join(out_dir, f"{enc_prefix}_1.pth"))
            torch.save(proj_m1_r.state_dict(), os.path.join(out_dir, "proj_m1_r.pth"))
            torch.save(proj_m2_r.state_dict(), os.path.join(out_dir, "proj_m2_r.pth"))
            if proj_m1_u is not None:
                torch.save(proj_m1_u.state_dict(), os.path.join(out_dir, "proj_m1_u.pth"))
            if proj_m2_u is not None:
                torch.save(proj_m2_u.state_dict(), os.path.join(out_dir, "proj_m2_u.pth"))
            print(f"  Saved to {out_dir}  (encoder_param={encoder_param})")
            return

        elif method == "factorcl_warmup":
            # L(λ) = 2λ·L_R + 2(1−λ)·(L_U1 + L_U2)/2
            # 2× rescaling: E[2λ]=1, E[2(1-λ)]=1 → each head gets 100% expected gradient.
            # L_R  = InfoNCE(proj_r(M1), proj_r(M2))            — same as FactorCL shared
            # L_U1 = NT-Xent(proj_u(M1_a), proj_u(M1_b))
            #      + 0.5·clamp(CLUB(proj_u(M1), proj_r(M1).detach()), min=0)
            # L_U2 = same for M2
            # CLUB explicitly removes shared info from the unique head.
            # Inference: z = normalize(λ·proj_r(h) + (1−λ)·proj_u(h))
            #
            # encoder_param="encoder"    → HyperNetEncoder: FiLM inside every conv block
            # encoder_param="projection" → plain AlexNetEncoder: only heads see λ
            use_hyp = (encoder_param == "encoder")
            if use_hyp:
                enc_m1 = HyperNetEncoder(latent_dim).to(device)
                enc_m2 = HyperNetEncoder(latent_dim).to(device)
            # else: enc_m1, enc_m2 already created above as AlexNetEncoder

            def _enc_fc(mod, x, lam):
                return mod(x, lam) if use_hyp else mod(x)

            proj_m1_r = ProjectionHead(latent_dim, proj_dim).to(device)
            proj_m2_r = ProjectionHead(latent_dim, proj_dim).to(device)
            proj_m1_u = None if _m2_only else ProjectionHead(latent_dim, proj_dim).to(device)
            proj_m2_u = None if _m1_only else ProjectionHead(latent_dim, proj_dim).to(device)
            club_m1 = None if _m2_only else CLUBInfoNCECritic(
                proj_dim, proj_dim, hidden_dim=512, layers=1, activation='relu').to(device)
            club_m2 = None if _m1_only else CLUBInfoNCECritic(
                proj_dim, proj_dim, hidden_dim=512, layers=1, activation='relu').to(device)

            _fc_unique_heads = [p for p in [proj_m1_u, proj_m2_u] if p is not None]
            enc_params_fc = _params(enc_m1, enc_m2, proj_m1_r, proj_m2_r, *_fc_unique_heads)
            opt_enc = optim.AdamW(enc_params_fc, lr=lr, weight_decay=1e-4)
            _fc_clubs = [c for c in [club_m1, club_m2] if c is not None]
            opt_critic = optim.AdamW(_params(*_fc_clubs), lr=lr, weight_decay=1e-4)
            sch_enc    = optim.lr_scheduler.CosineAnnealingLR(opt_enc,    T_max=epochs)
            sch_critic = optim.lr_scheduler.CosineAnnealingLR(opt_critic, T_max=epochs)
            writer_fc  = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

            if pareto_solver == "epo":
                from pareto_ssl.epo import EPOSolver as _EPO
                _sfc = _EPO(n_tasks=2,
                            n_params=sum(p.numel() for p in enc_params_fc))

            for epoch in range(1, epochs + 1):
                enc_total = 0.0; crit_total = 0.0
                for views in loader:
                    m1a, m1b = views[0][0].to(device), views[0][1].to(device)
                    m2a, m2b = views[1][0].to(device), views[1][1].to(device)

                    lam = _sample_lam(lam_dist, dirichlet_alpha, epoch, epochs)

                    h1a = _enc_fc(enc_m1, m1a, lam);  h1b = _enc_fc(enc_m1, m1b, lam)
                    h2a = _enc_fc(enc_m2, m2a, lam);  h2b = _enc_fc(enc_m2, m2b, lam)

                    z1_r = proj_m1_r(h1a);  z2_r = proj_m2_r(h2a)
                    z1_ua = proj_m1_u(h1a) if proj_m1_u is not None else None
                    z1_ub = proj_m1_u(h1b) if proj_m1_u is not None else None
                    z2_ua = proj_m2_u(h2a) if proj_m2_u is not None else None
                    z2_ub = proj_m2_u(h2b) if proj_m2_u is not None else None

                    # critic step
                    opt_critic.zero_grad()
                    crit_loss = torch.tensor(0.0, device=device)
                    if club_m1 is not None:
                        crit_loss = crit_loss + club_m1.learning_loss(
                            z1_ua.detach(), z1_r.detach())
                    if club_m2 is not None:
                        crit_loss = crit_loss + club_m2.learning_loss(
                            z2_ua.detach(), z2_r.detach())
                    crit_loss.backward()
                    torch.nn.utils.clip_grad_norm_(_params(*_fc_clubs), max_norm=1.0)
                    opt_critic.step()
                    crit_total += crit_loss.item()

                    # encoder step
                    opt_enc.zero_grad()
                    L_R = infonce_cross(z1_r, z2_r)
                    if _m1_only:
                        L_U = (nt_xent(z1_ua, z1_ub) +
                               LAM_CLUB * torch.clamp(club_m1(z1_ua, z1_r.detach()), min=0.0))
                    elif _m2_only:
                        L_U = (nt_xent(z2_ua, z2_ub) +
                               LAM_CLUB * torch.clamp(club_m2(z2_ua, z2_r.detach()), min=0.0))
                    else:
                        L_U1 = (nt_xent(z1_ua, z1_ub) +
                                LAM_CLUB * torch.clamp(club_m1(z1_ua, z1_r.detach()), min=0.0))
                        L_U2 = (nt_xent(z2_ua, z2_ub) +
                                LAM_CLUB * torch.clamp(club_m2(z2_ua, z2_r.detach()), min=0.0))
                        L_U  = (L_U1 + L_U2) / 2

                    if pareto_solver == "epo":
                        ray  = torch.tensor([lam, 1.0 - lam],
                                            device=device, dtype=torch.float32)
                        loss = _sfc(torch.stack([L_R, L_U]), ray, enc_params_fc)
                    else:
                        loss = 2 * lam * L_R + 2 * (1.0 - lam) * L_U

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(enc_params_fc, max_norm=1.0)
                    opt_enc.step()
                    enc_total += loss.item()

                sch_enc.step();  sch_critic.step()
                avg = enc_total / len(loader)
                writer_fc.add_scalar("loss/encoder", avg, epoch)
                writer_fc.add_scalar("loss/critic",  crit_total / len(loader), epoch)
                if epoch % 10 == 0 or epoch == 1:
                    print(f"  [{method}/{encoder_param}] epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

            writer_fc.close()
            enc_prefix = "hyp" if use_hyp else "enc"
            torch.save(enc_m1.state_dict(),    os.path.join(out_dir, f"{enc_prefix}_0.pth"))
            torch.save(enc_m2.state_dict(),    os.path.join(out_dir, f"{enc_prefix}_1.pth"))
            torch.save(proj_m1_r.state_dict(), os.path.join(out_dir, "proj_m1_r.pth"))
            torch.save(proj_m2_r.state_dict(), os.path.join(out_dir, "proj_m2_r.pth"))
            if proj_m1_u is not None:
                torch.save(proj_m1_u.state_dict(), os.path.join(out_dir, "proj_m1_u.pth"))
            if proj_m2_u is not None:
                torch.save(proj_m2_u.state_dict(), os.path.join(out_dir, "proj_m2_u.pth"))
            print(f"  Saved to {out_dir}  (encoder_param={encoder_param})")
            return

        elif method == "hyper_lambda":
            # Per batch: evaluate LAMBDA_GRID = {0, 0.25, 0.5, 0.75, 1.0} simultaneously.
            # L_total = mean_λ [ 2λ·L_R(z_λ) + 2(1-λ)·(L_U1+L_U2)(z_λ)/2 ]
            #         + α·L_smooth   (penalise non-linear variation along the curve)
            #         + β·L_diversity (penalise all z_λ collapsing to the same point)
            # 2× rescaling: E[2λ]=1, E[2(1-λ)]=1 → each head matches factorcl gradient.
            #
            # encoder_param="encoder"    → HyperNetEncoder: FiLM inside every conv block
            # encoder_param="projection" → plain AlexNetEncoder: only heads differ
            use_hyp = (encoder_param == "encoder")

            if use_hyp:
                enc_m1 = HyperNetEncoder(latent_dim).to(device)
                enc_m2 = HyperNetEncoder(latent_dim).to(device)
            # else: enc_m1, enc_m2 already created above as AlexNetEncoder

            proj_m1_r = ProjectionHead(latent_dim, proj_dim).to(device)
            proj_m2_r = ProjectionHead(latent_dim, proj_dim).to(device)
            proj_m1_u = None if _m2_only else ProjectionHead(latent_dim, proj_dim).to(device)
            proj_m2_u = None if _m1_only else ProjectionHead(latent_dim, proj_dim).to(device)
            club_m1 = None if _m2_only else CLUBInfoNCECritic(
                proj_dim, proj_dim, hidden_dim=512, layers=1, activation='relu').to(device)
            club_m2 = None if _m1_only else CLUBInfoNCECritic(
                proj_dim, proj_dim, hidden_dim=512, layers=1, activation='relu').to(device)

            _hl_unique = [p for p in [proj_m1_u, proj_m2_u] if p is not None]
            _hl_clubs  = [c for c in [club_m1, club_m2] if c is not None]
            enc_params = _params(enc_m1, enc_m2, proj_m1_r, proj_m2_r, *_hl_unique)
            opt_enc    = optim.AdamW(enc_params, lr=lr, weight_decay=1e-4)
            opt_critic = optim.AdamW(_params(*_hl_clubs), lr=lr, weight_decay=1e-4)
            sch_enc    = optim.lr_scheduler.CosineAnnealingLR(opt_enc,    T_max=epochs)
            sch_critic = optim.lr_scheduler.CosineAnnealingLR(opt_critic, T_max=epochs)
            writer_hl  = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

            alpha_smooth = ALPHA_SMOOTH   # weight for L_smooth   (pareto_config.yaml)
            beta_div     = BETA_DIV      # weight for L_diversity (pareto_config.yaml)

            if pareto_solver == "epo":
                from pareto_ssl.epo import EPOSolver as _EPO
                _shl = _EPO(n_tasks=2, n_params=sum(p.numel() for p in enc_params))

            def _enc(mod, x, lam):
                return mod(x, lam) if use_hyp else mod(x)

            for epoch in range(1, epochs + 1):
                enc_total = 0.0; crit_total = 0.0
                for views in loader:
                    m1a = views[0][0].to(device);  m1b = views[0][1].to(device)
                    m2a = views[1][0].to(device);  m2b = views[1][1].to(device)

                    # critic step: use one random lambda to keep it cheap
                    lam_c = _sample_lam(lam_dist, dirichlet_alpha, epoch, epochs)
                    h1a_c = _enc(enc_m1, m1a, lam_c).detach()
                    h2a_c = _enc(enc_m2, m2a, lam_c).detach()
                    z1_r_c = proj_m1_r(h1a_c).detach()
                    z2_r_c = proj_m2_r(h2a_c).detach()

                    opt_critic.zero_grad()
                    crit_loss = torch.tensor(0.0, device=device)
                    if club_m1 is not None:
                        z1_ua_c = proj_m1_u(h1a_c).detach()
                        crit_loss = crit_loss + club_m1.learning_loss(z1_ua_c, z1_r_c)
                    if club_m2 is not None:
                        z2_ua_c = proj_m2_u(h2a_c).detach()
                        crit_loss = crit_loss + club_m2.learning_loss(z2_ua_c, z2_r_c)
                    crit_loss.backward()
                    torch.nn.utils.clip_grad_norm_(_params(*_hl_clubs), 1.0)
                    opt_critic.step()
                    crit_total += crit_loss.item()

                    # encoder step: full LAMBDA_GRID
                    opt_enc.zero_grad()
                    task_loss = torch.tensor(0.0, device=device)
                    zs_m1, zs_m2 = [], []   # combined z_λ for smoothness/diversity
                    epo_losses_hl = []       # per-lambda EPO-weighted losses (EPO mode)

                    # When lam_dist='warmup', weight grid points by the active
                    # phase's Beta PDF so early training favours shared endpoints.
                    grid_w = (_warmup_grid_weights(epoch, epochs, LAMBDA_WARMUP_PHASES, LAMBDA_GRID)
                              if lam_dist == "warmup"
                              else [1.0 / len(LAMBDA_GRID)] * len(LAMBDA_GRID))

                    for w, lam in zip(grid_w, LAMBDA_GRID):
                        h1a = _enc(enc_m1, m1a, lam);  h1b = _enc(enc_m1, m1b, lam)
                        h2a = _enc(enc_m2, m2a, lam);  h2b = _enc(enc_m2, m2b, lam)

                        z1_r = proj_m1_r(h1a);  z2_r = proj_m2_r(h2a)
                        L_R  = infonce_cross(z1_r, z2_r)

                        if _m1_only:
                            z1_ua = proj_m1_u(h1a);  z1_ub = proj_m1_u(h1b)
                            L_U   = (nt_xent(z1_ua, z1_ub) +
                                     LAM_CLUB * torch.clamp(club_m1(z1_ua, z1_r.detach()), min=0.0))
                            z_rep_m1 = F.normalize(lam * z1_r + (1 - lam) * z1_ua, dim=-1)
                            z_rep_m2 = z_rep_m1  # mirror for regularisation symmetry
                        elif _m2_only:
                            z2_ua = proj_m2_u(h2a);  z2_ub = proj_m2_u(h2b)
                            L_U   = (nt_xent(z2_ua, z2_ub) +
                                     LAM_CLUB * torch.clamp(club_m2(z2_ua, z2_r.detach()), min=0.0))
                            z_rep_m2 = F.normalize(lam * z2_r + (1 - lam) * z2_ua, dim=-1)
                            z_rep_m1 = z_rep_m2
                        else:
                            z1_ua = proj_m1_u(h1a);  z1_ub = proj_m1_u(h1b)
                            z2_ua = proj_m2_u(h2a);  z2_ub = proj_m2_u(h2b)
                            L_U1  = (nt_xent(z1_ua, z1_ub) +
                                     LAM_CLUB * torch.clamp(club_m1(z1_ua, z1_r.detach()), min=0.0))
                            L_U2  = (nt_xent(z2_ua, z2_ub) +
                                     LAM_CLUB * torch.clamp(club_m2(z2_ua, z2_r.detach()), min=0.0))
                            L_U   = (L_U1 + L_U2) / 2
                            z_rep_m1 = F.normalize(lam * z1_r + (1 - lam) * z1_ua, dim=-1)
                            z_rep_m2 = F.normalize(lam * z2_r + (1 - lam) * z2_ua, dim=-1)

                        if pareto_solver == "ls":
                            task_loss += w * (2 * lam * L_R + 2 * (1.0 - lam) * L_U)
                        else:
                            ray_k = torch.tensor([lam, 1.0 - lam],
                                                 device=device, dtype=torch.float32)
                            epo_losses_hl.append(
                                (w, _shl(torch.stack([L_R, L_U]), ray_k, enc_params))
                            )

                        zs_m1.append(z_rep_m1)
                        zs_m2.append(z_rep_m2)

                    if pareto_solver == "epo":
                        # Weighted sum of per-preference EPO losses.
                        task_loss = sum(w * l for w, l in epo_losses_hl)

                    # L_smooth: penalise second-order non-linearity along the lambda curve
                    # z_{i+1} - 2*z_i + z_{i-1} ≈ 0 means linear interpolation
                    L_smooth = torch.tensor(0.0, device=device)
                    for i in range(1, len(LAMBDA_GRID) - 1):
                        L_smooth += (zs_m1[i+1] - 2*zs_m1[i] + zs_m1[i-1]).pow(2).sum(-1).mean()
                        L_smooth += (zs_m2[i+1] - 2*zs_m2[i] + zs_m2[i-1]).pow(2).sum(-1).mean()
                    L_smooth /= (2 * (len(LAMBDA_GRID) - 2))

                    # L_diversity: mean pairwise cosine similarity — minimise to push apart
                    sims, count = torch.tensor(0.0, device=device), 0
                    for i in range(len(LAMBDA_GRID)):
                        for j in range(i + 1, len(LAMBDA_GRID)):
                            sims += (zs_m1[i] * zs_m1[j]).sum(-1).mean()
                            sims += (zs_m2[i] * zs_m2[j]).sum(-1).mean()
                            count += 2
                    L_diversity = sims / count

                    loss = task_loss + alpha_smooth * L_smooth + beta_div * L_diversity
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(enc_params, 1.0)
                    opt_enc.step()
                    enc_total += loss.item()

                sch_enc.step();  sch_critic.step()
                avg = enc_total / len(loader)
                writer_hl.add_scalar("loss/encoder",   avg, epoch)
                writer_hl.add_scalar("loss/critic",    crit_total / len(loader), epoch)
                if epoch % 10 == 0 or epoch == 1:
                    print(f"  [hyper_lambda/{encoder_param}] epoch {epoch:3d}/{epochs}"
                          f"  loss={avg:.4f}")

            writer_hl.close()
            enc_prefix = "hyp" if use_hyp else "enc"
            torch.save(enc_m1.state_dict(),    os.path.join(out_dir, f"{enc_prefix}_0.pth"))
            torch.save(enc_m2.state_dict(),    os.path.join(out_dir, f"{enc_prefix}_1.pth"))
            torch.save(proj_m1_r.state_dict(), os.path.join(out_dir, "proj_m1_r.pth"))
            torch.save(proj_m2_r.state_dict(), os.path.join(out_dir, "proj_m2_r.pth"))
            if proj_m1_u is not None:
                torch.save(proj_m1_u.state_dict(), os.path.join(out_dir, "proj_m1_u.pth"))
            if proj_m2_u is not None:
                torch.save(proj_m2_u.state_dict(), os.path.join(out_dir, "proj_m2_u.pth"))
            print(f"  Saved to {out_dir}  (encoder_param={encoder_param})")
            return

        elif method == "simclr_grid_per_batch":
            # SimCLR loss on the LAMBDA_GRID (no CLUB) + L_smooth + L_diversity.
            # L(λ) = 2λ·InfoNCE(proj_r(M1), proj_r(M2))
            #      + 2(1-λ)·(NT-Xent(proj_u(M1)) + NT-Xent(proj_u(M2)))/2
            # Evaluated simultaneously over LAMBDA_GRID per batch, then summed.
            # No critic — unique head gets NT-Xent only, identical to simclr_single_per_batch.
            # encoder_param="encoder"    → HyperNetEncoder (FiLM in conv blocks)
            # encoder_param="projection" → plain AlexNetEncoder
            use_hyp = (encoder_param == "encoder")
            if use_hyp:
                enc_m1 = HyperNetEncoder(latent_dim).to(device)
                enc_m2 = HyperNetEncoder(latent_dim).to(device)

            proj_m1_r = ProjectionHead(latent_dim, proj_dim).to(device)
            proj_m2_r = ProjectionHead(latent_dim, proj_dim).to(device)
            proj_m1_u = None if _m2_only else ProjectionHead(latent_dim, proj_dim).to(device)
            proj_m2_u = None if _m1_only else ProjectionHead(latent_dim, proj_dim).to(device)

            _sh_unique = [p for p in [proj_m1_u, proj_m2_u] if p is not None]
            enc_params_sh = _params(enc_m1, enc_m2, proj_m1_r, proj_m2_r, *_sh_unique)
            opt_sh  = optim.AdamW(enc_params_sh, lr=lr, weight_decay=1e-4)
            sch_sh  = optim.lr_scheduler.CosineAnnealingLR(opt_sh, T_max=epochs)
            writer_sh = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

            alpha_smooth = ALPHA_SMOOTH
            beta_div     = BETA_DIV

            def _enc_sh(mod, x, lam):
                return mod(x, lam) if use_hyp else mod(x)

            for epoch in range(1, epochs + 1):
                total_sh = 0.0
                for views in loader:
                    m1a = views[0][0].to(device);  m1b = views[0][1].to(device)
                    m2a = views[1][0].to(device);  m2b = views[1][1].to(device)

                    opt_sh.zero_grad()
                    task_loss = torch.tensor(0.0, device=device)
                    zs_m1, zs_m2 = [], []

                    grid_w = (_warmup_grid_weights(epoch, epochs, LAMBDA_WARMUP_PHASES, LAMBDA_GRID)
                              if lam_dist == "warmup"
                              else [1.0 / len(LAMBDA_GRID)] * len(LAMBDA_GRID))

                    for w, lam in zip(grid_w, LAMBDA_GRID):
                        h1a = _enc_sh(enc_m1, m1a, lam);  h1b = _enc_sh(enc_m1, m1b, lam)
                        h2a = _enc_sh(enc_m2, m2a, lam);  h2b = _enc_sh(enc_m2, m2b, lam)

                        z1_r = proj_m1_r(h1a);  z2_r = proj_m2_r(h2a)
                        L_R  = infonce_cross(z1_r, z2_r)

                        if _m1_only:
                            z1_ua = proj_m1_u(h1a);  z1_ub = proj_m1_u(h1b)
                            L_U   = nt_xent(z1_ua, z1_ub)
                            z_rep_m1 = F.normalize(lam * z1_r + (1 - lam) * z1_ua, dim=-1)
                            z_rep_m2 = z_rep_m1
                        elif _m2_only:
                            z2_ua = proj_m2_u(h2a);  z2_ub = proj_m2_u(h2b)
                            L_U   = nt_xent(z2_ua, z2_ub)
                            z_rep_m2 = F.normalize(lam * z2_r + (1 - lam) * z2_ua, dim=-1)
                            z_rep_m1 = z_rep_m2
                        else:
                            z1_ua = proj_m1_u(h1a);  z1_ub = proj_m1_u(h1b)
                            z2_ua = proj_m2_u(h2a);  z2_ub = proj_m2_u(h2b)
                            L_U   = (nt_xent(z1_ua, z1_ub) + nt_xent(z2_ua, z2_ub)) / 2
                            z_rep_m1 = F.normalize(lam * z1_r + (1 - lam) * z1_ua, dim=-1)
                            z_rep_m2 = F.normalize(lam * z2_r + (1 - lam) * z2_ua, dim=-1)

                        task_loss += w * (2 * lam * L_R + 2 * (1.0 - lam) * L_U)
                        zs_m1.append(z_rep_m1)
                        zs_m2.append(z_rep_m2)

                    # L_smooth: penalise second-order non-linearity along the lambda curve
                    L_smooth = torch.tensor(0.0, device=device)
                    for i in range(1, len(LAMBDA_GRID) - 1):
                        L_smooth += (zs_m1[i+1] - 2*zs_m1[i] + zs_m1[i-1]).pow(2).sum(-1).mean()
                        L_smooth += (zs_m2[i+1] - 2*zs_m2[i] + zs_m2[i-1]).pow(2).sum(-1).mean()
                    L_smooth /= (2 * (len(LAMBDA_GRID) - 2))

                    # L_diversity: mean pairwise cosine similarity
                    sims, count = torch.tensor(0.0, device=device), 0
                    for i in range(len(LAMBDA_GRID)):
                        for j in range(i + 1, len(LAMBDA_GRID)):
                            sims += (zs_m1[i] * zs_m1[j]).sum(-1).mean()
                            sims += (zs_m2[i] * zs_m2[j]).sum(-1).mean()
                            count += 2
                    L_diversity = sims / count

                    loss = task_loss + alpha_smooth * L_smooth + beta_div * L_diversity
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(enc_params_sh, 1.0)
                    opt_sh.step()
                    total_sh += loss.item()

                sch_sh.step()
                avg = total_sh / len(loader)
                writer_sh.add_scalar("loss/train", avg, epoch)
                if epoch % 10 == 0 or epoch == 1:
                    print(f"  [simclr_grid_per_batch/{encoder_param}] epoch {epoch:3d}/{epochs}"
                          f"  loss={avg:.4f}")

            writer_sh.close()
            enc_prefix = "hyp" if use_hyp else "enc"
            torch.save(enc_m1.state_dict(),    os.path.join(out_dir, f"{enc_prefix}_0.pth"))
            torch.save(enc_m2.state_dict(),    os.path.join(out_dir, f"{enc_prefix}_1.pth"))
            torch.save(proj_m1_r.state_dict(), os.path.join(out_dir, "proj_m1_r.pth"))
            torch.save(proj_m2_r.state_dict(), os.path.join(out_dir, "proj_m2_r.pth"))
            if proj_m1_u is not None:
                torch.save(proj_m1_u.state_dict(), os.path.join(out_dir, "proj_m1_u.pth"))
            if proj_m2_u is not None:
                torch.save(proj_m2_u.state_dict(), os.path.join(out_dir, "proj_m2_u.pth"))
            print(f"  Saved to {out_dir}  (encoder_param={encoder_param})")
            return

        elif method in ("simclr_simplex4h", "simclr_simplex4h_norm", "simclr_simplex6h"):
            # Output-mixing simplex variants (no LoRA)
            # The shared-base `simclr_lora_simplex` interpolates WEIGHTS via
            # forward_mix. These interpolate head OUTPUTS instead:
            #   simplex4h      normalize(w_R*zhat_R + w_U*zhat_U)   2 heads/modality
            #   simplex4h_norm normalize(w_R*z_R    + w_U*z_U   )   mix RAW, norm once
            #   simplex6h      normalize(w_R*A + w_U1*B + w_U2*C)   3 heads/modality,
            #                  a TRUE 3-way blend where every coordinate reaches
            #                  every modality (4h gives m1 only (w_R,w_U1)).
            # On MOSEI these did not reproduce palora_enc_4h's interior sag, and
            # simplex4h_norm gave the best accuracy with the lowest CKA. Trifeature
            # is the positive control: it is the one benchmark where preference
            # selection demonstrably works (between/within ratio 15.3), so the
            # question here is whether these variants PRESERVE that.
            _s6 = (method == "simclr_simplex6h")
            _premix = (method == "simclr_simplex4h_norm")
            _keys = ("r", "u1", "u2") if _s6 else ("r", "u")
            proj_m1 = nn.ModuleDict({k: ProjectionHead(latent_dim, proj_dim) for k in _keys}).to(device)
            proj_m2 = nn.ModuleDict({k: ProjectionHead(latent_dim, proj_dim) for k in _keys}).to(device)
            opt_sx = optim.AdamW(_params(enc_m1, enc_m2, proj_m1, proj_m2),
                                 lr=lr, weight_decay=1e-4)
            sch_sx = optim.lr_scheduler.CosineAnnealingLR(opt_sx, T_max=epochs)
            writer_sx = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
            simplex_prefs = simplex_grid(simplex_side)
            print(f"  {method}: side={simplex_side} -> {len(simplex_prefs)} preferences, "
                  f"{'3 heads' if _s6 else '2 heads'}/modality, "
                  f"mix {'BEFORE' if _premix else 'after'} normalisation")

            def _mix(pd, h, w_r, w_u_own, w_u_other):
                if _s6:
                    if _premix:
                        return F.normalize(w_r * pd["r"].net(h) + w_u_own * pd["u1"].net(h)
                                           + w_u_other * pd["u2"].net(h), dim=-1)
                    return F.normalize(w_r * pd["r"](h) + w_u_own * pd["u1"](h)
                                       + w_u_other * pd["u2"](h), dim=-1)
                # 4-head: this modality sees only (w_R, its own w_U). Both can be 0 at
                # one simplex vertex, which would give normalize(0)=NaN — fall back to
                # an equal blend there (the shared-base form has no such degeneracy,
                # since (0,0) simply leaves W0 untouched).
                a, b = (w_r, w_u_own) if (w_r + w_u_own) > 0 else (0.5, 0.5)
                if _premix:
                    return F.normalize(a * pd["r"].net(h) + b * pd["u"].net(h), dim=-1)
                return F.normalize(a * pd["r"](h) + b * pd["u"](h), dim=-1)

            for epoch in range(1, epochs + 1):
                tot = 0.0
                for views in loader:
                    # loader yields views[modality][augmentation]; there is no third
                    # element — the label is not used for SSL training
                    x1a = views[0][0].to(device); x1b = views[0][1].to(device)
                    x2a = views[1][0].to(device); x2b = views[1][1].to(device)
                    h1a, h1b = enc_m1(x1a), enc_m1(x1b)
                    h2a, h2b = enc_m2(x2a), enc_m2(x2b)
                    tau = (epoch - 1) / max(1, epochs - 1)
                    prefs = (anneal_simplex(simplex_prefs, tau, annealing_temperature)
                             if preference_schedule == "annealed" else simplex_prefs)
                    opt_sx.zero_grad()
                    loss = 0.0
                    for (w_r, w_u1, w_u2) in prefs:
                        z1a = _mix(proj_m1, h1a, w_r, w_u1, w_u2)
                        z1b = _mix(proj_m1, h1b, w_r, w_u1, w_u2)
                        z2a = _mix(proj_m2, h2a, w_r, w_u2, w_u1)
                        z2b = _mix(proj_m2, h2b, w_r, w_u2, w_u1)
                        loss = loss + (w_r * infonce_cross(z1a, z2a)
                                       + w_u1 * nt_xent(z1a, z1b)
                                       + w_u2 * nt_xent(z2a, z2b))
                    loss = loss / len(prefs)
                    loss.backward(); opt_sx.step()
                    tot += loss.item()
                sch_sx.step()
                writer_sx.add_scalar("loss/total", tot / len(loader), epoch)
                if epoch % 10 == 0 or epoch == 1:
                    print(f"  epoch {epoch:3d}/{epochs}  loss={tot/len(loader):.4f}")
            writer_sx.close()
            os.makedirs(out_dir, exist_ok=True)
            torch.save(enc_m1.state_dict(), os.path.join(out_dir, "enc_0.pth"))
            torch.save(enc_m2.state_dict(), os.path.join(out_dir, "enc_1.pth"))
            # Saved under the plain-head names the eval path already loads; the
            # 6h third head gets its own name.
            torch.save(proj_m1["r"].state_dict(), os.path.join(out_dir, "proj_m1_r.pth"))
            torch.save(proj_m2["r"].state_dict(), os.path.join(out_dir, "proj_m2_r.pth"))
            torch.save(proj_m1[_keys[1]].state_dict(), os.path.join(out_dir, "proj_m1_u.pth"))
            torch.save(proj_m2[_keys[1]].state_dict(), os.path.join(out_dir, "proj_m2_u.pth"))
            if _s6:
                torch.save(proj_m1["u2"].state_dict(), os.path.join(out_dir, "proj_m1_u2.pth"))
                torch.save(proj_m2["u2"].state_dict(), os.path.join(out_dir, "proj_m2_u2.pth"))
            with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
                json.dump(dict(method=method, simplex_side=simplex_side,
                               premix=_premix, three_way=_s6, epochs=epochs,
                               seed=seed, latent_dim=latent_dim, proj_dim=proj_dim),
                          f, indent=2)
            print(f"  Saved to {out_dir}")
            return

        elif method in ("simclr_simplex_enc_decomp_R", "simclr_simplex_proj_decomp_R",
                        "simclr_simplex_both_decomp_R"):
            # simplex_enc_decomp_R (trifeature port)
            # THE PROBLEM this fixes: every arm above uses L_U = NT-Xent on one
            # modality, which asks only for augmentation-invariance and contains no
            # term referencing the other modality. So L_R captures SHARED and L_U
            # captures SHARED + UNIQUE -- nested, not competing. The lambda axis
            # between them is only as long as the unique part, which is why the corr
            # sweep collapses the u2 span 0.2511 -> 0.1979 -> 0.1488 as U2 becomes a
            # copy of R.
            #
            # THE FIX: a CLUB penalty pushing U away from R, within modality (exactly
            # FactorCL's club_v(z_vua, z_vr.detach())).
            #
            # STRUCTURE: lambda-dependence lives in the ENCODER (PaLoRA weight space,
            # so lambda -> W(lambda) -> z(lambda) -> L(lambda) holds and a symmetric
            # grid cannot collapse to equal weighting); cleanliness lives in the HEADS
            # (four PLAIN heads, one objective each). Readout is a component-wise
            # CONCAT with sqrt(lambda) block scaling -- see probe_lambda_cond.
            #
            lora_rank = 4
            # Per-site adapter scale. The both-sites arm applies alpha at the encoder AND
            # the heads, so at a shared alpha its total conditioning is roughly doubled --
            # measured head dW/W 1.47 against the enc arm's 0.20. Separate knobs make
            # "both, at half strength each" expressible; alpha alone does NOT pin dW/W
            # (the adapters learn their own scale), so the ablation reports the MEASURED
            # dW/W next to every row rather than assuming alpha controls it.
            _a_enc  = LORA_ALPHA_ENC[0]  if LORA_ALPHA_ENC[0]  is not None else lora_alpha
            _a_head = LORA_ALPHA_HEAD[0] if LORA_ALPHA_HEAD[0] is not None else lora_alpha
            _alpha_enc = _a_enc if preference_schedule == "annealed" else None
            _projvar = (method == "simclr_simplex_proj_decomp_R")
            _bothvar = (method == "simclr_simplex_both_decomp_R")
            # WHERE the preference acts, as two INDEPENDENT sites. The original code had
            # one boolean for both, which made enc and proj mutually exclusive:
            #   enc  : LoRA encoder, plain heads      (the reported STEER)
            #   proj : lambda-blind encoder, LoRA heads   (ablation, lambda-inert)
            #   both : LoRA in the encoder AND the heads  (this variant)
            # Note "both" applies lambda TWICE -- W(lambda) then head(lambda) -- so the
            # effective preference dependence is multiplicative and the vertices separate
            # more than their simplex coordinates suggest. Keep that in mind when reading
            # the readout, which assumes each block carries its lambda share.
            _enc_cond  = not _projvar            # encoder carries LoRA (enc, both)
            _head_cond = _projvar or _bothvar    # heads carry LoRA   (proj, both)
            # Where the preference acts. ENC: LoRA in the encoder, plain heads.
            # PROJ: lambda-blind encoder, LoRA in the four heads instead.
            #
            # PROJ cannot simply drop the adapters: with a lambda-blind encoder AND
            # plain heads every head output is identical at every preference, so
            #   mean_l [ l_R*L_R + l_U1*L_U1 + l_U2*L_U2 ] == lbar_R*L_R + ...
            # i.e. training at the MEAN preference. That is the collapse PaLoRA warns
            # about, and it would make this arm silently identical to a fixed-uniform
            # control rather than a test of it.
            # encoder: LoRA (enc, both) or plain (proj)
            if _enc_cond:
                enc_m1 = LoRAAlexNetEncoder(latent_dim, branches=2, rank=lora_rank,
                                            alpha=_alpha_enc).to(device)
                enc_m2 = LoRAAlexNetEncoder(latent_dim, branches=2, rank=lora_rank,
                                            alpha=_alpha_enc).to(device)
            else:
                enc_m1 = AlexNetEncoder(latent_dim).to(device)
                enc_m2 = AlexNetEncoder(latent_dim).to(device)
            # heads: LoRA (proj, both) or plain (enc). FOUR of them either way -- r/u per
            # modality, one objective each.
            if _head_cond:
                _mk = lambda: LoRADualProjectionHead(latent_dim, proj_dim,
                                                     rank=lora_rank, alpha=_a_head).to(device)
                proj_r1, proj_r2, proj_u1, proj_u2 = _mk(), _mk(), _mk(), _mk()
            else:
                proj_r1 = ProjectionHead(latent_dim, proj_dim).to(device)
                proj_r2 = ProjectionHead(latent_dim, proj_dim).to(device)
                proj_u1 = ProjectionHead(latent_dim, proj_dim).to(device)
                proj_u2 = ProjectionHead(latent_dim, proj_dim).to(device)
            club_1 = CLUBInfoNCECritic(proj_dim, proj_dim, 512, 1, "relu").to(device)
            club_2 = CLUBInfoNCECritic(proj_dim, proj_dim, 512, 1, "relu").to(device)

            _LK = ("A_r", "B_r", "A_u", "B_u")
            _mods = [enc_m1, enc_m2, proj_r1, proj_r2, proj_u1, proj_u2]
            opt_lr = optim.AdamW([
                {"params": [q for m in _mods for n, q in m.named_parameters()
                            if not any(k in n for k in _LK)], "lr": lr},
                {"params": [q for m in _mods for n, q in m.named_parameters()
                            if any(k in n for k in _LK)], "lr": lr * LORA_LR_MULT[0]},
            ], weight_decay=1e-4)
            opt_critic = optim.Adam(list(club_1.parameters()) + list(club_2.parameters()), lr=lr)
            sch_lr = optim.lr_scheduler.CosineAnnealingLR(opt_lr, T_max=epochs)
            sch_critic = optim.lr_scheduler.CosineAnnealingLR(opt_critic, T_max=epochs)
            writer_lr = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
            simplex_prefs = simplex_grid(simplex_side)
            steps_per_epoch = len(loader); total_steps = max(1, epochs * steps_per_epoch)
            # M preferences per minibatch (PaLoRA-style). 0 or >= |grid| = every
            # preference every step, which is the reported configuration.
            _M = prefs_per_batch or len(simplex_prefs)
            _pref_cycle = PrefCycle(len(simplex_prefs), _M, seed=seed)
            _pref_loss_sum = np.zeros(len(simplex_prefs))
            print(f"  {method}: side={simplex_side} -> {len(simplex_prefs)} prefs "
                  f"| M={_M}/step | anneal={anneal_mode}"
                  f" | enc_cond={_enc_cond} head_cond={_head_cond}"
                  f" + 2 CLUB critics | lam_club={lam_club} "
                  f"| readout=concat(3 x {proj_dim})")

            for epoch in range(1, epochs + 1):
                tot = crit_tot = 0.0
                club_raw_tot = clamp_tot = cka_tot = erank_tot = cosr_tot = 0.0
                nb = 0
                for batch_idx, views in enumerate(loader):
                    m1a = views[0][0].to(device); m1b = views[0][1].to(device)
                    m2a = views[1][0].to(device); m2b = views[1][1].to(device)
                    tau = (((epoch - 1) * steps_per_epoch + batch_idx) / (total_steps - 1)
                           if total_steps > 1 else 1.0)
                    tau = min(1.0, max(0.0, tau))
                    _idx = _pref_cycle.next()
                    _targets = [simplex_prefs[i] for i in _idx]
                    prefs = (anneal_simplex(_targets, tau, annealing_temperature,
                                            mode=anneal_mode)
                             if preference_schedule == "annealed" else _targets)
                    # conv trunk is lambda-blind -> features once, LoRA classifier per pref
                    if _enc_cond:
                        f1a = enc_m1.feat(m1a); f1b = enc_m1.feat(m1b)
                        f2a = enc_m2.feat(m2a); f2b = enc_m2.feat(m2b)
                    else:
                        f1a, f1b = enc_m1(m1a), enc_m1(m1b)
                        f2a, f2b = enc_m2(m2a), enc_m2(m2b)

                    # ---- 1. critic step on the DETACHED pooled batch ----
                    with torch.no_grad():
                        _u1, _r1, _u2, _r2 = [], [], [], []
                        for (w_r, w_u1, w_u2) in prefs:
                            _h1 = enc_m1.head_mix_from_feat(f1a, w_r, w_u1) if _enc_cond else f1a
                            _h2 = enc_m2.head_mix_from_feat(f2a, w_r, w_u2) if _enc_cond else f2a
                            if _head_cond:
                                _u1.append(proj_u1.forward_mix(_h1, w_r, w_u1))
                                _r1.append(proj_r1.forward_mix(_h1, w_r, w_u1))
                                _u2.append(proj_u2.forward_mix(_h2, w_r, w_u2))
                                _r2.append(proj_r2.forward_mix(_h2, w_r, w_u2))
                            else:
                                _u1.append(proj_u1(_h1)); _r1.append(proj_r1(_h1))
                                _u2.append(proj_u2(_h2)); _r2.append(proj_r2(_h2))
                    opt_critic.zero_grad()
                    _cl = (club_1.learning_loss(torch.cat(_u1), torch.cat(_r1)) +
                           club_2.learning_loss(torch.cat(_u2), torch.cat(_r2)))
                    _cl.backward(); opt_critic.step(); crit_tot += float(_cl.item())

                    # ---- 2. encoder + head step ----
                    opt_lr.zero_grad()
                    loss = 0.0
                    _braw = _bclamp = _bcka = _berank = _bcos = 0.0
                    for _pi, (w_r, w_u1, w_u2) in enumerate(prefs):
                        if _enc_cond:
                            h1a = enc_m1.head_mix_from_feat(f1a, w_r, w_u1)
                            h1b = enc_m1.head_mix_from_feat(f1b, w_r, w_u1)
                            h2a = enc_m2.head_mix_from_feat(f2a, w_r, w_u2)
                            h2b = enc_m2.head_mix_from_feat(f2b, w_r, w_u2)
                        else:
                            h1a, h1b, h2a, h2b = f1a, f1b, f2a, f2b
                        if _head_cond:
                            zr1 = proj_r1.forward_mix(h1a, w_r, w_u1)
                            zu1 = proj_u1.forward_mix(h1a, w_r, w_u1)
                            zr2 = proj_r2.forward_mix(h2a, w_r, w_u2)
                            zu2 = proj_u2.forward_mix(h2a, w_r, w_u2)
                            zu1b = proj_u1.forward_mix(h1b, w_r, w_u1)
                            zu2b = proj_u2.forward_mix(h2b, w_r, w_u2)
                        else:
                            zr1, zu1 = proj_r1(h1a), proj_u1(h1a)
                            zr2, zu2 = proj_r2(h2a), proj_u2(h2a)
                            zu1b, zu2b = proj_u1(h1b), proj_u2(h2b)
                        L_R  = infonce_cross(zr1, zr2)
                        L_U1 = nt_xent(zu1, zu1b)
                        L_U2 = nt_xent(zu2, zu2b)
                        raw1 = club_1(zu1, zr1.detach()); raw2 = club_2(zu2, zr2.detach())
                        use1 = torch.clamp(raw1, min=0.0); use2 = torch.clamp(raw2, min=0.0)
                        L_U1 = L_U1 + lam_club * use1
                        L_U2 = L_U2 + lam_club * use2
                        _pl = (w_r * L_R + w_u1 * L_U1 + w_u2 * L_U2)
                        loss = loss + _pl
                        # attribute the step's loss to the TARGET preference, not the
                        # annealed one -- prefs[_pi] is the annealed image of _idx[_pi].
                        _pref_loss_sum[_idx[_pi]] += float(_pl.item())
                        _braw   += float(raw1.item() + raw2.item()) / 2
                        _bclamp += (int(raw1.item() <= 0) + int(raw2.item() <= 0)) / 2
                        with torch.no_grad():
                            _bcka   += (_lin_cka_t(zu1, zr1) + _lin_cka_t(zu2, zr2)) / 2
                            _berank += (_eff_rank_t(zu1) + _eff_rank_t(zu2)) / 2
                            _bcos   += float(F.cosine_similarity(zr1, zr2, dim=-1).mean())
                    loss = loss / len(prefs)
                    loss.backward(); opt_lr.step()
                    tot += float(loss.item()); nb += 1
                    _np = len(prefs)
                    club_raw_tot += _braw / _np; clamp_tot += _bclamp / _np
                    cka_tot += _bcka / _np; erank_tot += _berank / _np; cosr_tot += _bcos / _np
                sch_lr.step(); sch_critic.step()
                _n = max(1, nb)
                writer_lr.add_scalar("loss/total",        tot / _n,          epoch)
                writer_lr.add_scalar("loss/critic",       crit_tot / _n,     epoch)
                writer_lr.add_scalar("club/raw",          club_raw_tot / _n, epoch)
                writer_lr.add_scalar("club/clamp_frac",   clamp_tot / _n,    epoch)
                writer_lr.add_scalar("decomp/cka_u_r",    cka_tot / _n,      epoch)
                writer_lr.add_scalar("decomp/eff_rank_u", erank_tot / _n,    epoch)
                writer_lr.add_scalar("decomp/cos_r1_r2",  cosr_tot / _n,     epoch)
                if epoch % 10 == 0 or epoch == 1:
                    _cf = clamp_tot / _n; _er = erank_tot / _n
                    _v = ("CLUB INERT (clamped)" if _cf > 0.8 else
                          "collapse risk" if _er < 2.0 else "binding")
                    print(f"  epoch {epoch:3d}/{epochs}  loss={tot/_n:.4f}  "
                          f"crit={crit_tot/_n:.4f}  club_raw={club_raw_tot/_n:+.4f}  "
                          f"clamp={_cf:.2f}  cka(u,r)={cka_tot/_n:.4f}  effrank_u={_er:.1f}  "
                          f"cos(r1,r2)={cosr_tot/_n:+.3f}  [{_v}]")
            writer_lr.close()
            _lk = ("A_r", "B_r", "A_u", "B_u")
            # ACHIEVED conditioning strength at each site, in units of the base weight
            # norm. alpha does not pin this -- the adapters learn their own scale (the
            # both arm hit 1.47 at the same alpha the enc arm runs 0.20 at) -- so it is
            # measured, printed AND stored, and every ablation row is read next to it.
            def _dw_ratio(layers):
                vals = []
                for l in layers:
                    w = float(l.linear.weight.norm())
                    if w > 0:
                        vals.append(max(float((l.A_r @ l.B_r).norm()),
                                        float((l.A_u @ l.B_u).norm())) / w)
                return max(vals) if vals else None
            _hd = _dw_ratio([l for h in (proj_r1, proj_r2, proj_u1, proj_u2)
                             for l in (h.layer1, h.layer2, h.layer3)]) if _head_cond else None
            _ed = _dw_ratio([e.classifier for e in (enc_m1, enc_m2)]) if _enc_cond else None
            print(f"  dW/W  encoder={_ed if _ed is None else round(_ed, 4)}  "
                  f"heads={_hd if _hd is None else round(_hd, 4)}"
                  f"   [enc arm reaches ~0.20; >1 means the adapters dominate the base]")
            torch.save(enc_m1.state_dict(),  os.path.join(out_dir, "lora_enc_0.pth"))
            torch.save(enc_m2.state_dict(),  os.path.join(out_dir, "lora_enc_1.pth"))
            torch.save(proj_r1.state_dict(), os.path.join(out_dir, "proj_m1_r.pth"))
            torch.save(proj_r2.state_dict(), os.path.join(out_dir, "proj_m2_r.pth"))
            torch.save(proj_u1.state_dict(), os.path.join(out_dir, "proj_m1_u.pth"))
            torch.save(proj_u2.state_dict(), os.path.join(out_dir, "proj_m2_u.pth"))
            with open(os.path.join(out_dir, "train_meta.json"), "w") as _f:
                json.dump(dict(method=method, simplex_side=simplex_side,
                               decomp_R=True, decomp_R_proj=_projvar,
                               decomp_R_sites=([s for s, on in
                                                (("enc", _enc_cond), ("proj", _head_cond)) if on]),
                               lora_rank=lora_rank, lora_alpha=lora_alpha,
                               lam_club=lam_club,
                               prefs_per_batch=_M, anneal_mode=anneal_mode,
                               pref_visits=[int(c) for c in _pref_cycle.counts],
                               pref_loss_mean=[float(t / max(1, c)) for t, c in
                                               zip(_pref_loss_sum, _pref_cycle.counts)],
                               readout="concat_sqrt_lambda", proj_dim=proj_dim,
                               latent_dim=latent_dim, epochs=epochs, seed=seed,
                               modality_pair=_ds_pair(), augment=_get_aug(),
                               lora_alpha_enc=_a_enc, lora_alpha_head=_a_head,
                               lora_lr_mult=LORA_LR_MULT[0],
                               dw_ratio_enc=_ed, dw_ratio_head=_hd), _f, indent=2)
            print(f"  Saved → {out_dir}")
            return

        elif method in ("simclr_lora_linear", "simclr_lora_curve", "simclr_lora_simplex",
                        "simclr_lora_enc_linear", "simclr_lora_enc_curve",
                        "simclr_lora_enc_simplex"):
            # Approach 4 (lora_linear / lora_curve): LoRA projection head, frozen AlexNet
            # Approach 5 (lora_enc_*):               LoRA on both encoder classifier + proj
            #
            # linear variants: 2-branch (R, U), endpoint training only
            # curve variants:  3-branch (R, U, M), fix3 = 30-ep endpoint warmup then blended
            is_curve   = "curve"   in method
            is_enc     = "enc"     in method
            # LoRA-Simplex: 3 objectives (R, U1, U2) on the 2-simplex, PaLoRA-style.
            # Each modality head keeps 2 branches (R + its own U); only the mixing
            # coefficients become 3-dimensional. Linear only — no curve variant.
            is_simplex = "simplex" in method
            branches  = 3 if is_curve else 2
            warmup_ep = 30 if is_curve else 0
            lora_rank = 4

            # PaLoRA α/r scaling — applied only for annealed (fix8 / simplex) runs
            _alpha_enc = lora_alpha if preference_schedule == "annealed" else None
            if is_enc:
                enc_m1 = LoRAAlexNetEncoder(latent_dim, branches=branches, rank=lora_rank,
                                            alpha=_alpha_enc).to(device)
                enc_m2 = LoRAAlexNetEncoder(latent_dim, branches=branches, rank=lora_rank,
                                            alpha=_alpha_enc).to(device)
            # else: enc_m1, enc_m2 already created above as AlexNetEncoder

            ProjCls = LoRATriProjectionHead if is_curve else LoRADualProjectionHead
            # PaLoRA α/r scaling — Fix 8 (annealed) only; base runs keep α=None → scale 1.0
            _alpha = lora_alpha if preference_schedule == "annealed" else None
            lora_proj_m1 = ProjCls(latent_dim, proj_dim, rank=lora_rank, alpha=_alpha).to(device)
            lora_proj_m2 = ProjCls(latent_dim, proj_dim, rank=lora_rank, alpha=_alpha).to(device)

            _LORA_KEYS = ("A_r", "B_r", "A_u", "B_u", "A_m", "B_m")
            def _lora_params(*mods):
                return [p for m in mods for n, p in m.named_parameters()
                        if any(k in n for k in _LORA_KEYS)]
            def _base_params(*mods):
                return [p for m in mods for n, p in m.named_parameters()
                        if not any(k in n for k in _LORA_KEYS)]

            all_mods = [enc_m1, enc_m2, lora_proj_m1, lora_proj_m2]
            opt_lr = optim.AdamW([
                {"params": _base_params(*all_mods), "lr": lr},
                {"params": _lora_params(*all_mods), "lr": lr * 10},
            ], weight_decay=1e-4)
            sch_lr = optim.lr_scheduler.CosineAnnealingLR(opt_lr, T_max=epochs)
            writer_lr = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

            # Fix 8: PaLoRA multi-preference setup
            # LoRA-Simplex: deterministic side-5 triangular grid (15 preferences)
            if is_simplex:
                simplex_prefs = simplex_grid(simplex_side)
                print(f"  LoRA-Simplex: side={simplex_side} grid → {len(simplex_prefs)} preferences "
                      f"| schedule={preference_schedule} | Q={annealing_temperature} | alpha={lora_alpha}")

            _pref_on = (preference_schedule != "single")
            steps_per_epoch = len(loader)
            total_steps = max(1, epochs * steps_per_epoch)
            if _pref_on:
                if num_preferences < 1:
                    raise ValueError("num_preferences (M) must be >= 1")
                if annealing_temperature <= 0:
                    raise ValueError("annealing_temperature (Q) must be > 0")
                print(f"  Fix 8: M={num_preferences} | schedule={preference_schedule} "
                      f"| Q={annealing_temperature} | alpha={lora_alpha}")

            for epoch in range(1, epochs + 1):
                total_lr = 0.0
                in_warmup = (is_curve and epoch <= warmup_ep)
                ep_tau = 0.0; ep_lmin = ep_lmax = ep_lmean = 0.5; ep_pref = False
                for batch_idx, views in enumerate(loader):
                    m1a = views[0][0].to(device); m1b = views[0][1].to(device)
                    m2a = views[1][0].to(device); m2b = views[1][1].to(device)
                    lam = sample_lambda(lam_dist)
                    _multi = (_pref_on and not in_warmup)

                    opt_lr.zero_grad()

                    if is_simplex:
                        # LoRA-Simplex: 3 objectives (R, U1, U2) on the 2-simplex
                        # Encoder is λ-blind → h computed once; only the two heads are
                        # rerun per preference. M forwards, ONE backward, ONE step.
                        if is_enc:
                            # encoder-LoRA: conv trunk is λ-blind → features once,
                            # only the LoRA classifier is rerun per preference
                            f1a = enc_m1.feat(m1a); f1b = enc_m1.feat(m1b)
                            f2a = enc_m2.feat(m2a); f2b = enc_m2.feat(m2b)
                        else:
                            h1a = enc_m1(m1a); h1b = enc_m1(m1b)
                            h2a = enc_m2(m2a); h2b = enc_m2(m2b)
                        tau = (((epoch - 1) * steps_per_epoch + batch_idx) / (total_steps - 1)
                               if total_steps > 1 else 1.0)
                        tau = min(1.0, max(0.0, tau))
                        prefs = (anneal_simplex(simplex_prefs, tau, annealing_temperature)
                                 if preference_schedule == "annealed" else simplex_prefs)
                        loss = 0.0
                        for (w_r, w_u1, w_u2) in prefs:
                            if is_enc:
                                h1a = enc_m1.head_mix_from_feat(f1a, w_r, w_u1)
                                h1b = enc_m1.head_mix_from_feat(f1b, w_r, w_u1)
                                h2a = enc_m2.head_mix_from_feat(f2a, w_r, w_u2)
                                h2b = enc_m2.head_mix_from_feat(f2b, w_r, w_u2)
                            # head m1 sees (λ_R, λ_U1); head m2 sees (λ_R, λ_U2)
                            z1a = lora_proj_m1.forward_mix(h1a, w_r, w_u1)
                            z1b = lora_proj_m1.forward_mix(h1b, w_r, w_u1)
                            z2a = lora_proj_m2.forward_mix(h2a, w_r, w_u2)
                            z2b = lora_proj_m2.forward_mix(h2b, w_r, w_u2)
                            L_R  = infonce_cross(z1a, z2a)
                            L_U1 = nt_xent(z1a, z1b)
                            L_U2 = nt_xent(z2a, z2b)
                            # simplex scalarization — coefficients already sum to 1
                            loss = loss + (w_r * L_R + w_u1 * L_U1 + w_u2 * L_U2)
                        loss = loss / len(prefs)
                        ep_tau = tau; ep_pref = True
                        _r = [p[0] for p in prefs]
                        ep_lmin = min(_r); ep_lmax = max(_r); ep_lmean = sum(_r) / len(_r)
                    elif _multi:
                        # Fix 8: PaLoRA M-preference annealed step
                        # λ-blind encoder → h computed once; heads rerun per λ_m.
                        if is_enc:
                            # encoder-LoRA: λ-blind conv trunk → features once,
                            # LoRA classifier rerun per preference
                            f1a = enc_m1.feat(m1a); f1b = enc_m1.feat(m1b)
                            f2a = enc_m2.feat(m2a); f2b = enc_m2.feat(m2b)
                        else:
                            h1a = enc_m1(m1a); h1b = enc_m1(m1b)
                            h2a = enc_m2(m2a); h2b = enc_m2(m2b)
                        tau = (((epoch - 1) * steps_per_epoch + batch_idx) / (total_steps - 1)
                               if total_steps > 1 else 1.0)
                        tau = min(1.0, max(0.0, tau))
                        if preference_schedule == "annealed":
                            lams_t = _anneal_preferences(num_preferences, tau, annealing_temperature, device)
                        else:  # "fixed"
                            lams_t = torch.linspace(0.0, 1.0, num_preferences, device=device)
                        lams = lams_t.tolist()
                        loss = 0.0
                        for lam_m in lams:
                            if is_enc:
                                h1a = enc_m1.head_from_feat(f1a, lam_m)
                                h1b = enc_m1.head_from_feat(f1b, lam_m)
                                h2a = enc_m2.head_from_feat(f2a, lam_m)
                                h2b = enc_m2.head_from_feat(f2b, lam_m)
                            z1a = lora_proj_m1(h1a, lam_m); z1b = lora_proj_m1(h1b, lam_m)
                            z2a = lora_proj_m2(h2a, lam_m); z2b = lora_proj_m2(h2b, lam_m)
                            L_R = infonce_cross(z1a, z2a)
                            # Train BOTH unique branches (U1 and U2) symmetrically — matches
                            # the base endpoint path and the multibench fix8. (The blended
                            # curve path only contrasts modality 1, which starved U2.)
                            L_U = (nt_xent(z1a, z1b) + nt_xent(z2a, z2b)) / 2
                            loss = loss + (2.0 * lam_m * L_R + 2.0 * (1.0 - lam_m) * L_U)
                        loss = loss / num_preferences                 # average over M
                        ep_tau = tau; ep_pref = True
                        ep_lmin = min(lams); ep_lmax = max(lams); ep_lmean = sum(lams) / len(lams)
                    elif in_warmup or not is_curve:
                        # Endpoint training: R and U branches, gradient-isolated
                        if is_enc:
                            h1a_r = enc_m1.forward_r(m1a); h1b_r = enc_m1.forward_r(m1b)
                            h2a_r = enc_m2.forward_r(m2a)
                            h1a_u = enc_m1.forward_u(m1a); h1b_u = enc_m1.forward_u(m1b)
                            h2a_u = enc_m2.forward_u(m2a); h2b_u = enc_m2.forward_u(m2b)
                        else:
                            h1a_r = h1a_u = enc_m1(m1a)
                            h1b_r = h1b_u = enc_m1(m1b)
                            h2a_r = h2a_u = enc_m2(m2a)
                            h2b_u          = enc_m2(m2b)
                        z1_r  = lora_proj_m1.forward_r(h1a_r)
                        z2_r  = lora_proj_m2.forward_r(h2a_r)
                        z1_u  = lora_proj_m1.forward_u(h1a_u)
                        z1_ub = lora_proj_m1.forward_u(h1b_u)
                        z2_u  = lora_proj_m2.forward_u(h2a_u)
                        z2_ub = lora_proj_m2.forward_u(h2b_u)
                        L_R = infonce_cross(z1_r, z2_r)
                        L_U = (nt_xent(z1_u, z1_ub) + nt_xent(z2_u, z2_ub)) / 2
                    else:
                        # Blended curve training: forward(h, lam) everywhere
                        if is_enc:
                            h1a = enc_m1(m1a, lam); h1b = enc_m1(m1b, lam)
                            h2a = enc_m2(m2a, lam); h2b = enc_m2(m2b, lam)
                        else:
                            h1a = enc_m1(m1a); h1b = enc_m1(m1b)
                            h2a = enc_m2(m2a); h2b = enc_m2(m2b)
                        z1  = lora_proj_m1(h1a, lam); z1b = lora_proj_m1(h1b, lam)
                        z2  = lora_proj_m2(h2a, lam)
                        L_R = infonce_cross(z1, z2)
                        L_U = nt_xent(z1, z1b)

                    if not _multi and not is_simplex:
                        loss = 2 * lam * L_R + 2 * (1.0 - lam) * L_U
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for m in all_mods for p in m.parameters()], max_norm=1.0)
                    opt_lr.step()
                    total_lr += loss.item()

                sch_lr.step()
                avg = total_lr / len(loader)
                writer_lr.add_scalar("loss/train", avg, epoch)
                if ep_pref:   # Fix 8 preference / annealing state
                    writer_lr.add_scalar("pref/tau",      ep_tau,   epoch)
                    writer_lr.add_scalar("pref/lam_min",  ep_lmin,  epoch)
                    writer_lr.add_scalar("pref/lam_max",  ep_lmax,  epoch)
                    writer_lr.add_scalar("pref/lam_mean", ep_lmean, epoch)
                phase = "fix8" if ep_pref else ("warmup" if in_warmup else "curve" if is_curve else "train")
                if epoch % 10 == 0 or epoch == 1:
                    pstr = (f"  τ={ep_tau:.3f} λ∈[{ep_lmin:.3f},{ep_lmax:.3f}] λ̄={ep_lmean:.3f}"
                            if ep_pref else "")
                    print(f"  [{method}|{phase}] epoch {epoch:3d}/{epochs}  loss={avg:.4f}{pstr}")

            writer_lr.close()
            enc_key = "lora_enc" if is_enc else "enc"
            torch.save(enc_m1.state_dict(),       os.path.join(out_dir, f"{enc_key}_0.pth"))
            torch.save(enc_m2.state_dict(),       os.path.join(out_dir, f"{enc_key}_1.pth"))
            torch.save(lora_proj_m1.state_dict(), os.path.join(out_dir, "lora_proj_m1.pth"))
            torch.save(lora_proj_m2.state_dict(), os.path.join(out_dir, "lora_proj_m2.pth"))
            import json as _json
            with open(os.path.join(out_dir, "train_meta.json"), "w") as _f:
                _meta = {"method": method, "branches": branches,
                         "rank": lora_rank, "latent_dim": latent_dim,
                         "proj_dim": proj_dim,
                         "modality_pair": _ds_pair()}
                if is_simplex:
                    _meta.update({"simplex_side": simplex_side,
                                  "preference_schedule": preference_schedule,
                                  "annealing_temperature": annealing_temperature,
                                  "lora_alpha": lora_alpha})
                _json.dump(_meta, _f, indent=2)
            print(f"  Saved to {out_dir}")
            return

        else:
            raise ValueError(f"Unknown method: {method}")

        optimizer = optim.AdamW(_params(*opt_modules), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

        for epoch in range(1, epochs + 1):
            total = 0.0
            for views in loader:
                optimizer.zero_grad()
                loss = step(views)
                loss.backward()
                optimizer.step()
                total += loss.item()
            scheduler.step()
            avg = total / len(loader)
            writer.add_scalar("loss/train", avg, epoch)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  [{method}] epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

        writer.close()
        torch.save(enc_m1.state_dict(), os.path.join(out_dir, "enc_0.pth"))
        torch.save(enc_m2.state_dict(), os.path.join(out_dir, "enc_1.pth"))

    print(f"  Saved to {out_dir}")


# Feature extraction

@torch.no_grad()
def _extract_pairs(enc_m1, enc_m2, loader, device, mode, mmfusion=None,
                   proj_r=None, proj_u=None, task=None, proj_m2_u=None):
    """
    mode = mmfusion_joint    : MMFusion([M1,M2]) CLS token → (B, 512)
    mode = unimodal_m1       : enc_m1(M1)                 → (B, latent_dim)
    mode = unimodal_m2       : enc_m2(M2)                 → (B, latent_dim)
    mode = factorcl_heads    :
      share   → proj_r(enc_m1(M1))
      unique1 → proj_u(enc_m1(M1))
      unique2 → proj_m2_u(enc_m2(M2))
    one-modality modes (force a single modality for ALL tasks)
    mode = one_mod_m1        : enc_m1(M1) for all tasks
    mode = one_mod_m2        : enc_m2(M2) for all tasks
    mode = mmfusion_m1_only  : MMFusion([M1,M2], mask=[[True,False]]) — M1 input only
    mode = mmfusion_m2_only  : MMFusion([M1,M2], mask=[[False,True]]) — M2 input only
    mode = factorcl_m1_only  : F.normalize(cat[proj_r(enc_m1(M1)), proj_u(enc_m1(M1))])
    mode = factorcl_m2_only  : F.normalize(cat[proj_r(enc_m2(M2)), proj_m2_u(enc_m2(M2))])
    """
    for m in [enc_m1, enc_m2, mmfusion, proj_r, proj_u, proj_m2_u]:
        if m is not None:
            m.eval()

    feats, lbls = [], []
    for imgs, batch_lbls in loader:
        m1 = imgs[0].to(device)
        m2 = imgs[1].to(device)
        if mode == "mmfusion_joint":
            z = mmfusion([m1, m2])
        elif mode == "unimodal_m2":
            z = enc_m2(m2)
        elif mode == "factorcl_heads":
            if task == "share":
                z = proj_r(enc_m1(m1))
            elif task == "unique2":
                z = proj_m2_u(enc_m2(m2))
            else:
                z = proj_u(enc_m1(m1))
        elif mode == "factorcl_concat":
            # Faithful to FactorCL (Liang et al.) MultiBench deployment:
            # concatenate ALL heads from BOTH modalities into one vector,
            # identical for every downstream task — the linear probe decides
            # which factors matter.
            z = F.normalize(torch.cat([
                proj_r(enc_m1(m1)),    # shared info from M1
                proj_u(enc_m1(m1)),    # unique info from M1
                proj_r(enc_m2(m2)),    # shared info from M2  (proj_r is cross-modal)
                proj_m2_u(enc_m2(m2)), # unique info from M2
            ], dim=-1), dim=-1)
        # one-modality modes
        elif mode == "one_mod_m1":
            z = enc_m1(m1)
        elif mode == "one_mod_m2":
            z = enc_m2(m2)
        elif mode == "mmfusion_m1_only":
            z = mmfusion([m1, m2], mask_modalities=[True, False])
        elif mode == "mmfusion_m2_only":
            z = mmfusion([m1, m2], mask_modalities=[False, True])
        elif mode == "factorcl_m1_only":
            z = F.normalize(torch.cat([proj_r(enc_m1(m1)), proj_u(enc_m1(m1))], dim=-1), dim=-1)
        elif mode == "factorcl_m2_only":
            z = F.normalize(torch.cat([proj_r(enc_m2(m2)), proj_m2_u(enc_m2(m2))], dim=-1), dim=-1)
        else:
            z = enc_m1(m1)
        feats.append(z.cpu().numpy())
        lbls.append(batch_lbls.cpu().numpy() if isinstance(batch_lbls, torch.Tensor)
                    else np.asarray(batch_lbls))
    return np.concatenate(feats), np.concatenate(lbls)


# Linear probe

def _kshot_subsample(feats, labels, k: int, seed: int = 42):
    """Keep at most k samples per class."""
    rng = np.random.default_rng(seed)
    idx = np.concatenate([
        rng.choice(np.where(labels == c)[0],
                   size=min(k, (labels == c).sum()), replace=False)
        for c in np.unique(labels)
    ])
    return feats[idx], labels[idx]


def _probe(tr_f, tr_l, te_f, te_l, k_shot: int = 0,
           Cs=(1e-2, 0.1, 1.0, 10.0)):
    if k_shot > 0:
        tr_f, tr_l = _kshot_subsample(tr_f, tr_l, k_shot)

    sc  = StandardScaler()
    tr  = sc.fit_transform(tr_f)
    te  = sc.transform(te_f)
    n   = len(tr)
    rng = np.random.default_rng(42)
    val_idx   = rng.choice(n, size=max(1, n // 5), replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)
    best_C, best_val = Cs[0], -1.0
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
        clf.fit(tr[train_idx], tr_l[train_idx])
        v = clf.score(tr[val_idx], tr_l[val_idx])
        if v > best_val:
            best_val, best_C = v, C
    clf = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
    clf.fit(tr, tr_l)
    return clf.score(te, te_l)


# Probing

def probe_method(method: str, enc_dir: str, data_dir: str, device: str,
                 latent_dim: int, batch_size: int, k_shot: int = 0,
                 proj_dim: int = 256) -> dict:
    """
    K-shot pair probe for all methods.
    k_shot=0  → use all available probe training pairs (may saturate at 1.0)
    k_shot=10 → 10 pairs per class (100 total) — forces meaningful differentiation
    """
    mode = INFERENCE_MODE[method]

    def _loader(task, split):
        return DataLoader(
            PairProbeDataset(data_dir, split=split, task=task),
            batch_size=batch_size, shuffle=False, num_workers=4,
        )

    enc_m1 = enc_m2 = mmfusion = proj_r = proj_u = proj_m2_u = None

    if mode == "mmfusion_joint":
        mmfusion = _make_mmfusion(device)
        mmfusion.load_state_dict(
            torch.load(os.path.join(enc_dir, "mmfusion.pth"),
                       map_location=device, weights_only=True))
    elif mode in ("factorcl_heads", "factorcl_concat"):
        def _load_enc(name):
            enc = AlexNetEncoder(latent_dim).to(device)
            enc.load_state_dict(torch.load(os.path.join(enc_dir, name),
                                           map_location=device, weights_only=True))
            return enc
        def _load_head(name):
            h = ProjectionHead(latent_dim, proj_dim).to(device)
            h.load_state_dict(torch.load(os.path.join(enc_dir, name),
                                         map_location=device, weights_only=True))
            return h
        enc_m1   = _load_enc("enc_0.pth")
        enc_m2   = _load_enc("enc_1.pth")
        proj_r   = _load_head("proj_r.pth")
        proj_u   = _load_head("proj_u.pth")
        proj_m2_u = _load_head("proj_m2_u.pth")
    else:
        def _load_enc(name):
            enc = AlexNetEncoder(latent_dim).to(device)
            enc.load_state_dict(
                torch.load(os.path.join(enc_dir, name),
                           map_location=device, weights_only=True))
            return enc
        enc_m1 = _load_enc("enc_0.pth")
        enc_m2 = _load_enc("enc_1.pth")

    results = {}
    for task in TASKS:
        if mode == "unimodal_split":
            task_mode = "unimodal_m2" if task == "unique2" else "unimodal_m1"
        else:
            task_mode = mode

        tr_f, tr_l = _extract_pairs(enc_m1, enc_m2, _loader(task, "train"),
                                     device, task_mode, mmfusion, proj_r, proj_u, task,
                                     proj_m2_u=proj_m2_u)
        te_f, te_l = _extract_pairs(enc_m1, enc_m2, _loader(task, "test"),
                                     device, task_mode, mmfusion, proj_r, proj_u, task,
                                     proj_m2_u=proj_m2_u)
        acc = _probe(tr_f, tr_l, te_f, te_l, k_shot=k_shot)
        results[task] = round(float(acc), 4)
        shot_str = f"  [{k_shot}-shot]" if k_shot > 0 else ""
        print(f"    {task} ({task_mode}){shot_str}: {acc:.4f}")

    return results


def probe_one_modality_fixed(method: str, enc_dir: str, data_dir: str, device: str,
                             latent_dim: int, batch_size: int, k_shot: int = 0,
                             proj_dim: int = 256) -> dict:
    """
    One-modality inference for fixed (non-lambda) methods.
    Probes all tasks using ONLY M1, then ONLY M2.
    Returns {"m1": {"share", "unique1", "unique2"}, "m2": {...}}.

    Mode mapping per method:
      unimodal_split  → one_mod_m1 / one_mod_m2     (simclr_both, clip)
      mmfusion_joint  → mmfusion_m1_only / _m2_only  (gmc, comm)
      factorcl_heads  → factorcl_m1_only / _m2_only  (factorcl_heads)
      factorcl_concat → factorcl_m1_only / _m2_only  (factorcl)
    """
    mode = INFERENCE_MODE[method]

    def _loader(task, split):
        return DataLoader(
            PairProbeDataset(data_dir, split=split, task=task),
            batch_size=batch_size, shuffle=False, num_workers=4,
        )

    enc_m1 = enc_m2 = mmfusion = proj_r = proj_u = proj_m2_u = None

    if mode == "mmfusion_joint":
        mmfusion = _make_mmfusion(device)
        mmfusion.load_state_dict(
            torch.load(os.path.join(enc_dir, "mmfusion.pth"),
                       map_location=device, weights_only=True))
        mode_m1, mode_m2 = "mmfusion_m1_only", "mmfusion_m2_only"
    elif mode in ("factorcl_heads", "factorcl_concat"):
        def _load_enc(name):
            enc = AlexNetEncoder(latent_dim).to(device)
            enc.load_state_dict(torch.load(os.path.join(enc_dir, name),
                                           map_location=device, weights_only=True))
            return enc
        def _load_head(name):
            h = ProjectionHead(latent_dim, proj_dim).to(device)
            h.load_state_dict(torch.load(os.path.join(enc_dir, name),
                                         map_location=device, weights_only=True))
            return h
        enc_m1    = _load_enc("enc_0.pth")
        enc_m2    = _load_enc("enc_1.pth")
        proj_r    = _load_head("proj_r.pth")
        proj_u    = _load_head("proj_u.pth")
        proj_m2_u = _load_head("proj_m2_u.pth")
        mode_m1, mode_m2 = "factorcl_m1_only", "factorcl_m2_only"
    else:
        # unimodal_split: simclr_both, clip, random
        def _load_enc(name):
            enc = AlexNetEncoder(latent_dim).to(device)
            enc.load_state_dict(torch.load(os.path.join(enc_dir, name),
                                           map_location=device, weights_only=True))
            return enc
        enc_m1 = _load_enc("enc_0.pth")
        enc_m2 = _load_enc("enc_1.pth")
        mode_m1, mode_m2 = "one_mod_m1", "one_mod_m2"

    results = {}
    for mod_name, mod_mode in [("m1", mode_m1), ("m2", mode_m2)]:
        print(f"\n  [one_modality / {mod_name}]")
        res = {}
        for task in TASKS:
            tr_f, tr_l = _extract_pairs(enc_m1, enc_m2, _loader(task, "train"),
                                        device, mod_mode, mmfusion, proj_r, proj_u,
                                        task, proj_m2_u=proj_m2_u)
            te_f, te_l = _extract_pairs(enc_m1, enc_m2, _loader(task, "test"),
                                        device, mod_mode, mmfusion, proj_r, proj_u,
                                        task, proj_m2_u=proj_m2_u)
            acc = _probe(tr_f, tr_l, te_f, te_l, k_shot=k_shot)
            res[task] = round(float(acc), 4)
            print(f"    {task}: {acc:.4f}")
        results[mod_name] = res

    return results


# Lambda-conditioned probe (sweep over lambda values)

def _lin_cka_t(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA on a batch, as a float. Same formula as linear_cka below and as
    multibench's, so the numbers are directly comparable.

    CLUB reports its own estimate of I(u;r), but that is only meaningful when the
    critic is well fit. This measures the EFFECT the penalty should have -- how
    aligned u and r actually are -- independently of CLUB. It must FALL."""
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    xy = (X.T @ Y).pow(2).sum()
    xx = (X.T @ X).pow(2).sum().sqrt(); yy = (Y.T @ Y).pow(2).sum().sqrt()
    return float(xy / (xx * yy)) if float(xx * yy) > 0 else 0.0


def _eff_rank_t(Z: torch.Tensor) -> float:
    """Participation ratio of the covariance eigenvalues. Guards the opposite
    failure: too large a lam_club drives z_u to a constant, which trivially
    decorrelates it while destroying the representation. Collapse toward 1 means
    lam_club is too big."""
    Z = Z - Z.mean(0, keepdim=True)
    ev = torch.clamp(torch.linalg.eigvalsh(torch.cov(Z.T.float())), min=0.0)
    s1, s2 = ev.sum(), (ev ** 2).sum()
    return float(s1 * s1 / s2) if float(s2) > 0 else 0.0


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between representation matrices X, Y (n x d).

    Byte-for-byte the same formula as multibench/probe.py:linear_cka, so the
    trifeature adjacent-lambda CKA is directly comparable to the 0.92-0.97 band
    measured on MOSEI. Do not "improve" it here without changing it there too.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 0 else 0.0


# Readout used by the decomp_R family. "amp" is the sqrt(lambda) amplitude concat
# the trifeature results were produced with; "dim" allocates round(lambda*proj_dim)
# DIMENSIONS per block instead.
#
# Why "dim" exists: a LINEAR probe computes sum_b (w_b * sqrt(lam_b)) * b and can
# rescale w_b by 1/sqrt(lam_b), recovering any attenuated block exactly. So under
# "amp" only a block being EXACTLY zero changes the readout, the 15 preferences
# collapse to at most 7 distinct supports, and the INTERIOR of the simplex is not
# resolved. Measured on MOSEI: ~85% of the spread across preferences was explained
# by support alone; switching to "dim" roughly doubled the within-support spread.
# The three vertices are identical under both readouts (all D dims to one block).
_READOUT = ["amp"]
# Which split the probe REPORTS on. "test" is the historical default. The placement x
# strength ablation selects its winner on "val" and only then reports test, so the
# comparison across 9 configurations never touches the test split.
_EVAL_SPLIT = ["test"]
# Adapter knobs, set from the CLI. LORA_LR_MULT was hard-coded at 10: LoRA A is
# zero-initialised, so with a shared lr the adapters barely move inside a fixed epoch
# budget. It has never been swept, and it is a confound for any arm that adds a SECOND
# set of adapters (the both arm then has twice as many parameters at 10x lr).
LORA_ALPHA_ENC = [None]
LORA_ALPHA_HEAD = [None]
LORA_LR_MULT = [10.0]


# Which coordinates of a block survive its dimension budget. The budget itself is
# set by lambda (_dim_split); this only decides WHICH coordinates realise it.
#
#   dim       first k. Deterministic and parameter-free, but arbitrary -- nothing in
#             training orders a head's coordinates.
#   dim_rand  a fixed random subset. The control: if it matches 'dim', coordinate
#             choice is immaterial. (On MOSEI at fixed lambda it did: -0.10 balanced
#             accuracy, -0.05 AUC, both n.s. over 25 task x seed cells.)
#   dim_pca   top-k principal components of the block. Retains the most variance for
#             the budget, but the probe standardises each coordinate afterwards, which
#             whitens a skewed spectrum and amplifies noise directions -- on MOSEI it
#             was significantly WORSE (-0.46, p=0.034).
#
# LEAKAGE GUARD: a PCA basis is fitted on the probe's FITTING split and reused on
# test. _PCA_CAN_FIT is set from the split name by the caller, so a missing basis at
# test raises rather than silently refitting on test data.
_PCA_BASIS = {}
_PCA_CAN_FIT = [False]
_DIM_RAND_SEED = 0


def reset_readout_state():
    """Clear fitted PCA bases. Call between checkpoints."""
    _PCA_BASIS.clear()


def set_readout(name):
    """Pick the decomp_R readout when driving the library directly.

    main() does this from --readout, whose default is "dim". An importing caller
    never goes through main(), so without this it gets the module default "amp"
    and the simplex interior stays unresolved.
    """
    if name not in ("amp", "dim", "dim_rand", "dim_pca"):
        raise ValueError(f"unknown readout {name!r}")
    _READOUT[0] = name
    return name


def get_readout():
    return _READOUT[0]


def _select_dims(B, k, readout, key):
    if readout == "dim":
        return B[:, :k]
    if readout == "dim_rand":
        g = torch.Generator().manual_seed(_DIM_RAND_SEED + 1000 * key[1])
        idx = torch.randperm(B.shape[-1], generator=g)[:k].sort().values
        return B[:, idx]
    if readout == "dim_pca":
        if key not in _PCA_BASIS:
            if not _PCA_CAN_FIT[0]:
                raise RuntimeError(
                    f"dim_pca: no basis for block {key[1]} at lambda {key[0]} and "
                    f"fitting is disabled on this split -- refusing to fit on test.")
            X = B - B.mean(0, keepdim=True)
            C = (X.T @ X) / max(len(X) - 1, 1)
            _, V = torch.linalg.eigh(C.double())
            _PCA_BASIS[key] = (B.mean(0, keepdim=True), V.flip(-1).float())
        mu, V = _PCA_BASIS[key]
        return (B - mu) @ V[:, :k]
    raise ValueError(f"unknown readout {readout!r}")


def _dim_split(weights, D):
    """Largest-remainder allocation of D dimensions across the blocks."""
    raw = [w * D for w in weights]
    base = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - base[i],
                    reverse=True)[:D - sum(base)]:
        base[i] += 1
    return base


def probe_lambda_cond(enc_dir: str, data_dir: str, device: str,
                      latent_dim: int, batch_size: int, k_shot: int = 0,
                      proj_dim: int = 256, modality: str = None,
                      mode: str = "probe") -> dict:
    """
    Sweep λ ∈ {0.0, 0.1, …, 1.0} and probe all tasks.

    modality=None  → bimodal: share/unique1 from M1, unique2 from M2  (default)
    modality='m1'  → M1-only: ALL tasks probed using enc_m1 path
    modality='m2'  → M2-only: ALL tasks probed using enc_m2 path

    Returns {"lambdas": [...], "share": [...], "unique1": [...], "unique2": [...]}.
    """
    use_hyp      = os.path.exists(os.path.join(enc_dir, "hyp_0.pth"))
    use_lora_enc = os.path.exists(os.path.join(enc_dir, "lora_enc_0.pth"))
    use_lora     = os.path.exists(os.path.join(enc_dir, "lora_proj_m1.pth"))
    # output-mixing simplex variants: mix-before-normalise, and the 6h third head
    _mp = os.path.join(enc_dir, "train_meta.json")
    _tm = json.load(open(_mp)) if os.path.exists(_mp) else {}
    _premix_eval = bool(_tm.get("premix", False))
    _three_way   = bool(_tm.get("three_way", False))
    _decomp_r    = bool(_tm.get("decomp_R", False))
    _decomp_proj = bool(_tm.get("decomp_R_proj", False))
    # Sites the preference acts on. decomp_R_sites is written by newer runs; older
    # checkpoints only have decomp_R_proj, where proj => heads-only, else encoder-only.
    _sites = _tm.get("decomp_R_sites") or (["proj"] if _decomp_proj else ["enc"])
    _enc_cond, _head_cond = ("enc" in _sites), ("proj" in _sites)
    # The proj variant writes its (plain) encoder under the same lora_enc_*.pth name,
    # so the filename sniff above would build a LoRAAlexNetEncoder for an
    # AlexNetEncoder checkpoint and fail on the state dict. train_meta is
    # authoritative; the filename is not.
    if not _enc_cond:
        use_lora_enc = False

    # Load LoRA meta (branches + rank) if present
    _meta_path = os.path.join(enc_dir, "train_meta.json")
    _lora_meta = {}
    if os.path.exists(_meta_path):
        import json as _json
        with open(_meta_path) as _mf:
            _lora_meta = _json.load(_mf)
    _lora_branches = _lora_meta.get("branches", 2)
    _lora_rank     = _lora_meta.get("rank", 4)

    def _load_enc(name):
        if use_lora_enc:
            enc = LoRAAlexNetEncoder(latent_dim, branches=_lora_branches,
                                     rank=_lora_rank).to(device)
        elif use_hyp:
            enc = HyperNetEncoder(latent_dim).to(device)
        else:
            enc = AlexNetEncoder(latent_dim).to(device)
        enc.load_state_dict(torch.load(os.path.join(enc_dir, name),
                                       map_location=device, weights_only=True))
        return enc

    def _load_head(name):
        path = os.path.join(enc_dir, name)
        if not os.path.exists(path):
            return None
        if _head_cond:
            # LoRA heads (proj / both) are stored under the same proj_m*_{r,u}.pth names
            # as the plain variant. rank/alpha must match training exactly: `scale` is
            # a plain float attribute, not a parameter, so load_state_dict would leave
            # a mismatched alpha at 1.0 and the probe would score a different model.
            h = LoRADualProjectionHead(latent_dim, proj_dim,
                                       rank=_tm.get("lora_rank", 4),
                                       alpha=_tm.get("lora_alpha")).to(device)
        else:
            h = ProjectionHead(latent_dim, proj_dim).to(device)
        h.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return h

    def _load_lora_proj(name):
        ProjCls = LoRATriProjectionHead if _lora_branches == 3 else LoRADualProjectionHead
        h = ProjCls(latent_dim, proj_dim, rank=_lora_rank).to(device)
        h.load_state_dict(torch.load(os.path.join(enc_dir, name),
                                     map_location=device, weights_only=True))
        return h

    # The decomp_R proj variant trains a PLAIN AlexNetEncoder but saves it under the
    # shared decomp_R filename (lora_enc_*.pth). use_lora_enc is forced False above so
    # the right CLASS is built; the FILENAME still has to be the one on disk.
    enc_prefix = ("lora_enc" if (use_lora_enc or _decomp_r)
                  else ("hyp" if use_hyp else "enc"))
    enc_m1    = _load_enc(f"{enc_prefix}_0.pth")
    enc_m2    = _load_enc(f"{enc_prefix}_1.pth")
    if use_lora:
        lora_proj_m1 = _load_lora_proj("lora_proj_m1.pth")
        lora_proj_m2 = _load_lora_proj("lora_proj_m2.pth")
        proj_m1_r = proj_m1_u = proj_m2_r = proj_m2_u = None
        # bound in BOTH branches: _extract closes over these, so leaving them unset
        # on the LoRA path raises NameError from inside the closure
        proj_m1_u2 = proj_m2_u2 = None
    else:
        lora_proj_m1 = lora_proj_m2 = None
        proj_m1_r = _load_head("proj_m1_r.pth")
        proj_m1_u = _load_head("proj_m1_u.pth")
        proj_m1_u2 = _load_head("proj_m1_u2.pth") if _three_way else None
        proj_m2_u2 = _load_head("proj_m2_u2.pth") if _three_way else None
        proj_m2_r = _load_head("proj_m2_r.pth")
        proj_m2_u = _load_head("proj_m2_u.pth")

    for m in [enc_m1, enc_m2, proj_m1_r, proj_m1_u, proj_m2_r, proj_m2_u,
              lora_proj_m1, lora_proj_m2]:
        if m is not None:
            m.eval()

    def _loader(task, split):
        return DataLoader(
            PairProbeDataset(data_dir, split=split, task=task),
            batch_size=batch_size, shuffle=False, num_workers=4,
        )

    def _fwd_enc(enc, x, lam):
        if use_lora_enc:
            # LoRA-Simplex passes a (w_R, w_U) pair — apply the two coefficients
            # independently, mirroring how the encoder classifier was trained.
            if isinstance(lam, tuple):
                w_r, w_u = lam
                return enc.head_mix_from_feat(enc.feat(x), w_r, w_u)
            return enc(x, lam)       # LoRAAlexNetEncoder.forward(x, lam)
        elif use_hyp:
            return enc(x, lam)       # HyperNetEncoder.forward(x, lam)
        else:
            return enc(x)            # AlexNetEncoder.forward(x)

    def _fwd_proj(proj_lora, proj_r, proj_u, h, lam, proj_u2=None, w_u2=None):
        # lam is either a scalar (1-D λ sweep) or a (w_R, w_U) pair (LoRA-Simplex),
        # in which case the two coefficients are applied independently.
        if isinstance(lam, tuple):
            w_r, w_u = lam
            if use_lora:
                return proj_lora.forward_mix(h, w_r, w_u)
            if proj_u2 is not None and w_u2 is not None:
                # simplex6h: true 3-way blend over this modality's three heads
                if _premix_eval:
                    return F.normalize(w_r * proj_r.net(h) + w_u * proj_u.net(h)
                                       + w_u2 * proj_u2.net(h), dim=-1)
                return F.normalize(w_r * proj_r(h) + w_u * proj_u(h)
                                   + w_u2 * proj_u2(h), dim=-1)
            # 4-head: guard the (0,0) vertex exactly as training does
            if (w_r + w_u) <= 0:
                w_r, w_u = 0.5, 0.5
            if _premix_eval:
                return F.normalize(w_r * proj_r.net(h) + w_u * proj_u.net(h), dim=-1)
            return F.normalize(w_r * proj_r(h) + w_u * proj_u(h), dim=-1)
        if use_lora:
            return proj_lora(h, lam)   # LoRADualProjectionHead.forward(h, lam)
        else:
            return F.normalize(lam * proj_r(h) + (1.0 - lam) * proj_u(h), dim=-1)

    if _decomp_r:
        # simplex_enc_decomp_R readout
        # Component-wise CONCAT, not a weighted sum into one space. The sum is what
        # leaks: under simplex6h, z_t = norm(B_t(h_t)) still emitted a full text
        # representation at EVERY preference, so selecting the provably-empty vision
        # axis cost nothing. Here the blocks are separate and a zero coefficient
        # zeroes its block outright.
        #
        #   z(lam) = normalize([ s_R*R^ || s_U1*U1 || s_U2*U2 ]),  s = sqrt(lam)
        #   R^ = normalize((r1(h1) + r2(h2)) / 2)
        #
        # sqrt(lam), not lam: a linear probe responds to variance and energy goes as
        # amplitude^2, so amplitude=lam would give energy shares proportional to
        # lam^2 -- (0.25,0.50,0.25) would land at (1/6, 2/3, 1/6). With sqrt(lam) the
        # energy share of each block equals lam exactly. Vertices are unaffected.
        # R is renormalised after averaging: two unit vectors at angle theta average
        # to norm cos(theta/2) < 1, which would under-weight the shared block.
        import math as _math
        # _load_head already builds the right class from train_meta (LoRA for the
        # proj variant, plain otherwise).
        _pr1 = _load_head("proj_m1_r.pth"); _pr2 = _load_head("proj_m2_r.pth")
        _pu1 = _load_head("proj_m1_u.pth"); _pu2 = _load_head("proj_m2_u.pth")
        for _m in (_pr1, _pr2, _pu1, _pu2):
            if _m is None:
                raise RuntimeError("decomp_R run is missing one of "
                                   "proj_m{1,2}_{r,u}.pth in " + enc_dir)

        # Encoder features do NOT depend on the preference: for the proj variant the
        # encoder is lambda-blind outright, and for the enc variant the conv trunk is
        # lambda-blind (only head_mix_from_feat sees lambda). Recomputing them for all
        # 15 preferences meant 15x3x2 = 90 passes over the images; caching per
        # (task, split) cuts that to 6.
        _FCACHE = {}

        @torch.no_grad()
        def _feats(task, split):
            key = (task, split)
            if key not in _FCACHE:
                H1, H2, L = [], [], []
                for imgs, bl in _loader(task, split):
                    m1 = imgs[0].to(device); m2 = imgs[1].to(device)
                    if _enc_cond:
                        H1.append(enc_m1.feat(m1)); H2.append(enc_m2.feat(m2))
                    else:
                        H1.append(enc_m1(m1)); H2.append(enc_m2(m2))
                    L.append(bl.cpu().numpy() if isinstance(bl, torch.Tensor)
                             else np.asarray(bl))
                _FCACHE[key] = (torch.cat(H1), torch.cat(H2), np.concatenate(L))
            return _FCACHE[key]

        @torch.no_grad()
        def _extract(task, split, lam):
            w_r, w_u1, w_u2 = lam
            s_r, s_u1, s_u2 = _math.sqrt(w_r), _math.sqrt(w_u1), _math.sqrt(w_u2)
            H1, H2, y = _feats(task, split)
            feats = []
            for i in range(0, H1.shape[0], batch_size):
                f1 = H1[i:i + batch_size]; f2 = H2[i:i + batch_size]
                # each modality sees only its OWN unique coefficient at BOTH sites --
                # the same routing as training, or the probe scores a model never fit.
                h1 = enc_m1.head_mix_from_feat(f1, w_r, w_u1) if _enc_cond else f1
                h2 = enc_m2.head_mix_from_feat(f2, w_r, w_u2) if _enc_cond else f2
                if _head_cond:
                    _R1 = _pr1.forward_mix(h1, w_r, w_u1)
                    _R2 = _pr2.forward_mix(h2, w_r, w_u2)
                    _U1 = _pu1.forward_mix(h1, w_r, w_u1)
                    _U2 = _pu2.forward_mix(h2, w_r, w_u2)
                else:
                    _R1, _R2, _U1, _U2 = _pr1(h1), _pr2(h2), _pu1(h1), _pu2(h2)
                R = F.normalize((_R1 + _R2) / 2.0, dim=-1)
                # A PCA basis cannot be fitted on one batch, so the blocks are
                # collected and the readout is applied once on the whole split. The
                # other readouts are row-wise, so this is identical for them.
                feats.append((R.cpu(), _U1.cpu(), _U2.cpu()))
            _PCA_CAN_FIT[0] = (split != "test")
            R = torch.cat([b[0] for b in feats])
            U1 = torch.cat([b[1] for b in feats])
            U2 = torch.cat([b[2] for b in feats])
            if _READOUT[0].startswith("dim"):
                _n = _dim_split((w_r, w_u1, w_u2), R.shape[-1])
                _parts = [_select_dims(b, k, _READOUT[0], key=(lam, bi))
                          for bi, (b, k) in enumerate(zip((R, U1, U2), _n)) if k > 0]
                z = F.normalize(torch.cat(_parts, dim=-1), dim=-1)
            else:
                z = F.normalize(torch.cat([s_r * R, s_u1 * U1, s_u2 * U2], dim=-1), dim=-1)
            return z.numpy(), y
    else:
      @torch.no_grad()
      def _extract(task, split, lam):
        # LoRA-Simplex: a 3-tuple (λ_R, λ_U1, λ_U2) routes (λ_R, λ_U1) to head m1
        # and (λ_R, λ_U2) to head m2 — exactly how the model was trained.
        # Scalars (1-D λ sweep) pass through unchanged.
        if isinstance(lam, tuple) and len(lam) == 3:
            lam_m1, lam_m2 = (lam[0], lam[1]), (lam[0], lam[2])
        else:
            lam_m1 = lam_m2 = lam
        feats, lbls = [], []
        for imgs, batch_lbls in _loader(task, split):
            m1 = imgs[0].to(device)
            m2 = imgs[1].to(device)
            if modality == 'm1':
                h = _fwd_enc(enc_m1, m1, lam_m1)
                z = _fwd_proj(lora_proj_m1, proj_m1_r, proj_m1_u, h, lam_m1,
                              proj_m1_u2, (lam[2] if _three_way and isinstance(lam, tuple) and len(lam)==3 else None))
            elif modality == 'm2':
                h = _fwd_enc(enc_m2, m2, lam_m2)
                z = _fwd_proj(lora_proj_m2, proj_m2_r, proj_m2_u, h, lam_m2,
                              proj_m2_u2, (lam[1] if _three_way and isinstance(lam, tuple) and len(lam)==3 else None))
            else:
                # bimodal: task-specific routing
                if task == "unique2":
                    h = _fwd_enc(enc_m2, m2, lam_m2)
                    z = _fwd_proj(lora_proj_m2, proj_m2_r, proj_m2_u, h, lam_m2,
                              proj_m2_u2, (lam[1] if _three_way and isinstance(lam, tuple) and len(lam)==3 else None))
                else:
                    h = _fwd_enc(enc_m1, m1, lam_m1)
                    z = _fwd_proj(lora_proj_m1, proj_m1_r, proj_m1_u, h, lam_m1,
                              proj_m1_u2, (lam[2] if _three_way and isinstance(lam, tuple) and len(lam)==3 else None))
            feats.append(z.cpu().numpy())
            lbls.append(batch_lbls.cpu().numpy() if isinstance(batch_lbls, torch.Tensor)
                        else np.asarray(batch_lbls))
        return np.concatenate(feats), np.concatenate(lbls)

    # LoRA-Simplex checkpoints record their grid side in train_meta.json — sweep the
    # 2-simplex (λ_R, λ_U1, λ_U2) instead of the 1-D λ line, so the evaluated points
    # are exactly the preferences the model was trained on.
    _meta_side = None
    _meta_f = os.path.join(enc_dir, "train_meta.json")
    if os.path.exists(_meta_f):
        try:
            with open(_meta_f) as _mf:
                _meta_side = json.load(_mf).get("simplex_side")
        except Exception:
            _meta_side = None

    # -- CKA mode (H7) --------------------------------------------------------
    # Measures how far the representation actually moves per grid step, so the
    # trifeature number can be read against MOSEI's adjacent-lambda band of
    # 0.92-0.97. To be comparable it MUST reproduce multibench run_cka_matrix
    # exactly (multibench/probe.py:1466):
    #   * walk the R -> U EDGE, (t, (1-t)/2, (1-t)/2) over CKA_MATRIX_GRID -- NOT
    #     the raster enumeration of simplex_grid, whose consecutive entries jump
    #     across the triangle and would make "adjacent" meaningless;
    #   * compute CKA per MODALITY and average, not on the two concatenated.
    # PairProbeDataset builds its pair list independently of the task (same rng
    # seed, grouped by R), so task="share" and task="unique2" return the SAME
    # rows -- the first routed through M1, the second through M2.
    if mode == "cka":
        CKA_MATRIX_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
        if _meta_side:
            pts = [(t, (1.0 - t) / 2.0, (1.0 - t) / 2.0) for t in CKA_MATRIX_GRID]
        else:
            pts = list(CKA_MATRIX_GRID)
        # For every mode whose ENCODER is preference-blind (everything except
        # lora_enc / hypernet), h does not depend on the preference -- only the
        # heads do. Decode the images ONCE and re-apply the heads per grid point.
        # Without this the job re-decodes ~53k PNGs from lustre five times over
        # and blows a 1 h walltime; multibench does the same thing via
        # enc_is_blind (probe.py:1486).
        _enc_blind = not (use_lora_enc or use_hyp)
        _hcache = {}

        @torch.no_grad()
        def _feats(task):
            """Encoder outputs for one task's pair list, computed once."""
            if task in _hcache:
                return _hcache[task]
            hs = []
            for imgs, _ in _loader(task, _EVAL_SPLIT[0]):
                x = imgs[1 if task == "unique2" else 0].to(device)
                hs.append((enc_m2 if task == "unique2" else enc_m1)(x))
            _hcache[task] = torch.cat(hs)
            return _hcache[task]

        @torch.no_grad()
        def _z_at(task, pt):
            """z for one task at one preference, reusing the cached features.

            Always read out through the plain 'dim' slice. CKA compares z(p_i) against
            z(p_j), so a per-preference fitted PCA basis would make the comparison
            partly about the BASES rather than the representations; and this path
            extracts the TEST split with no preceding train extraction, so a fitted
            readout could only get a basis by fitting on test -- which the leakage
            guard refuses. The slice needs no fitting. (Same fix as probe.py's CKA
            routines, applied here before it bites.)"""
            _ro_saved = _READOUT[0]
            if _ro_saved.startswith("dim"):
                _READOUT[0] = "dim"
            try:
                if not _enc_blind:
                    return _extract(task, _EVAL_SPLIT[0], pt)[0]
                h = _feats(task)
                if isinstance(pt, tuple) and len(pt) == 3:
                    lam = (pt[0], pt[2]) if task == "unique2" else (pt[0], pt[1])
                    u2w = (pt[1] if task == "unique2" else pt[2]) if _three_way else None
                else:
                    lam, u2w = pt, None
                if task == "unique2":
                    z = _fwd_proj(lora_proj_m2, proj_m2_r, proj_m2_u, h, lam,
                                  proj_m2_u2, u2w)
                else:
                    z = _fwd_proj(lora_proj_m1, proj_m1_r, proj_m1_u, h, lam,
                                  proj_m1_u2, u2w)
                return z.cpu().numpy()
            finally:
                _READOUT[0] = _ro_saved

        Z1, Z2 = [], []
        for pt in pts:
            z1 = _z_at("share",   pt)   # M1 path
            z2 = _z_at("unique2", pt)   # M2 path
            Z1.append(z1); Z2.append(z2)
            t0 = pt[0] if isinstance(pt, tuple) else pt
            print(f"  [cka] t={t0:.2f}  m1={z1.shape}  m2={z2.shape}"
                  f"{'  (cached encoder)' if _enc_blind else ''}", flush=True)
        n = len(pts)
        mat1 = [[linear_cka(Z1[i], Z1[j]) for j in range(n)] for i in range(n)]
        mat2 = [[linear_cka(Z2[i], Z2[j]) for j in range(n)] for i in range(n)]
        mat  = [[(mat1[i][j] + mat2[i][j]) / 2 for j in range(n)] for i in range(n)]
        adj  = [mat[i][i + 1] for i in range(n - 1)]
        off  = [mat[i][j] for i in range(n) for j in range(n) if i != j]
        return {
            "mode": "cka",
            "grid": [list(pt) if isinstance(pt, tuple) else pt for pt in pts],
            "adjacent_cka": [round(x, 4) for x in adj],
            "adjacent_cka_mean": round(float(np.mean(adj)), 4),
            "endpoint_cka": round(mat[0][n - 1], 4),
            "adjacent_cka_m1": [round(mat1[i][i + 1], 4) for i in range(n - 1)],
            "adjacent_cka_m2": [round(mat2[i][i + 1], 4) for i in range(n - 1)],
            "endpoint_cka_m1": round(mat1[0][n - 1], 4),
            "endpoint_cka_m2": round(mat2[0][n - 1], 4),
            "matrix": [[round(x, 4) for x in row] for row in mat],
            "all_pairs_mean": round(float(np.mean(off)), 4),
            "all_pairs_min": round(float(np.min(off)), 4),
            "n_eval": int(Z1[0].shape[0]),
            "embed_dim": int(Z1[0].shape[1]),
        }

    if _meta_side:
        prefs = simplex_grid(int(_meta_side))
        curve = {task: [] for task in TASKS}
        for p in prefs:
            print(f"  [simplex] probing (R,U1,U2)=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})")
            for task in TASKS:
                tr_f, tr_l = _extract(task, "train", p)
                te_f, te_l = _extract(task, _EVAL_SPLIT[0], p)
                acc = _probe(tr_f, tr_l, te_f, te_l, k_shot=k_shot)
                curve[task].append(round(float(acc), 4))
                print(f"    {task}: {acc:.4f}")
        return {"preferences": [list(p) for p in prefs],
                "lambdas": [round(p[0], 4) for p in prefs],  # λ_R, for plotting compat
                "simplex_side": int(_meta_side), **curve}

    lambdas = [round(v, 1) for v in np.arange(0.0, 1.01, 0.1)]
    curve = {task: [] for task in TASKS}

    for lam in lambdas:
        print(f"  [lambda_cond] probing lambda={lam:.1f}")
        for task in TASKS:
            tr_f, tr_l = _extract(task, "train", lam)
            te_f, te_l = _extract(task, _EVAL_SPLIT[0], lam)
            acc = _probe(tr_f, tr_l, te_f, te_l, k_shot=k_shot)
            curve[task].append(round(float(acc), 4))
            print(f"    {task}: {acc:.4f}")

    return {"lambdas": lambdas, **curve}


def probe_one_modality(enc_dir: str, data_dir: str, device: str,
                       latent_dim: int, batch_size: int, k_shot: int = 0,
                       proj_dim: int = 256) -> dict:
    """
    One-modality inference: sweep λ using only M1 (or only M2) encoder+heads.

    For each modality m ∈ {m1, m2} and each λ:
      z = F.normalize(λ·proj_r(enc(x_m)) + (1-λ)·proj_u(enc(x_m)))
    Probes all three tasks (share, unique1, unique2) from that single modality.

    Returns {"m1": {"lambdas", "share", "unique1", "unique2"},
             "m2": {"lambdas", "share", "unique1", "unique2"}}.

    Note: unique2 probed from M1 should be near chance (sanity check for
    disentanglement), and unique1 probed from M2 likewise.
    """
    use_hyp = os.path.exists(os.path.join(enc_dir, "hyp_0.pth"))

    def _load_enc(name):
        enc = HyperNetEncoder(latent_dim) if use_hyp else AlexNetEncoder(latent_dim)
        enc = enc.to(device)
        enc.load_state_dict(torch.load(os.path.join(enc_dir, name),
                                       map_location=device, weights_only=True))
        return enc

    def _load_head(name):
        h = ProjectionHead(latent_dim, proj_dim).to(device)
        h.load_state_dict(torch.load(os.path.join(enc_dir, name),
                                     map_location=device, weights_only=True))
        return h

    enc_prefix = "hyp" if use_hyp else "enc"
    enc_m1    = _load_enc(f"{enc_prefix}_0.pth")
    enc_m2    = _load_enc(f"{enc_prefix}_1.pth")
    proj_m1_r = _load_head("proj_m1_r.pth")
    proj_m1_u = _load_head("proj_m1_u.pth")
    proj_m2_r = _load_head("proj_m2_r.pth")
    proj_m2_u = _load_head("proj_m2_u.pth")

    for m in [enc_m1, enc_m2, proj_m1_r, proj_m1_u, proj_m2_r, proj_m2_u]:
        m.eval()

    def _loader(task, split):
        return DataLoader(
            PairProbeDataset(data_dir, split=split, task=task),
            batch_size=batch_size, shuffle=False, num_workers=4,
        )

    def _fwd(enc, x, lam):
        return enc(x, lam) if use_hyp else enc(x)

    @torch.no_grad()
    def _extract_m(mod_idx, enc, proj_r, proj_u, task, split, lam):
        feats, lbls = [], []
        for imgs, batch_lbls in _loader(task, split):
            x = imgs[mod_idx].to(device)
            h = _fwd(enc, x, lam)
            z = F.normalize(lam * proj_r(h) + (1.0 - lam) * proj_u(h), dim=-1)
            feats.append(z.cpu().numpy())
            lbls.append(batch_lbls.cpu().numpy() if isinstance(batch_lbls, torch.Tensor)
                        else np.asarray(batch_lbls))
        return np.concatenate(feats), np.concatenate(lbls)

    lambdas = [round(v, 1) for v in np.arange(0.0, 1.01, 0.1)]
    results = {}

    for mod_name, mod_idx, enc, pr, pu in [
        ("m1", 0, enc_m1, proj_m1_r, proj_m1_u),
        ("m2", 1, enc_m2, proj_m2_r, proj_m2_u),
    ]:
        curve = {task: [] for task in TASKS}
        print(f"\n  [one_modality / {mod_name}]")
        for lam in lambdas:
            print(f"    lambda={lam:.1f}", end="")
            for task in TASKS:
                tr_f, tr_l = _extract_m(mod_idx, enc, pr, pu, task, "train", lam)
                te_f, te_l = _extract_m(mod_idx, enc, pr, pu, task, "test",  lam)
                acc = _probe(tr_f, tr_l, te_f, te_l, k_shot=k_shot)
                curve[task].append(round(float(acc), 4))
                print(f"  {task}={acc:.3f}", end="")
            print()
        results[mod_name] = {"lambdas": lambdas, **curve}

    return results


# Pareto metrics

def _nondominated_mask(pts: np.ndarray) -> np.ndarray:
    """Boolean mask: True = point is not dominated by any other point in pts."""
    m  = len(pts)
    nd = np.ones(m, dtype=bool)
    for i in range(m):
        if not nd[i]:
            continue
        diff = pts[nd] - pts[i]                          # (k, 3)
        dom  = np.all(diff >= 0, axis=1) & np.any(diff > 0, axis=1)
        self_eq = np.all(diff == 0, axis=1)
        if np.any(dom & ~self_eq):
            nd[i] = False
    return nd


def _hv_mc(pts: np.ndarray, samples: np.ndarray) -> float:
    """MC hypervolume: fraction of [0,1]^3 dominated by pts (maximisation, ref=0)."""
    if len(pts) == 0:
        return 0.0
    covered = np.any(
        np.all(samples[:, None, :] <= pts[None, :, :], axis=2),
        axis=1,
    )
    return float(covered.mean())


def _hv_2d_exact(pts_2d: np.ndarray) -> float:
    """
    Exact 2D hypervolume with ref=(0,0) for maximisation.

    pts_2d : (N, 2) array of (share, U_avg) points.
    Returns the area of the region dominated by the nondominated front.
    Algorithm: keep ND points, sort by x ascending (→ y descending),
    then sum staircase rectangles: HV = Σ (x_i - x_{i-1}) * y_i.
    """
    if len(pts_2d) == 0:
        return 0.0
    nd   = pts_2d[_nondominated_mask(pts_2d)]
    nd   = nd[nd[:, 0].argsort()]          # sort by share ascending
    hv   = 0.0
    prev = 0.0
    for x, y in nd:
        hv  += (x - prev) * y
        prev = x
    return float(hv)


# Fixed methods used to define HV_baselines — the reference front.
# factorcl_heads is the oracle (task-specific heads) — shown but not in comparison.
_BASELINE_METHODS  = {"simclr_both", "clip", "gmc", "comm", "factorcl"}
_DISPLAY_ONLY      = {"factorcl_heads"}   # shown but not compared


def compute_all_pareto_metrics(all_results: dict, out_dir: str) -> dict:
    """
    Compute Pareto metrics for ALL methods in all_results.

    Space     : 2D (share, U_avg), U_avg = (unique1 + unique2) / 2
    HV        : exact 2D, ref = (0, 0)
    Baselines : simclr_both, clip, gmc, comm, factorcl_concat
                (factorcl and random excluded from the reference front)

    Fixed baselines → hv_single (share × U_avg), is_dominated
    Fixed anchors   → same + PCG vs baseline front  (factorcl, random, …)
    Lambda methods  → HV_lambda, HV_all, PCG, uniformity_score,
                      nondominated_ratio, best_U1, best_U2, best_U
                      + per-lambda CSV in out_dir/{method}/pareto_metrics.csv
    """
    import csv as _csv

    # collect 2D points
    fixed_pts  = {}   # method → np.array([share, U_avg])
    lam_arrays = {}   # method → np.array (L, 2)
    lam_raw    = {}   # method → full curve dict

    for method, data in all_results.items():
        if "probes" in data:
            p = data["probes"]
            u_avg = (p.get("unique1", 0) + p.get("unique2", 0)) / 2
            fixed_pts[method] = np.array([p.get("share", 0), u_avg])
        elif "curve" in data:
            c = data["curve"]
            u_avgs = [(c["unique1"][i] + c["unique2"][i]) / 2
                      for i in range(len(c["lambdas"]))]
            lam_arrays[method] = np.array(
                [[c["share"][i], u_avgs[i]] for i in range(len(c["lambdas"]))]
            )
            lam_raw[method] = c

    # baseline front
    base_pts_list = [pt for m, pt in fixed_pts.items() if m in _BASELINE_METHODS]
    if base_pts_list:
        base_arr     = np.stack(base_pts_list)
        HV_baselines = _hv_2d_exact(base_arr)
    else:
        base_arr, HV_baselines = np.zeros((0, 2)), 0.0

    # all points combined (dominance check)
    all_list = list(fixed_pts.values())
    for arr in lam_arrays.values():
        all_list.extend(arr)
    all_combined = np.stack(all_list) if all_list else np.zeros((0, 2))

    metrics = {}

    # fixed methods (baselines + anchors)
    for method, pt in fixed_pts.items():
        others = all_combined[~np.all(all_combined == pt, axis=1)]
        is_dom = bool(
            len(others) > 0 and
            np.any(np.all(others >= pt, axis=1) & np.any(others > pt, axis=1))
        )
        # PCG: how much this single point extends the baseline front
        combined_with = np.vstack([base_arr, pt[None]]) if len(base_arr) else pt[None]
        PCG_single    = _hv_2d_exact(combined_with) - HV_baselines

        role = ("baseline"     if method in _BASELINE_METHODS else
                "display_only" if method in _DISPLAY_ONLY    else
                "baseline")   # fallback

        metrics[method] = {
            "type":          "fixed",
            "role":          role,
            "hv_single":     float(pt[0] * pt[1]),
            "HV_baselines":  HV_baselines,
            "is_dominated":  is_dom,
        }
        if role == "baseline":
            metrics[method]["PCG"] = PCG_single

    # lambda methods
    for method, lam_pts_2d in lam_arrays.items():
        c       = lam_raw[method]
        lambdas = c["lambdas"]
        shares  = c["share"]
        u1s     = c["unique1"]
        u2s     = c["unique2"]
        n_lam   = len(lambdas)
        U_avgs  = [(u1s[i] + u2s[i]) / 2 for i in range(n_lam)]

        nd_mask = _nondominated_mask(lam_pts_2d)
        nd_pts  = lam_pts_2d[nd_mask]
        nd_lams = [lambdas[i] for i in range(n_lam) if nd_mask[i]]
        nondominated_ratio = float(nd_mask.sum() / n_lam)

        HV_lambda = _hv_2d_exact(nd_pts)
        combined  = np.vstack([nd_pts, base_arr]) if len(base_arr) else nd_pts
        HV_all    = _hv_2d_exact(combined)
        PCG       = HV_all - HV_baselines

        eps    = 1e-9
        rhos   = [shares[i] / (shares[i] + U_avgs[i] + eps) for i in range(n_lam)]
        errors = [abs(rhos[i] - lambdas[i]) for i in range(n_lam)]
        uniformity_score = 1.0 - float(np.mean(errors))

        best_U1    = float(np.max(u1s))
        best_U2    = float(np.max(u2s))
        best_U     = (best_U1 + best_U2) / 2

        enc_dir  = os.path.join(out_dir, method)
        os.makedirs(enc_dir, exist_ok=True)
        csv_path = os.path.join(enc_dir, "pareto_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["lambda", "share", "unique1", "unique2", "U_avg",
                        "rho", "uniformity_error", "dominated"])
            for i in range(n_lam):
                w.writerow([f"{lambdas[i]:.2f}", f"{shares[i]:.4f}",
                             f"{u1s[i]:.4f}",    f"{u2s[i]:.4f}",
                             f"{U_avgs[i]:.4f}", f"{rhos[i]:.4f}",
                             f"{errors[i]:.4f}", int(~nd_mask[i])])

        metrics[method] = {
            "type":                 "lambda",
            "HV_lambda":            HV_lambda,
            "HV_baselines":         HV_baselines,
            "HV_all":               HV_all,
            "PCG":                  PCG,
            "uniformity_score":     uniformity_score,
            "nondominated_ratio":   nondominated_ratio,
            "nondominated_lambdas": nd_lams,
            "best_U1":              best_U1,
            "best_U2":              best_U2,
            "best_U":               best_U,
        }

    # print
    W = 91
    print(f"\n{'─'*W}")
    print(f"  Pareto Metrics  (2D exact HV, space=(share,U_avg), ref=(0,0))")
    print(f"  Baselines: {', '.join(sorted(_BASELINE_METHODS))}")
    print(f"{'─'*W}")

    print(f"\n  Fixed baselines")
    print(f"  {'method':<26} {'hv_single':>9}  {'PCG':>7}  {'dom':>3}")
    print(f"  {'-'*50}")
    for m, mt in metrics.items():
        if mt["type"] == "fixed" and mt["role"] == "baseline":
            dom = "Y" if mt["is_dominated"] else "N"
            print(f"  {m:<26} {mt['hv_single']:>9.4f}  {mt['PCG']:>+7.4f}  {dom:>3}")
    print(f"  {'HV_baselines (ND)':<26} {HV_baselines:>9.4f}")

    display_only = {m: mt for m, mt in metrics.items()
                    if mt["type"] == "fixed" and mt["role"] == "display_only"}
    if display_only:
        print(f"\n  Display-only  (shown for reference, not used in comparison)")
        print(f"  {'method':<26} {'hv_single':>9}  {'dom':>3}")
        print(f"  {'-'*42}")
        for m, mt in display_only.items():
            dom = "Y" if mt["is_dominated"] else "N"
            print(f"  {m:<26} {mt['hv_single']:>9.4f}  {dom:>3}")

    print(f"\n  Lambda methods  (anchors — compared against baseline front)")
    print(f"  {'method':<26} {'HV_λ':>6}  {'HV_all':>6}  {'PCG':>7}  "
          f"{'Unif':>5}  {'ND':>4}  {'bestU1':>7}  {'bestU2':>7}  {'bestU':>6}")
    print(f"  {'-'*W}")
    for m, mt in metrics.items():
        if mt["type"] == "lambda":
            print(f"  {m:<26} {mt['HV_lambda']:>6.4f}  {mt['HV_all']:>6.4f}  "
                  f"{mt['PCG']:>+7.4f}  {mt['uniformity_score']:>5.3f}  "
                  f"{mt['nondominated_ratio']:>4.2f}  "
                  f"{mt['best_U1']:>7.4f}  {mt['best_U2']:>7.4f}  {mt['best_U']:>6.4f}")
    print(f"{'─'*W}")

    return metrics


# _DEAD_CODE_STUB — kept for reference only
def compute_pareto_metrics(curve: dict, fixed_results: dict,
                           out_dir: str, method: str,
                           n_mc: int = 200_000, seed: int = 42) -> dict:
    """
    Compute Pareto quality metrics for one lambda sweep curve.

    Parameters
    ----------
    curve        : dict with keys "lambdas", "share", "unique1", "unique2"
    fixed_results: all_results entries that have "probes" (the baselines)
    out_dir      : base results directory; CSV saved to out_dir/method/
    method       : method name (used for labelling and CSV path)
    n_mc         : Monte Carlo samples for hypervolume approximation
    seed         : RNG seed for reproducibility

    Returns
    -------
    dict with scalar metrics: HV_lambda, HV_baselines, HV_all, PCG,
                              uniformity_score, nondominated_ratio
    """
    import csv as _csv

    lambdas = curve["lambdas"]
    shares  = curve["share"]
    u1s     = curve["unique1"]
    u2s     = curve["unique2"]
    n_lam   = len(lambdas)

    # (L, 3) array — one row per lambda point
    lam_pts = np.array([[shares[i], u1s[i], u2s[i]] for i in range(n_lam)])

    # helpers
    def _nondominated_mask(pts):
        """Boolean mask: True = point is NOT dominated by any other point."""
        m = len(pts)
        nd = np.ones(m, dtype=bool)
        for i in range(m):
            if not nd[i]:
                continue
            others = pts[nd]
            # does any other ND point dominate pts[i]?
            # dominated = another point >= on ALL dims AND > on at LEAST one
            diff = others - pts[i]          # (k, 3)
            dom = np.all(diff >= 0, axis=1) & np.any(diff > 0, axis=1)
            # exclude self-comparison (diff == 0 on all dims)
            self_mask = np.all(diff == 0, axis=1)
            if np.any(dom & ~self_mask):
                nd[i] = False
        return nd

    def _hv_mc(pts, samples):
        """MC hypervolume: fraction of [0,1]^3 dominated by pts (maximisation, ref=0)."""
        if len(pts) == 0:
            return 0.0
        # sample s is covered if ∃ p ∈ pts such that s ≤ p on all dims
        covered = np.any(
            np.all(samples[:, None, :] <= pts[None, :, :], axis=2),
            axis=1,
        )
        return float(covered.mean())

    # shared MC samples (same seed → comparable HV values)
    rng     = np.random.default_rng(seed)
    samples = rng.uniform(0.0, 1.0, (n_mc, 3))

    # 1. Nondominated lambda points
    nd_mask = _nondominated_mask(lam_pts)
    nd_pts  = lam_pts[nd_mask]
    nondominated_ratio = nd_mask.sum() / n_lam
    nd_lambdas = [lambdas[i] for i in range(n_lam) if nd_mask[i]]

    # 2. Hypervolume
    HV_lambda = _hv_mc(nd_pts, samples)

    # Collect fixed-method points (baselines)
    base_list = []
    for data in fixed_results.values():
        if "probes" in data:
            p = data["probes"]
            base_list.append([p.get("share", 0), p.get("unique1", 0), p.get("unique2", 0)])
    base_pts = np.array(base_list) if base_list else np.zeros((0, 3))

    nd_base = base_pts[_nondominated_mask(base_pts)] if len(base_pts) else base_pts
    HV_baselines = _hv_mc(nd_base, samples)

    # Combined ND front (lambda ND + baselines)
    all_pts  = np.vstack([nd_pts, base_pts]) if len(base_pts) else nd_pts
    nd_all   = all_pts[_nondominated_mask(all_pts)]
    HV_all   = _hv_mc(nd_all, samples)

    PCG = HV_all - HV_baselines

    # 3. Uniformity / preference alignment
    eps    = 1e-9
    U_avgs = [(u1s[i] + u2s[i]) / 2          for i in range(n_lam)]
    rhos   = [shares[i] / (shares[i] + U_avgs[i] + eps) for i in range(n_lam)]
    errors = [abs(rhos[i] - lambdas[i])       for i in range(n_lam)]
    uniformity_score = 1.0 - float(np.mean(errors))

    # 4. Save CSV
    enc_dir  = os.path.join(out_dir, method)
    os.makedirs(enc_dir, exist_ok=True)
    csv_path = os.path.join(enc_dir, "pareto_metrics.csv")

    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["lambda", "share", "unique1", "unique2",
                    "U_avg", "rho", "uniformity_error", "dominated"])
        for i in range(n_lam):
            w.writerow([
                f"{lambdas[i]:.2f}",
                f"{shares[i]:.4f}",
                f"{u1s[i]:.4f}",
                f"{u2s[i]:.4f}",
                f"{U_avgs[i]:.4f}",
                f"{rhos[i]:.4f}",
                f"{errors[i]:.4f}",
                int(~nd_mask[i]),
            ])

    # 5. Print summary
    sep = "-" * 44
    print(f"\n  ┌─ Pareto metrics — {method} {'─' * max(0, 22 - len(method))}┐")
    print(f"  │  HV_lambda          {HV_lambda:.4f}")
    print(f"  │  HV_baselines       {HV_baselines:.4f}")
    print(f"  │  HV_all             {HV_all:.4f}")
    print(f"  │  PCG                {PCG:+.4f}")
    print(f"  │  Uniformity_score   {uniformity_score:.4f}")
    print(f"  │  Nondominated_ratio {nondominated_ratio:.2f}  "
          f"({nd_mask.sum()}/{n_lam} pts, λ={nd_lambdas})")
    print(f"  └{'─' * 42}┘")
    print(f"  CSV → {csv_path}")

    return {
        "HV_lambda":          HV_lambda,
        "HV_baselines":       HV_baselines,
        "HV_all":             HV_all,
        "PCG":                PCG,
        "uniformity_score":   uniformity_score,
        "nondominated_ratio": float(nondominated_ratio),
    }


# Run logger

class RunLogger:
    """
    Creates  <out_dir>/runs/run_<timestamp>/
    and writes a human-readable run_log.txt there.
    Also copies pareto_config.yaml and the Pareto figure into the run folder.
    """

    def __init__(self, out_dir: str, args):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(out_dir, "runs", f"run_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Copy config
        shutil.copy2(str(_CFG_PATH), os.path.join(self.run_dir, "pareto_config.yaml"))

        self.log_path = os.path.join(self.run_dir, "run_log.txt")
        self._rows: list[tuple] = []
        self._lam_dist = args.lam_dist

        cli_str = " \\\n  ".join(
            f"--{k.replace('_', '-')} {v}"
            for k, v in vars(args).items()
            if v is not None and v is not False
        ) or "(all defaults)"

        with open(self.log_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write(f"Run  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"out_dir      : {out_dir}\n")
            f.write(f"methods      : {' '.join(args.methods)}\n\n")
            f.write("--- Config (pareto_config.yaml) ---\n")
            f.write(f"pareto_solver    : {args.pareto_solver}\n")
            f.write(f"lam_dist         : {args.lam_dist if args.lam_dist is not None else 'auto (warmup for factorcl_warmup, uniform otherwise)'}\n")
            f.write(f"dirichlet_alpha  : {args.dirichlet_alpha}\n")
            f.write(f"lambda_grid      : {LAMBDA_GRID}\n")
            f.write(f"epochs           : {args.epochs}\n")
            f.write(f"lr               : {args.lr}\n")
            f.write(f"batch_size       : {args.batch_size}\n")
            f.write(f"latent_dim       : {args.latent_dim}\n")
            f.write(f"proj_dim         : {args.proj_dim}\n")
            f.write(f"probe_shots      : {args.probe_shots}\n")
            f.write(f"lam_club         : {LAM_CLUB}\n")
            f.write(f"alpha_smooth     : {ALPHA_SMOOTH}\n")
            f.write(f"beta_div         : {BETA_DIV}\n\n")
            f.write(f"--- CLI overrides ---\n{cli_str}\n\n")
            f.write("--- Results ---\n")
            f.write(f"{'method':<26} {'share':>7} {'unique1':>8} {'unique2':>8}  notes\n")
            f.write("-" * 66 + "\n")

    def _lam_note(self, method: str) -> str:
        """Distribution tag for lambda methods trained in this run."""
        if self._lam_dist is not None:
            dist = self._lam_dist
        else:
            dist = "warmup" if method == "factorcl_warmup" else DEFAULT_LAM_DIST
        return f"[{dist}]"

    def log_method(self, method: str, result: dict):
        if "curve" in result:
            curve = result["curve"]
            s  = max(curve.get("share",   [0]))
            u1 = max(curve.get("unique1", [0]))
            u2 = max(curve.get("unique2", [0]))
            note = f"oracle-best {self._lam_note(method)}"
        else:
            p  = result.get("probes", {})
            s  = p.get("share",   0)
            u1 = p.get("unique1", 0)
            u2 = p.get("unique2", 0)
            note = "fixed"
        self._rows.append((method, s, u1, u2, note))
        with open(self.log_path, "a") as f:
            f.write(f"{method:<26} {s:>7.4f} {u1:>8.4f} {u2:>8.4f}  {note}\n")
            if "curve" in result:
                curve = result["curve"]
                lams  = curve.get("lambdas", [])
                sh    = curve.get("share",   [0] * len(lams))
                u1s   = curve.get("unique1", [0] * len(lams))
                u2s   = curve.get("unique2", [0] * len(lams))
                f.write(f"  {'λ':>5}  {'share':>7}  {'unique1':>8}  {'unique2':>8}\n")
                f.write(f"  {'-'*37}\n")
                for i, lam in enumerate(lams):
                    f.write(f"  {lam:>5.2f}  {sh[i]:>7.4f}  {u1s[i]:>8.4f}  {u2s[i]:>8.4f}\n")
                f.write("\n")

    def finalize(self, results_json: str, plot_path: str, all_results: dict):
        if os.path.exists(plot_path):
            shutil.copy2(plot_path, os.path.join(self.run_dir, "pareto_frontier.png"))
        with open(self.log_path, "a") as f:

            # performance table
            f.write("\n--- Final comparison (all methods) ---\n")
            f.write(f"{'method':<26} {'share':>7} {'unique1':>8} {'unique2':>8}  notes\n")
            f.write("-" * 66 + "\n")
            for method, data in all_results.items():
                this_run = any(r[0] == method for r in self._rows)
                if "curve" in data:
                    curve = data["curve"]
                    s  = max(curve.get("share",   [0]))
                    u1 = max(curve.get("unique1", [0]))
                    u2 = max(curve.get("unique2", [0]))
                    dist_tag = f" {self._lam_note(method)}" if this_run else ""
                    note = f"oracle-best{dist_tag}"
                else:
                    p  = data.get("probes", {})
                    s  = p.get("share",   0)
                    u1 = p.get("unique1", 0)
                    u2 = p.get("unique2", 0)
                    note = "fixed"
                tag = "  ◄ this run" if this_run else ""
                f.write(f"{method:<26} {s:>7.4f} {u1:>8.4f} {u2:>8.4f}  {note}{tag}\n")

            # Pareto metrics table
            f.write("\n--- Pareto Metrics  (2D exact HV, space=(share,U_avg), ref=(0,0)) ---\n")

            baseline_names = ", ".join(sorted(_BASELINE_METHODS))
            f.write(f"\nFixed baselines  [hv_s = share × U_avg  |  baselines: {baseline_names}]\n")
            f.write(f"{'method':<26} {'hv_single':>9}  {'dominated':>9}\n")
            f.write("-" * 48 + "\n")
            for method, data in all_results.items():
                mt = data.get("pareto_metrics", {})
                if mt.get("type") == "fixed" and mt.get("role") == "baseline":
                    dom = "Yes" if mt.get("is_dominated", False) else "No"
                    f.write(f"{method:<26} {mt.get('hv_single', 0):>9.4f}  {dom:>9}\n")
            hv_b = next((d.get("pareto_metrics", {}).get("HV_baselines", 0)
                         for d in all_results.values()
                         if d.get("pareto_metrics", {}).get("type") == "fixed"), 0)
            f.write(f"{'HV_baselines (ND)':<26} {hv_b:>9.4f}\n")

            f.write(f"\nDisplay-only  (shown for reference, not used in comparison)\n")
            f.write(f"{'method':<26} {'hv_single':>9}  {'dominated':>9}\n")
            f.write("-" * 48 + "\n")
            for method, data in all_results.items():
                mt = data.get("pareto_metrics", {})
                if mt.get("type") == "fixed" and mt.get("role") == "display_only":
                    dom = "Yes" if mt.get("is_dominated", False) else "No"
                    f.write(f"{method:<26} {mt.get('hv_single', 0):>9.4f}  {dom:>9}\n")

            f.write(f"\nLambda methods  (anchors — compared against baseline front)\n")
            f.write(f"{'method':<26} {'HV_λ':>6}  {'HV_all':>6}  {'PCG':>7}  "
                    f"{'Unif':>5}  {'ND':>4}  {'bestU1':>7}  {'bestU2':>7}  "
                    f"{'bestU':>6}  notes\n")
            f.write("-" * 90 + "\n")
            for method, data in all_results.items():
                mt = data.get("pareto_metrics", {})
                if mt.get("type") == "lambda":
                    this_run = any(r[0] == method for r in self._rows)
                    dist_tag = f" {self._lam_note(method)}" if this_run else ""
                    tag      = "  ◄ this run" if this_run else ""
                    f.write(
                        f"{method:<26} {mt.get('HV_lambda', 0):>6.4f}  "
                        f"{mt.get('HV_all', 0):>6.4f}  "
                        f"{mt.get('PCG', 0):>+7.4f}  "
                        f"{mt.get('uniformity_score', 0):>5.3f}  "
                        f"{mt.get('nondominated_ratio', 0):>4.2f}  "
                        f"{mt.get('best_U1', 0):>7.4f}  "
                        f"{mt.get('best_U2', 0):>7.4f}  "
                        f"{mt.get('best_U', 0):>6.4f}{dist_tag}{tag}\n"
                    )

            f.write("\n--- Files ---\n")
            f.write(f"results.json    : {results_json}\n")
            f.write(f"pareto_frontier : {plot_path}\n")
            f.write(f"run_dir         : {self.run_dir}\n")
            f.write("=" * 60 + "\n")
        print(f"\nRun log saved to {self.run_dir}")


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default=DEFAULT_DATA_DIR)
    parser.add_argument("--out_dir",     default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--batch_size",  type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--latent_dim",  type=int,   default=DEFAULT_LATENT_DIM)
    parser.add_argument("--proj_dim",    type=int,   default=DEFAULT_PROJ_DIM)
    parser.add_argument("--probe_shots", type=int,   default=DEFAULT_PROBE_SHOTS,
                        help="K-shot probe: K pairs per class for training. 0=all pairs.")
    parser.add_argument("--device",      default=DEFAULT_DEVICE)
    parser.add_argument("--methods",     nargs="+",
                        default=DEFAULT_METHODS_LIST if DEFAULT_METHODS_LIST
                                else list(METHODS.keys()))
    parser.add_argument("--encoder_param", default=DEFAULT_ENCODER_PARAM,
                        choices=["projection", "encoder"],
                        help="hyper_lambda only — 'projection': condition heads only; "
                             "'encoder': FiLM inside the AlexNet conv blocks")
    parser.add_argument("--lam_dist", default=None,
                        choices=["uniform", "dirichlet", "beta", "warmup"],
                        help="Distribution for drawing λ during training. "
                             "'uniform' | 'dirichlet' (Beta(α,α)) | 'beta' (Beta(α,β)) | "
                             "'warmup' (phased curriculum from lambda_warmup.phases in config). "
                             "If omitted: factorcl_warmup defaults to 'warmup', every other "
                             "method defaults to the config default — so --methods can mix "
                             "factorcl_warmup with other λ-methods in one call and each gets "
                             "its natural default. Pass this explicitly to override that "
                             "per-method default uniformly for every method in --methods.")
    parser.add_argument("--dirichlet_alpha", type=float, default=DEFAULT_DIRICHLET_ALPHA,
                        help="Concentration param α for Dirichlet/Beta sampling "
                             "(only used when --lam_dist dirichlet). "
                             "α<1→bimodal, α=1→uniform, α>1→unimodal.")
    parser.add_argument("--pareto_solver", default=DEFAULT_PARETO_SOLVER,
                        choices=["ls", "epo"],
                        help="'ls': 2λ·L_R + 2(1-λ)·L_U  |  'epo': Exact Pareto LP.")
    parser.add_argument("--lora_alpha_enc", type=float, default=None,
                        help="adapter scale for the ENCODER (default: --lora_alpha)")
    parser.add_argument("--lora_alpha_head", type=float, default=None,
                        help="adapter scale for the four HEADS (default: --lora_alpha)")
    parser.add_argument("--lora_lr_mult", type=float, default=10.0,
                        help="lr multiplier for LoRA parameters (historical default 10, "
                             "never swept; relevant because the both-sites arm has twice "
                             "as many parameters in that group)")
    parser.add_argument("--eval_split", default="test", choices=["val", "test"],
                        help="split the probe REPORTS on. Select configurations on val; "
                             "report the winner once on test.")
    parser.add_argument("--augment", default="",
                        help="per-modality augmentation override, e.g. 'M3:nocolor'. The "
                             "default SimCLR recipe applies ColorJitter(0.4) p=0.8 and "
                             "RandomGrayscale p=0.2, which trains a colour-unique head to "
                             "DISCARD colour (measured: U1 0.782 bits, shape leak 2.333). "
                             "'nocolor' keeps crop+flip only.")
    parser.add_argument("--modality_pair", default="M1,M2",
                        help="which two rendered modalities form the pair: M1 "
                             "(shape+deformation), M2 (shape+texture), M3 (shape+colour). "
                             "Default M1,M2 reproduces every existing result; M3,M2 gives "
                             "unique1 = colour, unique2 = texture against the same shared shape.")
    parser.add_argument("--seed",          type=int, default=DEFAULT_TRAIN_SEED,
                        help="Training seed (torch/numpy/random/CUDA).")
    parser.add_argument("--readout",
                        choices=["amp", "dim", "dim_rand", "dim_pca"],
                        default="dim",
                        help="decomp_R readout. amp = sqrt(lambda) amplitude concat "
                             "(a linear probe can undo it, so only the SUPPORT matters "
                             "and the simplex interior is unresolved). dim = allocate "
                             "round(lambda*proj_dim) DIMENSIONS per block, which no "
                             "reweighting can recover. Vertices are identical either way.")
    parser.add_argument("--skip_training",  action="store_true", default=DEFAULT_SKIP_TRAINING)
    parser.add_argument("--plot_only",      action="store_true", default=DEFAULT_PLOT_ONLY)
    parser.add_argument("--modality",       default=None, choices=["m1", "m2"],
                        help="Single-modality mode: train loss and probe both adapt to the "
                             "specified modality. Results stored as {method}_m1 / {method}_m2.")
    parser.add_argument("--lambda_cka",     action="store_true", default=False,
                        help="H7: measure adjacent-lambda / endpoint CKA on existing "
                             "checkpoints and exit. No training, no probing.")
    parser.add_argument("--reprobe_all",    action="store_true", default=DEFAULT_REPROBE_ALL,
                        help="Re-probe every checkpointed method at --probe_shots after training.")
    parser.add_argument("--alpha_smooth",  type=float, default=None,
                        help="Override alpha_smooth (L_smooth weight) from config. "
                             "Set 0.0 to ablate L_smooth.")
    parser.add_argument("--beta_div",      type=float, default=None,
                        help="Override beta_div (L_diversity weight) from config. "
                             "Set 0.0 to ablate L_diversity.")
    parser.add_argument("--result_key",    default=None,
                        help="Override the results.json key (default: derived from --methods). "
                             "Use for ablation variants that share a --methods value but must "
                             "be stored separately (e.g. simclr_hyper_no_smooth).")
    # Fix 8: PaLoRA multi-preference training (LoRA methods only)
    parser.add_argument("--num_preferences", type=int, default=1,
                        help="M preferences per minibatch (Fix 8). 1 = single-λ baseline")
    parser.add_argument("--preference_schedule", choices=["single", "fixed", "annealed"],
                        default="single",
                        help="single = baseline; fixed = M evenly-spaced λ; "
                             "annealed = PaLoRA center-to-edge schedule")
    parser.add_argument("--annealing_temperature", type=float, default=1.0,
                        help="Q in the PaLoRA annealing schedule (>0)")
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="PaLoRA α; LoRA scale = α/r. Applied only when annealed")
    parser.add_argument("--lam_club", type=float, default=1.0,
                        help="simplex_enc_decomp_R: weight on the CLUB penalty that "
                             "decorrelates each U head from its R head. 0 = readout-only ablation.")
    parser.add_argument("--simplex_side", type=int, default=5,
                        help="LoRA-Simplex: points per edge of the triangular grid "
                             "(side=5 → 15 preferences on the 2-simplex)")
    parser.add_argument("--prefs_per_batch", type=int, default=0,
                        help="decomp_R: M preferences per minibatch, drawn by a balanced "
                             "shuffled-cycle scheduler so every grid point gets the same "
                             "number of updates. 0 (default) or >= |grid| = every "
                             "preference every step, i.e. the REPORTED configuration "
                             "(M=15 on the side-5 grid). Lower M cuts the per-step cost "
                             "proportionally at the same number of optimiser steps.")
    parser.add_argument("--anneal_mode", choices=["power", "linear"], default="power",
                        help="decomp_R: how annealed preferences travel from the centroid "
                             "to their target. power (default) reproduces every existing "
                             "run; linear interpolates centroid->target, which is the only "
                             "one of the two that actually moves the vertices.")
    args = parser.parse_args()
    # Pair selection must happen before any dataset is constructed; it changes which
    # images are loaded AND which label 'unique1'/'unique2' refer to.
    _pair = set_modality_pair(args.modality_pair)
    LORA_ALPHA_ENC[0]  = args.lora_alpha_enc
    LORA_ALPHA_HEAD[0] = args.lora_alpha_head
    LORA_LR_MULT[0]    = args.lora_lr_mult
    _EVAL_SPLIT[0]     = args.eval_split
    if args.eval_split != "test":
        print(f"[eval_split] probing on '{args.eval_split}' -- test untouched")
    _aug = _set_aug(args.augment)
    if args.augment:
        print(f"[augment] {_aug}")
    if _pair != ['M1', 'M2']:
        print(f"[modality-pair] {_pair[0]}/{_pair[1]} -> unique1={pair_unique_keys()[0]}, "
              f"unique2={pair_unique_keys()[1]}")

    _READOUT[0] = args.readout

    # Allow CLI to override L_smooth / L_diversity weights (ablation use)
    global ALPHA_SMOOTH, BETA_DIV
    if args.alpha_smooth is not None:
        ALPHA_SMOOTH = args.alpha_smooth
    if args.beta_div is not None:
        BETA_DIV = args.beta_div

    os.makedirs(args.out_dir, exist_ok=True)
    # results.json is the canonical name -- merge_results / merge_seeds /
    # compare_results / results_table / aggregate_seeds all hardcode it, so writing
    # elsewhere makes a run invisible to them. The readout is therefore recorded
    # INSIDE the file instead of in its name (see _stamp below); a file with no
    # "_readout" key predates this and is amplitude. A per-readout copy is also
    # written so two readouts can coexist in one directory for comparison.
    results_path = os.path.join(args.out_dir, "results.json")
    results_path_ro = os.path.join(args.out_dir, f"results_readout_{args.readout}.json")

    def _dump(obj):
        # Stamp the readout INSIDE each method entry, not at the top level: every
        # aggregator iterates all_results.items() and treats each key as a method,
        # so a top-level "_readout" key would be read as a method whose value is a
        # string and crash on data["probes"]. Per-entry keys are inert to them.
        obj = {k: ({**v, "readout": args.readout} if isinstance(v, dict) else v)
               for k, v in obj.items()}
        for _p in (results_path, results_path_ro):
            with open(_p, "w") as _f:
                json.dump(obj, _f, indent=2)

    if args.plot_only:
        with open(results_path) as f:
            all_results = json.load(f)
        from pareto_ssl.plot_pareto import plot_pareto
        plot_pareto(all_results, args.out_dir)
        return

    all_results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)

    run_log = RunLogger(args.out_dir, args)

    for method in args.methods:
        if method not in METHODS:
            print(f"Unknown method: {method}, skipping."); continue

        print(f"\n{'='*60}")
        print(f"  {method}  —  {METHODS[method]}")
        print(f"{'='*60}")
        # When --modality is set: checkpoint lives in {method}_m1 / {method}_m2,
        # the result key also carries the suffix, and training+probing adapt fully.
        mod_suffix  = f"_{args.modality}" if args.modality else ""
        if args.result_key is not None:
            if len(args.methods) != 1:
                raise ValueError("--result_key requires exactly one --methods value")
            result_key = args.result_key
        else:
            # Fix 8 (annealed) runs get a distinct "_fix8" key so they merge/plot as
            # their own method next to the base LoRA runs (never overwrite them).
            # LoRA-Simplex is excluded: annealing is intrinsic to it (not an ablation
            # layered on a base run), and its own method name already distinguishes
            # it — appending "_fix8" would break merge_results.py, which requires
            # folder name == result key.
            fix8_sfx = ("_fix8" if args.preference_schedule == "annealed"
                        and "simplex" not in method else "")
            result_key = f"{method}{fix8_sfx}{mod_suffix}"
        # Checkpoints are keyed by result_key (not bare method name): two runs of
        # the same method with different flags (e.g. --encoder_param, --alpha_smooth)
        # must never share a checkpoint dir, or probing can silently load a stale,
        # mismatched checkpoint left over from a previous ablation variant.
        checkpoint_method = _CHECKPOINT_ALIAS.get(method)
        if checkpoint_method is not None:
            enc_dir = os.path.join(args.out_dir, f"{checkpoint_method}{mod_suffix}")
        else:
            enc_dir = os.path.join(args.out_dir, result_key)

        # Per-method λ-distribution default: factorcl_warmup defaults to its
        # namesake curriculum, every other method to the config default —
        # unless --lam_dist was passed explicitly, which then applies to all
        # methods in --methods uniformly (needed for ablations that compare
        # factorcl_warmup under uniform vs warmup sampling).
        effective_lam_dist = (args.lam_dist if args.lam_dist is not None
                               else ("warmup" if method == "factorcl_warmup" else DEFAULT_LAM_DIST))

        if not args.skip_training:
            if checkpoint_method is not None:
                print(f"  Reusing checkpoints from '{checkpoint_method}{mod_suffix}' — no training needed")
            else:
                train_encoders(
                    method=method, data_dir=args.data_dir, out_dir=enc_dir,
                    device=args.device, epochs=args.epochs, lr=args.lr,
                    batch_size=args.batch_size, latent_dim=args.latent_dim,
                    proj_dim=args.proj_dim, encoder_param=args.encoder_param,
                    lam_dist=effective_lam_dist, dirichlet_alpha=args.dirichlet_alpha,
                    pareto_solver=args.pareto_solver, seed=args.seed,
                    modality=args.modality,
                    num_preferences=args.num_preferences,
                    preference_schedule=args.preference_schedule,
                    annealing_temperature=args.annealing_temperature,
                    lora_alpha=args.lora_alpha,
                    simplex_side=args.simplex_side,
                    prefs_per_batch=args.prefs_per_batch,
                    anneal_mode=args.anneal_mode,
                    lam_club=args.lam_club,
                )
        else:
            if not (os.path.isdir(enc_dir) and
                    any(f.endswith(".pth") for f in os.listdir(enc_dir))):
                print(f"  No checkpoint found — skipping probe for {result_key}")
                continue

        shot_str = f", {args.probe_shots}-shot" if args.probe_shots > 0 else ""
        if method in LAMBDA_METHODS:
            mod_label = f" [{args.modality}-only]" if args.modality else ""
            print(f"  Probing (lambda sweep 0→1{mod_label}{shot_str})...")
            curve = probe_lambda_cond(
                enc_dir=enc_dir, data_dir=args.data_dir,
                device=args.device, latent_dim=args.latent_dim,
                batch_size=args.batch_size, k_shot=args.probe_shots,
                proj_dim=args.proj_dim, modality=args.modality,
            )
            desc = METHODS[method] + (f" [{args.modality}-only inference]" if args.modality else "")
            all_results[result_key] = {"description": desc, "curve": curve}
        else:
            if args.modality:
                # Fixed method + single-modality probe
                print(f"  Probing (one_modality / {args.modality}{shot_str})...")
                om = probe_one_modality_fixed(
                    method=method, enc_dir=enc_dir, data_dir=args.data_dir,
                    device=args.device, latent_dim=args.latent_dim,
                    batch_size=args.batch_size, k_shot=args.probe_shots,
                    proj_dim=args.proj_dim,
                )
                desc = METHODS[method] + f" [{args.modality}-only inference]"
                all_results[result_key] = {
                    "description": desc,
                    "probes": om[args.modality],
                }
            else:
                mode_str = INFERENCE_MODE[method]
                print(f"  Probing ({mode_str}{shot_str})...")
                probes = probe_method(
                    method=method, enc_dir=enc_dir, data_dir=args.data_dir,
                    device=args.device, latent_dim=args.latent_dim,
                    batch_size=args.batch_size, k_shot=args.probe_shots,
                    proj_dim=args.proj_dim,
                )
                all_results[result_key] = {"description": METHODS[method], "probes": probes}

        _dump(all_results)
        run_log.log_method(result_key, all_results[result_key])

    # H7: lambda-CKA only
    if args.lambda_cka:
        cka_path = os.path.join(args.out_dir, "lambda_cka.json")
        out = json.load(open(cka_path)) if os.path.exists(cka_path) else {}
        targets = [m for m in args.methods if m in LAMBDA_METHODS]
        if not targets:
            print(f"  --lambda_cka: none of {args.methods} are lambda methods "
                  f"({sorted(LAMBDA_METHODS)}) — nothing to do.")
            return
        for method in targets:
            enc_dir = os.path.join(args.out_dir, _CHECKPOINT_ALIAS.get(method, method))
            if not os.path.isdir(enc_dir):
                print(f"  [{method}] no checkpoint dir at {enc_dir} — skipping")
                continue
            print(f"\n  [{method}] lambda-CKA on {enc_dir}")
            try:
                out[method] = probe_lambda_cond(
                    enc_dir=enc_dir, data_dir=args.data_dir,
                    device=args.device, latent_dim=args.latent_dim,
                    batch_size=args.batch_size, proj_dim=args.proj_dim,
                    mode="cka",
                )
                r = out[method]
                print(f"    adjacent-CKA mean = {r['adjacent_cka_mean']:.4f}   "
                      f"endpoint = {r['endpoint_cka']:.4f}   "
                      f"all-pairs min = {r['all_pairs_min']:.4f}")
            except Exception as exc:
                print(f"  [{method}] CKA FAILED — {type(exc).__name__}: {exc}")
                continue
            with open(cka_path, "w") as f:
                json.dump(out, f, indent=2)
        print(f"\n  wrote {cka_path}")
        print(f"  MOSEI reference band for comparison: adjacent 0.92-0.97, "
              f"endpoint 0.38-0.71")
        return

    if args.reprobe_all:
        # Any subdirectory of out_dir that contains at least one .pth file
        reprobe_methods = [
            m for m in METHODS
            if os.path.isdir(os.path.join(args.out_dir, _CHECKPOINT_ALIAS.get(m, m)))
            and any(
                f.endswith(".pth")
                for f in os.listdir(os.path.join(args.out_dir, _CHECKPOINT_ALIAS.get(m, m)))
            )
        ]
        shot_str = f"{args.probe_shots}-shot" if args.probe_shots > 0 else "all pairs"
        print(f"\n{'─'*60}")
        print(f"  --reprobe_all: re-probing {len(reprobe_methods)} methods "
              f"at {shot_str} (same images, seed=42)")
        print(f"  {reprobe_methods}")
        print(f"{'─'*60}")
        for method in reprobe_methods:
            enc_dir = os.path.join(args.out_dir, _CHECKPOINT_ALIAS.get(method, method))
            s_str = f", {shot_str}"
            try:
                if method in LAMBDA_METHODS:
                    print(f"  [{method}] probing (lambda sweep 0→1{s_str})...")
                    curve = probe_lambda_cond(
                        enc_dir=enc_dir, data_dir=args.data_dir,
                        device=args.device, latent_dim=args.latent_dim,
                        batch_size=args.batch_size, k_shot=args.probe_shots,
                        proj_dim=args.proj_dim,
                    )
                    all_results[method] = {"description": METHODS[method], "curve": curve}
                else:
                    mode_str = INFERENCE_MODE[method]
                    print(f"  [{method}] probing ({mode_str}{s_str})...")
                    probes = probe_method(
                        method=method, enc_dir=enc_dir, data_dir=args.data_dir,
                        device=args.device, latent_dim=args.latent_dim,
                        batch_size=args.batch_size, k_shot=args.probe_shots,
                        proj_dim=args.proj_dim,
                    )
                    all_results[method] = {"description": METHODS[method], "probes": probes}
            except Exception as exc:
                print(f"  [{method}] PROBE FAILED — {exc}  (skipping)")
                continue
            # Save incrementally so a crash midway doesn't lose earlier results
            _dump(all_results)

    print(f"\n{'Method':<16} {'share':>6} {'unique1':>8} {'unique2':>8}"
          f"  ({args.probe_shots}-shot probe, AlexNet/MMFusion)")
    print("-" * 50)
    for m, d in all_results.items():
        if "curve" in d:
            curve = d["curve"]
            best = {t: max(curve[t]) for t in TASKS}
            print(f"{m:<16} {best.get('share',0):>6.4f} {best.get('unique1',0):>8.4f}"
                  f" {best.get('unique2',0):>8.4f}  ({m} oracle-best)")
        else:
            p = d["probes"]
            print(f"{m:<16} {p.get('share',0):>6.4f} {p.get('unique1',0):>8.4f}"
                  f" {p.get('unique2',0):>8.4f}")

    # Pareto metrics for ALL methods
    pareto_metrics = compute_all_pareto_metrics(all_results, args.out_dir)
    for method, mt in pareto_metrics.items():
        all_results[method]["pareto_metrics"] = mt
    _dump(all_results)

    from pareto_ssl.plot_pareto import plot_pareto
    plot_pareto(all_results, args.out_dir)
    run_log.finalize(results_path, os.path.join(args.out_dir, "pareto_frontier.png"), all_results)


if __name__ == "__main__":
    main()
