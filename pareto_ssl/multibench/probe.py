"""
Linear Evaluation Probe — MultiBench (MOSI / UR-FUNNY)
=======================================================
Three modes:

  validation : full valid set → 80/20 split to select best λ → fit LR on full valid
               → Top-1 accuracy (%) on test

  fewshot    : K ∈ {5, 10, 20, 50} shots from valid
               → LOO-kNN λ selection on K shots (no test data touched)
               → kNN prediction on test, averaged over 5 seeds

  cka        : CKA diagnostic — extracts z at λ=0 and λ=1 on test set,
               computes linear CKA between them per modality and overall.
               Low CKA  → heads extract genuinely different structure → Pareto curve is real.
               High CKA → heads collapse to same representation → λ sweep is cosmetic.
               Run this BEFORE trusting any Pareto curve results.

Usage:
  cd <repo root>

  python pareto_ssl/multibench/probe.py \\
      --enc_dir pareto_ssl/multibench/results/humor/approach2/factorcl_warmup_seed42 \\
      --dataset humor --approach 2 --mode cka

  python pareto_ssl/multibench/probe.py \\
      --enc_dir pareto_ssl/multibench/results/humor/approach2/factorcl_warmup_seed42 \\
      --dataset humor --approach 2 --mode validation

  python pareto_ssl/multibench/probe.py \\
      --enc_dir pareto_ssl/multibench/results/humor/approach2/factorcl_warmup_seed42 \\
      --dataset humor --approach 2 --mode fewshot
"""

import argparse
import functools
import math
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Path setup
_PROG = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROG))
sys.path.insert(0, str(_PROG / "CoMM"))

from pareto_ssl.multibench.affect_data import (
    AffectDataset, collate_affect, data_path as _affect_path)
from pareto_ssl.networks import (ProjectionHead, DualFiLMProjectionHead,
                                 DualFiLMProjectionHeadPreNorm,
                                 LoRADualProjectionHead, LoRATriProjectionHead,
                                 LoRATransformerEncoder)
from pareto_ssl.benchmark import DEFAULT_PROJ_DIM

try:
    from models.mmfusion import FusionTransformer as _CommFusionTransformer
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "The MultiBench pipeline builds on CoMM's Transformer, which is a separate "
        "project and is not redistributed here.\n"
        "Clone it next to this repository:\n"
        "    git clone https://github.com/Duplums/CoMM\n"
        "so that <repo root>/CoMM/models/ exists, then re-run.\n"
        "The Trifeature pipeline (pareto_ssl/benchmark.py) does not need it."
    ) from _e

from pareto_ssl.multibench.benchmark_multibench import (
    simplex_grid,
    duo_grid,
    enc_film_mode_for,
    FEAT_DIM, MODALITIES, ENC_DIM_XFMR, FILM_MODES,
    _actual_enc_dim, _make_encoders, FusionEncoder,
    TransformerSeqEncoder, CommMLPHead, LoRACommMLPHead,
    _make_fusion_xfmr, _image_root,
)
from pareto_ssl.multibench.image_backends import is_image_dataset, image_probe_loader
from pareto_ssl.multibench.multitask_registry import (
    is_multitask_dataset, probe_loader as mt_probe_loader, usable_tasks, dataset_class,
)

def _lamstr(lam, w=6, p=2):
    """Format a preference for printing: scalar λ or a simplex tuple."""
    if isinstance(lam, (tuple, list)):
        return "(" + ",".join(f"{float(c):.2f}" for c in lam) + ")"
    return f"{float(lam):>{w}.{p}f}"


_SIMPLEX_ENC = [False]   # approach-5 simplex: encoder is preference-conditioned
_MT_TASK = [None]        # which mosei_multitask label the probe is scoring
_SIMPLEX_SIDE = [5]      # overwritten from train_meta.json in _load_models

# Readout for the decomp_R family: how the preference is applied to the three
# blocks of z(lambda).
#   "amp" : z = normalize([sqrt(l_R)*R || sqrt(l_v)*U1 || sqrt(l_t)*U2])  (original)
#   "dim" : each block contributes round(l_t * proj_dim) DIMENSIONS, unscaled.
#   "none": no per-block weighting at all -- the three blocks are concatenated at
#           full width. lambda then acts ONLY through W(lambda) inside the LoRA
#           encoder/heads, so any variation across preferences is attributable to
#           the weight-space conditioning and nothing else. This is the clean
#           isolation of the PaLoRA mechanism; "amp" and "dim" both mix that
#           channel with a readout-side effect.
#
# Why "dim" exists: under "amp" a LINEAR probe can undo the scaling exactly
# (w_block -> w_block/sqrt(l_block)), and the probe even sweeps C to find the
# regularisation that makes that cheapest. So amplitude weighting is invisible and
# only a block being EXACTLY zero changes anything -- the 15 preferences collapse
# to 7 distinct supports. Measured: 85% of the spread across preferences is
# explained by support alone, 0.76pt within a support group. Allocating DIMENSIONS
# removes information that no reweighting can recover, so lambda becomes a real
# continuous axis. Total width stays ~proj_dim, so the budget is fixed and only its
# split changes.
_READOUT = ["amp"]   # amp | dim | dim_rand | dim_pca | none
# Cosmetic descriptions for the banner. Looked up with .get(), never indexed:
# a read-out missing a blurb must not crash the probe (it did -- KeyError on
# 'dim_rand', after the value was added to argparse but not here).
_READOUT_BLURB = {
    "amp":      "  (sqrt-lambda amplitude scaling)",
    "dim":      "  (dimension allocation — lambda removes DIMS, not amplitude)",
    "dim_rand": "  (dimension allocation, RANDOM coordinates — control for 'dim')",
    "dim_pca":  "  (dimension allocation, top principal components per block)",
    "dim_split":"  (dimension allocation; R budget SPLIT across per-modality shared vectors, not averaged)",
    "none":     "  (no readout weighting — lambda acts ONLY via W(lambda))",
}

# Optional FIXED preference: when set, run_validation reports test accuracy at THIS
# lambda instead of at the per-seed validation argmax.
#
# Why it matters: the argmax is taken over 15 preferences on a 373-sample validation
# split, where one sample is 0.27 points. Measured on ENC lc0.5/dim, the preference
# (0.00,0.25,0.75) ranks 1st or 2nd in all five seeds (worst rank 2) yet loses the
# argmax in two of them -- by 1.88 and by 0.27 points, the latter being a single
# validation sample. Reporting "one fixed operating point evaluated on every seed"
# is both more stable and more honest than per-seed argmax selection, and unlike
# dropping disagreeing seeds it discards nothing.
_FIXED_LAM = [None]

# Constants
LAMBDA_GRID              = [round(l, 2) for l in np.arange(0.0, 1.01, 0.1)]
FEWSHOT_K                = [5, 10, 20, 50]
FEWSHOT_SEEDS            = [42, 43, 44, 45, 46]
KNN_K                    = 5
LR_CS                    = (1e-2, 0.1, 1.0, 10.0)
CURVE_PROOF_INTERMED_LAMS = [0.05, 0.15, 0.35, 0.55, 0.65, 0.85]


# Helpers: load models

class _CommEncWrapper(nn.Module):
    """Wraps TransformerSeqEncoder + FusionTransformer into a single λ-blind encoder.
    Returns (B, 40) — same shape as TransformerEncoder — so _extract_backbone works unchanged."""
    def __init__(self, enc, fusion_xfmr):
        super().__init__()
        self.enc = enc
        self.fusion_xfmr = fusion_xfmr

    def forward(self, x, lam=None):
        return self.fusion_xfmr([self.enc(x)])


def _apply_enc_width(meta, approach):
    """Rebuild-time encoder width, taken from train_meta.

    _make_encoders/_actual_enc_dim read benchmark_multibench.ENC_DIM_XFMR at CALL
    time, while probe.py bound its own copy at import time -- so both must be set.
    Every path that reconstructs an encoder has to call this: run_fixed did not,
    which meant a --enc_width 70 baseline was rebuilt at the default 40 and every
    layer failed with a size mismatch on load.
    """
    _w = meta.get("enc_width") or (meta.get("enc_dim") if approach != 1 else None)
    if _w and int(_w) != ENC_DIM_XFMR:
        import pareto_ssl.multibench.benchmark_multibench as _bm
        _w = int(_w)
        _bm.ENC_DIM_XFMR = _w
        globals()["ENC_DIM_XFMR"] = _w
        LoRATransformerEncoder._D = _w   # ap5 encoder keeps its own width
        print(f"  encoder width from train_meta -> {_w}")
    return _w


def _load_models(enc_dir, approach, dataset, enc_dim, proj_dim, device,
                 enc_ckpt_v=None, enc_ckpt_t=None, film_mode="none"):
    # Always read proj_dim and film_mode from train_meta.json if it exists
    meta_path = os.path.join(enc_dir, "train_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        proj_dim  = meta.get("proj_dim",  proj_dim)
        film_mode = meta.get("film_mode", film_mode)
        enc_dim   = meta.get("enc_dim",   enc_dim)

    # Rebuild the encoder at the width it was TRAINED with. The trainer records the
    # true width in enc_dim, but _actual_enc_dim() returns the module constant
    # ENC_DIM_XFMR for every approach except 1 — so a capacity-sweep checkpoint was
    # loaded into a 40-wide model and failed with size-mismatch on every layer.
    _apply_enc_width(meta, approach)
    if meta.get("simplex_side"):
        _SIMPLEX_SIDE[0] = int(meta["simplex_side"])
    _SIMPLEX_ENC[0] = (meta.get("film_mode") == "simplex" and approach == 5)

    adim   = _actual_enc_dim(approach, enc_dim)

    # Output-mixing variants: dispatch on film_mode BEFORE the approach split
    # These were previously handled inside `if approach == 5:`, which was wrong —
    # simplex4h/simplex4h_norm/simplex6h run on approach 4, so they fell through to
    # the LoRA path and tried to load proj_v.pth (a shared-base head they never
    # save). Only palora_enc needs a LoRA-conditioned encoder; the simplex4h*/6h
    # encoder is λ-blind, which enc_film_mode_for already reports.
    _fm = meta.get("film_mode", film_mode)
    if _fm in ("simplex_enc_decomp_R", "simplex_both_decomp_R"):
        # Four plain heads (r/u per modality) PLUS a preference-conditioned LoRA
        # encoder -- so it needs the approach-5 encoder path AND the four-head head
        # path. It is NOT in the simplex4h*/6h list just below, because those keep a
        # lambda-BLIND encoder; decomp_R is routed by the palora_enc branch further
        # down instead, which builds the LoRA encoder and loads proj_{v,t}_{r,u}.pth.
        # Missing that routing made the probe look for proj_v.pth and fail.
        # simplex_both_decomp_R is here too: its encoder IS preference-conditioned, and
        # without this flag _extract_backbone calls enc(x) instead of enc.forward_mix,
        # i.e. probes a lambda-blind encoder for a model trained with a conditioned one.
        _SIMPLEX_ENC[0] = True
    if _fm in ("simplex4h", "simplex4h_norm", "simplex6h", "duo_txt",
               "simplex_proj_decomp_R", "simplex_both_decomp_R"):
        enc_v, enc_t = _make_encoders(approach, dataset, enc_dim, device,
                                      enc_film_mode_for(_fm, approach),
                                      plain_encoder=(_fm == "simplex_proj_decomp_R"))
        for enc, nm in ((enc_v, enc_ckpt_v or os.path.join(enc_dir, "enc_v.pth")),
                        (enc_t, enc_ckpt_t or os.path.join(enc_dir, "enc_t.pth"))):
            if os.path.exists(nm):
                enc.load_state_dict(torch.load(nm, map_location=device, weights_only=True))
            enc.eval()
        if _fm == "simplex6h":
            def _ld3(name):                       # three plain heads in a ModuleDict
                d = torch.nn.ModuleDict({k: ProjectionHead(adim, proj_dim)
                                         for k in ("r", "u1", "u2")}).to(device)
                fp = os.path.join(enc_dir, name)
                if os.path.exists(fp):
                    d.load_state_dict(torch.load(fp, map_location=device, weights_only=True))
                d.eval()
                return d
            return (enc_v, enc_t, _ld3("proj_v.pth"), _ld3("proj_t.pth"), None, None)
        def _ld4b(name):                          # four independent heads
            if _fm in ("simplex_proj_decomp_R", "simplex_both_decomp_R"):
                # LoRA heads: the preference acts HERE, so rank/alpha must match
                # training exactly. `scale` is a plain float attribute, not a
                # parameter, so load_state_dict would silently leave a mismatched
                # alpha at 1.0 and the probe would evaluate a different model.
                h = LoRADualProjectionHead(adim, proj_dim,
                                           rank=meta.get("lora_rank", 4),
                                           alpha=meta.get("lora_alpha")).to(device)
            else:
                h = ProjectionHead(adim, proj_dim).to(device)
            fp = os.path.join(enc_dir, name)
            if os.path.exists(fp):
                h.load_state_dict(torch.load(fp, map_location=device, weights_only=True))
            h.eval()
            return h
        return (enc_v, enc_t, _ld4b("proj_v_r.pth"), _ld4b("proj_t_r.pth"),
                _ld4b("proj_v_u.pth"), _ld4b("proj_t_u.pth"))

    if approach == 5:
        # LoRA encoder + LoRA proj head
        branches = meta.get("lora_branches", 2)
        rank     = meta.get("lora_rank",   4)
        rank_m   = meta.get("lora_rank_m", rank)
        # α/r must match training or the probe rebuilds a DIFFERENT model: `scale`
        # is a plain float attribute (not a parameter), so load_state_dict would
        # silently leave it at 1.0. palora_enc puts α on the encoder as well.
        _alpha     = meta.get("lora_alpha")
        _enc_alpha = _alpha if meta.get("film_mode") == "palora_enc" else None
        if is_image_dataset(dataset):
            # Image datasets have no FEAT_DIM entry and no token sequence: they use
            # CNN encoders, with LoRA on the adapter when the film_mode asks for a
            # preference-conditioned encoder. _make_encoders owns that decision, so
            # route through it rather than duplicating it here — otherwise this
            # branch KeyErrors on FEAT_DIM["avmnist"] and, if that were patched
            # around, would build a Transformer for a CNN checkpoint.
            enc_v, enc_t = _make_encoders(
                approach, dataset, enc_dim, device,
                enc_film_mode_for(meta.get("film_mode", film_mode), approach),
                lora_branches=branches, lora_rank=rank, lora_rank_m=rank_m,
                enc_alpha=_enc_alpha)
        else:
            fv = FEAT_DIM[dataset]["vision"]; ft = FEAT_DIM[dataset]["text"]
            enc_v = LoRATransformerEncoder(fv, branches=branches, rank=rank, rank_m=rank_m,
                                           alpha=_enc_alpha).to(device)
            enc_t = LoRATransformerEncoder(ft, branches=branches, rank=rank, rank_m=rank_m,
                                           alpha=_enc_alpha).to(device)
        ckpt_v = enc_ckpt_v or os.path.join(enc_dir, "enc_v.pth")
        ckpt_t = enc_ckpt_t or os.path.join(enc_dir, "enc_t.pth")
        if os.path.exists(ckpt_v):
            enc_v.load_state_dict(torch.load(ckpt_v, map_location=device, weights_only=True))
        if os.path.exists(ckpt_t):
            enc_t.load_state_dict(torch.load(ckpt_t, map_location=device, weights_only=True))
        enc_v.eval(); enc_t.eval()
        if meta.get("film_mode") in ("palora_enc", "simplex_enc_decomp_R"):
            # palora_enc / simplex4h*: FOUR SEPARATE plain heads, mixed at the
            # OUTPUT. palora_enc conditions the encoder too; simplex4h* keep the
            # encoder λ-blind and differ only in whether the mix happens before or
            # after each head's L2-normalisation.
            # Returning all four makes _apply_heads take its 4-head interpolation
            # branch, i.e. z(λ)=normalize(λ·z_R+(1-λ)·z_U) — exactly as trained.
            def _ld4(name):
                h = ProjectionHead(adim, proj_dim).to(device)
                fp = os.path.join(enc_dir, name)
                if os.path.exists(fp):
                    h.load_state_dict(torch.load(fp, map_location=device, weights_only=True))
                h.eval()
                return h
            return (enc_v, enc_t, _ld4("proj_v_r.pth"), _ld4("proj_t_r.pth"),
                    _ld4("proj_v_u.pth"), _ld4("proj_t_u.pth"))
        if meta.get("film_mode") == "simplex6h":
            # three plain heads per modality, saved as one ModuleDict each
            def _ld3(name):
                d = torch.nn.ModuleDict({k: ProjectionHead(adim, proj_dim)
                                         for k in ("r", "u1", "u2")}).to(device)
                fp = os.path.join(enc_dir, name)
                if os.path.exists(fp):
                    d.load_state_dict(torch.load(fp, map_location=device, weights_only=True))
                d.eval()
                return d
            return (enc_v, enc_t, _ld3("proj_v.pth"), _ld3("proj_t.pth"), None, None)
        HEAD_CLS = LoRATriProjectionHead if branches == 3 else LoRADualProjectionHead
        head_kw  = dict(rank=rank, rank_m=rank_m) if branches == 3 else dict(rank=rank)
        if _alpha:
            head_kw["alpha"] = _alpha
        def _load_h5(name):
            h = HEAD_CLS(adim, proj_dim, **head_kw).to(device)
            h.load_state_dict(torch.load(os.path.join(enc_dir, name), map_location=device, weights_only=True))
            h.eval()
            return h
        return (enc_v, enc_t, _load_h5("proj_v.pth"), _load_h5("proj_t.pth"), None, None)

    arch = meta.get("arch", "factorcl_based")

    # comm_based approach 3/4: TransformerSeqEncoder + FusionTransformer + LoRACommMLPHead
    if arch == "comm_based" and approach in (3, 4):
        fv, ft = FEAT_DIM[dataset]["vision"], FEAT_DIM[dataset]["text"]
        enc_v_raw = TransformerSeqEncoder(fv).to(device)
        enc_t_raw = TransformerSeqEncoder(ft).to(device)
        ckpt_v = enc_ckpt_v or os.path.join(enc_dir, "enc_v.pth")
        ckpt_t = enc_ckpt_t or os.path.join(enc_dir, "enc_t.pth")
        if os.path.exists(ckpt_v):
            enc_v_raw.load_state_dict(torch.load(ckpt_v, map_location=device, weights_only=True))
        if os.path.exists(ckpt_t):
            enc_t_raw.load_state_dict(torch.load(ckpt_t, map_location=device, weights_only=True))
        enc_v_raw.eval(); enc_t_raw.eval()
        fusion_xfmr = _make_fusion_xfmr(device)
        fx_path = os.path.join(enc_dir, "fusion_xfmr.pth")
        if os.path.exists(fx_path):
            fusion_xfmr.load_state_dict(torch.load(fx_path, map_location=device, weights_only=True))
        fusion_xfmr.eval()
        enc_v = _CommEncWrapper(enc_v_raw, fusion_xfmr)
        enc_t = _CommEncWrapper(enc_t_raw, fusion_xfmr)
        rank       = meta.get("lora_rank",    4)
        rank_m     = meta.get("lora_rank_m", rank)
        lora_lyrs  = meta.get("lora_layers", 3)
        if "lora_branches" in meta:
            branches = meta["lora_branches"]
        else:
            _sd = torch.load(os.path.join(enc_dir, "proj_v.pth"), map_location="cpu", weights_only=True)
            branches = 3 if any("A_m" in k for k in _sd) else 2
        def _load_lora_comm(name):
            h = LoRACommMLPHead(adim, 512, proj_dim, rank=rank, rank_m=rank_m, branches=branches, lora_layers=lora_lyrs).to(device)
            h.load_state_dict(torch.load(os.path.join(enc_dir, name), map_location=device, weights_only=True))
            h.eval()
            return h
        return (enc_v, enc_t, _load_lora_comm("proj_v.pth"), _load_lora_comm("proj_t.pth"), None, None)

    # comm_based approach 1/2: 4-head setup (simclr or factorcl_warmup)
    # gmc/comm/factorcl go through run_fixed, not here. Only simclr/factorcl_warmup reach _load_models.
    if arch == "comm_based":
        fv, ft = FEAT_DIM[dataset]["vision"], FEAT_DIM[dataset]["text"]
        enc_v_raw = TransformerSeqEncoder(fv).to(device)
        enc_t_raw = TransformerSeqEncoder(ft).to(device)
        ckpt_v = enc_ckpt_v or os.path.join(enc_dir, "enc_v.pth")
        ckpt_t = enc_ckpt_t or os.path.join(enc_dir, "enc_t.pth")
        if os.path.exists(ckpt_v):
            enc_v_raw.load_state_dict(torch.load(ckpt_v, map_location=device, weights_only=True))
        if os.path.exists(ckpt_t):
            enc_t_raw.load_state_dict(torch.load(ckpt_t, map_location=device, weights_only=True))
        enc_v_raw.eval(); enc_t_raw.eval()
        fusion_xfmr = _make_fusion_xfmr(device)
        fx_path = os.path.join(enc_dir, "fusion_xfmr.pth")
        if os.path.exists(fx_path):
            fusion_xfmr.load_state_dict(torch.load(fx_path, map_location=device, weights_only=True))
        fusion_xfmr.eval()
        enc_v = _CommEncWrapper(enc_v_raw, fusion_xfmr)
        enc_t = _CommEncWrapper(enc_t_raw, fusion_xfmr)
        cb_method = meta.get("method", "simclr_single_per_batch")
        if cb_method == "simclr_single_per_batch":
            def _load_head_cb(name):
                h = CommMLPHead(adim, 512, proj_dim).to(device)
                h.load_state_dict(torch.load(os.path.join(enc_dir, name), map_location=device, weights_only=True))
                h.eval(); return h
        else:  # factorcl_warmup — ProjectionHead (no BN)
            def _load_head_cb(name):
                h = ProjectionHead(adim, proj_dim).to(device)
                h.load_state_dict(torch.load(os.path.join(enc_dir, name), map_location=device, weights_only=True))
                h.eval(); return h
        return (enc_v, enc_t,
                _load_head_cb("proj_v_r.pth"), _load_head_cb("proj_t_r.pth"),
                _load_head_cb("proj_v_u.pth"), _load_head_cb("proj_t_u.pth"))

    enc_film_mode = enc_film_mode_for(film_mode, approach)
    enc_v, enc_t = _make_encoders(approach, dataset, enc_dim, device, enc_film_mode)

    ckpt_v = enc_ckpt_v or os.path.join(enc_dir, "enc_v.pth")
    ckpt_t = enc_ckpt_t or os.path.join(enc_dir, "enc_t.pth")
    if os.path.exists(ckpt_v):
        enc_v.load_state_dict(torch.load(ckpt_v, map_location=device, weights_only=True))
    if os.path.exists(ckpt_t):
        enc_t.load_state_dict(torch.load(ckpt_t, map_location=device, weights_only=True))
    enc_v.eval(); enc_t.eval()

    # approaches 3/4 use a single projection head per modality (proj_v_u=None signals this)
    if approach in (3, 4):
        if approach == 3:
            HEAD_CLS = DualFiLMProjectionHeadPreNorm if meta.get("proj_prenorm") else DualFiLMProjectionHead
            head_kw  = {}
        else:
            rank   = meta.get("lora_rank",   4)
            rank_m = meta.get("lora_rank_m", rank)
            if "lora_branches" in meta:
                branches = meta["lora_branches"]
            else:
                # auto-detect: peek at checkpoint keys; fall back to 3 (LoRA-curve default)
                _pv = os.path.join(enc_dir, "proj_v.pth")
                if os.path.exists(_pv):
                    _sd = torch.load(_pv, map_location="cpu", weights_only=True)
                    branches = 3 if any("A_m" in k for k in _sd) else 2
                else:
                    branches = 3
            HEAD_CLS = LoRATriProjectionHead if branches == 3 else LoRADualProjectionHead
            head_kw  = (dict(rank=rank, rank_m=rank_m) if branches == 3 else dict(rank=rank))
            if meta.get("lora_alpha"):      # match the α/r used at training time
                head_kw["alpha"] = meta["lora_alpha"]
        def _load_head4(name):
            h = HEAD_CLS(adim, proj_dim, **head_kw).to(device)
            h.load_state_dict(torch.load(os.path.join(enc_dir, name), map_location=device, weights_only=True))
            h.eval()
            return h
        return (enc_v, enc_t, _load_head4("proj_v.pth"), _load_head4("proj_t.pth"), None, None)
    else:
        def _load_head(name):
            h = ProjectionHead(adim, proj_dim).to(device)
            h.load_state_dict(torch.load(os.path.join(enc_dir, name), map_location=device, weights_only=True))
            h.eval()
            return h
        return (enc_v, enc_t,
                _load_head("proj_v_r.pth"), _load_head("proj_t_r.pth"),
                _load_head("proj_v_u.pth"), _load_head("proj_t_u.pth"))


# Helpers: feature extraction

def _pref_endpoints(film_mode):
    """The two preferences a CKA diagnostic should compare.

    For a 1-D λ these are the scalars 0.0 (pure unique) and 1.0 (pure shared).
    A simplex run has no such scalars — a preference is (λ_R, λ_U_vis, λ_U_txt) —
    and passing 0.0/1.0 anyway fell through to the scalar branch of
    _extract_backbone, whose encoder call returned None and crashed the whole
    probe before validation ever ran.

    The simplex analogues keep the same meaning:
      pure shared -> (1, 0, 0): both heads see (λ_R=1, λ_U=0)
      pure unique -> (0, .5, .5): vision sees (0, .5) and text (0, .5), i.e. both
        modalities fully on their OWN unique coordinate. The vertex (0,1,0) is NOT
        the right choice — it would hand text (0, 0) and leave it undefined.
    """
    if film_mode in ("simplex_enc_decomp_R", "simplex_proj_decomp_R",
                     "simplex_both_decomp_R"):
        # the R<->U axis: pure shared vs both modalities on their own unique block
        return (0.0, 0.5, 0.5), (1.0, 0.0, 0.0)
    if film_mode == "duo_txt":
        # 1-D controller on the (R, U_text) edge. The endpoints are the two ends of
        # THAT edge, not the simplex ones: (0,0,1) is pure text-uniqueness and is a
        # preference duo_txt actually trains on, whereas (0,.5,.5) is off its grid
        # entirely and would measure a mixture the model never saw.
        return (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
    if film_mode in ("simplex", "simplex4h", "simplex4h_norm", "simplex6h"):
        return (0.0, 0.5, 0.5), (1.0, 0.0, 0.0)
    return 0.0, 1.0


# Raw inputs cached per (dataset, split, task), keyed so a multitask run that
# switches label cannot read another task's y.
#
# WHY: a preference-conditioned encoder needs one forward per lambda, so
# _get_all_z calls _extract_backbone once for each of the 15 grid points, on each
# split. Rebuilding the loader inside it meant the MOSEI pickle was re-read and
# four dataloader workers were forked THIRTY times per probe -- minutes of pure
# I/O, and a worker that dies takes the whole job with it
# ("DataLoader worker exited unexpectedly"). The tensors are identical across
# lambda; only the encoder differs. So materialise them once and iterate the list.
_RAW_CACHE = {}
_RAW_CACHE_CAP = float(os.environ.get("PROBE_RAW_CACHE_GB", "8")) * (1024 ** 3)

# Dataloader workers. Default 0 -- the probe datasets are pickles already resident
# in RAM, so __getitem__ is a numpy index and worker processes buy nothing while
# costing a shared-memory segment per batch. On Jean-Zay that segment is what runs
# out: "DataLoader worker killed by signal: Bus error ... out of shared memory".
# Override with PROBE_NUM_WORKERS if a dataset ever needs real prefetching.
_NW = int(os.environ.get("PROBE_NUM_WORKERS", "0"))


def _build_probe_loader(dataset, split, batch_size):
    """One place that knows how to construct the probe loader for any dataset."""
    if is_multitask_dataset(dataset):
        return mt_probe_loader(dataset, _image_root(dataset), split, batch_size,
                               task=_MT_TASK[0], num_workers=_NW,
                               modalities=MODALITIES)
    if is_image_dataset(dataset):
        return image_probe_loader(dataset, _image_root(dataset), split, batch_size, _NW)
    ds = AffectDataset(_affect_path(dataset), dataset, split=split,
                       modalities=MODALITIES, task="classification")
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=_NW, pin_memory=True, collate_fn=collate_affect)


def _raw_batches(dataset, split, batch_size):
    """Iterable of batches for (dataset, split, task), materialised once if it fits.

    Returns a list of (v, t, y) when cached, or a freshly built loader when the
    tensors would exceed PROBE_RAW_CACHE_GB. The over-budget verdict is memoised as
    a None sentinel so we do not re-attempt materialisation -- and pay for a
    half-filled loader -- on every subsequent lambda.
    """
    key = (dataset, split, _MT_TASK[0] if is_multitask_dataset(dataset) else None)
    if key in _RAW_CACHE:
        cached = _RAW_CACHE[key]
        return cached if cached is not None else _build_probe_loader(dataset, split, batch_size)
    out, nbytes = [], 0
    for X, y in _build_probe_loader(dataset, split, batch_size):
        v = X[0].float().cpu(); t = X[1].float().cpu()
        nbytes += v.element_size() * v.nelement() + t.element_size() * t.nelement()
        if nbytes > _RAW_CACHE_CAP:
            print(f"  [probe] raw inputs for {dataset}/{split} exceed "
                  f"{_RAW_CACHE_CAP/1024**3:.1f} GB — streaming instead")
            _RAW_CACHE[key] = None
            return _build_probe_loader(dataset, split, batch_size)
        out.append((v, t, y))
    _RAW_CACHE[key] = out
    return out


@torch.no_grad()
def _extract_backbone(enc_v, enc_t, dataset, split, batch_size, device, lam=None):
    """Run encoder → (h_v, h_t, labels) on CPU. Pass lam for FiLM-conditioned encoders."""
    loader = _raw_batches(dataset, split, batch_size)
    h_vs, h_ts, ys = [], [], []
    for _item in tqdm(loader, desc=f"  encoder [{split}] lam={lam}", leave=False):
        X, y = ((_item[0], _item[1]), _item[2]) if len(_item) == 3 else _item
        v = X[0].float().to(device)
        t = X[1].float().to(device)
        if isinstance(lam, tuple) and len(lam) == 3 and _SIMPLEX_ENC[0]:
            # simplex-enc: vision encoder gets (λ_R, λ_U_vision), text (λ_R, λ_U_text)
            h_vs.append(enc_v.forward_mix(v, lam[0], lam[1]).cpu())
            h_ts.append(enc_t.forward_mix(t, lam[0], lam[2]).cpu())
        elif isinstance(lam, tuple):
            # simplex on the heads only — the encoder is λ-blind, call it plainly
            h_vs.append(enc_v(v).cpu())
            h_ts.append(enc_t(t).cpu())
        else:
            o_v = enc_v(v, lam) if lam is not None else enc_v(v)
            o_t = enc_t(t, lam) if lam is not None else enc_t(t)
            if o_v is None or o_t is None:
                raise RuntimeError(
                    f"encoder returned None for lam={lam!r} (type {type(lam).__name__}). "
                    f"This means a scalar λ was passed to a preference-conditioned "
                    f"encoder, or vice versa — check _pref_endpoints/film_mode wiring.")
            h_vs.append(o_v.cpu())
            h_ts.append(o_t.cpu())
        ys.append(y.numpy().reshape(-1) if isinstance(y, torch.Tensor) else np.asarray(y).reshape(-1))
    return torch.cat(h_vs), torch.cat(h_ts), np.concatenate(ys)


def _neutral_mask(dataset, split):
    """Non-neutral Acc2 support (mosi/mosei only).

    Returns a boolean array aligned with the probe feature order — True where the
    *continuous* sentiment != 0 (i.e. strictly positive or strictly negative). On
    the non-neutral subset the existing (>0 → 1, ≤0 → 0) binarization already equals
    the sign-based labels, so callers just subset with this mask. Returns None for
    datasets without a continuous-sentiment / neutral notion (humor, mustard, images).
    """
    if dataset not in ("mosi", "mosei"):
        return None
    mb = AffectDataset(_affect_path(dataset), dataset, split=split,
                       modalities=MODALITIES, task="classification")
    raw = np.asarray(mb.data["labels"]).reshape(-1)
    return raw != 0.0


def _acc2_variants(z_tr, l_tr, z_te, l_te, dataset, tr_splits=("valid",)):
    """Return {zero_included, non_neutral(optional), n_dropped_*} test-accuracy dict.

    zero_included = current metric (neutrals kept in the negative class).
    non_neutral   = drop sentiment==0 from BOTH train and test, then Acc2 by sign.
    """
    m = _lr_metrics(z_tr, l_tr, z_te, l_te)
    out = {"zero_included": m["acc"],
           "lr_C": _LAST_C[0], "lr_C_at_grid_edge": _LAST_C_AT_EDGE[0],
           "lr_C_grid": list(_lr_grid()), "scaler": _SCALER[0],
           "balanced_acc": m["balanced_acc"],
           "macro_f1":     m["macro_f1"],
           "auc":          m["auc"],
           "majority":     m["majority"],
           "class_weight": m["class_weight"]}
    # The mask must describe the rows actually in z_tr. It was hard-wired to "valid",
    # and the length guard below then SILENTLY dropped non-neutral whenever the probe
    # was fit on anything else -- no error, just a missing number.
    _ms = [_neutral_mask(dataset, _s) for _s in tr_splits]
    m_tr = None if any(_x is None for _x in _ms) else np.concatenate(_ms)
    m_te = _neutral_mask(dataset, "test")
    if m_tr is not None and m_te is not None and len(m_tr) == len(l_tr) and len(m_te) == len(l_te):
        acc_nn = _lr_accuracy(z_tr[m_tr], l_tr[m_tr], z_te[m_te], l_te[m_te]) * 100
        out["non_neutral"]   = round(acc_nn, 2)
        out["n_dropped_val"]  = int((~m_tr).sum())
        out["n_dropped_test"] = int((~m_te).sum())
    return out


def _get_all_z(enc_v, enc_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u,
               dataset, split, film_mode, batch_size, device, modality="both"):
    """
    Compute z(λ) for all λ in LAMBDA_GRID.
    film_mode=none : encoder runs once, heads interpolated per λ
    film_mode=token/full : encoder re-runs with each λ (more expensive but necessary)
    Returns: z_dict {lam: np.ndarray}, labels np.ndarray
    """
    FIXED_LAM_MODALITIES = ("shared", "shared_vision", "shared_text", "unique_vision", "unique_text")
    if film_mode in ("simplex", "simplex4h", "simplex4h_norm", "simplex6h", "duo_txt",
                     "simplex_enc_decomp_R", "simplex_proj_decomp_R",
                     "simplex_both_decomp_R"):
        # (λ_R, λ_U_vision, λ_U_text) triangular grid — same one used in training
        grid = (duo_grid(_SIMPLEX_SIDE[0]) if film_mode == "duo_txt"
                else simplex_grid(_SIMPLEX_SIDE[0]))
    else:
        grid = [1.0] if modality in FIXED_LAM_MODALITIES else LAMBDA_GRID
    # λ-blind encoder: none (ap1/2) or proj (ap3) — extract backbone once
    # λ-conditioned encoder: token/full (ap1/2/3) — re-run per λ
    # Only the probe's FITTING split may fit a PCA read-out basis; test reuses it.
    # Set here because _apply_heads never sees the split name.
    _PCA_CAN_FIT[0] = (split != "test")
    enc_is_blind = film_mode in ("none", "proj", "palora_proj")
    if enc_is_blind:
        h_v, h_t, labels = _extract_backbone(enc_v, enc_t, dataset, split, batch_size, device)
        z_dict = {lam: _apply_heads(h_v, h_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u, lam, device,
                                     modality=modality, film_mode=film_mode)
                  for lam in grid}
    else:
        z_dict = {}
        labels = None
        for lam in grid:
            h_v, h_t, l = _extract_backbone(enc_v, enc_t, dataset, split, batch_size, device, lam=lam)
            z_dict[lam] = _apply_heads(h_v, h_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u, lam, device,
                                        modality=modality, film_mode=film_mode)
            if labels is None:
                labels = l
    return z_dict, labels


# which coordinates of a block survive the dimension budget
#
# The budget itself (how many dims each of R/U1/U2 gets) is set by lambda. This
# only decides WHICH coordinates of a block are kept once that number is fixed.
#
#   dim       first n coordinates. Deterministic and parameter-free, but arbitrary:
#             the heads are plain MLPs, so nothing orders their coordinates and the
#             first n is effectively a fixed random subset.
#   dim_rand  a different fixed random subset. The control for "dim": if the two
#             agree, coordinate choice is immaterial and the arbitrariness is benign.
#   dim_pca   top n principal components of the block. Retains the most variance for
#             a given budget, at the cost of a fitted read-out.
#
# LEAKAGE GUARD. A PCA basis must be fitted on the split the probe is FITTED on and
# then applied unchanged to test. _PCA_CAN_FIT is set by _get_all_z from the split
# name, so a basis can never be fitted on test: if one is missing when scoring test,
# this raises instead of silently refitting.
_PCA_BASIS = {}
_PCA_CAN_FIT = [False]
# Fixed comparison width for the paper: every method is reported at 64 dims, so a
# method whose native embedding is wider also gets a PCA-reduced number at 64.
MATCH_DIM    = 64
# --save_z <path.npz>: dump the exact representations run_fixed probes, so offline
# analyses (CKA spread, per-task information profile) use the SAME z as the reported
# numbers rather than a re-implementation of the extraction.
SAVE_Z = [None]
# Where the linear probe is FIT. "valid" (default) is the historical protocol and keeps
# every published number fixed. It is non-standard: on MOSI it fits a 64-d probe on 229
# samples and selects lambda from 46. Standard linear evaluation (CLIP, SimCLR) fits on
# TRAIN, chooses C and lambda on VALID, reports once on TEST:
#   "train"    : fit on train; C and lambda chosen on the full valid split
#   "trainval" : same selection, then the final probe is refit on train+valid
# Every method compared on a dataset MUST use the same mode.
_PROBE_FIT = ["valid"]
# When set to (X_fit, y_fit, X_sel, y_sel), _lr_predict chooses C by fitting on X_fit and
# scoring on X_sel -- an explicit validation set -- instead of an internal 80/20 split.
_C_SEL = [None]
_PCA_SHARED  = [False]    # False = one basis per (lambda, block) -- the
                          # original behaviour and the numbers reported.
                          # True shares one basis per block across lambda:
                          # tested, and it did NOT help (R2 0.264 -> 0.230,
                          # p 0.147 -> 0.304), so it is opt-in only.
_DIM_RAND_SEED = 0

# Per-coordinate standardisation, on by default (the original protocol).
#
# WHY IT IS A SWITCH: StandardScaler divides every coordinate by its own sd. In a
# PCA basis those coordinates ARE the principal components, so that division is
# exactly a whitening -- it rescales the weakest directions up to match the
# strongest, amplifying noise. There is no "whitening step" inside dim_pca to
# remove; the whitening IS the scaler. Turning it off lets the eigenvalue ordering
# survive into the classifier, which is what selecting principal directions is
# supposed to buy.
#
# Set it for ALL readouts when comparing them, never for one: a readout scored
# without the scaler against one scored with it confounds the readout with the
# protocol.
_SCALER = ["standard"]        # standard | center | none
_TUNE_C  = [False]            # tune C in the lambda sweep (off = published behaviour)
_WIDE_C  = [False]            # narrow C grid (1e-2, 0.1, 1, 10) is the DEFAULT and is
                              # what every reported number uses. The wide grid was
                              # tested end-to-end: it changed no result by more than
                              # 0.6 points, so the saturation it fixes is empirically
                              # inert. --wide_c opts in.
_LAST_C  = [None]             # C chosen by the most recent _lr_predict
_LAST_C_AT_EDGE = [False]     # ... and whether it sat on a grid endpoint


class _NoScaler:
    """StandardScaler interface, identity transform."""
    def fit_transform(self, X): return X
    def transform(self, X):     return X


def _scaler():
    """Probe preprocessing.

      standard  StandardScaler()                       -- centre and divide by sd
      center    StandardScaler(with_std=False)         -- centre only
      none      identity

    'center' is the right control for dim_pca: dividing by the per-coordinate sd is
    exactly a whitening when the coordinates are principal components, which erases
    the eigenvalue ordering that selecting principal directions is meant to provide.
    Centring keeps that ordering while still giving the solver a zero-mean input.
    """
    m = _SCALER[0]
    if m == "standard":
        return StandardScaler()
    if m == "center":
        return StandardScaler(with_mean=True, with_std=False)
    if m == "none":
        return _NoScaler()
    raise ValueError(f"unknown scaler {m!r}")


# Regularisation grid. Without the sd division the feature scale can shift by orders
# of magnitude, so the optimum leaves the standardised grid; widening it for the
# non-standard scalers keeps C a tuned nuisance parameter rather than a confound
# that silently favours whichever arm happens to match the old strength.
# Widened after instrumentation showed C saturating: with the published grid
# (0.01, 0.1, 1.0) two thirds of fits pinned to the top endpoint, i.e. the optimum
# lay above it and the probe was systematically over-regularised. Extending to 1e3
# still left ~47% at the ceiling, so the range now reaches 1e6. That is expected
# rather than pathological -- z is L2-normalised, so the penalty acts on
# tiny-norm features and needs a large C to be weak. Any grid whose selected-C
# histogram peaks at an ENDPOINT is mis-specified: the value is recorded so this
# is checkable rather than invisible.
# Spans BOTH regimes. Instrumentation showed standard-scaled features preferring
# C <= 1 while centre-only features prefer 1e2-1e3, so a grid covering only one
# end truncates the other and the two protocols are not comparably tuned. Every
# method must find an INTERIOR optimum here; check lr_C_hist before trusting any
# comparison built on top of it.
_LR_GRID_WIDE = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4)


def _lr_grid():
    return _LR_GRID_WIDE if _WIDE_C[0] else (
        LR_CS if _SCALER[0] == "standard" else (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3))


def reset_readout_state():
    """Clear the fitted PCA bases. Call between checkpoints."""
    _PCA_BASIS.clear()


def _cka_uses_dim_slice(fn):
    """Force the unfitted 'dim' slice for any CKA routine.

    CKA compares z(lambda_i) against z(lambda_j). Under dim_pca each lambda would
    carry its own fitted basis, so the similarity would partly measure the
    difference between BASES rather than between representations. The CKA routines
    also extract the TEST split directly and run before any validation sweep, so a
    fitted read-out could only obtain a basis by fitting on test -- which the
    leakage guard correctly refuses.

    Applied as a decorator rather than at the call site: there are three CKA entry
    points (run_cka, run_cka_curve, run_cka_matrix) reachable from several modes,
    and guarding one of them is how this was missed the first time.
    """
    @functools.wraps(fn)
    def _wrapped(*a, **kw):
        saved = _READOUT[0]
        if saved.startswith("dim"):
            _READOUT[0] = "dim"
        try:
            return fn(*a, **kw)
        finally:
            _READOUT[0] = saved
    return _wrapped


def _select_dims(B, n, readout, key):
    if readout == "dim":
        return B[:, :n]
    if readout == "dim_rand":
        g = torch.Generator().manual_seed(_DIM_RAND_SEED + 1000 * key[1])
        idx = torch.randperm(B.shape[-1], generator=g)[:n].sort().values
        return B[:, idx]
    if readout == "dim_pca":
        # Default: one basis per (lambda, block) -- 45 on a 15-point grid. This puts
        # z(lam_i) and z(lam_j) in different coordinate systems, which looked like a
        # likely cause of dim_pca's compressed between-task separation. Sharing one
        # basis per block was implemented and tested (--pca_shared) and did NOT help:
        # between-task 0.481 -> 0.487 but within-task 0.359 -> 0.398, so R2 fell
        # 0.264 -> 0.230 and p rose 0.147 -> 0.304. The weakness is intrinsic to
        # ordering coordinates by variance, not to how the bases are cached.
        bkey = key[1] if _PCA_SHARED[0] else key
        if bkey not in _PCA_BASIS:
            if not _PCA_CAN_FIT[0]:
                raise RuntimeError(
                    f"dim_pca: no basis for block {key[1]} (key {bkey!r}) and fitting "
                    f"is disabled on this split -- refusing to fit on test.")
            X = B - B.mean(0, keepdim=True)
            # eigh on the (D x D) covariance: D=64, so this is microseconds.
            C = (X.T @ X) / max(len(X) - 1, 1)
            _, V = torch.linalg.eigh(C.double())
            _PCA_BASIS[bkey] = (B.mean(0, keepdim=True), V.flip(-1).float())
        mu, V = _PCA_BASIS[bkey]
        return (B - mu) @ V[:, :n]
    raise ValueError(f"unknown readout {readout!r}")


@torch.no_grad()
def _apply_heads(h_v, h_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u,
                 lam: float, device: str, chunk: int = 512,
                 modality: str = "both", film_mode: str = "none"):
    """Apply projection heads at fixed λ → numpy representation.

    film_mode="proj": proj_v_r / proj_t_r are DualFiLMProjectionHead — called with explicit lam.
      shared_vision / shared = λ=1.0, unique_vision / unique_text = λ=0.0, rest sweep lam.

    Other film_modes: 4-head interpolation as before.
      both          = concat(z_v, z_t)  [default, λ swept]
      vision        = z_v only          [λ swept]
      text          = z_t only          [λ swept]
      shared        = concat(proj_v_r, proj_t_r)  [λ=1 forced, both shared heads]
      shared_vision = proj_v_r(hv) only           [vision shared head alone]
      shared_text   = proj_t_r(ht) only           [text shared head alone]
      unique_vision = proj_v_u(hv) only           [vision unique head alone]
      unique_text   = proj_t_u(ht) only           [text unique head alone]
    """
    zs, n = [], h_v.shape[0]
    _blocks = []          # decomp_R only: (R, U1, U2, R_vision, R_text) per chunk, full width
    for i in range(0, n, chunk):
        hv = h_v[i:i+chunk].to(device)
        ht = h_t[i:i+chunk].to(device)
        if film_mode in ("simplex_enc_decomp_R", "simplex_proj_decomp_R",
                     "simplex_both_decomp_R"):
            # component-wise readout
            # NOT a weighted sum into one space (that is what leaks): three
            # SEPARATE blocks, each scaled by its own coefficient.
            #
            #   z(λ) = normalize([ s_R·R̂ ‖ s_U1·Û1 ‖ s_U2·Û2 ])
            #   R̂ = normalize( (r1(h1) + r2(h2)) / 2 )
            #
            # At λ=(0,1,0) the R and U2 blocks are EXACTLY zero, so only
            # M1-unique content reaches the probe. Under simplex6h the text
            # stream survived every preference, which is why selecting the
            # provably-empty vision axis cost nothing.
            #
            # SCALING: s = sqrt(λ), not λ. A linear probe responds to variance,
            # and energy goes as amplitude², so amplitude=λ would give energy
            # shares ∝ λ² -- (0.25,0.50,0.25) would land at (1/6, 2/3, 1/6)
            # rather than (1/4, 1/2, 1/4). With sqrt(λ) the energy share of each
            # block equals λ exactly, which is what "the representation contains
            # λ_t of component t" should mean. Vertices are unaffected.
            #
            # R is re-normalised after averaging: two unit vectors at angle θ
            # average to norm cos(θ/2) < 1, which would systematically
            # under-weight the shared block at every preference.
            w_r, w_u1, w_u2 = lam
            s_r, s_u1, s_u2 = math.sqrt(w_r), math.sqrt(w_u1), math.sqrt(w_u2)
            if film_mode in ("simplex_proj_decomp_R", "simplex_both_decomp_R"):
                # LoRA heads carry the preference (the encoder is λ-blind), and each
                # modality sees only its OWN unique coefficient — the same routing
                # used in training, or the probe evaluates a model that was never fit.
                _rv = proj_v_r.forward_mix(hv, w_r, w_u1)
                _rt = proj_t_r.forward_mix(ht, w_r, w_u2)
                U1  = proj_v_u.forward_mix(hv, w_r, w_u1)
                U2  = proj_t_u.forward_mix(ht, w_r, w_u2)
            else:
                _rv, _rt = proj_v_r(hv), proj_t_r(ht)
                U1, U2   = proj_v_u(hv), proj_t_u(ht)
            R = F.normalize((_rv + _rt) / 2.0, dim=-1)
            # R AVERAGES the two per-modality shared vectors, so the probe never sees
            # modality-resolved shared content -- the component where vision and text
            # disagree is destroyed. FactorCL keeps its shared pair concatenated, which
            # is the one structural difference left after width, the dim budget and
            # block-dropping were ruled out. Keep _rv/_rt so readout="dim_split" can
            # concatenate them instead. Averaging stays the default.
            # The dimension-selecting read-outs need the WHOLE split before they can
            # act (a PCA basis cannot be fitted on a 512-row chunk), so the blocks are
            # collected here and the read-out is applied once after the loop. Every
            # read-out is row-wise, so this is bit-identical to applying it per chunk.
            _blocks.append((R.cpu(), U1.cpu(), U2.cpu(), _rv.cpu(), _rt.cpu()))
            continue
        if isinstance(proj_v_r, torch.nn.ModuleDict):
            # simplex6h: three plain heads per modality, true 3-way blend. Keyed on
            # the head type rather than on lam, because a scalar lam would otherwise
            # fall through to the DualFiLM branch and call ModuleDict(hv, lam).
            w_r, w_u1, w_u2 = (lam if isinstance(lam, (tuple, list)) and len(lam) == 3
                               else (float(lam), (1.0 - float(lam)) / 2, (1.0 - float(lam)) / 2))
            _m3 = lambda pd, h: F.normalize(w_r * pd["r"](h) + w_u1 * pd["u1"](h)
                                            + w_u2 * pd["u2"](h), dim=-1)
            z_v, z_t = _m3(proj_v_r, hv), _m3(proj_t_r, ht)
            if modality in ("both", "shared"):
                zs.append(torch.cat([z_v, z_t], dim=-1).cpu().numpy())
            elif modality in ("text", "shared_text", "unique_text"):
                zs.append(z_t.cpu().numpy())
            else:
                zs.append(z_v.cpu().numpy())
        elif (proj_v_u is not None and isinstance(lam, tuple) and len(lam) == 3
                and film_mode in ("simplex4h", "simplex4h_norm", "duo_txt")):
            # 4-head simplex: vision mixes its two heads with (w_R, w_U_vision),
            # text with (w_R, w_U_text). simplex4h_norm mixes the RAW head outputs
            # and normalises once; simplex4h blends the already-unit vectors.
            w_r, w_u1, w_u2 = lam
            # same degenerate-vertex guard as training: (0,0) -> equal blend
            wv_r, wv_u = (0.5, 0.5) if w_r + w_u1 <= 0 else (w_r, w_u1)
            wt_r, wt_u = (0.5, 0.5) if w_r + w_u2 <= 0 else (w_r, w_u2)
            if film_mode in ("simplex4h_norm", "duo_txt"):   # mix RAW, normalise once
                z_v = F.normalize(wv_r * proj_v_r.net(hv) + wv_u * proj_v_u.net(hv), dim=-1)
                z_t = F.normalize(wt_r * proj_t_r.net(ht) + wt_u * proj_t_u.net(ht), dim=-1)
            else:
                z_v = F.normalize(wv_r * proj_v_r(hv) + wv_u * proj_v_u(hv), dim=-1)
                z_t = F.normalize(wt_r * proj_t_r(ht) + wt_u * proj_t_u(ht), dim=-1)
            if modality in ("both", "shared"):
                zs.append(torch.cat([z_v, z_t], dim=-1).cpu().numpy())
            elif modality in ("text", "shared_text", "unique_text"):
                zs.append(z_t.cpu().numpy())
            else:
                zs.append(z_v.cpu().numpy())
        elif proj_v_u is None and isinstance(lam, tuple) and len(lam) == 3:
            # simplex: vision head sees (λ_R, λ_U_vision), text head (λ_R, λ_U_text)
            w_r, w_u1, w_u2 = lam
            z_v = proj_v_r.forward_mix(hv, w_r, w_u1)
            z_t = proj_t_r.forward_mix(ht, w_r, w_u2)
            if modality in ("both", "shared"):
                zs.append(torch.cat([z_v, z_t], dim=-1).cpu().numpy())
            elif modality in ("text", "shared_text", "unique_text"):
                zs.append(z_t.cpu().numpy())
            else:
                zs.append(z_v.cpu().numpy())
        elif proj_v_u is None:
            # DualFiLMProjectionHead (approach 3, any film_mode): proj_v_r=vision head, proj_t_r=text head
            lam_v = 1.0 if modality in ("shared", "shared_vision") else (0.0 if modality == "unique_vision" else lam)
            lam_t = 1.0 if modality in ("shared", "shared_text")   else (0.0 if modality == "unique_text"   else lam)
            z_v = proj_v_r(hv, lam_v)   # DualFiLMProjectionHead already L2-normalises
            z_t = proj_t_r(ht, lam_t)
            if modality in ("both", "shared"):
                zs.append(torch.cat([z_v, z_t], dim=-1).cpu().numpy())
            elif modality in ("text", "shared_text", "unique_text"):
                zs.append(z_t.cpu().numpy())
            else:   # vision / shared_vision / unique_vision
                zs.append(z_v.cpu().numpy())
        elif modality == "shared_vision":
            zs.append(F.normalize(proj_v_r(hv), dim=-1).cpu().numpy())
        elif modality == "shared_text":
            zs.append(F.normalize(proj_t_r(ht), dim=-1).cpu().numpy())
        elif modality == "unique_vision":
            zs.append(F.normalize(proj_v_u(hv), dim=-1).cpu().numpy())
        elif modality == "unique_text":
            zs.append(F.normalize(proj_t_u(ht), dim=-1).cpu().numpy())
        else:
            eff_lam = 1.0 if modality == "shared" else lam
            z_v = F.normalize(eff_lam * proj_v_r(hv) + (1.0 - eff_lam) * proj_v_u(hv), dim=-1)
            z_t = F.normalize(eff_lam * proj_t_r(ht) + (1.0 - eff_lam) * proj_t_u(ht), dim=-1)
            if modality == "vision":
                zs.append(z_v.cpu().numpy())
            elif modality == "text":
                zs.append(z_t.cpu().numpy())
            else:  # both or shared
                zs.append(torch.cat([z_v, z_t], dim=-1).cpu().numpy())
    if _blocks:
        # decomp_R read-out, applied once on the full split
        R  = torch.cat([b[0] for b in _blocks])
        U1 = torch.cat([b[1] for b in _blocks])
        U2 = torch.cat([b[2] for b in _blocks])
        Rv = torch.cat([b[3] for b in _blocks])
        Rt = torch.cat([b[4] for b in _blocks])
        ro = _READOUT[0]
        if ro == "none":
            # lambda already shaped R/U1/U2 through W(lambda); apply nothing on top.
            # Every preference yields the same 3*proj_dim width, so the support-
            # masking artefact that dominates "amp" cannot occur.
            return F.normalize(torch.cat([R, U1, U2], dim=-1), dim=-1).numpy()
        if ro == "amp":
            _w = lam if isinstance(lam, (tuple, list)) else (float(lam), 0.0, 0.0)
            s_r, s_u1, s_u2 = (math.sqrt(max(c, 0.0)) for c in _w)
            return F.normalize(torch.cat([s_r * R, s_u1 * U1, s_u2 * U2], dim=-1),
                               dim=-1).numpy()
        # dim / dim_rand / dim_pca all allocate the SAME per-block budget; they
        # differ only in WHICH coordinates of each block are kept.
        w_r, w_u1, w_u2 = lam
        D = R.shape[-1]
        raw  = [w_r * D, w_u1 * D, w_u2 * D]
        base = [int(x) for x in raw]
        for _i in sorted(range(3), key=lambda i: raw[i] - base[i],
                         reverse=True)[:D - sum(base)]:
            base[_i] += 1
        if ro == "dim_split":
            # Same per-block budget as "dim"; the R share is split evenly between the
            # vision and text shared vectors instead of spent on their mean. Total
            # width is unchanged, so this is a like-for-like swap of averaging for
            # concatenation.
            n_r = base[0]
            n_rv = (n_r + 1) // 2
            parts = []
            for B, nb, bi in ((Rv, n_rv, 0), (Rt, n_r - n_rv, 1), (U1, base[1], 2), (U2, base[2], 3)):
                if nb > 0:
                    parts.append(_select_dims(B, nb, "dim", key=(tuple(lam), bi)))
            return F.normalize(torch.cat(parts, dim=-1), dim=-1).numpy()
        parts = []
        for bi, (B, nb) in enumerate(zip((R, U1, U2), base)):
            if nb <= 0:
                continue
            parts.append(_select_dims(B, nb, ro, key=(tuple(lam), bi)))
        return F.normalize(torch.cat(parts, dim=-1), dim=-1).numpy()
    return np.concatenate(zs)


# Helpers: classifiers

def _majority_baseline(labels):
    """Accuracy of always predicting the most frequent class (%).

    The floor any classifier must clear. On an imbalanced label this is high by
    construction — MOSEI `disgust` is 13% positive, so a constant predictor scores
    87% and a plain-accuracy number below that means the probe learned nothing
    useful, however large it looks.
    """
    l = np.asarray(labels).reshape(-1)
    if len(l) == 0:
        return float("nan")
    _, counts = np.unique(l, return_counts=True)
    return float(counts.max() / len(l)) * 100


def _lr_metrics(tr_z, tr_l, te_z, te_l, seed=42):
    """Prevalence-robust metrics for one train/test split.

    Plain accuracy is not comparable across labels with different positive rates,
    so balanced accuracy (mean per-class recall; chance = 50% regardless of
    prevalence) and macro-F1 are reported alongside it, with the majority-class
    floor for reference.
    """
    pred, clf, sc = _lr_predict(tr_z, tr_l, te_z, seed=seed)
    te_l = np.asarray(te_l).reshape(-1)
    # AUC is threshold-free: it says whether the representation RANKS positives
    # above negatives, independently of where the decision boundary sits. AUC~0.5
    # with low balanced accuracy means the features are uninformative; AUC>>0.5
    # with low balanced accuracy means only the threshold is wrong.
    auc = float("nan")
    try:
        if len(np.unique(te_l)) == 2:
            auc = float(roc_auc_score(te_l, clf.predict_proba(sc.transform(te_z))[:, 1]))
    except Exception:
        pass
    return {
        "acc":          round(float((pred == te_l).mean()) * 100, 2),
        "balanced_acc": round(float(balanced_accuracy_score(te_l, pred)) * 100, 2),
        "macro_f1":     round(float(f1_score(te_l, pred, average="macro",
                                             zero_division=0)) * 100, 2),
        "auc":          (round(auc * 100, 2) if auc == auc else None),
        "majority":     round(_majority_baseline(te_l), 2),
        "class_weight": "balanced" if _is_skewed(tr_l) else None,
    }


def _is_skewed(labels, thresh=55.0):
    """True when the majority class exceeds `thresh`% — i.e. plain accuracy and an
    unweighted loss are both dominated by one class.

    Threshold is 55, not 60. MOSI's majority class is 59.62%, which cleared the old
    60.0 bar by 0.38 points and so was probed with plain accuracy and an unweighted
    classifier. The consequence was visible in the results: 5-seed Acc2
    zero-included came out at 59.07 +- 0.87, i.e. *below* the 59.62 constant
    predictor, while balanced accuracy on the same runs was 58.24 against a 50.0
    chance floor. At 55 MOSI is handled correctly; MOSEI (51.0) and humor (50.66)
    stay unweighted, so no previously reported number on those datasets moves."""
    return _majority_baseline(labels) >= thresh


def _lr_predict(tr_z, tr_l, te_z, seed=42):
    """Tune C on 20% of train, refit on full train, return test PREDICTIONS.

    Split out from _lr_accuracy so accuracy and the prevalence-robust metrics are
    computed from one and the same fitted classifier rather than two independent
    fits.
    """
    if _C_SEL[0] is not None:
        _Xf, _yf, _Xs, _ys = _C_SEL[0]
        _yf = np.asarray(_yf).reshape(-1); _ys = np.asarray(_ys).reshape(-1)
        _ssc = _scaler(); _Xf_s = _ssc.fit_transform(_Xf); _Xs_s = _ssc.transform(_Xs)
        _skew = _is_skewed(_yf); _cw0 = "balanced" if _skew else None
        _grid = _lr_grid(); best_C, best_val = _grid[0], -1.0
        for C in _grid:
            _c = LogisticRegression(C=C, max_iter=1000, random_state=seed,
                                    class_weight=_cw0).fit(_Xf_s, _yf)
            v = (balanced_accuracy_score(_ys, _c.predict(_Xs_s)) if _skew
                 else _c.score(_Xs_s, _ys))
            if v > best_val:
                best_val, best_C = v, C
        _LAST_C[0] = best_C
        _LAST_C_AT_EDGE[0] = best_C in (_grid[0], _grid[-1])
        sc = _scaler(); tr = sc.fit_transform(tr_z); te = sc.transform(te_z)
        clf = LogisticRegression(C=best_C, max_iter=1000, random_state=seed,
                                 class_weight="balanced" if _is_skewed(tr_l) else None)
        clf.fit(tr, tr_l)
        return clf.predict(te), clf, sc
    sc = _scaler()
    tr = sc.fit_transform(tr_z)
    te = sc.transform(te_z)
    n  = len(tr)
    rng = np.random.default_rng(seed)
    val_idx   = rng.choice(n, size=max(1, n // 5), replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)
    # On a skewed label an unweighted logistic loss is dominated by the majority
    # class, so the fitted model under-predicts positives and balanced accuracy
    # collapses to ~50% EVEN IF the features are informative. Tuning C on plain
    # accuracy compounds it by rewarding whichever C best mimics the majority.
    # Both are corrected when the label is skewed.
    skew = _is_skewed(tr_l)
    cw = "balanced" if skew else None
    score = ((lambda c, X, y: balanced_accuracy_score(y, c.predict(X))) if skew
             else (lambda c, X, y: c.score(X, y)))
    _grid = _lr_grid()
    best_C, best_val = _grid[0], -1.0
    for C in _grid:
        clf = LogisticRegression(C=C, max_iter=1000, random_state=seed, class_weight=cw)
        clf.fit(tr[train_idx], tr_l[train_idx])
        v = score(clf, tr[val_idx], tr_l[val_idx])
        if v > best_val:
            best_val, best_C = v, C
    # Record it: a C pinned to an endpoint of the grid means the optimum lies
    # OUTSIDE it, so the probe is mis-regularised and any A/B on top of it is
    # confounded. Cannot be diagnosed unless the value is kept.
    _LAST_C[0] = best_C
    _LAST_C_AT_EDGE[0] = best_C in (_grid[0], _grid[-1])
    clf = LogisticRegression(C=best_C, max_iter=1000, random_state=seed, class_weight=cw)
    clf.fit(tr, tr_l)
    return clf.predict(te), clf, sc


def _lr_predict_labels(tr_z, tr_l, te_z, seed=42):
    return _lr_predict(tr_z, tr_l, te_z, seed=seed)[0]


def _lr_accuracy(tr_z, tr_l, te_z, te_l, seed=42):
    """Tune C on 20% of train, fit on full train, return test Top-1 accuracy."""
    pred = _lr_predict_labels(tr_z, tr_l, te_z, seed=seed)
    return float((pred == np.asarray(te_l).reshape(-1)).mean())


def _knn_predict(z_support, l_support, z_query, k):
    """kNN (cosine, z already L2-normalised)."""
    k   = min(k, len(z_support))
    sim = z_query @ z_support.T
    top = np.argsort(sim, axis=1)[:, -k:]
    return np.array([np.bincount(l_support[row], minlength=2).argmax() for row in top])


def _knn_loo_acc(z_support, l_support, k):
    """Leave-one-out kNN accuracy on support — used for λ selection in few-shot."""
    n   = len(z_support)
    k   = min(k, n - 1)
    sim = z_support @ z_support.T
    np.fill_diagonal(sim, -np.inf)
    top  = np.argsort(sim, axis=1)[:, -k:]
    pred = np.array([np.bincount(l_support[row], minlength=2).argmax() for row in top])
    return float((pred == l_support).mean())


# Fixed baseline mode (gmc / comm / factorcl — no lambda sweep)

def run_fixed(enc_dir, dataset, approach, enc_dim, proj_dim, batch_size, device,
              matched_only: bool = False):
    """matched_only=True → FAIR-COMPARISON protocol, identical to the λ-methods:
    probe fitted on the VALID split, C tuned the same way (_lr_accuracy), plain
    top-1 accuracy on test, plus the Acc2 variants. The train-fitted /
    balanced-accuracy CoMM protocol is skipped entirely (not just hidden), so a
    λ-blind baseline and a λ-method are evaluated by exactly the same rule.
    """
    """
    Probe with a single fixed representation — no lambda sweep.

    factorcl_based arch:
      gmc / comm   : z = FusionEncoder(enc_v(v), enc_t(t))
      factorcl     : z = concat(proj_v_r(enc_v(v)), proj_t_r(enc_t(t)))
                     also computes CKA(z_R, z_U) if proj_v_u.pth / proj_t_u.pth exist.

    comm_based arch:
      gmc / comm   : z = mlp_head(FusionTransformer([enc_v(v), enc_t(t)]))   (joint)
      factorcl     : z = concat(proj_v_r(FusionTransformer([enc_v(v)])),
                                proj_t_r(FusionTransformer([enc_t(t)])))
    """
    _has_u    = False  # set True for factorcl_based factorcl when U heads exist
    _extract_u = None
    _extract_r = None
    _extract_group = None

    meta_path = os.path.join(enc_dir, "train_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    method   = meta.get("method",   "gmc")
    # train_meta must NOT override the dataset: it records what the model was TRAINED
    # on, while `dataset` is what we are now evaluating against. Conflating them made
    # every mosei_multitask probe of a mosei-trained baseline silently score MOSEI
    # sentiment instead -- all five task directories received the identical number.
    # The architecture fields below DO come from train_meta, which is correct: those
    # describe the checkpoint. Only the read-out target follows the CLI.
    if meta.get("dataset") and meta["dataset"] != dataset:
        print(f"  note: checkpoint trained on '{meta['dataset']}', evaluating on "
              f"'{dataset}' (labels from the latter)")
    _apply_enc_width(meta, approach)      # before ANY encoder is constructed
    adim     = meta.get("enc_dim",  ENC_DIM_XFMR)
    proj_dim = meta.get("proj_dim", proj_dim)
    arch     = meta.get("arch",     "factorcl_based")

    def _ld(path, module):
        module.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        module.eval()
        return module

    def _loader(split):
        # Mirror _extract_backbone's dispatch. MultiBench raises
        # "Dataset not implemented: avmnist" for the image datasets, which made the
        # matched protocol — the ONLY protocol that works for the joint-encoder
        # baselines (gmc/comm/factorcl) — unusable on avmnist/enrico.
        if is_multitask_dataset(dataset):
            return mt_probe_loader(dataset, _image_root(dataset), split, batch_size,
                                   task=_MT_TASK[0], modalities=MODALITIES)
        if is_image_dataset(dataset):
            return image_probe_loader(dataset, _image_root(dataset), split, batch_size, _NW)
        ds = AffectDataset(_affect_path(dataset), dataset, split=split,
                           modalities=MODALITIES, task="classification")
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=_NW,
                          pin_memory=True, collate_fn=collate_affect)

    # comm_based
    if arch == "comm_based":
        fv = FEAT_DIM[dataset]["vision"]; ft = FEAT_DIM[dataset]["text"]
        enc_v = _ld(os.path.join(enc_dir, "enc_v.pth"), TransformerSeqEncoder(fv).to(device))
        enc_t = _ld(os.path.join(enc_dir, "enc_t.pth"), TransformerSeqEncoder(ft).to(device))
        fusion_xfmr = _ld(
            os.path.join(enc_dir, "fusion_xfmr.pth"),
            _CommFusionTransformer(width=ENC_DIM_XFMR, n_heads=8, n_layers=1,
                                   fusion="concat", pool="cls", batch_first=True).to(device),
        )

        if method in ("gmc", "comm"):
            mlp_head = _ld(os.path.join(enc_dir, "mlp_head.pth"),
                           CommMLPHead(adim, 512, proj_dim).to(device))
            def _extract(split):
                zs, ys = [], []
                with torch.no_grad():
                    for X, y in _loader(split):
                        v = X[0].float().to(device); t = X[1].float().to(device)
                        z = mlp_head(fusion_xfmr([enc_v(v), enc_t(t)]))
                        zs.append(z.cpu().numpy())
                        ys.append(y.numpy().reshape(-1) if isinstance(y, torch.Tensor) else np.asarray(y).reshape(-1))
                return np.concatenate(zs), np.concatenate(ys)
        else:  # factorcl comm_based — heads saved as ProjectionHead (no BN)
            proj_v_r = _ld(os.path.join(enc_dir, "proj_v_r.pth"), ProjectionHead(adim, proj_dim).to(device))
            proj_t_r = _ld(os.path.join(enc_dir, "proj_t_r.pth"), ProjectionHead(adim, proj_dim).to(device))
            def _extract(split):
                zs, ys = [], []
                with torch.no_grad():
                    for X, y in _loader(split):
                        v = X[0].float().to(device); t = X[1].float().to(device)
                        z = torch.cat([proj_v_r(fusion_xfmr([enc_v(v)])),
                                       proj_t_r(fusion_xfmr([enc_t(t)]))], dim=-1)
                        zs.append(z.cpu().numpy())
                        ys.append(y.numpy().reshape(-1) if isinstance(y, torch.Tensor) else np.asarray(y).reshape(-1))
                return np.concatenate(zs), np.concatenate(ys)

    # factorcl_based
    else:
        enc_v, enc_t = _make_encoders(approach, dataset, enc_dim, device, film_mode="none")
        _ld(os.path.join(enc_dir, "enc_v.pth"), enc_v)
        _ld(os.path.join(enc_dir, "enc_t.pth"), enc_t)

        if method in ("clip", "cross_self"):
            # z = concat(proj_v(enc_v(v)), proj_t(enc_t(t)))  ->  2 * proj_dim
            proj_v = _ld(os.path.join(enc_dir, "proj_v.pth"), ProjectionHead(adim, proj_dim).to(device))
            proj_t = _ld(os.path.join(enc_dir, "proj_t.pth"), ProjectionHead(adim, proj_dim).to(device))
            print(f"  {method} embedding: 2 x {proj_dim} = {2 * proj_dim} dims")
            def _extract(split):
                zs, ys = [], []
                with torch.no_grad():
                    for X, y in _loader(split):
                        v = X[0].float().to(device); t = X[1].float().to(device)
                        z = torch.cat([proj_v(enc_v(v)), proj_t(enc_t(t))], dim=-1)
                        zs.append(z.cpu().numpy())
                        ys.append(y.numpy().reshape(-1) if isinstance(y, torch.Tensor) else np.asarray(y).reshape(-1))
                return np.concatenate(zs), np.concatenate(ys)
        elif method == "factorcl":
            # Official FactorCLSSL.get_embedding() concatenates FIVE heads per
            # modality, in this order, and does NOT L2-normalise them:
            #   infonce_x1x2 (shared) | club_x1x2 | infonce_x{1,2}y (unique)
            #   | infonce_x1x2_cond   | club_x1x2_cond
            # z = cat(x1_reps) (+) cat(x2_reps)  ->  10 * proj_dim.
            # Probing only the shared pair evaluates FactorCL's SHARED
            # representation alone and discards the unique + conditional
            # components that are the paper's contribution -- which penalises it
            # exactly on the tasks where uniqueness matters. Runs trained before
            # the 10-head save was added only have proj_{v,t}_{r,u}.pth, so fall
            # back to the old 2-head embedding rather than crash.
            _ORDER = ("r", "cl", "u", "cr", "ccl")
            # --factorcl_official trains unnormalised mlp_head(d,d). `normalize` holds no
            # parameters, so the state_dict is identical and a wrong setting would load
            # SILENTLY and mis-score -- it must come from meta, never from a CLI flag.
            _nrm = not meta.get("head_no_norm", False)
            _PH  = lambda i, o: ProjectionHead(i, o, normalize=_nrm).to(device)
            _heads, _missing = {}, []
            for _m in ("v", "t"):
                for _n in _ORDER:
                    _p = os.path.join(enc_dir, f"proj_{_m}_{_n}.pth")
                    if os.path.exists(_p):
                        _heads[(_m, _n)] = _ld(_p, _PH(adim, proj_dim))
                    else:
                        _missing.append(f"proj_{_m}_{_n}.pth")
            _official = len(_heads) == 10
            # Trained compression head (--fusion_out at train time). When present it
            # IS the representation: 10*proj_dim -> its out_dim, learned by NT-Xent
            # like every other method's head, so no PCA control is needed.
            _out_p  = os.path.join(enc_dir, "proj_out.pth")
            _fc_out = meta.get("factorcl_out")
            proj_out = (_ld(_out_p, _PH(10 * proj_dim, int(_fc_out)))
                        if (_official and _fc_out and os.path.exists(_out_p)) else None)
            proj_v_r, proj_t_r = _heads[("v", "r")], _heads[("t", "r")]
            _has_u = ("v", "u") in _heads and ("t", "u") in _heads
            if _has_u:
                proj_v_u, proj_t_u = _heads[("v", "u")], _heads[("t", "u")]
            # Paper grouping (Appendix D.1): Z_S_i concatenates the heads for
            #   I_NCE(X1;X2) and I_NCE-CLUB(X1;X2|.)            -> r, ccl   (2 per modality)
            # Z_U_i concatenates the heads for
            #   I_NCE(Xi;Xi'), I_NCE-CLUB(X1;X2), I_NCE(X1;X2|.) -> u, cl, cr (3 per modality)
            # so Z_S = [Z_S1||Z_S2] is 4d and Z_U = [Z_U1||Z_U2] is 6d. This reproduces
            # the paper's own Table 4 ablation (FACTORCL-S / -U1 / -U2).
            _GRP = {"S": (("v", "r"), ("v", "ccl"), ("t", "r"), ("t", "ccl")),
                    "U": (("v", "u"), ("v", "cl"), ("v", "cr"),
                          ("t", "u"), ("t", "cl"), ("t", "cr"))}
            # Per-head-type probes (both modalities, 2*proj_dim each). loss/critic sits
            # at ~0 for the whole run, which is the exact value a CONSTANT critic gives
            # -- so either the CLUB critics never learn, or the cl/ccl heads minimise the
            # CLUB bound by collapsing to carry nothing. A head at the majority baseline
            # here means collapse; a head well above it means the critic simply failed.
            for _n in _ORDER:
                _GRP[f"head_{_n}"] = (("v", _n), ("t", _n))
            if _official and proj_out is None:
                def _extract_group(split, g):
                    keys = _GRP[g]; zs = []
                    with torch.no_grad():
                        for X, _ in _loader(split):
                            v = X[0].float().to(device); t = X[1].float().to(device)
                            hv, ht = enc_v(v), enc_t(t)
                            zs.append(torch.cat(
                                [_heads[k](hv if k[0] == "v" else ht) for k in keys],
                                dim=-1).cpu().numpy())
                    return np.concatenate(zs)
            if _official:
                print(f"  factorcl embedding: OFFICIAL 5 heads/modality "
                      f"-> 10 x {proj_dim} = {10 * proj_dim} dims"
                      + (f" -> trained head -> {int(_fc_out)} dims"
                         if proj_out is not None else " (no compression head)")
                      + ("  [OFFICIAL: heads unnormalised]" if not _nrm
                         else "  [heads L2-normalised]"))
            else:
                print(f"  factorcl embedding: LEGACY shared-pair only "
                      f"-> 2 x {proj_dim} = {2 * proj_dim} dims "
                      f"(missing {len(_missing)}: {', '.join(_missing)})")
            def _extract(split):
                zs, ys = [], []
                with torch.no_grad():
                    for X, y in _loader(split):
                        v = X[0].float().to(device); t = X[1].float().to(device)
                        hv, ht = enc_v(v), enc_t(t)
                        if _official:
                            z = torch.cat([_heads[("v", n)](hv) for n in _ORDER]
                                          + [_heads[("t", n)](ht) for n in _ORDER], dim=-1)
                            if proj_out is not None:
                                z = proj_out(z)
                        else:
                            z = torch.cat([F.normalize(proj_v_r(hv), dim=-1),
                                           F.normalize(proj_t_r(ht), dim=-1)], dim=-1)
                        zs.append(z.cpu().numpy())
                        ys.append(y.numpy().reshape(-1) if isinstance(y, torch.Tensor) else np.asarray(y).reshape(-1))
                return np.concatenate(zs), np.concatenate(ys)
            if _has_u:
                def _extract_r(split):
                    # The shared pair ALONE. cka_r_u must contrast shared against
                    # unique; once _extract returns the official 10-block embedding
                    # it can no longer stand in for the R side.
                    zs = []
                    with torch.no_grad():
                        for X, _ in _loader(split):
                            v = X[0].float().to(device); t = X[1].float().to(device)
                            zs.append(torch.cat([F.normalize(proj_v_r(enc_v(v)), dim=-1),
                                                 F.normalize(proj_t_r(enc_t(t)), dim=-1)],
                                                dim=-1).cpu().numpy())
                    return np.concatenate(zs)
                def _extract_u(split):
                    zs = []
                    with torch.no_grad():
                        for X, _ in _loader(split):
                            v = X[0].float().to(device); t = X[1].float().to(device)
                            z = torch.cat([F.normalize(proj_v_u(enc_v(v)), dim=-1),
                                           F.normalize(proj_t_u(enc_t(t)), dim=-1)], dim=-1)
                            zs.append(z.cpu().numpy())
                    return np.concatenate(zs)
        elif not os.path.exists(os.path.join(enc_dir, "fusion.pth")):
            # A preference-conditioned method (lora / simplex / palora / ...) probed
            # in fixed/matched mode. run_fixed was written only for the lambda-blind
            # baselines, whose `else` branch loads a GMC/CoMM fusion module; these
            # runs have no fusion.pth. Evaluate them at ONE fixed preference — the
            # simplex barycentre, or lambda=0.5 for a 1-D lambda — which is exactly
            # what "fixed" means for a lambda-method and keeps the metric identical
            # to the baselines.
            _meta_f = os.path.join(enc_dir, "train_meta.json")
            _fm = "none"
            if os.path.exists(_meta_f):
                _fm = json.load(open(_meta_f)).get("film_mode", "none")
            _mods = _load_models(enc_dir, approach, dataset, enc_dim, proj_dim,
                                 device, None, None, _fm)
            _ev, _et, _pvr, _ptr, _pvu, _ptu = _mods
            _grid = (duo_grid(_SIMPLEX_SIDE[0]) if _fm == "duo_txt" else
                     simplex_grid(_SIMPLEX_SIDE[0]) if _fm in (
                         "simplex", "simplex4h", "simplex4h_norm", "simplex6h",
                         "simplex_enc_decomp_R", "simplex_proj_decomp_R",
                         "simplex_both_decomp_R") else None)
            _fixed_lam = (min(_grid, key=lambda g: max(g) - min(g)) if _grid else 0.5)
            print(f"  fixed-preference probe: film_mode={_fm}  lambda={_lamstr(_fixed_lam)}")
            def _extract(split):
                h_v, h_t, y = _extract_backbone(_ev, _et, dataset, split,
                                                batch_size, device,
                                                lam=(_fixed_lam if _grid else None))
                z = _apply_heads(h_v, h_t, _pvr, _pvu, _ptr, _ptu, lam=_fixed_lam,
                                 device=device, film_mode=_fm)
                return z, y
        else:
            # rebuild the fusion head at the shape it was TRAINED with; a capacity-
            # matched CoMM/GMC has an MLP head and a different output dim, and
            # loading it into the default Linear(2*adim, adim) fails on every key.
            _fm_meta = os.path.join(enc_dir, "train_meta.json")
            _fmj = json.load(open(_fm_meta)) if os.path.exists(_fm_meta) else {}
            fusion = _ld(os.path.join(enc_dir, "fusion.pth"),
                         FusionEncoder(adim, _fmj.get("fusion_out") or adim,
                                       hidden=_fmj.get("fusion_hidden")).to(device))
            def _extract(split):
                zs, ys = [], []
                with torch.no_grad():
                    for X, y in _loader(split):
                        v = X[0].float().to(device); t = X[1].float().to(device)
                        zs.append(fusion(enc_v(v), enc_t(t)).cpu().numpy())
                        ys.append(y.numpy().reshape(-1) if isinstance(y, torch.Tensor) else np.asarray(y).reshape(-1))
                return np.concatenate(zs), np.concatenate(ys)

    # CoMM protocol: LogisticRegressionCV with balanced_accuracy on train split, eval on test
    # NOTE: LogisticRegressionCV.score() honours `scoring`, so test_acc below is
    # BALANCED accuracy fitted on the TRAIN split — a different metric AND a
    # different fitting split from the λ-methods (`validation` mode: plain top-1,
    # probe fitted on the VALID split). The matched-protocol number computed further
    # down is the one to use when comparing baselines against the λ-methods.
    z_test,  y_test  = _extract("test")

    test_acc = val_acc = None
    if not matched_only:
        z_train, y_train = _extract("train")
        scaler    = _scaler()
        z_train_s = scaler.fit_transform(z_train)
        z_test_s  = scaler.transform(z_test)

        clf = LogisticRegressionCV(Cs=5, max_iter=1000, scoring="balanced_accuracy", n_jobs=4, cv=5)
        clf.fit(z_train_s, y_train)

        test_acc = float(clf.score(z_test_s,  y_test))  * 100
        val_acc  = float(clf.score(z_train_s, y_train)) * 100

    # CKA(z_R, z_U) for FactorCL: measures disentanglement between redundant and unique heads
    cka_r_u = None
    if _has_u and _extract_u is not None:
        z_u_test = _extract_u("test")
        z_r_test = _extract_r("test") if _extract_r is not None else z_test
        cka_r_u  = round(float(linear_cka(z_r_test, z_u_test)), 4)
        print(f"  CKA(z_R, z_U) = {cka_r_u:.4f}  [{_interpret_cka(cka_r_u)}]")

    # Matched protocol (identical to `validation` mode, minus the λ sweep)
    # Same fitting split (valid), same classifier + C tuning (_lr_accuracy), same
    # metric (plain top-1 on test). This is the number that is directly comparable
    # to the λ-methods; λ-blind baselines simply have a single representation.
    z_valid, y_valid = _extract("valid")
    _pf = _PROBE_FIT[0]
    z_train_pf = y_train_pf = None
    if _pf in ("train", "trainval"):
        z_train_pf, y_train_pf = _extract("train")
        print(f"  probe_fit={_pf}: probe fit on TRAIN (n={len(y_train_pf)}); C chosen on valid "
              f"(n={len(y_valid)})" + ("; final refit on train+valid" if _pf == "trainval" else ""))
        _C_SEL[0] = (z_train_pf, y_train_pf, z_valid, y_valid)
        try:
            if _pf == "train":
                acc2 = _acc2_variants(z_train_pf, y_train_pf, z_test, y_test, dataset,
                                      tr_splits=("train",))
            else:
                acc2 = _acc2_variants(np.concatenate([z_train_pf, z_valid]),
                                      np.concatenate([y_train_pf, y_valid]), z_test, y_test,
                                      dataset, tr_splits=("train", "valid"))
        finally:
            _C_SEL[0] = None
    else:
        acc2 = _acc2_variants(z_valid, y_valid, z_test, y_test, dataset)
    acc2["probe_fit"] = _pf

    def _variants_pf(zv, zt, get_train):
        """_acc2_variants under the ACTIVE probe_fit protocol.

        Used by the sub-representation probes below (FactorCL S/U groups, PCA
        width match) so they are fitted on the same split as the headline number.
        `get_train` is lazy: it is only called when the protocol needs it.
        """
        if _pf not in ("train", "trainval"):
            return _acc2_variants(zv, y_valid, zt, y_test, dataset)
        ztr = get_train()
        _C_SEL[0] = (ztr, y_train_pf, zv, y_valid)
        try:
            if _pf == "train":
                out = _acc2_variants(ztr, y_train_pf, zt, y_test, dataset,
                                     tr_splits=("train",))
            else:
                out = _acc2_variants(np.concatenate([ztr, zv]),
                                     np.concatenate([y_train_pf, y_valid]),
                                     zt, y_test, dataset, tr_splits=("train", "valid"))
        finally:
            _C_SEL[0] = None
        out["probe_fit"] = _pf
        return out

    if SAVE_Z[0]:
        _extra = {}
        if is_multitask_dataset(dataset):
            # task=None returns the raw 7-label vector; `task` changes only the label,
            # never which samples are kept, so this order matches z exactly. Asserted.
            for _sp, _zz in (("valid", z_valid), ("test", z_test)):
                _ds = dataset_class(dataset)(_image_root(dataset), split=_sp, task=None)
                _Y7 = np.stack([np.asarray(_ds._y(_i)).reshape(-1) for _i in range(len(_ds))])
                assert len(_Y7) == len(_zz), f"label/z misalignment on {_sp}: {len(_Y7)} vs {len(_zz)}"
                _extra[f"y7_{_sp}"] = _Y7
        os.makedirs(os.path.dirname(os.path.abspath(SAVE_Z[0])), exist_ok=True)
        np.savez_compressed(SAVE_Z[0], z_valid=z_valid, z_test=z_test,
                            y_valid=np.asarray(y_valid), y_test=np.asarray(y_test), **_extra)
        print(f"  saved representations -> {SAVE_Z[0]}  z_test {z_test.shape}"
              + (f"  y7 {_extra['y7_test'].shape}" if _extra else ""))

    # model-selection signal (never touches test)
    # The matched protocol fits the probe on valid and reports test, so nothing here
    # can rank two CHECKPOINTS without peeking at test. That matters for the
    # "why not train N baselines with different loss weights?" control: taking the
    # max over N test scores is an oracle, and comparing it against STEER's
    # validation-selected consensus lambda is not like-for-like. k-fold CV inside the
    # valid split gives an honest selection score at ~zero extra cost.
    val_cv = None
    try:
        from sklearn.model_selection import StratifiedKFold
        _yv = np.asarray(y_valid).reshape(-1)
        if len(np.unique(_yv)) > 1:
            _k = min(5, int(np.min(np.bincount(_yv.astype(int)))))
            if _k >= 2:
                _sc = []
                for _tr, _te in StratifiedKFold(_k, shuffle=True, random_state=0).split(z_valid, _yv):
                    _sc.append(_lr_metrics(z_valid[_tr], _yv[_tr], z_valid[_te], _yv[_te])["acc"])
                val_cv = float(np.mean(_sc))
                print(f"  val_cv ({_k}-fold within valid, for model selection): {val_cv:.2f}")
    except Exception as _e:
        print(f"  val_cv unavailable: {_e}")
    test_top1 = acc2["zero_included"]

    # width-matched variant
    # The official FactorCL embedding is 10 * proj_dim, so at proj_dim=40 it enters
    # the probe at 400 dims against the 64 every other method gets. Report BOTH: the
    # native representation (what the method actually produces) and a PCA reduction
    # to MATCH_DIM, so neither "we crippled it" nor "it got 6x the width" applies.
    # The basis is fitted on the VALID split only -- the same split the probe is fitted
    # on -- and applied to test, so no test information enters the projection.
    # FactorCL S / U ablation (paper Table 4)
    factorcl_ablation = None
    if _extract_group is not None:
        factorcl_ablation = {}
        for _g in ("S", "U") + tuple(f"head_{n}" for n in _ORDER):
            _zv = _extract_group("valid", _g); _zt = _extract_group("test", _g)
            # Same fitting split as the main probe -- otherwise --probe_fit train
            # reports a train-fit FactorCL-full against valid-fit S/U sub-probes.
            _a  = _variants_pf(_zv, _zt, lambda: _extract_group("train", _g))
            factorcl_ablation[_g] = dict(dim=int(_zv.shape[1]), **_a)
            print(f"  FactorCL-{_g:9s}: {_zv.shape[1]:3d} dims  "
                  f"top1 {_a['zero_included']:6.2f}  (majority {_a['majority']:.2f})")
        print(f"  FactorCL-full: {z_valid.shape[1]:3d} dims  top1 {acc2['zero_included']:.2f}")

    acc2_matched = None
    if z_valid.shape[1] > MATCH_DIM:
        # Basis fitted on the split the probe is fitted on (never on test), so the
        # width match stays on the same protocol as every other number in the file.
        _zb = z_train_pf if _pf in ("train", "trainval") else z_valid
        _mu = _zb.mean(0, keepdims=True)
        _X  = _zb - _mu
        _C  = (_X.T @ _X) / max(len(_X) - 1, 1)
        _V  = np.linalg.eigh(_C)[1][:, ::-1][:, :MATCH_DIM]
        acc2_matched = _variants_pf((z_valid - _mu) @ _V, (z_test - _mu) @ _V,
                                    lambda: (z_train_pf - _mu) @ _V)
        print(f"  width-matched (PCA {z_valid.shape[1]} -> {MATCH_DIM}): "
              f"top1 {acc2_matched['zero_included']:.2f} "
              f"(native {acc2['zero_included']:.2f})")
    if not matched_only:
        print(f"  [{method}|{arch}] CoMM-protocol (balanced acc, fit on train) test={test_acc:.1f}%")
    print(f"  [{method}|{arch}] FAIR protocol (top-1, probe fit on valid)  test={test_top1:.1f}%")
    if "non_neutral" in acc2:
        print(f"  [{method}|{arch}] FAIR Acc2 non-neutral                      test={acc2['non_neutral']:.1f}%")
    result = {
        "mode":     "matched" if matched_only else "fixed",
        "method":   method,
        "dataset":  dataset,
        "approach": approach,
        "arch":     arch,
        # Fair-comparison protocol — identical rule as the λ-methods.
        "metric":        "top1_accuracy",
        "protocol":      "fit_on_valid_top1",
        "test_top1_acc": test_top1,
        "acc2":          acc2,
    }
    if not matched_only:
        # Legacy CoMM protocol kept only when explicitly running --mode fixed.
        result["metric_comm"]   = "balanced_accuracy"
        result["val_acc"]       = round(val_acc,  2)
        result["test_acc"]      = round(test_acc, 2)
    if cka_r_u is not None:
        result["cka_r_u"] = cka_r_u
    result["embed_dim"] = int(z_valid.shape[1])
    if val_cv is not None:
        result["val_cv"] = val_cv
    if factorcl_ablation is not None:
        result["factorcl_ablation"] = factorcl_ablation
    if acc2_matched is not None:
        result["acc2_match%d" % MATCH_DIM] = acc2_matched
        result["test_top1_acc_match%d" % MATCH_DIM] = acc2_matched["zero_included"]
    return result


# Validation mode

def run_validation(enc_dir, dataset, approach, enc_dim, proj_dim,
                   batch_size, device, enc_ckpt_v, enc_ckpt_t, film_mode="none",
                   modality="both"):
    """
    Full valid set → 80/20 split to select best λ (LR) → fit on full valid → Top-1 (%) on test.
    modality: 'both' | 'vision' | 'text' | 'shared'
    """
    enc_v, enc_t, proj_v_r, proj_t_r, proj_v_u, proj_t_u = _load_models(
        enc_dir, approach, dataset, enc_dim, proj_dim, device, enc_ckpt_v, enc_ckpt_t, film_mode)

    print(f"  Computing z(λ) for all λ — film_mode={film_mode} modality={modality} (valid + test)...")
    z_val, l_val = _get_all_z(enc_v, enc_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u,
                               dataset, "valid", film_mode, batch_size, device, modality=modality)
    z_te,  l_te  = _get_all_z(enc_v, enc_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u,
                               dataset, "test",  film_mode, batch_size, device, modality=modality)

    _pf = _PROBE_FIT[0]
    _trainfit = _pf in ("train", "trainval")
    if _trainfit:
        z_tr, l_tr = _get_all_z(enc_v, enc_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u,
                                dataset, "train", film_mode, batch_size, device, modality=modality)
        print(f"  probe_fit={_pf}: probe fit on TRAIN (n={len(l_tr)}); C and lambda chosen "
              f"on the FULL valid split (n={len(l_val)})"
              + ("; final probe refit on train+valid" if _pf == "trainval" else ""))

    # 80/20 split of valid for λ selection
    n_val = len(l_val)
    rng   = np.random.default_rng(42)
    val_idx   = rng.choice(n_val, size=max(1, n_val // 5), replace=False)
    train_idx = np.setdiff1d(np.arange(n_val), val_idx)

    print(f"\n  λ sweep on valid (train={len(train_idx)}, val={len(val_idx)}):")
    print(f"  {'λ':>6}   {'val acc (%)':>12}")
    print(f"  {'─'*6}   {'─'*12}")

    lam_grid = sorted(z_val.keys())
    val_accs = []
    # Plain accuracy is a poor selection criterion on an imbalanced label: it is
    # maximised by whatever best mimics the majority class. Balanced accuracy
    # (mean per-class recall, chance = 50% at any prevalence) is recorded per λ
    # alongside it, and drives selection when the label is skewed.
    val_baccs = []
    # Boolean mask over the validation split: True where continuous sentiment != 0.
    # None for datasets with no neutral notion, in which case the column is omitted.
    _nn_val = _neutral_mask(dataset, "valid")
    if _nn_val is not None and len(_nn_val) != n_val:
        print(f"  [warn] neutral mask has {len(_nn_val)} rows but valid has {n_val}"
              f" — skipping validation non-neutral")
        _nn_val = None
    val_nns = []
    for lam in lam_grid:
        if _trainfit:
            sc  = _scaler()
            tr  = sc.fit_transform(z_tr[lam])
            va  = sc.transform(z_val[lam])
            _sk = _is_skewed(l_tr)
            _cwt = "balanced" if _sk else None
            _bs, pred = -1.0, None
            for _C in _lr_grid():           # C chosen on valid, jointly with lambda
                _c = LogisticRegression(C=_C, max_iter=1000, random_state=42,
                                        class_weight=_cwt).fit(tr, l_tr)
                _p = _c.predict(va)
                _s = (balanced_accuracy_score(l_val, _p) if _sk
                      else float((_p == l_val).mean()))
                if _s > _bs:
                    _bs, pred = _s, _p
            truth = l_val
            _sel_idx = np.arange(n_val)
        else:
            _sel_idx = val_idx
            sc  = _scaler()
            tr  = sc.fit_transform(z_val[lam][train_idx])
            va  = sc.transform(z_val[lam][val_idx])
            _cw = "balanced" if _is_skewed(l_val[train_idx]) else None
            if _TUNE_C[0]:
                # C chosen on the FITTING half only. Off by default: the published
                # lambda curves were produced at a fixed C=1.0 and must not move.
                _best, _bv = None, -1.0
                for _C in _lr_grid():
                    _c = LogisticRegression(C=_C, max_iter=1000, random_state=42,
                                            class_weight=_cw).fit(tr, l_val[train_idx])
                    _v = _c.score(tr, l_val[train_idx])
                    if _v > _bv:
                        _bv, _best = _v, _c
                clf = _best
            else:
                clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                                         class_weight=_cw).fit(tr, l_val[train_idx])
            pred = clf.predict(va)
            truth = l_val[val_idx]
        acc = float((pred == truth).mean())
        val_accs.append(round(acc * 100, 2))
        val_baccs.append(round(float(balanced_accuracy_score(truth, pred)) * 100, 2))
        # Non-neutral Acc2 on the SELECTION split. acc2 below reports the same
        # metric on test; this is its validation counterpart, so a hyper-parameter
        # can be chosen on non-neutral without ever reading the test split.
        if _nn_val is not None:
            m = _nn_val[_sel_idx]
            val_nns.append(round(float((pred[m] == truth[m]).mean()) * 100, 2)
                           if m.any() else float("nan"))
        print(f"  {_lamstr(lam):>16s}   {val_accs[-1]:>11.2f}   bacc={val_baccs[-1]:>6.2f}")

    # Skew is measured on the selection split itself, not on dataset-wide rates.
    _maj = _majority_baseline(l_val[np.arange(n_val) if _trainfit else val_idx])
    _skewed = _maj >= 60.0
    _sel = val_baccs if _skewed else val_accs
    if _skewed:
        print(f"  label is skewed (majority baseline {_maj:.1f}%) "
              f"-> selecting lambda on BALANCED accuracy")

    if _FIXED_LAM[0] is not None:
        best_lam = min(lam_grid, key=lambda l: sum(
            (a - b) ** 2 for a, b in zip(
                (l if isinstance(l, tuple) else (float(l),)),
                _FIXED_LAM[0][:len(l) if isinstance(l, tuple) else 1])))
        best_idx = lam_grid.index(best_lam)
        print(f"  FIXED preference requested: {_lamstr(_FIXED_LAM[0])} "
              f"-> using grid point {_lamstr(best_lam)} "
              f"(validation argmax would have been {_lamstr(lam_grid[int(np.argmax(_sel))])})")
    else:
        best_idx = int(np.argmax(_sel))
        best_lam = lam_grid[best_idx]
    print(f"  {'─'*6}   {'─'*12}")
    print(f"  Best λ={_lamstr(best_lam)}  val={val_accs[best_idx]:.2f}%  → evaluating on test...")

    if _trainfit:
        _C_SEL[0] = (z_tr[best_lam], l_tr, z_val[best_lam], l_val)
        try:
            if _pf == "train":
                acc2 = _acc2_variants(z_tr[best_lam], l_tr, z_te[best_lam], l_te, dataset,
                                      tr_splits=("train",))
            else:
                acc2 = _acc2_variants(np.concatenate([z_tr[best_lam], z_val[best_lam]]),
                                      np.concatenate([l_tr, l_val]), z_te[best_lam], l_te,
                                      dataset, tr_splits=("train", "valid"))
        finally:
            _C_SEL[0] = None
    else:
        acc2 = _acc2_variants(z_val[best_lam], l_val, z_te[best_lam], l_te, dataset)
    acc2["probe_fit"] = _pf
    test_acc = acc2["zero_included"]
    print(f"\n  Test Top-1 accuracy (zero-included) = {test_acc:.2f}%")
    if "non_neutral" in acc2:
        print(f"  Test Acc2 (non-neutral, dropped {acc2['n_dropped_test']} test / "
              f"{acc2['n_dropped_val']} val) = {acc2['non_neutral']:.2f}%")

    return {
        "mode":          "validation",
        "dataset":       dataset,
        "approach":      approach,
        "modality":      modality,
        "lambdas":       lam_grid,
        "val_acc_pct":   val_accs,
        "val_bacc_pct":  val_baccs,
        "selection_metric": "balanced_acc" if _skewed else "acc",
        "majority_baseline_val": round(_maj, 2),
        "best_lambda":   best_lam,
        "best_val_acc":  val_accs[best_idx],
        "best_val_bacc": val_baccs[best_idx],
        "val_nn_pct":    (val_nns or None),
        "best_val_nn":   (val_nns[best_idx] if val_nns else None),
        "test_top1_acc": test_acc,          # zero-included (unchanged; backward-compatible)
        "acc2":          acc2,              # {zero_included, non_neutral?, n_dropped_*}
    }


# Validation-CV mode (gate experiment)

def run_backbone(enc_dir, dataset, approach, enc_dim, proj_dim,
                 batch_size, device, enc_ckpt_v, enc_ckpt_t, film_mode="none"):
    """IDENTICAL-PROTOCOL evaluation: probe concat(enc_v(v), enc_t(t)) for every method.

    Why this mode exists
    --------------------
    The other modes evaluate each method through its own output head, which differs
    by architecture and is therefore not literally the same protocol:

        CoMM / GMC   fusion(enc_v, enc_t)          40-d   (joint models; no per-modality head exists)
        FactorCL     concat(z_v, z_t)              80-d
        lambda-arms  concat(z_v, z_t)             128-d

    Two robustness checks say this does not change the ranking (FactorCL 40-d vs
    80-d: -0.05; our arms 128-d vs 64-d: -0.69, i.e. smaller is better). But
    "insensitive to protocol" is a weaker claim to defend than "same protocol",
    so this mode removes the objection entirely: EVERY method is probed on the
    concatenated frozen ENCODER outputs, all heads discarded. Every method saves
    enc_v.pth / enc_t.pth, so the input is 2 * enc_dim for all of them.

    This is also the conventional SSL evaluation (SimCLR et al. probe the backbone
    and discard the projection head), so it needs no special pleading.

    IMPORTANT LIMITATION
    --------------------
    For approach-4 arms the encoder is preference-BLIND (enc_film_mode_for returns
    "none"), so z does not depend on lambda and this mode reports ONE number per
    run. That is the right question for the headline table -- "does
    preference-conditioned TRAINING yield a better encoder?" -- but it cannot show
    lambda selection. Keep the head-level modes for every lambda analysis.
    For approach 5 / palora_enc the encoder IS conditioned, so lam is passed.
    """
    enc_v, enc_t, *_ = _load_models(enc_dir, approach, dataset, enc_dim, proj_dim,
                                    device, enc_ckpt_v, enc_ckpt_t, film_mode)
    enc_film = enc_film_mode_for(film_mode, approach)
    lam = None
    if enc_film not in ("none", ""):
        lam = _pref_endpoints(film_mode)[1]     # redundancy endpoint for conditioned encoders
        print(f"  encoder is preference-conditioned ({enc_film}); probing at lam={_lamstr(lam)}")
    else:
        print(f"  encoder is preference-blind; one representation per run")

    def _z(split):
        h_v, h_t, y = _extract_backbone(enc_v, enc_t, dataset, split,
                                        batch_size, device, lam=lam)
        return np.hstack([h_v.numpy(), h_t.numpy()]), y

    z_val, l_val = _z("valid")
    z_te,  l_te  = _z("test")
    print(f"  probe input: {z_val.shape[1]}-d  (2 x enc_dim, heads discarded)")

    res = _lr_metrics(z_val, l_val, z_te, l_te)
    acc2 = _acc2_variants(z_val, l_val, z_te, l_te, dataset)
    out = {"mode": "backbone", "dataset": dataset, "approach": approach,
           "film_mode": film_mode, "probe_dim": int(z_val.shape[1]),
           "lam": (list(lam) if isinstance(lam, tuple) else lam),
           "metrics": res, "acc2": acc2}
    print(f"  test: " + "  ".join(f"{k}={v}" for k, v in res.items()
                                  if isinstance(v, (int, float))))
    return out


def run_validation_cv(enc_dir, dataset, approach, enc_dim, proj_dim,
                      batch_size, device, enc_ckpt_v, enc_ckpt_t, film_mode="none",
                      modality="both", n_folds=5):
    """
    k-fold CV on the full valid set to select λ* robustly.
    Also records test_acc at every λ (for curve visualisation — not used for selection).
    Output includes the full val mean/std curve and test curve per λ.
    """
    from sklearn.model_selection import StratifiedKFold

    enc_v, enc_t, proj_v_r, proj_t_r, proj_v_u, proj_t_u = _load_models(
        enc_dir, approach, dataset, enc_dim, proj_dim, device, enc_ckpt_v, enc_ckpt_t, film_mode)

    print(f"  Computing z(λ) — film_mode={film_mode} modality={modality} ...")
    z_val, l_val = _get_all_z(enc_v, enc_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u,
                               dataset, "valid", film_mode, batch_size, device, modality=modality)
    z_te,  l_te  = _get_all_z(enc_v, enc_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u,
                               dataset, "test",  film_mode, batch_size, device, modality=modality)

    lam_grid = sorted(z_val.keys())
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    print(f"\n  {n_folds}-fold CV on full valid ({len(l_val)} samples) | test_acc at each λ:")
    print(f"  {'λ':>5}   {'val mean':>9}   {'val std':>8}   {'test acc':>9}")
    print(f"  {'─'*5}   {'─'*9}   {'─'*8}   {'─'*9}")

    val_means, val_stds, test_accs_all = [], [], []
    for lam in lam_grid:
        Z = z_val[lam]
        fold_accs = []
        for tr_idx, va_idx in skf.split(Z, l_val):
            sc  = _scaler()
            tr  = sc.fit_transform(Z[tr_idx])
            va  = sc.transform(Z[va_idx])
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            clf.fit(tr, l_val[tr_idx])
            fold_accs.append(float(clf.score(va, l_val[va_idx])) * 100)
        vm = float(np.mean(fold_accs))
        vs = float(np.std(fold_accs))
        ta = _lr_accuracy(z_val[lam], l_val, z_te[lam], l_te) * 100
        val_means.append(round(vm, 2))
        val_stds.append(round(vs, 2))
        test_accs_all.append(round(ta, 2))
        print(f"  {_lamstr(lam):>16s}   {vm:>9.2f}   {vs:>8.2f}   {ta:>9.2f}")

    best_idx = int(np.argmax(val_means))
    best_lam = lam_grid[best_idx]
    lam_half_idx = int(np.argmin([abs(l - 0.5) for l in lam_grid]))

    prominence_vs_worst = round(val_means[best_idx] - min(val_means), 2)
    prominence_vs_half  = round(val_means[best_idx] - val_means[lam_half_idx], 2)

    print(f"  {'─'*5}   {'─'*9}   {'─'*8}   {'─'*9}")
    print(f"  λ*={_lamstr(best_lam)}  val={val_means[best_idx]:.2f}%  test={test_accs_all[best_idx]:.2f}%")
    print(f"  peak prominence vs worst: +{prominence_vs_worst:.2f}%  vs λ=0.5: +{prominence_vs_half:.2f}%")

    acc2 = _acc2_variants(z_val[best_lam], l_val, z_te[best_lam], l_te, dataset)
    if "non_neutral" in acc2:
        print(f"  Acc2 non-neutral = {acc2['non_neutral']:.2f}%  (zero-included {acc2['zero_included']:.2f}%)")

    return {
        "mode":                 "validation_cv",
        "dataset":              dataset,
        "approach":             approach,
        "modality":             modality,
        "n_folds":              n_folds,
        "lambdas":              lam_grid,
        "val_acc_mean_pct":     val_means,
        "val_acc_std_pct":      val_stds,
        "test_acc_pct":         test_accs_all,
        "best_lambda":          best_lam,
        "best_val_acc":         val_means[best_idx],
        "test_top1_acc":        test_accs_all[best_idx],   # zero-included (backward-compatible)
        "acc2":                 acc2,                       # {zero_included, non_neutral?, n_dropped_*}
        "prominence_vs_worst":  prominence_vs_worst,
        "prominence_vs_half":   prominence_vs_half,
    }


# Ensemble mode (fix4)

def run_ensemble(enc_dir, dataset, approach, enc_dim, proj_dim,
                 batch_size, device, enc_ckpt_v, enc_ckpt_t, film_mode="proj",
                 modality="both"):
    """
    Average z(λ) across all λ in the grid, train one linear probe on the
    averaged features.  Eliminates the 45-sample val-argmax instability (fix4).
    """
    enc_v, enc_t, proj_v_r, proj_t_r, proj_v_u, proj_t_u = _load_models(
        enc_dir, approach, dataset, enc_dim, proj_dim, device, enc_ckpt_v, enc_ckpt_t, film_mode)

    print(f"  Computing z(λ) for all λ — film_mode={film_mode} modality={modality} (valid + test)...")
    z_val, l_val = _get_all_z(enc_v, enc_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u,
                               dataset, "valid", film_mode, batch_size, device, modality=modality)
    z_te,  l_te  = _get_all_z(enc_v, enc_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u,
                               dataset, "test",  film_mode, batch_size, device, modality=modality)

    lam_grid = sorted(z_val.keys())
    print(f"  Ensembling over {len(lam_grid)} λ values: {lam_grid}")

    z_avg_val = np.mean([z_val[lam] for lam in lam_grid], axis=0)
    z_avg_te  = np.mean([z_te[lam]  for lam in lam_grid], axis=0)

    test_acc = _lr_accuracy(z_avg_val, l_val, z_avg_te, l_te) * 100
    print(f"\n  Ensemble Test Top-1 accuracy = {test_acc:.2f}%")

    return {
        "mode":          "ensemble",
        "dataset":       dataset,
        "approach":      approach,
        "modality":      modality,
        "lam_grid":      lam_grid,
        "n_lambdas":     len(lam_grid),
        "test_top1_acc": round(test_acc, 2),
    }


# Few-shot mode

def run_fewshot(enc_dir, dataset, approach, enc_dim, proj_dim,
                batch_size, device, enc_ckpt_v, enc_ckpt_t, film_mode="none",
                modality="both"):
    """
    K ∈ {5,10,20,50} shots from valid → LOO-kNN λ selection → kNN on test.
    Averaged over 5 seeds. Reports Top-1 accuracy (%).
    """
    enc_v, enc_t, proj_v_r, proj_t_r, proj_v_u, proj_t_u = _load_models(
        enc_dir, approach, dataset, enc_dim, proj_dim, device, enc_ckpt_v, enc_ckpt_t, film_mode)

    print(f"  Computing z(λ) for all λ — film_mode={film_mode} modality={modality} (valid + test)...")
    z_val_all, l_val = _get_all_z(enc_v, enc_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u,
                                   dataset, "valid", film_mode, batch_size, device, modality=modality)
    z_te_all,  l_te  = _get_all_z(enc_v, enc_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u,
                                   dataset, "test",  film_mode, batch_size, device, modality=modality)

    n_val   = len(l_val)
    K_results = {}

    for K in FEWSHOT_K:
        k_nn = min(KNN_K, K - 1) if K > 1 else 1
        print(f"\n  ── K={K} few-shot (kNN, k={k_nn}) ──")
        seed_results = []

        for seed in FEWSHOT_SEEDS:
            rng      = np.random.default_rng(seed)
            shot_idx = rng.choice(n_val, size=K, replace=False)

            # λ selection: LOO-kNN on K shots (test data not touched)
            loo_accs = []
            for lam in LAMBDA_GRID:
                score = _knn_loo_acc(z_val_all[lam][shot_idx], l_val[shot_idx], k=k_nn)
                loo_accs.append(round(score * 100, 2))

            best_idx = int(np.argmax(loo_accs))
            best_lam = LAMBDA_GRID[best_idx]

            pred_te  = _knn_predict(z_val_all[best_lam][shot_idx], l_val[shot_idx],
                                    z_te_all[best_lam], k=k_nn)
            acc      = float((pred_te == l_te).mean()) * 100

            print(f"    seed={seed}  LOO-kNN λ sweep ({K} shots):")
            for lam, sc in zip(LAMBDA_GRID, loo_accs):
                marker = " ←" if lam == best_lam else ""
                print(f"      λ={_lamstr(lam)}  loo_acc={sc:.2f}%{marker}")
            print(f"    → best_λ={_lamstr(best_lam)}  test_acc={acc:.2f}%")

            seed_results.append({
                "seed":         seed,
                "best_lambda":  best_lam,
                "loo_acc_pct":  dict(zip([str(l) for l in LAMBDA_GRID], loo_accs)),
                "test_top1_acc": round(acc, 2),
            })

        test_accs = [r["test_top1_acc"] for r in seed_results]
        print(f"\n  K={K} summary:  acc={np.mean(test_accs):.2f}%±{np.std(test_accs):.2f}%")
        K_results[str(K)] = {
            "k_nn":          k_nn,
            "seeds":         seed_results,
            "mean_top1_acc": round(float(np.mean(test_accs)), 2),
            "std_top1_acc":  round(float(np.std(test_accs)),  2),
        }

    return {"mode": "fewshot", "dataset": dataset, "approach": approach, "K_results": K_results}


# CKA

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear Centered Kernel Alignment between representation matrices X, Y (n × d).
    CKA(X,Y) = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    Range [0, 1]. 1 = identical geometry, 0 = completely different.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 0 else 0.0


def _interpret_cka(cka: float) -> str:
    if cka >= 0.85:
        return "HIGH — heads are NOT differentiating. λ sweep will be cosmetic."
    if cka >= 0.60:
        return "MEDIUM — partial separation. Pareto curve may have weak structure."
    return "LOW — heads extract genuinely different sub-structure. Pareto curve is real."


@_cka_uses_dim_slice
def run_cka(enc_dir, dataset, approach, enc_dim, proj_dim,
            batch_size, device, enc_ckpt_v=None, enc_ckpt_t=None, out_dir=None,
            film_mode="none"):
    """
    Extract z at λ=0 (pure unique) and λ=1 (pure shared) on test set.
    Compute linear CKA between them per modality and overall.
    """
    enc_v, enc_t, proj_v_r, proj_t_r, proj_v_u, proj_t_u = _load_models(
        enc_dir, approach, dataset, enc_dim, proj_dim, device, enc_ckpt_v, enc_ckpt_t, film_mode)

    # film_mode must come from train_meta, not the CLI default. _load_models already
    # reads it internally, but run_cka's own copy stayed at whatever was passed in --
    # so probing without --film_mode silently took the lambda-blind branch and called
    # a preference-conditioned encoder with no preference:
    #   TypeError: LoRATransformerEncoder.forward() missing 1 required positional
    #   argument: 'lam'
    _meta_p = os.path.join(enc_dir, "train_meta.json")
    if os.path.exists(_meta_p):
        film_mode = json.load(open(_meta_p)).get("film_mode", film_mode)
    P0, P1 = _pref_endpoints(film_mode)
    print(f"  Extracting encoder features (test set) — film_mode={film_mode}, "
          f"endpoints {_lamstr(P0)} vs {_lamstr(P1)}...")
    if film_mode in ("none", "proj"):
        h_v0, h_t0, _ = _extract_backbone(enc_v, enc_t, dataset, "test", batch_size, device)
        h_v1, h_t1 = h_v0, h_t0
    else:
        h_v0, h_t0, _ = _extract_backbone(enc_v, enc_t, dataset, "test", batch_size, device, lam=P0)
        h_v1, h_t1, _ = _extract_backbone(enc_v, enc_t, dataset, "test", batch_size, device, lam=P1)

    # z at the "pure unique" endpoint (only the U coordinate contributes)
    z0_v = _apply_heads(h_v0, h_t0, proj_v_r, proj_v_u, proj_t_r, proj_t_u, lam=P0, device=device, film_mode=film_mode)
    # z at the "pure shared" endpoint (only the R coordinate contributes)
    z1_v = _apply_heads(h_v1, h_t1, proj_v_r, proj_v_u, proj_t_r, proj_t_u, lam=P1, device=device, film_mode=film_mode)

    # Also compute per-modality CKA separately
    proj_dim_half = z0_v.shape[1] // 2
    z0_vision = z0_v[:, :proj_dim_half];  z1_vision = z1_v[:, :proj_dim_half]
    z0_text   = z0_v[:, proj_dim_half:];  z1_text   = z1_v[:, proj_dim_half:]

    cka_vision  = linear_cka(z0_vision, z1_vision)
    cka_text    = linear_cka(z0_text,   z1_text)
    cka_overall = linear_cka(z0_v,      z1_v)

    print()
    print(f"  {'='*56}")
    print(f"  CKA Diagnostic — {dataset.upper()} | approach={approach}")
    print(f"  Comparing z{_lamstr(P0)} [pure unique] vs z{_lamstr(P1)} [pure shared]")
    print(f"  {'='*56}")
    print(f"  CKA vision  : {cka_vision:.4f}   {_interpret_cka(cka_vision)}")
    print(f"  CKA text    : {cka_text:.4f}   {_interpret_cka(cka_text)}")
    print(f"  CKA overall : {cka_overall:.4f}   {_interpret_cka(cka_overall)}")
    print(f"  {'='*56}")
    print()
    if cka_overall >= 0.85:
        print("  ACTION: Encoder collapsed shared and unique into one representation.")
        print("          The Pareto curve will be cosmetic — λ does not matter.")
        print("          → Stabilize the encoder before running full probing.")
    elif cka_overall >= 0.60:
        print("  ACTION: Partial separation. Run validation probe but interpret carefully.")
    else:
        print("  ACTION: Good separation. Proceed with full validation + fewshot probing.")
    print()

    results = {
        "mode":        "cka",
        "dataset":     dataset,
        "approach":    approach,
        "cka_vision":  round(cka_vision,  4),
        "cka_text":    round(cka_text,    4),
        "cka_overall": round(cka_overall, 4),
        "interpretation": _interpret_cka(cka_overall),
    }
    save_dir = out_dir or enc_dir
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "probe_cka.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results → {out_path}")
    return results


# CKA over checkpoints

@_cka_uses_dim_slice
def run_cka_curve(enc_dir, dataset, approach, enc_dim, proj_dim,
                  batch_size, device, enc_ckpt_v=None, enc_ckpt_t=None, out_dir=None,
                  film_mode="none"):
    """
    Sweep over periodic training checkpoints (checkpoints/ep*/), compute CKA
    at each one. Reveals whether head separation grows, collapses, or oscillates.
    """
    ckpt_root = os.path.join(enc_dir, "checkpoints")
    if not os.path.isdir(ckpt_root):
        print(f"  [ERROR] No checkpoints dir at {ckpt_root}")
        print("  Train again with the updated benchmark_multibench.py to get periodic checkpoints.")
        return {}

    ep_dirs = sorted(
        [d for d in Path(ckpt_root).iterdir() if d.is_dir() and d.name.startswith("ep")],
        key=lambda d: int(d.name[2:]),
    )
    if not ep_dirs:
        print(f"  [ERROR] No ep* subdirs found in {ckpt_root}")
        return {}

    print(f"  Found {len(ep_dirs)} checkpoints: {[d.name for d in ep_dirs]}")

    meta_cka = {}
    _meta_p = os.path.join(enc_dir, "train_meta.json")
    if os.path.exists(_meta_p):
        with open(_meta_p) as _f:
            meta_cka = json.load(_f)
        proj_dim = meta_cka.get("proj_dim", proj_dim)
        enc_dim  = meta_cka.get("enc_dim",  enc_dim)

    h_v_cache, h_t_cache = None, None
    if film_mode in ("none", "proj"):
        print("  Extracting encoder features (test set) once...")
        enc_v_f, enc_t_f, _, _, _, _ = _load_models(
            enc_dir, approach, dataset, enc_dim, proj_dim, device, enc_ckpt_v, enc_ckpt_t, film_mode)
        h_v_cache, h_t_cache, _ = _extract_backbone(enc_v_f, enc_t_f, dataset, "test", batch_size, device)
        del enc_v_f, enc_t_f

    curve = []
    BAR = 20
    print()
    print(f"  {'Epoch':<8} {'CKA-V':>7} {'CKA-T':>7} {'CKA-all':>9}  Trend")
    print(f"  {'-'*60}")

    for ep_dir in ep_dirs:
        epoch = int(ep_dir.name[2:])
        adim = enc_dim if approach == 1 else 40
        if approach in (3, 4):
            if approach == 3:
                HEAD_CLS = DualFiLMProjectionHeadPreNorm if meta_cka.get("proj_prenorm") else DualFiLMProjectionHead
                _hkw = {}
            else:
                _branches = meta_cka.get("lora_branches", 3)
                _rank     = meta_cka.get("lora_rank",     4)
                _rank_m   = meta_cka.get("lora_rank_m",   _rank)
                HEAD_CLS  = LoRATriProjectionHead if _branches == 3 else LoRADualProjectionHead
                _hkw      = (dict(rank=_rank, rank_m=_rank_m) if _branches == 3 else dict(rank=_rank))
            proj_v_r = HEAD_CLS(adim, proj_dim, **_hkw).to(device)
            proj_t_r = HEAD_CLS(adim, proj_dim, **_hkw).to(device)
            proj_v_u = proj_t_u = None
            for name, mod in [("proj_v", proj_v_r), ("proj_t", proj_t_r)]:
                w = os.path.join(ep_dir, f"{name}.pth")
                if os.path.exists(w):
                    mod.load_state_dict(torch.load(w, map_location=device, weights_only=True))
                mod.eval()
        else:
            proj_v_r = ProjectionHead(adim, proj_dim).to(device)
            proj_t_r = ProjectionHead(adim, proj_dim).to(device)
            proj_v_u = ProjectionHead(adim, proj_dim).to(device)
            proj_t_u = ProjectionHead(adim, proj_dim).to(device)
            for name, mod in [("proj_v_r", proj_v_r), ("proj_t_r", proj_t_r),
                               ("proj_v_u", proj_v_u), ("proj_t_u", proj_t_u)]:
                w = os.path.join(ep_dir, f"{name}.pth")
                if os.path.exists(w):
                    mod.load_state_dict(torch.load(w, map_location=device, weights_only=True))
                mod.eval()

        cP0, cP1 = _pref_endpoints(film_mode)
        if film_mode in ("none", "proj"):
            h_v, h_t = h_v_cache, h_t_cache
            z0 = _apply_heads(h_v, h_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u, lam=cP0, device=device, film_mode=film_mode)
            z1 = _apply_heads(h_v, h_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u, lam=cP1, device=device, film_mode=film_mode)
        else:
            # Load encoder from this checkpoint and extract at both endpoints
            enc_v_ep, enc_t_ep = _make_encoders(approach, dataset, enc_dim, device,
                                                enc_film_mode_for(film_mode, approach))
            for enc, name in [(enc_v_ep, "enc_v"), (enc_t_ep, "enc_t")]:
                w = os.path.join(ep_dir, f"{name}.pth")
                if os.path.exists(w):
                    enc.load_state_dict(torch.load(w, map_location=device, weights_only=True))
                enc.eval()
            with torch.no_grad():
                h_v0, h_t0, _ = _extract_backbone(enc_v_ep, enc_t_ep, dataset, "test", batch_size, device, lam=cP0)
                h_v1, h_t1, _ = _extract_backbone(enc_v_ep, enc_t_ep, dataset, "test", batch_size, device, lam=cP1)
            del enc_v_ep, enc_t_ep
            z0 = _apply_heads(h_v0, h_t0, proj_v_r, proj_v_u, proj_t_r, proj_t_u, lam=cP0, device=device, film_mode=film_mode)
            z1 = _apply_heads(h_v1, h_t1, proj_v_r, proj_v_u, proj_t_r, proj_t_u, lam=cP1, device=device, film_mode=film_mode)

        half = z0.shape[1] // 2
        cka_v   = linear_cka(z0[:, :half], z1[:, :half])
        cka_t   = linear_cka(z0[:, half:], z1[:, half:])
        cka_all = linear_cka(z0, z1)

        bar_len = int(cka_all * BAR)
        bar     = "█" * bar_len + "░" * (BAR - bar_len)
        print(f"  ep{epoch:<6d} {cka_v:>7.4f} {cka_t:>7.4f} {cka_all:>9.4f}  [{bar}]")

        curve.append({"epoch": epoch, "cka_vision": round(cka_v, 4),
                      "cka_text": round(cka_t, 4), "cka_overall": round(cka_all, 4)})

    # Quick trend summary
    if len(curve) >= 2:
        delta = curve[-1]["cka_overall"] - curve[0]["cka_overall"]
        trend = "GROWING (collapsing)" if delta > 0.05 else "SHRINKING (separating)" if delta < -0.05 else "STABLE"
        print(f"\n  CKA overall Δ from ep{curve[0]['epoch']} → ep{curve[-1]['epoch']}: {delta:+.4f}  ({trend})")
        variance = float(np.std([c["cka_overall"] for c in curve]))
        osc = "HIGH oscillation — encoder is unstable" if variance > 0.08 else \
              "MODERATE oscillation" if variance > 0.04 else "LOW oscillation — encoder is stable"
        print(f"  Std across checkpoints: {variance:.4f}  ({osc})")

    print()
    results = {
        "mode": "cka_curve", "dataset": dataset, "approach": approach, "curve": curve,
    }
    save_dir = out_dir or enc_dir
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "probe_cka_curve.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results → {out_path}")
    return results


# CKA matrix over λ grid

CKA_MATRIX_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

@_cka_uses_dim_slice
def run_cka_matrix(enc_dir, dataset, approach, enc_dim, proj_dim,
                   batch_size, device, enc_ckpt_v=None, enc_ckpt_t=None,
                   film_mode="proj", out_dir=None):
    """
    Compute pairwise CKA(z(λ_i), z(λ_j)) for λ in {0, 0.25, 0.5, 0.75, 1.0}.

    Interpretation guide:
      Learned curve  : off-diagonal values decay smoothly with |λ_i - λ_j|.
                       z(0.5) is NOT equidistant from z(0) and z(1).
      Fake curve     : z(0.5) ≈ linear interpolation of z(0) and z(1) →
                       CKA(z(0), z(0.5)) ≈ CKA(z(0), z(1)) + 0.5·gap.
      Key diagnostic : CKA(z(0), z(0.5)) vs CKA(z(0), z(1)).
                       If the ratio deviates from ~0.5 → curvature exists.
    """
    enc_v, enc_t, proj_v_r, proj_t_r, proj_v_u, proj_t_u = _load_models(
        enc_dir, approach, dataset, enc_dim, proj_dim, device,
        enc_ckpt_v, enc_ckpt_t, film_mode)

    enc_is_blind = film_mode in ("none", "proj", "palora_proj")
    if enc_is_blind:
        h_v_base, h_t_base, _ = _extract_backbone(enc_v, enc_t, dataset, "test", batch_size, device)

    # A simplex preference is a 3-vector, so the scalar 1-D grid is not a path
    # through its space. Walk the R -> U edge instead: (t, (1-t)/2, (1-t)/2)
    # reproduces exactly the 1-D endpoints at t=1 and t=0, so the curvature
    # diagnostic below keeps its meaning.
    if film_mode in ("simplex", "simplex4h", "simplex4h_norm", "simplex6h", "duo_txt",
                     "simplex_enc_decomp_R", "simplex_proj_decomp_R",
                     "simplex_both_decomp_R"):
        grid = [(t, (1.0 - t) / 2.0, (1.0 - t) / 2.0) for t in CKA_MATRIX_GRID]
    else:
        grid = CKA_MATRIX_GRID
    n = len(grid)
    half = None  # inferred from first batch — avoids proj_dim mismatch

    # Extract z(λ) for each λ — split into vision and text halves
    zv, zt = {}, {}
    for lam in grid:
        if enc_is_blind:
            h_v, h_t = h_v_base, h_t_base
        else:
            h_v, h_t, _ = _extract_backbone(enc_v, enc_t, dataset, "test", batch_size, device, lam=lam)
        z_both = _apply_heads(h_v, h_t, proj_v_r, proj_v_u, proj_t_r, proj_t_u,
                              lam=lam, device=device, film_mode=film_mode)
        if half is None:
            half = z_both.shape[1] // 2
        zv[lam] = z_both[:, :half]
        zt[lam] = z_both[:, half:]

    # Pairwise CKA matrices
    mat_v = [[round(linear_cka(zv[grid[i]], zv[grid[j]]), 4) for j in range(n)] for i in range(n)]
    mat_t = [[round(linear_cka(zt[grid[i]], zt[grid[j]]), 4) for j in range(n)] for i in range(n)]
    mat_all = [[round((mat_v[i][j] + mat_t[i][j]) / 2, 4) for j in range(n)] for i in range(n)]

    # Print
    lam_strs = [_lamstr(l, p=2) for l in grid]
    col_w = 7
    header = "  λ      | " + "  ".join(f"{s:>{col_w}}" for s in lam_strs)
    sep    = "  " + "-" * (len(header) - 2)

    for label, mat in [("VISION", mat_v), ("TEXT", mat_t), ("OVERALL", mat_all)]:
        print(f"\n  CKA MATRIX [{label}] — {dataset.upper()} ap{approach}")
        print(header); print(sep)
        for i, lam_i in enumerate(grid):
            row = "  ".join(f"{mat[i][j]:>{col_w}.4f}" for j in range(n))
            print(f"  {_lamstr(lam_i):>10s}   | {row}")

    # Key diagnostic: curvature indicator
    # For a pure linear interpolation z(0.5) = 0.5*z(0)+0.5*z(1),
    # CKA(z(0),z(0.5)) should be ~midway between 1 and CKA(z(0),z(1)).
    # Deviation from this = evidence of real curvature.
    mid = n // 2  # index of λ=0.5
    cka_00 = mat_all[0][0]    # = 1.0
    cka_01 = mat_all[0][n-1]  # CKA(z(0), z(1))
    cka_05 = mat_all[0][mid]  # CKA(z(0), z(0.5))
    linear_pred = (cka_00 + cka_01) / 2
    curvature   = cka_05 - linear_pred
    print(f"\n  Curvature indicator: CKA(z0,z0.5)={cka_05:.4f}  linear_pred={linear_pred:.4f}  "
          f"deviation={curvature:+.4f}")
    if abs(curvature) < 0.02:
        print("  → z(0.5) is consistent with linear interpolation (no detected curvature)")
    elif curvature > 0:
        print("  → z(0.5) is CLOSER to z(0) than expected (asymmetric curve toward unique)")
    else:
        print("  → z(0.5) is FARTHER from z(0) than expected (genuine interior curvature)")

    results = {
        "mode": "cka_matrix", "dataset": dataset, "approach": approach,
        "lam_grid": grid, "matrix_vision": mat_v, "matrix_text": mat_t,
        "matrix_overall": mat_all, "curvature_indicator": round(curvature, 4),
    }
    save_dir = out_dir or enc_dir
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "probe_cka_matrix.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results → {out_path}")
    return results


# LoRA branch diagnostic

def run_lora_diag(enc_dir, dataset, approach, enc_dim, proj_dim,
                  batch_size, device, enc_ckpt_v=None, enc_ckpt_t=None,
                  film_mode="proj", out_dir=None):
    """
    Per-layer Frobenius norm diagnostic for approach 4 LoRA heads.

    Reports ‖ΔW_R‖_F, ‖ΔW_U‖_F, ‖ΔW_R−ΔW_U‖_F for all heads.
    For 3-branch runs also reports ‖ΔW_M‖_F.

    Interpretation:
      ‖ΔW_M‖_F → 0         : Pareto frontier is locally linear (informative result)
      ‖ΔW_M‖_F >> 0        : curvature branch active, frontier is genuinely curved
      ‖ΔW_R−ΔW_U‖_F >> 0  : endpoint branches well separated
    """
    if approach != 4:
        print("  lora_diag is only meaningful for approach 4.")
        return {}

    meta_path = os.path.join(enc_dir, "train_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        proj_dim = meta.get("proj_dim", proj_dim)
        enc_dim  = meta.get("enc_dim",  enc_dim)

    branches = meta.get("lora_branches", 3)
    rank     = meta.get("lora_rank",     4)
    rank_m   = meta.get("lora_rank_m",   rank)
    adim = _actual_enc_dim(approach, enc_dim)
    if branches == 3:
        HEAD_CLS, hkw = LoRATriProjectionHead, dict(rank=rank, rank_m=rank_m)
    else:
        HEAD_CLS, hkw = LoRADualProjectionHead, dict(rank=rank)

    head_v = HEAD_CLS(adim, proj_dim, **hkw).to(device)
    head_t = HEAD_CLS(adim, proj_dim, **hkw).to(device)
    for name, mod in [("proj_v", head_v), ("proj_t", head_t)]:
        w = os.path.join(enc_dir, f"{name}.pth")
        mod.load_state_dict(torch.load(w, map_location=device, weights_only=True))
        mod.eval()

    has_m = branches == 3
    hdr_m = f" {'‖ΔW_M‖_F':>12}" if has_m else ""
    print(f"\n  LoRA branch diagnostic — {dataset} ap{approach} ({branches}-branch)")
    print(f"  {'Layer':<10} {'modal':<7} {'‖ΔW_R‖_F':>12} {'‖ΔW_U‖_F':>12} {'‖ΔW_R−ΔW_U‖_F':>16}{hdr_m}")
    print(f"  {'-'*(62 + (13 if has_m else 0))}")

    layer_results = []
    for mod_name, head in [("vision", head_v), ("text", head_t)]:
        for i, layer in enumerate([head.layer1, head.layer2, head.layer3], 1):
            with torch.no_grad():
                dW_r = (layer.A_r @ layer.B_r).cpu()
                dW_u = (layer.A_u @ layer.B_u).cpu()
                norm_r    = dW_r.norm(p="fro").item()
                norm_u    = dW_u.norm(p="fro").item()
                norm_diff = (dW_r - dW_u).norm(p="fro").item()
                row = {"layer": i, "modality": mod_name,
                       "norm_r": round(norm_r, 6), "norm_u": round(norm_u, 6),
                       "norm_diff": round(norm_diff, 6)}
                m_str = ""
                if has_m:
                    dW_m   = (layer.A_m @ layer.B_m).cpu()
                    norm_m = dW_m.norm(p="fro").item()
                    row["norm_m"] = round(norm_m, 6)
                    m_str = f" {norm_m:>12.4f}"
            print(f"  {'layer'+str(i):<10} {mod_name:<7} {norm_r:>12.4f} {norm_u:>12.4f} {norm_diff:>16.4f}{m_str}")
            layer_results.append(row)

    results = {"mode": "lora_diag", "dataset": dataset, "approach": approach,
               "lora_branches": branches, "enc_dir": enc_dir, "layers": layer_results}
    save_dir = out_dir or enc_dir
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "probe_lora_diag.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results → {out_path}")
    return results


# Curve proof helpers

@torch.no_grad()
def _cpz_lam(h_v, h_t, proj_v, proj_t, lam, device, chunk=512):
    """[z_v || z_t] using forward(h, lam) — uses ΔW_M for LoRATriProjectionHead."""
    zv, zt = [], []
    for i in range(0, h_v.shape[0], chunk):
        zv.append(proj_v(h_v[i:i+chunk].to(device), lam).cpu())
        zt.append(proj_t(h_t[i:i+chunk].to(device), lam).cpu())
    return np.concatenate([torch.cat(zv).numpy(), torch.cat(zt).numpy()], axis=1)


@torch.no_grad()
def _cpz_interp(h_v, h_t, proj_v, proj_t, lam, device, chunk=512):
    """normalize(lam*z_r + (1-lam)*z_u) — bypasses ΔW_M branch entirely."""
    zv, zt = [], []
    for i in range(0, h_v.shape[0], chunk):
        hv = h_v[i:i+chunk].to(device)
        ht = h_t[i:i+chunk].to(device)
        zv.append(F.normalize(lam * proj_v.forward_r(hv) + (1-lam) * proj_v.forward_u(hv), dim=-1).cpu())
        zt.append(F.normalize(lam * proj_t.forward_r(ht) + (1-lam) * proj_t.forward_u(ht), dim=-1).cpu())
    return np.concatenate([torch.cat(zv).numpy(), torch.cat(zt).numpy()], axis=1)


def run_curve_proof(enc_dir, dataset, approach, enc_dim, proj_dim,
                    batch_size, device, enc_ckpt_v=None, enc_ckpt_t=None,
                    film_mode="proj", out_dir=None, linear_dir=None):
    """
    4 experiments comparing LoRA-curve (fix3, enc_dir) vs LoRA-linear (linear_dir).

    E1 — Does ΔW_M contribute at λ=0.5?
         curve forward(h,0.5) vs normalize(0.5*z_r+0.5*z_u) [no M branch].
    E2 — ΔW_M norm: ‖ΔW_M‖ as % of ‖ΔW_R−ΔW_U‖ per layer.
    E3 — CKA curvature: deviation of z(0.5) from linear prediction.
    E4 — Accuracy at intermediate λ ∉ {0,0.1,...,1}.
    """
    if not linear_dir:
        print("  ERROR: curve_proof mode requires --linear_dir")
        return {}
    if approach != 4:
        print("  curve_proof mode is only for approach 4")
        return {}

    print(f"  curve  : {enc_dir}")
    print(f"  linear : {linear_dir}")

    enc_v_c, enc_t_c, proj_v_c, proj_t_c, _, _ = _load_models(
        enc_dir,    approach, dataset, enc_dim, proj_dim, device, enc_ckpt_v, enc_ckpt_t, film_mode)
    enc_v_l, enc_t_l, proj_v_l, proj_t_l, _, _ = _load_models(
        linear_dir, approach, dataset, enc_dim, proj_dim, device, film_mode=film_mode)

    print("  Extracting backbone features (valid + test) for both models...")
    hv_c_val, ht_c_val, l_val = _extract_backbone(enc_v_c, enc_t_c, dataset, "valid", batch_size, device)
    hv_c_te,  ht_c_te,  l_te  = _extract_backbone(enc_v_c, enc_t_c, dataset, "test",  batch_size, device)
    hv_l_val, ht_l_val, _     = _extract_backbone(enc_v_l, enc_t_l, dataset, "valid", batch_size, device)
    hv_l_te,  ht_l_te,  _     = _extract_backbone(enc_v_l, enc_t_l, dataset, "test",  batch_size, device)

    res = {"mode": "curve_proof", "dataset": dataset, "approach": approach,
           "curve_dir": enc_dir, "linear_dir": linear_dir}

    # E1
    print("\n  [E1] ΔW_M contribution at λ=0.5")
    lam = 0.5
    ac = _lr_accuracy(
        _cpz_lam(hv_c_val, ht_c_val, proj_v_c, proj_t_c, lam, device), l_val,
        _cpz_lam(hv_c_te,  ht_c_te,  proj_v_c, proj_t_c, lam, device), l_te) * 100
    ai = _lr_accuracy(
        _cpz_interp(hv_c_val, ht_c_val, proj_v_c, proj_t_c, lam, device), l_val,
        _cpz_interp(hv_c_te,  ht_c_te,  proj_v_c, proj_t_c, lam, device), l_te) * 100
    al = _lr_accuracy(
        _cpz_lam(hv_l_val, ht_l_val, proj_v_l, proj_t_l, lam, device), l_val,
        _cpz_lam(hv_l_te,  ht_l_te,  proj_v_l, proj_t_l, lam, device), l_te) * 100
    dm = ac - ai
    dv = ac - al
    e1v = "✓ M adds value" if dm > 0.5 else ("~ marginal" if dm > -0.5 else "✗ M hurts")
    print(f"    curve  forward(h,0.5)  : {ac:.2f}%  [uses ΔW_M]")
    print(f"    curve  0.5·z_r+0.5·z_u: {ai:.2f}%  [bypasses ΔW_M]")
    print(f"    linear forward(h,0.5)  : {al:.2f}%")
    print(f"    ΔW_M adds {dm:+.2f}pp vs interp  |  curve vs linear: {dv:+.2f}pp  |  {e1v}")
    res["E1"] = {"acc_curve": round(ac,2), "acc_interp": round(ai,2), "acc_linear": round(al,2),
                 "delta_M_pp": round(dm,2), "delta_vs_linear_pp": round(dv,2), "verdict": e1v}

    # E2
    print("\n  [E2] ΔW_M norm per layer")
    meta_path = os.path.join(enc_dir, "train_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    rank   = meta.get("lora_rank",   4)
    rank_m = meta.get("lora_rank_m", rank)
    adim   = _actual_enc_dim(approach, meta.get("enc_dim", enc_dim))
    pdim   = meta.get("proj_dim", proj_dim)
    e2_layers = []
    print(f"    {'Layer':<8} {'Modal':<7} {'‖ΔW_R‖':>9} {'‖ΔW_U‖':>9} {'‖ΔW_R-U‖':>10} {'‖ΔW_M‖':>9} {'M%':>7}")
    print(f"    {'-'*56}")
    for mname, hfile in [("vision", "proj_v.pth"), ("text", "proj_t.pth")]:
        head = LoRATriProjectionHead(adim, pdim, rank=rank, rank_m=rank_m).to(device)
        head.load_state_dict(torch.load(os.path.join(enc_dir, hfile),
                                        map_location=device, weights_only=True))
        head.eval()
        for i, layer in enumerate([head.layer1, head.layer2, head.layer3], 1):
            with torch.no_grad():
                dWr = layer.A_r @ layer.B_r
                dWu = layer.A_u @ layer.B_u
                dWm = layer.A_m @ layer.B_m
                nr = dWr.norm(p="fro").item()
                nu = dWu.norm(p="fro").item()
                nd = (dWr - dWu).norm(p="fro").item()
                nm = dWm.norm(p="fro").item()
                pct = (nm / nd * 100) if nd > 1e-8 else 0.0
            print(f"    L{i:<7} {mname:<7} {nr:>9.4f} {nu:>9.4f} {nd:>10.4f} {nm:>9.4f} {pct:>6.1f}%")
            e2_layers.append({"layer": i, "modality": mname,
                               "norm_r": round(nr,4), "norm_u": round(nu,4),
                               "norm_diff": round(nd,4), "norm_m": round(nm,4),
                               "m_pct_of_diff": round(pct,2)})
    avg_pct = float(np.mean([r["m_pct_of_diff"] for r in e2_layers]))
    avg_nm  = float(np.mean([r["norm_m"]        for r in e2_layers]))
    avg_nd  = float(np.mean([r["norm_diff"]      for r in e2_layers]))
    e2v = ("✓ M active (>10%)" if avg_pct > 10
           else ("~ small (2-10%)" if avg_pct > 2 else "✗ M near-zero (<2%)"))
    print(f"    avg ‖ΔW_M‖={avg_nm:.4f}  M={avg_pct:.1f}% of endpoint diff  {e2v}")
    res["E2"] = {"layers": e2_layers, "avg_norm_m": round(avg_nm,4),
                 "avg_norm_diff": round(avg_nd,4), "avg_m_pct_of_diff": round(avg_pct,2),
                 "verdict": e2v}

    # E3
    print("\n  [E3] Trajectory curvature (CKA-based)")
    print("       deviation = CKA(z0,z0.5) − (1+CKA(z0,z1))/2")
    print("       negative = z(0.5) farther from z(0) than linear predicts = real curvature")
    cka_lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    e3 = {}
    for label, hv_te, ht_te, pv, pt in [
        ("curve",  hv_c_te, ht_c_te, proj_v_c, proj_t_c),
        ("linear", hv_l_te, ht_l_te, proj_v_l, proj_t_l),
    ]:
        zs = {lm: _cpz_lam(hv_te, ht_te, pv, pt, lm, device) for lm in cka_lams}
        half = zs[0.0].shape[1] // 2
        c01 = linear_cka(zs[0.0], zs[1.0])
        c05 = linear_cka(zs[0.0], zs[0.5])
        pred = (1.0 + c01) / 2
        dev  = c05 - pred
        c01v = linear_cka(zs[0.0][:, :half], zs[1.0][:, :half])
        c05v = linear_cka(zs[0.0][:, :half], zs[0.5][:, :half])
        devv = c05v - (1.0 + c01v) / 2
        c01t = linear_cka(zs[0.0][:, half:], zs[1.0][:, half:])
        c05t = linear_cka(zs[0.0][:, half:], zs[0.5][:, half:])
        devt = c05t - (1.0 + c01t) / 2
        print(f"    [{label}]  CKA(z0,z1)={c01:.4f}  CKA(z0,z0.5)={c05:.4f}  pred={pred:.4f}  dev={dev:+.4f}")
        print(f"      vision dev={devv:+.4f}   text dev={devt:+.4f}")
        e3[label] = {"deviation_overall": round(dev,4), "cka_z0_z1_overall": round(c01,4),
                     "cka_z0_z05_overall": round(c05,4), "linear_pred_overall": round(pred,4),
                     "deviation_vision": round(devv,4), "cka_z0_z1_vision": round(c01v,4),
                     "cka_z0_z05_vision": round(c05v,4),
                     "deviation_text": round(devt,4), "cka_z0_z1_text": round(c01t,4),
                     "cka_z0_z05_text": round(c05t,4)}
    gain = e3["linear"]["deviation_overall"] - e3["curve"]["deviation_overall"]
    e3v = ("✓ Curve more curved than linear" if gain > 0.005
           else ("~ similar curvature" if abs(gain) <= 0.005 else "✗ Curve less curved than linear"))
    print(f"    curvature gain (linear_dev − curve_dev): {gain:+.4f}  {e3v}")
    e3["curvature_gain"] = round(gain, 4)
    e3["verdict"] = e3v
    res["E3"] = e3

    # E4
    print(f"\n  [E4] Accuracy at intermediate λ ∉ {{0,0.1,...,1}}")
    print(f"    {'λ':>6} | {'curve':>10} | {'linear':>10} | {'Δ':>7}")
    print(f"    {'-'*38}")
    ca, la = [], []
    for lm in CURVE_PROOF_INTERMED_LAMS:
        ac = _lr_accuracy(_cpz_lam(hv_c_val, ht_c_val, proj_v_c, proj_t_c, lm, device), l_val,
                          _cpz_lam(hv_c_te,  ht_c_te,  proj_v_c, proj_t_c, lm, device), l_te) * 100
        al = _lr_accuracy(_cpz_lam(hv_l_val, ht_l_val, proj_v_l, proj_t_l, lm, device), l_val,
                          _cpz_lam(hv_l_te,  ht_l_te,  proj_v_l, proj_t_l, lm, device), l_te) * 100
        ca.append(round(ac, 2)); la.append(round(al, 2))
        print(f"    {_lamstr(lm):>12s} | {ac:>9.2f}% | {al:>9.2f}% | {(ac-al):>+6.2f}pp")
    avgc = float(np.mean(ca)); avgl = float(np.mean(la)); avgd = avgc - avgl
    print(f"    {'-'*38}")
    print(f"    {'avg':>6} | {avgc:>9.2f}% | {avgl:>9.2f}% | {avgd:>+6.2f}pp")
    e4v = ("✓ Curve better at intermediate λ" if avgd > 1
           else ("~ similar" if abs(avgd) <= 1 else "✗ Linear better at intermediate λ"))
    print(f"    {e4v}")
    res["E4"] = {"lams": CURVE_PROOF_INTERMED_LAMS, "curve_accs": ca, "linear_accs": la,
                 "avg_curve": round(avgc,2), "avg_linear": round(avgl,2),
                 "avg_delta_pp": round(avgd,2), "verdict": e4v}

    save_dir = out_dir or enc_dir
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "probe_curve_proof.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  Results → {out_path}")
    return res


# CLI

def main():
    p = argparse.ArgumentParser(
        description="Linear evaluation probe for Pareto SSL MultiBench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--enc_dir",    required=True)
    p.add_argument("--dataset",    choices=["mosi", "humor", "mustard", "mosei", "avmnist", "enrico", "mosei_multitask", "chsims"], required=True)
    p.add_argument("--task", default=None,
                   help="multitask datasets only: which label to probe -- mosei_multitask "
                        "(sentiment|happy|sad|anger|disgust) or chsims (M|T|V). Omit to loop "
                        "over all usable tasks.")
    p.add_argument("--approach",   type=int, choices=[1, 2, 3, 4, 5], required=True)
    p.add_argument("--mode",       choices=["validation", "validation_cv", "fewshot", "cka", "cka_curve", "cka_matrix", "lora_diag", "curve_proof", "fixed", "matched", "backbone"], default="validation")
    p.add_argument("--out_dir",    default=None,
                   help="Write probe_results.json here (default: enc_dir)")
    p.add_argument("--enc_dim",    type=int, default=128)
    p.add_argument("--proj_dim",   type=int, default=DEFAULT_PROJ_DIM)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--enc_ckpt_v", default=None)
    p.add_argument("--enc_ckpt_t", default=None)
    p.add_argument("--film_mode", choices=list(FILM_MODES), default="none",
                   help="Must match the film_mode used during training")
    p.add_argument("--fixed_lam", default=None,
                   help="Comma-separated preference, e.g. 0.0,0.25,0.75. Report test "
                        "at THIS lambda instead of the validation argmax. Nearest grid "
                        "point is used. Lets one operating point be evaluated across "
                        "every seed, which the per-seed argmax cannot do.")
    p.add_argument("--scaler", choices=["standard", "center", "none"],
                   default="standard",
                   help="probe preprocessing. 'standard' divides each coordinate by "
                        "its sd, which whitens a PCA basis and erases the eigenvalue "
                        "ordering; 'center' centres only. Apply the SAME setting to "
                        "every arm being compared.")
    p.add_argument("--pca_shared", action="store_true",
                   help="dim_pca: fit ONE PCA basis per block, shared across every "
                        "preference. Default is a separate basis per (lambda, block), "
                        "which is what the reported numbers use. Sharing was tested "
                        "and did not improve task-discriminability.")
    p.add_argument("--wide_c", action="store_true",
                   help="Widen the C grid to 1e-3 .. 1e4 (8 points). The default 4-point "
                        "grid saturates at its ceiling on 3-4 of 5 fits, but widening it "
                        "moved no reported number by more than 0.6, so it is off by "
                        "default and the reported results use the narrow grid.")
    p.add_argument("--probe_fit", choices=["valid", "train", "trainval"], default="valid",
                   help="Where the linear probe is fit. valid (default, historical): probe "
                        "on valid, lambda chosen on a 20%% slice of it. train: standard linear "
                        "evaluation -- fit on train, C and lambda chosen on valid, test once. "
                        "trainval: as train, then refit the final probe on train+valid. Use "
                        "the SAME mode for every method on a dataset.")
    p.add_argument("--save_z", default=None,
                   help="fixed/matched mode: write z_valid/z_test (+ the raw 7-label "
                        "vector for mosei_multitask) to this .npz for offline analysis")
    p.add_argument("--narrow_c", action="store_true",
                   help="No-op: the narrow grid is the default. Accepted so scripts "
                        "written while wide was briefly the default keep working.")
    p.add_argument("--tune_c", action="store_true",
                   help="tune the logistic C in the lambda sweep (default: fixed 1.0, "
                        "as published). Required when changing --scaler, or C becomes "
                        "a confound favouring whichever arm matches the old strength.")
    p.add_argument("--readout",
                   choices=["amp", "dim", "dim_rand", "dim_pca", "dim_split", "none"],
                   default="amp",
                   help="decomp_R readout: amp = sqrt(lambda) amplitude scaling "
                        "(original; a linear probe can undo it, so only the SUPPORT "
                        "matters). dim = allocate round(lambda*proj_dim) DIMENSIONS "
                        "per block, which no reweighting can recover. none = no "
                        "readout weighting at all, so lambda acts ONLY through "
                        "W(lambda) in the LoRA encoder/heads.")
    p.add_argument("--modality",
                   choices=["both", "vision", "text", "shared",
                            "shared_vision", "shared_text", "unique_vision", "unique_text"],
                   default="both",
                   help="What to probe: both|vision|text|shared|shared_vision|shared_text|unique_vision|unique_text")
    p.add_argument("--no_cka", action="store_true", help="Skip CKA diagnostic")
    p.add_argument("--linear_dir", default=None,
                   help="Path to LoRA-linear run dir (required for --mode curve_proof)")
    args = p.parse_args()
    _READOUT[0] = args.readout
    _SCALER[0] = args.scaler
    _WIDE_C[0] = args.wide_c and not args.narrow_c
    SAVE_Z[0] = args.save_z
    _PROBE_FIT[0] = args.probe_fit
    _PCA_SHARED[0] = args.pca_shared
    _TUNE_C[0] = args.tune_c
    if args.fixed_lam:
        _FIXED_LAM[0] = tuple(float(x) for x in args.fixed_lam.split(","))

    out_dir = args.out_dir or args.enc_dir
    # A non-default readout writes to its own directory: re-probing a checkpoint
    # must not silently overwrite the amplitude-readout results it is compared with.
    if args.readout != "amp":
        out_dir = os.path.join(out_dir, f"readout_{args.readout}")
    # The probe protocol is part of a run's identity, not just its readout: a
    # 'center' run and a 'standard' run of the same readout are different numbers
    # and must not land on the same path. Only non-default settings extend it, so
    # every existing result keeps its current location.
    _proto = []
    if args.probe_fit != "valid":
        _proto.append(f"probefit_{args.probe_fit}")
    if args.scaler != "standard":
        _proto.append(f"scaler{args.scaler}")
    if args.tune_c:
        _proto.append("tunec")
    if _WIDE_C[0]:
        _proto.append("widec")
    if args.pca_shared:
        _proto.append("pcashared")
    if _proto:
        out_dir = os.path.join(out_dir, "_".join(_proto))
    if args.fixed_lam:
        out_dir = os.path.join(out_dir, "fixedlam_" + args.fixed_lam.replace(",", "_"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Approach : {args.approach}")
    print(f"  Readout  : {args.readout}"
          + _READOUT_BLURB.get(args.readout, ""))
    print(f"  Mode     : {args.mode}")
    print(f"  Enc dir  : {args.enc_dir}")
    print(f"{'='*60}")

    kw = dict(enc_dir=args.enc_dir, dataset=args.dataset, approach=args.approach,
              enc_dim=args.enc_dim, proj_dim=args.proj_dim,
              batch_size=args.batch_size, device=args.device,
              enc_ckpt_v=args.enc_ckpt_v, enc_ckpt_t=args.enc_ckpt_t,
              film_mode=args.film_mode)
    kw_probe = {**kw, "modality": args.modality}

    # which label(s) to score
    # mosei_multitask carries 7 labels and exists precisely to ask whether
    # DIFFERENT tasks select DIFFERENT preferences. Scoring one label per
    # invocation would answer nothing, so the default is to sweep every usable
    # task. Each task writes into its own sub-directory using the SAME filenames
    # as a single-label run, so every downstream analysis script keeps working
    # unchanged — it just gets pointed at task_<name>/ instead of the run root.
    if is_multitask_dataset(args.dataset):
        tasks = [args.task] if args.task else usable_tasks(args.dataset)
        print(f"  Tasks    : {', '.join(tasks)}")
    else:
        if args.task:
            print(f"  (--task is mosei_multitask-only; ignored for {args.dataset})")
        tasks = [None]

    def _task_dir(t):
        if t is None:
            return out_dir
        d = os.path.join(out_dir, f"task_{t}")
        os.makedirs(d, exist_ok=True)
        return d

    def _banner(t):
        if t is not None:
            print(f"\n{'#'*60}\n#  task: {t}\n{'#'*60}")

    # backbone: identical protocol for EVERY method (concat of frozen encoders,
    # all heads discarded). Runs before the CKA skip-logic because it applies to
    # lambda-methods and joint baselines alike.
    if args.mode == "backbone":
        for t in tasks:
            _MT_TASK[0] = t
            _banner(t)
            _meta_f = os.path.join(args.enc_dir, "train_meta.json")
            _fm = (json.load(open(_meta_f)).get("film_mode", "none")
                   if os.path.exists(_meta_f) else "none")
            res = run_backbone(args.enc_dir, args.dataset, args.approach,
                               args.enc_dim, args.proj_dim, args.batch_size,
                               args.device, args.enc_ckpt_v, args.enc_ckpt_t,
                               film_mode=_fm)
            out_file = os.path.join(_task_dir(t), "probe_backbone.json")
            with open(out_file, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  Saved → {out_file}")
        return

    if args.mode in ("fixed", "matched"):
        for t in tasks:
            _MT_TASK[0] = t
            _banner(t)
            res = run_fixed(args.enc_dir, args.dataset, args.approach,
                            args.enc_dim, args.proj_dim, args.batch_size, args.device,
                            matched_only=(args.mode == "matched"))
            out_file = os.path.join(_task_dir(t), f"probe_{args.mode}.json")
            with open(out_file, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  Saved → {out_file}")
        return

    # CKA diagnostics compare z(λ) across λ, so they only mean anything for
    # λ-conditioned methods. The joint/baseline methods (gmc, comm, factorcl) have
    # no λ at all — running CKA on them is meaningless (and crashes on their head
    # layout), so skip it automatically regardless of --no_cka.
    _LAMBDA_BLIND = {"gmc", "comm", "factorcl", "clip", "cross_self"}
    _m_path = os.path.join(args.enc_dir, "train_meta.json")
    _train_method = None
    if os.path.exists(_m_path):
        with open(_m_path) as _f:
            _train_method = json.load(_f).get("method")
    _cka_ok = _train_method not in _LAMBDA_BLIND

    if args.mode in ("validation", "validation_cv", "fewshot") and not args.no_cka:
        if _cka_ok:
            # CKA compares representations across λ; it does not touch the labels,
            # so it is run ONCE for the run rather than repeated per task.
            #
            # It is always measured through the deterministic 'dim' slice, even when
            # the probe uses dim_rand/dim_pca. Two reasons. (1) CKA compares z(λ_i)
            # against z(λ_j); a per-λ fitted PCA basis would make that comparison
            # partly about the bases rather than the representations. (2) run_cka
            # extracts the TEST split directly and runs before any validation sweep,
            # so a fitted read-out could only get a basis by fitting on test -- which
            # the leakage guard correctly refuses. The slice needs no fitting.
            run_cka(**kw, out_dir=out_dir)
            run_cka_matrix(**kw, out_dir=out_dir)
        else:
            print(f"  Skipping CKA — '{_train_method}' is λ-blind (no λ to compare across).")

    for t in tasks:
        _MT_TASK[0] = t
        _banner(t)
        t_dir = _task_dir(t)

        if args.mode == "curve_proof":
            res = run_curve_proof(**kw, out_dir=t_dir, linear_dir=args.linear_dir)
        elif args.mode == "lora_diag":
            res = run_lora_diag(**kw, out_dir=t_dir)
        if args.mode == "validation":
            res = run_validation(**kw_probe)
            # auto per-modality probes
            for mod in ("vision", "text"):
                mod_res = run_validation(**{**kw, "modality": mod})
                mod_file = os.path.join(t_dir, f"probe_validation_{mod}.json")
                with open(mod_file, "w") as f:
                    json.dump(mod_res, f, indent=2)
                print(f"Results → {mod_file}")
        elif args.mode == "validation_cv":
            res = run_validation_cv(**kw_probe)
        elif args.mode == "fewshot":
            res = run_fewshot(**kw_probe)
        elif args.mode == "cka_curve":
            res = run_cka_curve(**kw, out_dir=t_dir)
        elif args.mode == "cka_matrix":
            res = run_cka_matrix(**kw, out_dir=t_dir)
        else:
            res = run_cka(**kw, out_dir=t_dir)

        if isinstance(res, dict) and t is not None:
            res["task"] = t
        mod_suffix = f"_{args.modality}" if args.modality != "both" else ""
        out_file = os.path.join(t_dir, f"probe_{args.mode}{mod_suffix}.json")
        with open(out_file, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nResults → {out_file}")


if __name__ == "__main__":
    main()
