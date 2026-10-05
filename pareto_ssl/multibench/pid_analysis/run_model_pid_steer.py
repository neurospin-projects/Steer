#!/usr/bin/env python3
"""Model PID of STEER's representation across the preference simplex (Trifeature).

PID plan, steps 1-4: show that moving lambda toward a vertex changes WHICH
information-theoretic interaction the model's prediction uses -- not merely that probe
accuracy moves. For every preference lambda on the 15-point simplex and every target
(R = shape, U1 = deformation, U2 = texture):

    z(lambda)  built exactly as probe_lambda_cond() builds it, `dim` read-out included
    yhat       out-of-fold prediction of the same linear probe the accuracy table uses
    PID        BROJA decomposition of I(X1, X2 ; yhat) into R / U1 / U2 / S  (CVX)

Sources are the two modalities' GENERATIVE FACTORS, the exact discrete description of
each input on this synthetic dataset -- no clustering, no estimator noise:

    X1 = (R, U1) of the M1 image        X2 = (R, U2) of the M2 image

The factor irrelevant to a target is dropped (U1/U2 for the R target, and so on) to keep
the convex program small; that is valid only while the dropped factor is independent of
everything kept, which is CHECKED against a shuffled null and aborts if violated.

Three things this file is careful about, each of which silently corrupts the result:

1. READ-OUT. lambda acts partly at probe time by allocating the dimension budget across
   the R/U1/U2 blocks. PID on full-width blocks cannot see that and measures the encoder
   alone -- the likely reason the earlier `simclr_lora_curve` sweep came out flat. So
   `_dim_split` / `_select_dims` from benchmark.py are applied here too.

2. PAIR SAMPLING. Pairs are (M1_i, M2_j) with R_i = R_j. Drawing them with replacement
   from a few hundred records over-represents some (R, U) combinations and manufactures
   dependence between factors that are independent by construction: 1861 pairs from the
   200-record test split scored 0.59 bits of shuffle-corrected I(U1;R), against -0.003
   over those same records. Pairs are therefore built over ALL records, with each record
   used about equally often (a permutation-shift scheme per R class).

3. FINITE-SAMPLE FLOOR. With |X1| up to 150, plug-in PID gives every component a positive
   offset even for a random target. Each cell also decomposes a SHUFFLED target; the
   quantity to interpret is raw minus that floor. Both are stored.

yhat is out-of-fold: records are split into K folds, the probe is fitted on pairs whose
two records both lie outside the fold and predicts pairs whose records both lie inside it,
so no record is ever in fit and evaluation at once. Pooling the folds restores the full
record diversity that keeps the factors independent.

Reference rows: the same decomposition with the TRUE label (dataset PID). By construction
that must be pure redundancy for R (log2 15 = 3.907 bits) and pure uniqueness for U1/U2
(log2 10 = 3.322) -- an end-to-end check on the whole pipeline before any model number is
read.

Usage:
  python pareto_ssl/multibench/pid_analysis/run_model_pid_steer.py \
      --enc-dir pareto_ssl/results_corr_sweep/_parts/corr00_simclr_simplex_enc_decomp_R_s42_lc05/simclr_simplex_enc_decomp_R \
      --data-dir data/tri15_corr00 --out results/steer_trifeature/steer_s42_lc05.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from pid_cvx import broja_pid, validate as validate_solver            # noqa: E402
from pareto_ssl.networks import (AlexNetEncoder, LoRAAlexNetEncoder,  # noqa: E402
                                 ProjectionHead)
from pareto_ssl.datasets import _EVAL, _remap                         # noqa: E402
from pareto_ssl.benchmark import (_dim_split, _select_dims,           # noqa: E402
                                  _kshot_subsample)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGETS = ["share", "unique1", "unique2"]
LOG2 = np.log(2.0)


# records and pairs
class RecordImages(Dataset):
    """One item per RECORD: its two paired images. Features are cached per record and
    pairs index into them, so N records cost 2N forwards instead of 2 per pair.

    The pair is whatever the checkpoint was TRAINED with (train_meta.json
    "modality_pair"): M1 shape+deformation, M2 shape+texture, M3 shape+colour. Reading
    M1/M2 unconditionally would quietly decompose the wrong images for a run trained on
    another pair -- and the accuracies would look plausible, since every modality shares
    the same R.
    """

    def __init__(self, data_dir, records, keys=("M1_path", "M2_path")):
        self.data_dir, self.records, self.keys = data_dir, records, keys

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return tuple(_EVAL(Image.open(_remap(r[k], self.data_dir)).convert("RGB"))
                     for k in self.keys)


def make_pairs(R, idx, per_record=2, seed=42):
    """Ordered (i, j) pairs with R_i == R_j, i != j, each record used equally often.

    Within an R class the members are permuted and paired against shifts of themselves,
    so every record appears exactly `per_record` times as M1 and `per_record` times as
    M2. Sampling with replacement instead is what manufactured the factor dependence
    described in the module docstring.
    """
    rng = np.random.default_rng(seed)
    pairs = []
    for r in np.unique(R[idx]):
        mem = idx[R[idx] == r]
        if len(mem) < 2:
            continue
        mem = rng.permutation(mem)
        for s in range(1, min(per_record, len(mem) - 1) + 1):
            pairs.extend(zip(mem, np.roll(mem, -s)))
    return np.asarray(pairs, dtype=int)


# model
def load_steer(enc_dir, device=DEVICE):
    """Mirrors the decomp_R branch of probe_lambda_cond()."""
    with open(os.path.join(enc_dir, "train_meta.json")) as f:
        meta = json.load(f)
    method = meta.get("method", "")
    if "decomp_R" not in method:
        raise SystemExit(f"{enc_dir} is method {method!r}, not a decomp_R (STEER) run")
    is_proj = "proj_decomp_R" in method
    latent, proj_dim = meta.get("latent_dim", 512), meta.get("proj_dim", 256)
    branches, rank = meta.get("branches", 2), meta.get("rank", 4)

    def _enc(name):
        e = (AlexNetEncoder(latent) if is_proj
             else LoRAAlexNetEncoder(latent, branches=branches, rank=rank)).to(device)
        e.load_state_dict(torch.load(os.path.join(enc_dir, name), map_location=device,
                                     weights_only=True))
        return e.eval()

    def _head(name):
        h = ProjectionHead(latent, proj_dim).to(device)
        h.load_state_dict(torch.load(os.path.join(enc_dir, name), map_location=device,
                                     weights_only=True))
        return h.eval()

    return dict(enc1=_enc("lora_enc_0.pth"), enc2=_enc("lora_enc_1.pth"),
                pr1=_head("proj_m1_r.pth"), pr2=_head("proj_m2_r.pth"),
                pu1=_head("proj_m1_u.pth"), pu2=_head("proj_m2_u.pth"),
                is_proj=is_proj, proj_dim=proj_dim, meta=meta,
                side=meta.get("simplex_side", 5), lam_club=meta.get("lam_club"))


@torch.no_grad()
def trunk_feats(model, loader, device=DEVICE):
    """lambda-independent encoder features, one row per record."""
    H1, H2 = [], []
    for m1, m2 in loader:
        m1, m2 = m1.to(device), m2.to(device)
        if model["is_proj"]:
            H1.append(model["enc1"](m1).cpu()); H2.append(model["enc2"](m2).cpu())
        else:
            H1.append(model["enc1"].feat(m1).cpu()); H2.append(model["enc2"].feat(m2).cpu())
    return torch.cat(H1), torch.cat(H2)


@torch.no_grad()
def blocks_at(model, H1, H2, lam, batch=1024, device=DEVICE):
    """Per-record projection blocks at this preference: (A=r1(h1), B=r2(h2), U1, U2)."""
    w_r, w_u1, w_u2 = lam
    A, B, U1, U2 = [], [], [], []
    for s in range(0, len(H1), batch):
        f1, f2 = H1[s:s + batch].to(device), H2[s:s + batch].to(device)
        if model["is_proj"]:
            h1, h2 = f1, f2
        else:
            h1 = model["enc1"].head_mix_from_feat(f1, w_r, w_u1)
            h2 = model["enc2"].head_mix_from_feat(f2, w_r, w_u2)
        A.append(model["pr1"](h1).cpu()); B.append(model["pr2"](h2).cpu())
        U1.append(model["pu1"](h1).cpu()); U2.append(model["pu2"](h2).cpu())
    return torch.cat(A), torch.cat(B), torch.cat(U1), torch.cat(U2)


def z_for_pairs(blocks, pairs, lam, readout="dim"):
    """z(lambda) for each pair, from cached per-record blocks (same maths as the probe)."""
    A, B, U1, U2 = blocks
    i, j = pairs[:, 0], pairs[:, 1]
    Rb = F.normalize((A[i] + B[j]) / 2.0, dim=-1)
    U1b, U2b = U1[i], U2[j]
    w_r, w_u1, w_u2 = lam
    if readout.startswith("dim"):
        n = _dim_split((w_r, w_u1, w_u2), Rb.shape[-1])
        parts = [_select_dims(b, k, readout, key=(tuple(lam), bi))
                 for bi, (b, k) in enumerate(zip((Rb, U1b, U2b), n)) if k > 0]
        z = F.normalize(torch.cat(parts, dim=-1), dim=-1)
    else:                                       # "amp": legacy sqrt(lambda) read-out
        z = F.normalize(torch.cat([math.sqrt(w_r) * Rb, math.sqrt(w_u1) * U1b,
                                   math.sqrt(w_u2) * U2b], dim=-1), dim=-1)
    return z.numpy()


def probe_predict(ztr, ytr, zte, Cs=(1e-2, 0.1, 1.0, 10.0), seed=42, k_shot=0):
    """benchmark._probe's protocol (C chosen on a held-out fifth), returning predictions.

    k_shot mirrors benchmark._probe: the Trifeature curves in the paper were probed with
    probe_shots=1, and benchmark's own docstring warns that k_shot=0 "may saturate at 1.0".
    It does: fitting on all 22k pairs decodes U1/U2 at 0.96-1.00 at EVERY preference, so
    yhat equals the label everywhere and the model PID collapses onto the dataset PID.
    Matching the reported protocol is what makes the trajectory meaningful.
    """
    if k_shot > 0:
        ztr, ytr = _kshot_subsample(ztr, ytr, k_shot, seed)
    sc = StandardScaler()
    tr, te = sc.fit_transform(ztr), sc.transform(zte)
    rng = np.random.default_rng(42)
    val = rng.choice(len(tr), size=max(1, len(tr) // 5), replace=False)
    fit = np.setdiff1d(np.arange(len(tr)), val)
    best_C, best = Cs[0], -1.0
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=1000, random_state=seed).fit(tr[fit], ytr[fit])
        v = clf.score(tr[val], ytr[val])
        if v > best:
            best, best_C = v, C
    clf = LogisticRegression(C=best_C, max_iter=1000, random_state=seed).fit(tr, ytr)
    return clf.predict(te), float(best_C)


# discrete sources
def _mi_raw(a, b):
    na, nb = a.max() + 1, b.max() + 1
    p = np.zeros((na, nb))
    np.add.at(p, (a, b), 1.0)
    p /= p.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum() / LOG2)


def _mi_excess(a, b, perms=50, seed=0):
    """I(a;b) minus its shuffled-null mean: plug-in MI reads ~0.35 bits on independent
    10- and 15-valued factors at n=300, which would fail an uncorrected check."""
    rng = np.random.default_rng(seed)
    return _mi_raw(a, b) - float(np.mean([_mi_raw(a, rng.permutation(b))
                                          for _ in range(perms)]))


def build_sources(target, R, U1, U2, collapse=True, tol=0.02, force=False):
    nU = int(max(U1.max(), U2.max())) + 1
    y = {"share": R, "unique1": U1, "unique2": U2}[target]
    checks = {}
    if not collapse:
        return R * nU + U1, R * nU + U2, y, checks
    if target == "share":
        checks = {"I(U1;R)": _mi_excess(U1, R), "I(U2;R)": _mi_excess(U2, R),
                  "I(U1;U2)": _mi_excess(U1, U2)}
        x1, x2 = R.copy(), R.copy()
    elif target == "unique1":
        checks = {"I(U2;U1)": _mi_excess(U2, U1), "I(U2;R)": _mi_excess(U2, R)}
        x1, x2 = R * nU + U1, R.copy()
    else:
        checks = {"I(U1;U2)": _mi_excess(U1, U2), "I(U1;R)": _mi_excess(U1, R)}
        x1, x2 = R.copy(), R * nU + U2
    worst = max(checks.values())
    if worst > tol and not force:
        raise SystemExit(f"[{target}] collapse unsafe: a dropped factor carries "
                         f"{worst:.3f} bits (shuffle-corrected) about what is kept "
                         f"({checks}). Use --no-collapse, or --force to override.")
    return x1, x2, y, checks


def joint(x1, x2, y):
    def _compact(a):
        u, inv = np.unique(a, return_inverse=True)
        return inv, len(u)
    i1, n1 = _compact(x1); i2, n2 = _compact(x2); iy, ny = _compact(y)
    p = np.zeros((n1, n2, ny))
    np.add.at(p, (i1, i2, iy), 1.0)
    return p / p.sum()


def pid_null(x1, x2, y, perms=1, seed=0):
    """PID of a shuffled target: the finite-sample floor for these alphabets."""
    if perms <= 0:
        return None
    rng = np.random.default_rng(seed)
    runs = [broja_pid(joint(x1, x2, rng.permutation(y))) for _ in range(perms)]
    return {k: float(np.mean([r[k] for r in runs]))
            for k in ("R", "U1", "U2", "S", "I_total")}


# sweep
def simplex_grid(side=5):
    step = 1.0 / (side - 1)
    return [(round(a * step, 4), round(b * step, 4), round(1 - a * step - b * step, 4))
            for a in range(side) for b in range(side - a)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--readout", default="dim", choices=["dim", "dim_rand", "amp"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--per-record", type=int, default=2,
                    help="how many times each record is used as M1 (and as M2)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--lambdas", default="all")
    ap.add_argument("--no-collapse", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--null-perms", type=int, default=1)
    ap.add_argument("--k-shot", type=int, default=1,
                    help="probe training pairs per class; 1 matches the reported "
                         "Trifeature protocol (probe_shots=1). 0 = all pairs, which "
                         "saturates and makes every preference look identical")
    ap.add_argument("--k-shot-repeats", type=int, default=5,
                    help="independent k-shot draws per fold; predictions are pooled, "
                         "so a 1-shot probe is not judged on a single lucky example")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modality-pair", default=None,
                    help="override the pair, e.g. M3,M2. Needed for checkpoints trained "
                         "before train_meta recorded modality_pair -- without it those "
                         "default to M1,M2 and the WRONG images are decomposed, which "
                         "looks plausible because all three modalities share the same R.")
    ap.add_argument("--dataset-pid-only", action="store_true",
                    help="stop after the true-label reference rows (pipeline check)")
    a = ap.parse_args()

    if not a.skip_validate:
        print("BROJA solver validation (bits):"); validate_solver()

    model = load_steer(a.enc_dir)
    grid = simplex_grid(model["side"])
    idxs = (list(range(len(grid))) if a.lambdas == "all"
            else [int(i) for i in a.lambdas.split(",")])
    targets = a.targets.split(",")
    print(f"checkpoint {a.enc_dir}\n  method={model['meta'].get('method')} "
          f"lam_club={model['lam_club']} proj_dim={model['proj_dim']} "
          f"grid={len(grid)} readout={a.readout} device={DEVICE}", flush=True)

    with open(os.path.join(a.data_dir, "metadata.json")) as f:
        records = json.load(f)
    _MOD_U = {"M1": "U1", "M2": "U2", "M3": "U3"}
    pair = list(model["meta"].get("modality_pair") or ["M1", "M2"])
    if a.modality_pair:
        cli = [x.strip().upper() for x in a.modality_pair.split(",")]
        if model["meta"].get("modality_pair") and list(model["meta"]["modality_pair"]) != cli:
            raise SystemExit(f"--modality-pair {cli} contradicts train_meta "
                             f"{model['meta']['modality_pair']}; refusing to guess")
        pair = cli
    if any(m not in _MOD_U for m in pair):
        raise SystemExit(f"train_meta modality_pair={pair} not understood")
    u1_key, u2_key = _MOD_U[pair[0]], _MOD_U[pair[1]]
    print(f"  modality pair {pair[0]}/{pair[1]} -> unique1={u1_key}, unique2={u2_key}",
          flush=True)
    Rrec = np.array([r["R"] for r in records])
    U1rec = np.array([r[u1_key] for r in records])
    U2rec = np.array([r[u2_key] for r in records])

    dl = DataLoader(RecordImages(a.data_dir, records,
                                 keys=(f"{pair[0]}_path", f"{pair[1]}_path")),
                    batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)
    t0 = time.time()
    H1, H2 = trunk_feats(model, dl)
    print(f"  {len(records)} records, trunk features in {time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(a.seed)
    folds = np.array_split(rng.permutation(len(records)), a.folds)
    fold_pairs = [make_pairs(Rrec, np.sort(f), a.per_record, a.seed + k)
                  for k, f in enumerate(folds)]
    train_pairs = [make_pairs(Rrec, np.sort(np.concatenate(folds[:k] + folds[k + 1:])),
                              a.per_record, a.seed + 100 + k) for k in range(a.folds)]
    n_eval = sum(len(p) for p in fold_pairs)
    print(f"  {a.folds} folds; {n_eval} out-of-fold pairs, "
          f"{sum(len(p) for p in train_pairs)} fit pairs")

    ev = np.concatenate(fold_pairs)
    Rp, U1p, U2p = Rrec[ev[:, 0]], U1rec[ev[:, 0]], U2rec[ev[:, 1]]

    out = dict(meta=dict(enc_dir=a.enc_dir, data_dir=a.data_dir, readout=a.readout,
                         folds=a.folds, per_record=a.per_record, n_records=len(records),
                         n_eval_pairs=int(n_eval), collapse=not a.no_collapse,
                         null_perms=a.null_perms, k_shot=a.k_shot,
                         k_shot_repeats=a.k_shot_repeats,
                         method=model["meta"].get("method"),
                         lam_club=model["lam_club"], ckpt_seed=model["meta"].get("seed"),
                         proj_dim=model["proj_dim"], device=DEVICE),
               grid=grid, dataset_pid={}, curve={})
    out["meta"]["modality_pair"] = pair
    out["meta"]["unique_labels"] = [u1_key, u2_key]

    src = {}
    for t in targets:
        x1, x2, y, checks = build_sources(t, Rp, U1p, U2p, not a.no_collapse, force=a.force)
        src[t] = (x1, x2)
        r = broja_pid(joint(x1, x2, y))
        r["null"] = pid_null(x1, x2, y, a.null_perms)
        out["dataset_pid"][t] = dict(r, independence_checks=checks)
        nl = r["null"] or dict(R=0, U1=0, U2=0, S=0)
        print(f"  dataset PID Y={t:8s} R={r['R'] - nl['R']:.3f} U1={r['U1'] - nl['U1']:.3f} "
              f"U2={r['U2'] - nl['U2']:.3f} S={r['S'] - nl['S']:.3f}   "
              f"(raw R={r['R']:.3f}, floor {nl['R']:.3f}; checks "
              f"{ {k: round(v, 3) for k, v in checks.items()} })")

    if a.dataset_pid_only:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {a.out} (dataset PID only)", flush=True)
        return

    ylab = {"share": (Rrec, Rp), "unique1": (U1rec, U1p), "unique2": (U2rec, U2p)}
    for gi in idxs:
        lam = grid[gi]
        blocks = blocks_at(model, H1, H2, lam)
        row = {}
        for t in targets:
            t0 = time.time()
            yrec = ylab[t][0]
            reps = max(1, a.k_shot_repeats if a.k_shot > 0 else 1)
            preds = []
            for rep in range(reps):
                yhat_r = np.empty(0, dtype=int)
                for k in range(a.folds):
                    ztr = z_for_pairs(blocks, train_pairs[k], lam, a.readout)
                    zte = z_for_pairs(blocks, fold_pairs[k], lam, a.readout)
                    ytr = yrec[train_pairs[k][:, 0 if t != "unique2" else 1]]
                    pred, C = probe_predict(ztr, ytr, zte, seed=a.seed + rep,
                                            k_shot=a.k_shot)
                    yhat_r = np.concatenate([yhat_r, pred])
                preds.append(yhat_r)
            # pool the draws: one k-shot probe is a single noisy sample of what the
            # representation makes decodable, and the PID needs the distribution
            yhat = np.concatenate(preds)
            acc = float(np.mean([ (p == ylab[t][1]).mean() for p in preds ]))
            x1, x2 = (np.tile(src[t][0], reps), np.tile(src[t][1], reps))
            r = broja_pid(joint(x1, x2, yhat))
            r["null"] = pid_null(x1, x2, yhat, a.null_perms)
            row[t] = dict(r, probe_acc=acc, probe_C=C, seconds=round(time.time() - t0, 1))
            nl = r["null"] or dict(R=0, U1=0, U2=0, S=0)
            print(f"  lam={lam} Y={t:8s} acc={acc:.3f}  R={r['R'] - nl['R']:.3f} "
                  f"U1={r['U1'] - nl['U1']:.3f} U2={r['U2'] - nl['U2']:.3f} "
                  f"S={r['S'] - nl['S']:.3f}  [{row[t]['seconds']}s {r['status']}]", flush=True)
        out["curve"][str(gi)] = dict(lam=lam, **row)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
