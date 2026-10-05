#!/usr/bin/env python3
"""Model PID of a FIXED-representation baseline on Trifeature (CLIP / GMC / CoMM / FactorCL).

Companion to run_model_pid_steer.py: same sources, same pairing, same 1-shot probe, same
BROJA/CVX decomposition -- only the representation differs. STEER gives 15 points (one per
preference); a baseline gives one point per TASK, so the figure reads "STEER spans a
region, the baselines sit at points".

The representation comes from benchmark._extract_pairs, i.e. the function the accuracy
tables themselves use, driven by pareto_config's inference_mode:

    gmc / comm       mmfusion_joint   -- ONE fused vector, identical for all three tasks
    factorcl         factorcl_concat  -- ONE concatenated vector, identical for all tasks
    clip             unimodal_split   -- M1 for share/unique1, M2 for unique2
    factorcl_heads   factorcl_heads   -- a DIFFERENT head per task (task oracle)

The last two are not single fixed representations: their read-out already depends on the
task, which is an advantage STEER's single z(lambda) does not get. Reported, but flagged
in the output as `task_dependent_readout`, and factorcl_heads is display-only in the
accuracy tables for exactly this reason.

Pairing, folds, k-shot probe and the shuffled-target floor are imported from the STEER
script so the two are comparable cell for cell.

Usage:
  python pareto_ssl/multibench/pid_analysis/run_model_pid_trifeature_baseline.py \
      --enc-dir pareto_ssl/results_corr_sweep/_parts/corr00_gmc_s42/gmc \
      --method gmc --data-dir data/tri15_corr00 \
      --out pareto_ssl/multibench/pid_analysis/results/trifeature_baselines/gmc_s42.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from run_model_pid_steer import (make_pairs, probe_predict, build_sources, joint,   # noqa: E402
                                 pid_null, RecordImages)
from pid_cvx import broja_pid, validate as validate_solver                          # noqa: E402
from pareto_ssl.benchmark import _extract_pairs, INFERENCE_MODE                     # noqa: E402
from pareto_ssl.networks import AlexNetEncoder, ProjectionHead                      # noqa: E402
from pareto_ssl.datasets import _EVAL, _remap                                       # noqa: E402
from PIL import Image                                                               # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TASKS = ["share", "unique1", "unique2"]
TASK_DEPENDENT = {"unimodal_split", "factorcl_heads"}


_IMG_CACHE = {}


def _load_images(data_dir, records):
    """Decode every record's two images ONCE, float16 on CPU (~840 MB each for 2800).

    Without this the pairs re-read PNGs for every fold and every k-shot repeat: 3 tasks x
    5 folds x 5 repeats over ~5600 pairs is ~400k decodes, hours of pure I/O for a few
    minutes of actual compute.
    """
    key = (data_dir, len(records))
    if key not in _IMG_CACHE:
        M1 = torch.stack([_EVAL(Image.open(_remap(r["M1_path"], data_dir)).convert("RGB"))
                          for r in records]).half()
        M2 = torch.stack([_EVAL(Image.open(_remap(r["M2_path"], data_dir)).convert("RGB"))
                          for r in records]).half()
        _IMG_CACHE[key] = (M1, M2)
        print(f"  cached images: M1 {tuple(M1.shape)} M2 {tuple(M2.shape)} "
              f"({(M1.numel() + M2.numel()) * 2 / 1e9:.2f} GB)", flush=True)
    return _IMG_CACHE[key]


class ExplicitPairs(Dataset):
    """([M1_i, M2_j], label) for a GIVEN array of (i, j) record pairs, from the cache.

    PairProbeDataset samples its own pairs per split; here the pairs must be the same ones
    the STEER run used, or the two PID numbers are computed on different samples.
    """

    def __init__(self, data_dir, records, pairs, labels):
        self.pairs, self.labels = pairs, labels
        self.M1, self.M2 = _load_images(data_dir, records)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        i, j = self.pairs[k]
        return [self.M1[i].float(), self.M2[j].float()], int(self.labels[k])


def load_modules(method, enc_dir, latent_dim, proj_dim, device=DEVICE):
    """Mirrors probe_method's loading branch for the fixed baselines."""
    mode = INFERENCE_MODE[method]
    enc_m1 = enc_m2 = mmfusion = proj_r = proj_u = proj_m2_u = None

    def _enc(name):
        e = AlexNetEncoder(latent_dim).to(device)
        e.load_state_dict(torch.load(os.path.join(enc_dir, name), map_location=device,
                                     weights_only=True))
        return e.eval()

    def _head(name):
        h = ProjectionHead(latent_dim, proj_dim).to(device)
        h.load_state_dict(torch.load(os.path.join(enc_dir, name), map_location=device,
                                     weights_only=True))
        return h.eval()

    if mode == "mmfusion_joint":
        from pareto_ssl.benchmark import _make_mmfusion
        mmfusion = _make_mmfusion(device)
        mmfusion.load_state_dict(torch.load(os.path.join(enc_dir, "mmfusion.pth"),
                                            map_location=device, weights_only=True))
        mmfusion.eval()
    elif mode in ("factorcl_heads", "factorcl_concat"):
        enc_m1, enc_m2 = _enc("enc_0.pth"), _enc("enc_1.pth")
        proj_r, proj_u, proj_m2_u = _head("proj_r.pth"), _head("proj_u.pth"), _head("proj_m2_u.pth")
    else:
        enc_m1, enc_m2 = _enc("enc_0.pth"), _enc("enc_1.pth")
    return mode, dict(enc_m1=enc_m1, enc_m2=enc_m2, mmfusion=mmfusion,
                      proj_r=proj_r, proj_u=proj_u, proj_m2_u=proj_m2_u)


def z_for(mods, mode, task, data_dir, records, pairs, labels, batch_size, workers):
    task_mode = ("unimodal_m2" if task == "unique2" else "unimodal_m1") \
        if mode == "unimodal_split" else mode
    dl = DataLoader(ExplicitPairs(data_dir, records, pairs, labels),
                    batch_size=batch_size, shuffle=False, num_workers=workers)
    z, y = _extract_pairs(mods["enc_m1"], mods["enc_m2"], dl, DEVICE, task_mode,
                          mods["mmfusion"], mods["proj_r"], mods["proj_u"], task,
                          proj_m2_u=mods["proj_m2_u"])
    return np.asarray(z), np.asarray(y), task_mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-dir", required=True)
    ap.add_argument("--method", required=True,
                    choices=["gmc", "comm", "clip", "factorcl", "factorcl_heads", "simclr_both"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--latent-dim", type=int, default=512)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--per-record", type=int, default=2)
    ap.add_argument("--k-shot", type=int, default=1)
    ap.add_argument("--k-shot-repeats", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--null-perms", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-validate", action="store_true")
    a = ap.parse_args()

    if not a.skip_validate:
        print("BROJA solver validation (bits):", flush=True); validate_solver()

    mode, mods = load_modules(a.method, a.enc_dir, a.latent_dim, a.proj_dim)
    print(f"{a.method}: inference_mode={mode}"
          + ("  [TASK-DEPENDENT read-out]" if mode in TASK_DEPENDENT else ""), flush=True)

    with open(os.path.join(a.data_dir, "metadata.json")) as f:
        records = json.load(f)
    R = np.array([r["R"] for r in records])
    U1 = np.array([r["U1"] for r in records])
    U2 = np.array([r["U2"] for r in records])

    rng = np.random.default_rng(a.seed)
    folds = np.array_split(rng.permutation(len(records)), a.folds)
    fold_pairs = [make_pairs(R, np.sort(f), a.per_record, a.seed + k) for k, f in enumerate(folds)]
    train_pairs = [make_pairs(R, np.sort(np.concatenate(folds[:k] + folds[k + 1:])),
                              a.per_record, a.seed + 100 + k) for k in range(a.folds)]
    ev = np.concatenate(fold_pairs)
    Rp, U1p, U2p = R[ev[:, 0]], U1[ev[:, 0]], U2[ev[:, 1]]
    print(f"  {len(records)} records, {len(ev)} out-of-fold pairs", flush=True)

    out = dict(meta=dict(enc_dir=a.enc_dir, method=a.method, inference_mode=mode,
                         task_dependent_readout=mode in TASK_DEPENDENT,
                         data_dir=a.data_dir, folds=a.folds, per_record=a.per_record,
                         k_shot=a.k_shot, k_shot_repeats=a.k_shot_repeats,
                         n_eval_pairs=int(len(ev)), null_perms=a.null_perms,
                         device=DEVICE), tasks={})

    lab = {"share": (R, Rp, 0), "unique1": (U1, U1p, 0), "unique2": (U2, U2p, 1)}
    zcache = {}
    for task in TASKS:
        t0 = time.time()
        yrec, ytrue, col = lab[task]
        reps = max(1, a.k_shot_repeats if a.k_shot > 0 else 1)
        # the representation depends on the task ONLY through task_mode (identical for
        # every task under mmfusion_joint / factorcl_concat), and not at all on the repeat
        tm = ("unimodal_m2" if task == "unique2" else "unimodal_m1") \
            if mode == "unimodal_split" else mode
        if tm not in zcache:
            zcache[tm] = [(z_for(mods, mode, task, a.data_dir, records, train_pairs[k],
                                 yrec[train_pairs[k][:, col]], a.batch_size, a.num_workers)[0],
                           z_for(mods, mode, task, a.data_dir, records, fold_pairs[k],
                                 yrec[fold_pairs[k][:, col]], a.batch_size, a.num_workers)[0])
                          for k in range(a.folds)]
        preds = []
        for rep in range(reps):
            yhat_r = np.empty(0, dtype=int)
            for k in range(a.folds):
                ztr, zte = zcache[tm][k]
                ytr = yrec[train_pairs[k][:, col]]
                pred, C = probe_predict(ztr, ytr, zte, seed=a.seed + rep, k_shot=a.k_shot)
                yhat_r = np.concatenate([yhat_r, pred])
            preds.append(yhat_r)
        yhat = np.concatenate(preds)
        acc = float(np.mean([(p == ytrue).mean() for p in preds]))
        x1, x2, _, _ = build_sources(task, Rp, U1p, U2p)
        x1, x2 = np.tile(x1, reps), np.tile(x2, reps)
        r = broja_pid(joint(x1, x2, yhat))
        r["null"] = pid_null(x1, x2, yhat, a.null_perms)
        nl = r["null"] or dict(R=0, U1=0, U2=0, S=0)
        out["tasks"][task] = dict(r, probe_acc=acc, seconds=round(time.time() - t0, 1))
        print(f"  Y={task:8s} acc={acc:.3f}  R={r['R'] - nl['R']:.3f} "
              f"U1={r['U1'] - nl['U1']:.3f} U2={r['U2'] - nl['U2']:.3f} "
              f"S={r['S'] - nl['S']:.3f}  [{out['tasks'][task]['seconds']}s]", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
