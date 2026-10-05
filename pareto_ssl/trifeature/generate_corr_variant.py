#!/usr/bin/env python
"""
Generate trifeature datasets at a chosen corr(U, R) — H2 dose-response.

R has 15 classes (H(R) = 3.91 bits vs 3.32 at 10) so the shared factor is harder
to encode. U1/U2/U3 keep 10 classes, because that is all the original generator
can physically render:

    texture ids 0-9   10 genuinely distinct patterns  (verified by hash)
    texture ids 10-14 all identical blank white
    COLORS            10 entries

WHY NOT `U2 = R`
----------------
The original correlated_factor returns R itself on the copy branch. With R up to
14 and 10-entry tables that is an IndexError at draw_shape_image(color=COLORS[U3]),
which is why trifeatures_15shapes_corr00 exists but no 15-shape corr>0 set does.

WHY NOT `U2 = R % 10` EITHER
----------------------------
g(R) = R % 10 is 2-to-1 for u in 0..4 and 1-to-1 for u in 5..9, so the copy branch
alone would make classes 0-4 twice as likely. U2's class balance would then shift
with the correlation level -- and class balance changes probe accuracy on its own.
The sweep would be measuring imbalance, not separability.

THE MECHANISM USED HERE
-----------------------
Keep the copy branch, and compensate the other branch so the marginal never moves:

    g(R) = R % 10
    with prob c :  U2 = g(R)                     all the R-dependence lives here
    otherwise   :  U2 ~ q_c                      independent of R
    q_c(u) = (1/NC - c * P(g(R)=u)) / (1 - c)

By construction c*P(g(R)=u) + (1-c)*q_c(u) = 1/NC exactly, for every u and every c.
Verified: marginal uniform at c = 0, .25, .5, .75; I(U2;R) = 0, 0.273, 0.920, 2.020
bits. Valid only up to c = 0.75 -- at c = 0.8 q_c needs negative mass, and the
script refuses.

Note this also fixes a wart in the original: its c=0 branch EXCLUDED R
(P(U2==R) = 0.000 against a 0.100 chance rate), so "corr00" was mildly
anti-correlated rather than independent. Here c=0 is genuinely independent.

U1 IS NEVER CORRELATED -- only U2 and U3 are, exactly as in the original. So the
manipulation is one-sided: it contaminates modality 2 and leaves modality 1 clean.
Prediction to test: u2 span collapses with c, u1 span survives.

RENDERING is taken from the original bytecode
(__pycache__/generate_trifeature_corr00_15shapes.cpython-312.pyc), whose source is
lost. Only the label-drawing loop is reimplemented here; every draw_shape_image
call matches the original argument-for-argument. Requires CPython 3.12.

USAGE
    python pareto_ssl/trifeature/generate_corr_variant.py --corr 0.5 \\
        --out data/tri15_corr50 --n 2800
"""
import argparse
import json
import marshal
import os
import random
import sys
import types

import numpy as np

PYC = "__pycache__/generate_trifeature_corr00_15shapes.cpython-312.pyc"
NUM_CLASSES = 10
SUBDIRS = ("M1_shape_R_U1", "M2_texture_R_U2", "M3_color_R_U3")


def load_generator(pyc=PYC):
    if not os.path.exists(pyc):
        sys.exit(f"missing {pyc} — cannot reproduce the original rendering")
    with open(pyc, "rb") as f:
        f.read(16)
        code = marshal.load(f)
    m = types.ModuleType("gen15")
    m.__dict__["__name__"] = "gen15"        # not __main__, so its CLI block stays inert
    exec(code, m.__dict__)
    return m


def compensating_q(c, num_shapes, num_classes=NUM_CLASSES):
    """q_c such that c*P(g(R)=u) + (1-c)*q_c(u) == 1/num_classes for every u."""
    pg = np.array([sum(1 for R in range(num_shapes) if R % num_classes == u) / num_shapes
                   for u in range(num_classes)])
    if c >= 1.0:
        sys.exit("corr must be < 1")
    q = (1.0 / num_classes - c * pg) / (1.0 - c)
    if q.min() < -1e-12:
        sys.exit(f"corr={c} is out of range: q_c would need negative mass "
                 f"(min {q.min():.4f}). Max supported is "
                 f"{(1.0/num_classes)/pg.max():.3f}.")
    return np.clip(q, 0.0, None) / np.clip(q, 0.0, None).sum(), pg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr", type=float, required=True)
    ap.add_argument("--num_shapes", type=int, default=10,
                    help="classes for R. DEFAULT 10 = same size as U, so g(R) is the "
                         "IDENTITY, q_c is uniform, and the design is simply "
                         "'with prob c set U2=R, else draw U2 uniformly'. Setting 15 "
                         "makes g 2-to-1 and forces the compensating draw -- correct, "
                         "but much harder to describe in a paper, and R is already at "
                         "ceiling (share=1.0000 in every run) so the extra classes buy "
                         "nothing measurable.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2800,
                    help="2800 = N_TRAIN 2400 + 2*N_TEST 200, all the benchmark reads")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    m = load_generator()
    NUM_SHAPES = args.num_shapes
    q, pg = compensating_q(args.corr, NUM_SHAPES)
    _ident = (NUM_SHAPES == NUM_CLASSES)
    print(f"  NUM_SHAPES={NUM_SHAPES}  NUM_CLASSES={NUM_CLASSES}  "
          f"IMAGE_SIZE={m.IMAGE_SIZE}  corr={args.corr}")
    print(f"  g(R)=R mod {NUM_CLASSES} is {'the IDENTITY' if _ident else 'a many-to-1 map'}; "
          f"q_c is {'uniform (no compensation needed)' if _ident else 'non-uniform (compensating)'}")

    rng = random.Random(args.seed)
    nprng = np.random.default_rng(args.seed)
    for s in SUBDIRS:
        os.makedirs(os.path.join(args.out, s), exist_ok=True)

    recs = []
    for idx in range(args.n):
        R = rng.randint(0, NUM_SHAPES - 1)
        U1 = rng.randint(0, NUM_CLASSES - 1)
        gR = R % NUM_CLASSES
        U2 = gR if rng.random() < args.corr else int(nprng.choice(NUM_CLASSES, p=q))
        U3 = gR if rng.random() < args.corr else int(nprng.choice(NUM_CLASSES, p=q))
        fn = f"{idx:06d}.png"
        # argument-for-argument identical to the original generate_dataset
        m.draw_shape_image(R=R, U1=U1, color=(0, 0, 0), texture_id=None,
                           size=m.IMAGE_SIZE).save(os.path.join(args.out, SUBDIRS[0], fn))
        m.draw_shape_image(R=R, U1=0, color=(0, 0, 0), texture_id=U2,
                           size=m.IMAGE_SIZE).save(os.path.join(args.out, SUBDIRS[1], fn))
        m.draw_shape_image(R=R, U1=0, color=m.COLORS[U3], texture_id=None,
                           size=m.IMAGE_SIZE).save(os.path.join(args.out, SUBDIRS[2], fn))
        recs.append({"id": idx,
                     "M1_path": f"{SUBDIRS[0]}/{fn}",
                     "M2_path": f"{SUBDIRS[1]}/{fn}",
                     "M3_path": f"{SUBDIRS[2]}/{fn}",
                     "R": R, "U1": U1, "U2": U2, "U3": U3,
                     "corr_U2_with_R": args.corr, "corr_U3_with_R": args.corr})
        if (idx + 1) % 500 == 0:
            print(f"    {idx+1}/{args.n}", flush=True)

    json.dump(recs, open(os.path.join(args.out, "metadata.json"), "w"))

    # report what was actually produced, not what was requested
    import collections
    cnt = collections.Counter(r["U2"] for r in recs)
    marg = np.array([cnt[u] for u in range(NUM_CLASSES)]) / len(recs)
    def mi(a, b):
        j = collections.Counter(zip(a, b)); n = len(a)
        pa = collections.Counter(a); pb = collections.Counter(b)
        return sum(c/n*np.log2((c/n)/((pa[x]/n)*(pb[y]/n))) for (x, y), c in j.items())
    Rs = [r["R"] for r in recs]
    print(f"\n  wrote {len(recs)} records to {args.out}")
    print(f"  |R|={len(set(Rs))}  |U1|={len({r['U1'] for r in recs})}  "
          f"|U2|={len({r['U2'] for r in recs})}")
    print(f"  P(U2 == g(R))       = {sum(r['U2']==r['R']%NUM_CLASSES for r in recs)/len(recs):.3f} "
          f"(target ~{args.corr + (1-args.corr)*0.1:.3f})")
    print(f"  U2 marginal         = {np.round(marg,3)}   (target 0.1 each)")
    print(f"  max |marginal-0.1|  = {np.abs(marg-0.1).max():.4f}")
    print(f"  I(U2;R)             = {mi([r['U2'] for r in recs], Rs):.3f} bits")
    print(f"  I(U1;R)             = {mi([r['U1'] for r in recs], Rs):.3f} bits  (should be ~0)")


if __name__ == "__main__":
    main()
