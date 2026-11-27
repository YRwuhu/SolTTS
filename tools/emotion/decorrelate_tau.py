#!/usr/bin/env python3
"""Attenuate the shared PC1 arousal axis across emotion direction vectors.

In single-speaker (and multi-speaker) emotion embeddings, the first singular vector
(PC1) often accounts for ~70% of total variance across emotions (shared arousal /
loudness shift), making contrasting emotions such as `happy` and `crying` share
>0.7 cosine similarity.

This script attenuates the PC1 component by `beta` (0.0 = unchanged, 1.0 = full
removal), re-normalizes each direction to unit L2 norm, and writes a drop-in
emotion vector directory (tau_*.f32 + manifest.json) ready for A/B evaluation.

Usage:
    python tools/emotion/decorrelate_tau.py \
        --in resources/emotion_vectors \
        --out _emotion_vectors_decorr70 \
        --beta 0.70
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Attenuate shared PC1 axis across emotion tau vectors")
    ap.add_argument("--in", dest="in_dir", default="resources/emotion_vectors")
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--beta", type=float, default=0.70, help="Fraction of PC1 projection to subtract (0..1)")
    ap.add_argument("--fit-ready-only", action="store_true",
                    help="Fit SVD PC1 using only 'ready' emotions (recommended to avoid noisy rare classes)")
    args = ap.parse_args()

    manifest_path = os.path.join(args.in_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    emotions = manifest["emotions"]
    names = list(emotions.keys())
    vecs = {}
    for e in names:
        p = os.path.join(args.in_dir, emotions[e]["file"])
        v = np.fromfile(p, dtype="<f4").astype(np.float64)
        v /= max(np.linalg.norm(v), 1e-12)
        vecs[e] = v

    fit_names = [e for e in names if not args.fit_ready_only or emotions[e].get("quality") == "ready"]
    M_fit = np.stack([vecs[e] for e in fit_names])
    _, S, Vt = np.linalg.svd(M_fit, full_matrices=False)
    pc1 = Vt[0]
    pc1_ratio = float((S[0] ** 2) / np.sum(S ** 2))
    print(f"SVD fit on {len(fit_names)} emotions: PC1 energy ratio = {pc1_ratio:.4f}, beta = {args.beta:.2f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_taus = {}
    for e in names:
        v = vecs[e]
        proj = float(v @ pc1)
        v_new = v - args.beta * proj * pc1
        norm = float(np.linalg.norm(v_new))
        if norm < 1e-9:
            v_new = v
            norm = 1.0
        v_new = (v_new / norm).astype("<f4")
        out_taus[e] = v_new.astype(np.float64)
        out_taus[e] /= np.linalg.norm(out_taus[e])
        v_new.tofile(os.path.join(args.out_dir, emotions[e]["file"]))

    # Pairwise stats before vs after
    def offdiag(d):
        arr = np.stack([d[e] for e in names])
        G = arr @ arr.T
        mask = ~np.eye(len(names), dtype=bool)
        vals = G[mask]
        return float(vals.mean()), float(vals.max()), float(vals.min())

    b_mean, b_max, b_min = offdiag(vecs)
    a_mean, a_max, a_min = offdiag(out_taus)
    print(f"Pairwise cosine before: mean={b_mean:+.3f} min={b_min:+.3f} max={b_max:+.3f}")
    print(f"Pairwise cosine after : mean={a_mean:+.3f} min={a_min:+.3f} max={a_max:+.3f}")
    for x, y in [("happy", "crying"), ("happy", "angry"), ("happy", "playful"), ("happy", "excited"), ("flustered", "excited"), ("shy", "lovestruck")]:
        if x in vecs and y in vecs:
            print(f"  {x:<10} vs {y:<10}: {float(vecs[x]@vecs[y]):+.3f} -> {float(out_taus[x]@out_taus[y]):+.3f}")

    manifest["method"] = f"{manifest.get('method', '')}; pc1_attenuation(beta={args.beta:.2f})"
    manifest["pc1_beta"] = args.beta
    manifest["pc1_energy_ratio"] = round(pc1_ratio, 4)
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    diag_src = os.path.join(args.in_dir, "diagnostics.json")
    if os.path.exists(diag_src):
        shutil.copyfile(diag_src, os.path.join(args.out_dir, "diagnostics.json"))

    print(f"Wrote decorrelated vectors to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
