"""Validate a candidate τ bundle and install it into every runtime resource location.

Usage:
    python -I install_emotion_vectors.py \
        --candidate E:\\AI\\SolTTS\\resources\\emotion_vectors_atri19_candidate \
        --apply

Without --apply it only validates and prints what would change.

Validation performed before any copy:
  * manifest format/version and expected 19-label schema
  * every declared file exists, is exactly 20480 float32 values, finite, non-zero, unit norm
  * baseline is never shipped as a vector
  * no file listed in the manifest escapes the directory

Targets are the three places the runtime can load from:
  resources/emotion_vectors          (development runs, ResourcesDir)
  SolTTS.Core/emotion_vectors        (copied to the build output as a fallback)
  SolTTS-package/resources/emotion_vectors and SolTTS-package/emotion_vectors (shipped app)

The existing 6-class bundle is kept in _backup/emotion_vectors_csemo6; this script does not
overwrite that backup.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

EMOTIONS = [
    "happy", "calm", "disgust", "shy", "fearful", "angry", "serious",
    "nervous", "flustered", "confused", "excited", "helpless", "worried",
    "surprised", "crying", "lovestruck", "embarrassed", "confident", "playful",
]
BASELINE = "calm"
EXPECTED_DIM = 20480


def fail(msg: str) -> None:
    print("ERROR:", msg, file=sys.stderr)
    raise SystemExit(2)


def validate(candidate: str) -> dict:
    import numpy as np

    manifest_path = os.path.join(candidate, "manifest.json")
    if not os.path.exists(manifest_path):
        fail(f"manifest.json not found in {candidate}")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    if manifest.get("format") != "soltts-emotion-tau":
        fail(f"unexpected format: {manifest.get('format')!r}")
    if int(manifest.get("version", 0)) < 2:
        fail(f"manifest version {manifest.get('version')} is too old (need >= 2)")
    if int(manifest.get("dim", 0)) != EXPECTED_DIM:
        fail(f"manifest dim {manifest.get('dim')} != {EXPECTED_DIM}")

    emotions = manifest.get("emotions") or {}
    if BASELINE in emotions:
        fail("manifest must not ship a vector for the baseline 'calm'")

    missing = [e for e in EMOTIONS if e != BASELINE and e not in emotions]
    if missing:
        print(f"warning: {len(missing)} labels have no vector and will be unavailable: {missing}")

    for emo, info in emotions.items():
        if emo not in EMOTIONS:
            fail(f"manifest lists unknown label {emo!r}")
        if emo == BASELINE:
            fail("baseline entry present")
        name = info.get("file") or f"tau_{emo}.f32"
        if "/" in name or "\\" in name or ".." in name:
            fail(f"{emo}: illegal file name {name!r}")
        path = os.path.join(candidate, name)
        if not os.path.exists(path):
            fail(f"{emo}: missing file {name}")
        size = os.path.getsize(path)
        if size != EXPECTED_DIM * 4:
            fail(f"{emo}: file size {size} != {EXPECTED_DIM * 4}")
        vec = np.fromfile(path, dtype="<f4")
        if not np.all(np.isfinite(vec)):
            fail(f"{emo}: contains NaN/Infinity")
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            fail(f"{emo}: zero vector")
        if abs(norm - 1.0) > 1e-3:
            fail(f"{emo}: not unit normalized (norm={norm})")

    print(f"validated {len(emotions)} vectors in {candidate}")
    print(f"labels without vectors: {missing if missing else 'none'}")
    return manifest


def install(candidate: str, targets: list[str], files: list[str]) -> None:
    for target in targets:
        parent = os.path.dirname(target.rstrip("\\/"))
        os.makedirs(parent, exist_ok=True)
        staging = tempfile.mkdtemp(prefix=".tau_install_", dir=parent)
        try:
            for f in files:
                shutil.copy2(os.path.join(candidate, f), os.path.join(staging, f))
            # Swap in the new set: remove stale tau_*.f32 so an unloaded label cannot linger.
            os.makedirs(target, exist_ok=True)
            for old in os.listdir(target):
                if old.endswith(".f32") or old in ("manifest.json", "diagnostics.json", "samples_meta.json"):
                    os.remove(os.path.join(target, old))
            for f in os.listdir(staging):
                shutil.move(os.path.join(staging, f), os.path.join(target, f))
            print(f"installed -> {target}")
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=r"E:\AI\SolTTS\resources\emotion_vectors_atri19_candidate")
    ap.add_argument("--soltts", default=r"E:\AI\SolTTS")
    ap.add_argument("--package", default=r"E:\AI\SolTTS-package")
    ap.add_argument("--apply", action="store_true", help="Actually install; without it, only validate")
    args = ap.parse_args()

    validate(args.candidate)
    files = sorted(os.listdir(args.candidate))
    files = [f for f in files if f.endswith(".f32") or f in ("manifest.json", "diagnostics.json")]
    if not args.apply:
        print("dry run; pass --apply to install into:")
        for t in _targets(args):
            print("  ", t)
        return 0

    install(args.candidate, _targets(args), files)
    return 0


def _targets(args) -> list[str]:
    return [
        os.path.join(args.soltts, "resources", "emotion_vectors"),
        os.path.join(args.soltts, "SolTTS.Core", "emotion_vectors"),
        os.path.join(args.package, "resources", "emotion_vectors"),
        os.path.join(args.package, "emotion_vectors"),
    ]


if __name__ == "__main__":
    sys.exit(main())