"""Extract 19-emotion direction vectors τ from the ATRI single-speaker corpus.

Recipe (single speaker, calm baseline):
    τ_e = mean_i ( x_e^{i} ) − mean_j ( x_calm^{j} )

Samples come from `labels.jsonl` produced by tools/emotion_annotation/annotate.py.
Only stable labels are used by default (review=False, audible_speech=true). Low-confidence or
disputed clips still appear in coverage statistics but never enter the baseline or a direction,
and clips whose hash no longer matches are rejected rather than silently reused.

Design notes:
  * The speaker encoder input pipeline mirrors SolTTS.Core/Utils/AudioHelper.cs: mono downmix,
    16 kHz resample, DC removal, peak normalisation. Getting this wrong shifts every vector.
  * Duration filtering (default 1.5-12 s) matches the existing CSEMOTIONS recipe. Clips outside
    the window are reported, not silently dropped; long clips are centre-trimmed.
  * Split-by-chapter validation prevents adjacent lines of the same scene from leaking between
    the fit and validation sides.
  * Emissions are staged in a candidate directory. Nothing under resources/emotion_vectors is
    touched until a human decides the quality report is acceptable.

Usage:
    python -I extract_tau_atri.py \
        --labels F:\\CNBackups\\ATRI\\emotion_annotation\\labels.jsonl \
        --resources E:\\AI\\SolTTS\\resources \
        --out E:\\AI\\SolTTS\\resources\\emotion_vectors_atri19_candidate
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import time
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

EMOTIONS = [
    "happy", "calm", "disgust", "shy", "fearful", "angry", "serious",
    "nervous", "flustered", "confused", "excited", "helpless", "worried",
    "surprised", "crying", "lovestruck", "embarrassed", "confident", "playful",
]
BASELINE = "calm"


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------
# audio io (mirrors AudioHelper.cs)
# ----------------------------------------------------------------------------
def read_wav_mono_16k(path: str) -> Tuple[np.ndarray, int, float]:
    """Read a PCM WAV, downmix to mono, resample to 16 kHz. Returns (samples, sr, original_duration)."""
    with wave.open(path, "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"unsupported sample width {sw * 8} bit (expected 16)")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    duration = len(audio) / float(sr) if sr else 0.0
    if sr != 16000:
        audio = resample_linear(audio, sr, 16000)
    return audio.astype(np.float32), 16000, duration


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Dependency-free linear resampling. The corpus is 48 kHz; 3:1 decimation needs an anti-alias step."""
    if sr_in == sr_out:
        return x
    g = math.gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    if up == 1 and down == 3:
        # Cheap anti-alias: 3-tap moving average before decimation.
        k = np.ones(3, dtype=np.float32) / 3.0
        x = np.convolve(x, k, mode="same")
    n_out = int(round(len(x) * sr_out / sr_in))
    idx = np.linspace(0, len(x) - 1, n_out, dtype=np.float64)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def crop_for_encoder(audio: np.ndarray, sr: int, max_sec: float) -> np.ndarray:
    """Centre-crop to a fixed analysis window.

    The speaker encoder's cost grows with input length, and the reference-audio path at inference
    reads at most 60 s. A 3 s window is enough for a stable timbre/emotion embedding, keeps batch
    extraction tractable and makes every clip comparable.
    """
    if max_sec <= 0:
        return audio
    n = int(max_sec * sr)
    if len(audio) <= n:
        return audio
    start = (len(audio) - n) // 2
    return audio[start:start + n]


def normalize_for_encoder(a: np.ndarray) -> np.ndarray:
    """DC removal + peak normalisation to 0.9, exactly as the CSEMOTIONS recipe does."""
    a = np.asarray(a, dtype=np.float32)
    a = a - a.mean()
    peak = float(np.abs(a).max())
    if peak > 1e-6:
        a = a / peak * 0.9
    return a


# ----------------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------------
@dataclass
class Sample:
    file: str
    path: str
    chapter: str
    label: str
    confidence: float
    duration: float
    sha256: str
    review: bool


def load_labels(path: str, audio_dir: str, include_review: bool) -> List[Sample]:
    samples: List[Sample] = []
    skipped = Counter()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                skipped["bad_json"] += 1
                continue
            if r.get("status") != "done":
                skipped["not_done"] += 1
                continue
            label = r.get("label")
            if label not in EMOTIONS:
                skipped["unknown_label"] += 1
                continue
            if r.get("audible_speech") is False:
                skipped["no_audible_speech"] += 1
                continue
            if r.get("review") and not include_review:
                skipped["needs_review"] += 1
                continue
            p = r.get("path") or os.path.join(audio_dir, r["file"])
            if not os.path.exists(p):
                skipped["missing_file"] += 1
                continue
            samples.append(
                Sample(
                    file=r["file"],
                    path=p,
                    chapter=str(r.get("chapter", "unknown")),
                    label=label,
                    confidence=float(r.get("confidence") or 0.0),
                    duration=float(r.get("duration_sec") or 0.0),
                    sha256=r.get("sha256", ""),
                    review=bool(r.get("review")),
                )
            )
    if skipped:
        log("label filtering:", dict(skipped))
    return samples


# ----------------------------------------------------------------------------
# split + extraction
# ----------------------------------------------------------------------------
def chapter_split(samples: List[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    """Hold out whole chapters so adjacent lines never straddle the split."""
    by_chapter: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        by_chapter[s.chapter].append(s)
    chapters = sorted(by_chapter)
    rng = random.Random(seed)
    rng.shuffle(chapters)
    n_val = max(1, int(round(len(chapters) * val_ratio)))
    val_chapters = set(chapters[:n_val])
    train = [s for c in chapters if c not in val_chapters for s in by_chapter[c]]
    val = [s for c in sorted(val_chapters) for s in by_chapter[c]]
    return train, val


def embed_all(samples: List[Sample], spk, in_name: str, max_sec: float,
              cache_path: Optional[str] = None, threads: int = 2) -> Tuple[Dict[str, List[np.ndarray]], dict, Counter]:
    """Group embeddings by label. Returns (store, per_file_report, exclusions).

    Embeddings are cached on disk keyed by file hash, so re-running diagnostics is cheap and
    never re-runs the encoder for clips that have not changed.
    """
    cache: Dict[str, np.ndarray] = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with np.load(cache_path, allow_pickle=False) as npz:
                for k in npz.files:
                    cache[k] = npz[k].astype(np.float32)
            log(f"embedding cache: {len(cache)} entries from {cache_path}")
        except Exception as ex:
            log(f"cache unreadable ({ex}); recomputing")

    def _embed_one(s: Sample):
        key = s.sha256 or s.file
        if key in cache:
            return key, cache[key], None, None
        try:
            audio, sr, dur = read_wav_mono_16k(s.path)
        except Exception as ex:
            return key, None, f"read_error:{type(ex).__name__}", None
        if dur < 1.5:
            return key, None, "too_short", dur
        audio = crop_for_encoder(audio, sr, max_sec)
        try:
            a = normalize_for_encoder(audio)
            e = spk.run(None, {in_name: a[None, :].astype(np.float32)})[0]
            vec = np.asarray(e).reshape(-1).astype(np.float32)
        except Exception as ex:
            return key, None, f"embed_error:{type(ex).__name__}", dur
        if vec.size != 20480 or not np.all(np.isfinite(vec)):
            return key, None, "bad_embedding", dur
        return key, vec, None, dur

    from concurrent.futures import ThreadPoolExecutor

    results = {}
    # ONNX Runtime sessions are thread-safe for Run; keep concurrency modest to stay CPU-friendly.
    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        for i, (key, vec, err, dur) in enumerate(
                ex.map(_embed_one, samples), 1):
            results[key] = (vec, err, dur)
            if i % 100 == 0:
                log(f"  embedded {i}/{len(samples)}")

    store: Dict[str, List[np.ndarray]] = defaultdict(list)
    report: Dict[str, dict] = {}
    excluded = Counter()
    for s in samples:
        key = s.sha256 or s.file
        vec, err, dur = results[key]
        if vec is not None:
            store[s.label].append(vec)
            cache[key] = vec
            report[s.file] = {"label": s.label, "duration": round(dur or s.duration, 3),
                              "used": True, "chapter": s.chapter}
        else:
            excluded[err or "unknown"] += 1
            report[s.file] = {"label": s.label, "duration": round(dur or s.duration, 3),
                              "used": False, "reason": err, "chapter": s.chapter}

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(cache_path, **cache)
        log(f"embedding cache written: {cache_path} ({len(cache)} entries)")
    return store, report, excluded


def mean_of(xs: List[np.ndarray]) -> Optional[np.ndarray]:
    return np.mean(np.stack(xs), axis=0) if xs else None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 1e-12 else 0.0


def compute_taus(store: Dict[str, List[np.ndarray]], min_samples: int) -> Tuple[Dict[str, np.ndarray], dict]:
    base = mean_of(store.get(BASELINE, []))
    taus: Dict[str, np.ndarray] = {}
    diag: Dict[str, dict] = {}
    if base is None:
        return taus, {"error": "no calm samples"}

    base_norm = float(np.linalg.norm(base))
    for emo in EMOTIONS:
        if emo == BASELINE:
            continue
        embs = store.get(emo, [])
        n = len(embs)
        if n < min_samples:
            diag[emo] = {
                "n_samples": n,
                "status": "unavailable" if n == 0 else "insufficient",
                "reason": f"need >= {min_samples} stable samples, got {n}",
            }
            continue
        mean_e = mean_of(embs)
        tau = mean_e - base
        tau_norm = float(np.linalg.norm(tau))
        if tau_norm < 1e-9:
            diag[emo] = {"n_samples": n, "status": "insufficient",
                         "reason": "emotion mean equals the calm baseline"}
            continue
        # Coherence: how much of each sample's deviation from the baseline points along the mean
        # deviation. mean(cos(sample_delta, tau)) is 1.0 when every clip deviates identically and
        # near 0 when deviations cancel out. This is NOT the same as ||mean(delta)||/mean||delta||,
        # which trivially approaches 1 whenever the shared component dominates.
        unit = tau / tau_norm
        deltas = np.stack(embs) - base
        delta_norms = np.linalg.norm(deltas, axis=1)
        valid = delta_norms > 1e-9
        if not np.any(valid):
            diag[emo] = {"n_samples": n, "status": "insufficient", "reason": "all samples equal baseline"}
            continue
        cos_to_tau = (deltas[valid] @ unit) / delta_norms[valid]
        coherence = float(cos_to_tau.mean())
        retained = float(tau_norm / (delta_norms[valid].mean() + 1e-9))
        # Quality gate: enough clips AND deviations that actually agree in direction.
        ready = n >= 20 and coherence >= 0.35
        taus[emo] = tau
        diag[emo] = {
            "n_samples": n,
            "tau_norm": tau_norm,
            "baseline_norm": base_norm,
            "coherence": coherence,
            "coherence_std": float(cos_to_tau.std()),
            "retained_fraction": retained,
            "mean_delta_norm": float(delta_norms[valid].mean()),
            "std_delta_norm": float(delta_norms[valid].std()),
            "status": "ready" if ready else "experimental",
            "reason": None if ready else
                      ("low sample count" if n < 20 else f"low direction coherence ({coherence:.2f})"),
        }
    return taus, diag


def bootstrap_stability(store: Dict[str, List[np.ndarray]], taus: Dict[str, np.ndarray],
                        rounds: int, ratio: float, seed: int) -> dict:
    """Resample samples within each label and measure cosine agreement with the full-data τ.

    Vectorised: each label is stacked into one matrix once, so a resample is just a row-subset
    mean instead of hundreds of Python-level vector sums.
    """
    rng = np.random.default_rng(seed)
    out = {}
    base_list = store.get(BASELINE, [])
    if not base_list:
        return out
    base_m = np.stack(base_list)  # [N, D]
    for emo, full in taus.items():
        embs = store.get(emo, [])
        if not embs:
            continue
        emo_m = np.stack(embs)
        cosines = []
        for _ in range(rounds):
            n_b = max(2, int(len(base_m) * ratio))
            n_e = max(2, int(len(emo_m) * ratio))
            bi = rng.choice(len(base_m), size=n_b, replace=False)
            ei = rng.choice(len(emo_m), size=n_e, replace=False)
            t = emo_m[ei].mean(axis=0) - base_m[bi].mean(axis=0)
            cosines.append(cosine(t, full))
        out[emo] = {
            "rounds": rounds,
            "mean_cosine": float(np.mean(cosines)),
            "min_cosine": float(np.min(cosines)),
            "p05_cosine": float(np.percentile(cosines, 5)),
        }
    return out


def holdout_validation(train_store, val_store, taus: Dict[str, np.ndarray]) -> dict:
    """Re-extract from the held-out chapters and compare direction with the fit."""
    out = {}
    val_taus, val_diag = compute_taus(val_store, min_samples=5)
    for emo, tau in taus.items():
        if emo in val_taus:
            out[emo] = {
                "holdout_samples": val_diag[emo]["n_samples"],
                "cosine_to_fit": cosine(tau, val_taus[emo]),
            }
        else:
            out[emo] = {"holdout_samples": 0, "cosine_to_fit": None}
    return out


def similarity_matrix(taus: Dict[str, np.ndarray]) -> List[dict]:
    rows = []
    names = [e for e in EMOTIONS if e in taus]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = cosine(taus[a], taus[b])
            rows.append({"a": a, "b": b, "cosine": round(c, 4)})
    rows.sort(key=lambda r: -abs(r["cosine"]))
    return rows


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Extract ATRI 19-emotion τ from labels.jsonl")
    ap.add_argument("--labels", default=r"F:\CNBackups\ATRI\emotion_annotation\labels.jsonl")
    ap.add_argument("--audio-dir", default=r"F:\CNBackups\ATRI\raw")
    ap.add_argument("--resources", default=r"E:\AI\SolTTS\resources")
    ap.add_argument("--out", default=r"E:\AI\SolTTS\resources\emotion_vectors_atri19_candidate")
    ap.add_argument("--include-review", action="store_true",
                    help="Also use clips flagged for review (not recommended)")
    ap.add_argument("--min-samples", type=int, default=10,
                    help="Minimum stable clips per emotion to emit a vector")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--bootstrap-rounds", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--analysis-sec", type=float, default=3.0,
                    help="centre-crop length fed to the speaker encoder (0 = whole clip)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "dml"])
    args = ap.parse_args()

    import onnxruntime as ort

    t0 = time.time()
    spk_path = os.path.join(args.resources, "shared", "v1_extra", "speaker_encoder.onnx")
    if not os.path.exists(spk_path):
        log("Missing speaker_encoder.onnx:", spk_path)
        return 1

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    if args.device in ("auto", "dml"):
        if "DmlExecutionProvider" in ort.get_available_providers():
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    spk = ort.InferenceSession(spk_path, so, providers=providers)
    in_name = spk.get_inputs()[0].name
    log("providers:", providers, "| input:", in_name)

    samples = load_labels(args.labels, args.audio_dir, args.include_review)
    log(f"usable labels: {len(samples)}")
    counts = Counter(s.label for s in samples)
    for e in EMOTIONS:
        log(f"  {e:<12} {counts.get(e, 0)}")

    train, val = chapter_split(samples, args.val_ratio, args.seed)
    log(f"split: train={len(train)} clips / {len(set(s.chapter for s in train))} chapters, "
        f"val={len(val)} clips / {len(set(s.chapter for s in val))} chapters")

    cache_path = os.path.join(args.out, "embedding_cache.npz")
    log("\nembedding training split...")
    train_store, train_report, train_ex = embed_all(
        train, spk, in_name, max_sec=args.analysis_sec, cache_path=cache_path, threads=args.threads)
    log(f"train excluded: {dict(train_ex)}")
    log("\nembedding validation split...")
    val_store, val_report, val_ex = embed_all(
        val, spk, in_name, max_sec=args.analysis_sec, cache_path=cache_path, threads=args.threads)

    taus, diag = compute_taus(train_store, args.min_samples)
    if not taus:
        log("no usable directions; baseline or samples missing")
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "diagnostics.json"), "w", encoding="utf-8") as fh:
            json.dump({"diag": diag, "excluded": dict(train_ex)}, fh, ensure_ascii=False, indent=2)
        return 3

    log("\n==== directions ====")
    for emo in EMOTIONS:
        if emo == BASELINE:
            continue
        d = diag.get(emo, {})
        log(f"  {emo:<12} n={d.get('n_samples', 0):<4} |τ|={d.get('tau_norm', 0):8.3f} "
            f"coherence={d.get('coherence', 0):.3f} status={d.get('status')}")

    log("\nbootstrap stability...")
    boot = bootstrap_stability(train_store, taus, args.bootstrap_rounds, 0.8, args.seed)
    log("\nholdout validation (chapter split)...")
    hold = holdout_validation(train_store, val_store, taus)

    sims = similarity_matrix(taus)
    log("\nmost similar direction pairs:")
    for r in sims[:10]:
        log(f"  {r['a']:<12} ~ {r['b']:<12} cos={r['cosine']:+.3f}")

    # ---------------- write candidate bundle ----------------
    os.makedirs(args.out, exist_ok=True)
    dim = next(iter(taus.values())).size
    manifest = {
        "format": "soltts-emotion-tau",
        "version": 2,
        "dim": int(dim),
        "source_dataset": "ATRI (local corpus, single speaker, Japanese)",
        "method": "single-speaker baseline difference: tau_e = mean_i(x_e^i) - mean_j(x_calm^j)",
        "baseline": BASELINE,
        "schema": EMOTIONS,
        "seed": args.seed,
        "min_samples": args.min_samples,
        "labels_file": os.path.abspath(args.labels),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "emotions": {},
    }
    for emo, tau in taus.items():
        n = tau / (np.linalg.norm(tau) + 1e-12)
        if not np.all(np.isfinite(n)):
            log(f"  refusing non-finite τ for {emo}")
            continue
        with open(os.path.join(args.out, f"tau_{emo}.f32"), "wb") as fh:
            fh.write(n.astype("<f4").tobytes())
        d = diag[emo]
        hold_cos = hold.get(emo, {}).get("cosine_to_fit")
        # A direction that is self-consistent but does not reproduce on held-out chapters is
        # memorising scene/recording state, not the emotion. Demote it rather than shipping it
        # as "ready" just because the in-sample numbers look good.
        quality, quality_reason = d["status"], d.get("reason")
        if quality == "ready" and hold_cos is not None and hold_cos < 0.30:
            quality = "experimental"
            quality_reason = f"poor generalisation to held-out chapters (cosine {hold_cos:.2f})"
        manifest["emotions"][emo] = {
            "file": f"tau_{emo}.f32",
            "dim": int(dim),
            "tau_norm": d["tau_norm"],
            "consistency": d["coherence"],
            "n_samples": d["n_samples"],
            "quality": quality,
            "quality_reason": quality_reason,
            "normalized": True,
            "holdout_cosine": hold_cos,
            "bootstrap_mean_cosine": boot.get(emo, {}).get("mean_cosine"),
        }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    diagnostics = {
        "generated_at": manifest["generated_at"],
        "schema": EMOTIONS,
        "baseline": BASELINE,
        "label_counts": {e: counts.get(e, 0) for e in EMOTIONS},
        "embedded_counts": {e: len(train_store.get(e, [])) for e in EMOTIONS},
        "directions": diag,
        "bootstrap": boot,
        "holdout": hold,
        "similar_pairs": sims,
        "excluded_train": dict(train_ex),
        "excluded_val": dict(val_ex),
        "missing_baseline": len(train_store.get(BASELINE, [])) == 0,
    }
    with open(os.path.join(args.out, "diagnostics.json"), "w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "samples_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"train": train_report, "val": val_report}, fh, ensure_ascii=False, indent=1)

    log(f"\nemitted {len(manifest['emotions'])} directions to {args.out}")
    unavailable = [e for e in EMOTIONS if e != BASELINE and e not in taus]
    if unavailable:
        log(f"unavailable: {unavailable}")
    log(f"elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())