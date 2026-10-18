"""Extracts emotion direction vectors τ from CSEMOTIONS.

Key design (paired-difference formula, preventing speaker identity from leaking in):
    τ_e = mean_{s,i} ( x_e^{s,i} - x_n^{s} )
where s is the speaker, i is the i-th sample for that speaker, and x_n^{s} is the same speaker's neutral mean.

Dataset size: about 59 samples per speaker per emotion (4160 / 10 / 7).
By default, N samples per speaker per emotion are used (default 20) to ensure audio decodes.

Usage:
    python extract_tau.py --parquet-dir E:\\AI\\_tmp --out E:\\AI\\SolTTS\\resources\\emotion_vectors
"""
import argparse
import glob
import io
import json
import os
import sys
import time

import numpy as np


def log(*a):
    print(*a, flush=True)


EMOTIONS = ["neutral", "happy", "angry", "sad", "surprise", "playfulness", "fearful"]
EMO_ALIAS = {
    "neutral": "neutral", "happy": "happy", "angry": "angry", "sad": "sad",
    "surprise": "surprise", "playful": "playfulness", "playfulness": "playfulness",
    "fearful": "fearful", "fear": "fearful",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default=r"E:\AI\_tmp")
    ap.add_argument("--resources", default=r"E:\AI\SolTTS\resources")
    ap.add_argument("--out", default=r"E:\AI\SolTTS\resources\emotion_vectors")
    ap.add_argument("--per-speaker-emotion", type=int, default=20)
    ap.add_argument("--min-sec", type=float, default=1.5)
    ap.add_argument("--max-sec", type=float, default=12.0)
    ap.add_argument("--threads", type=int, default=2, help="CPU thread count (low interference)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "dml"],
                    help="auto: prefer the DirectML discrete GPU, fall back to CPU on failure")
    ap.add_argument("--save-wav", action="store_true", help="Also export trimmed wav files (for manual listening)")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import onnxruntime as ort
    import soundfile as sf
    import soxr

    import faulthandler
    faulthandler.enable()

    t0 = time.time()
    spk_path = os.path.join(args.resources, "shared", "v1_extra", "speaker_encoder.onnx")
    if not os.path.exists(spk_path):
        log("Missing speaker_encoder.onnx:", spk_path)
        sys.exit(1)

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    if args.device in ("auto", "dml"):
        avail = ort.get_available_providers()
        if "DmlExecutionProvider" in avail:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif args.device == "dml":
            log("DML requested but unavailable; falling back to CPU")
    log("Using providers:", providers)
    spk = ort.InferenceSession(spk_path, so, providers=providers)
    in_name = spk.get_inputs()[0].name
    log(f"speaker_encoder input: {in_name}, output: {spk.get_outputs()[0].name} {spk.get_outputs()[0].shape}")

    files = sorted(glob.glob(os.path.join(args.parquet_dir, "csemo_*.parquet")))
    if not files:
        log("No parquet shards found:", args.parquet_dir)
        sys.exit(1)
    log("Shards:", [os.path.basename(f) for f in files])

    def emb_of(audio, sr):
        """audio -> 16k -> speaker encoder -> 20480-dim embedding"""
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != 16000:
            a = soxr.resample(a, sr, 16000).astype(np.float32)
        a = a - a.mean()
        peak = float(np.abs(a).max())
        if peak > 1e-6:
            a = a / peak * 0.9
        e = spk.run(None, {in_name: a[None, :].astype(np.float32)})[0]
        return np.asarray(e).reshape(-1).astype(np.float32)

    # ---------- 1. Collect samples ----------
    # per speaker -> per emotion -> list of embeddings
    store = {}          # speaker -> {emotion: [emb,...]}
    meta = []           # Records provenance for traceability/listening
    counts = {}

    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        log(f"\n[{fi+1}/{len(files)}] {os.path.basename(f)}: {pf.metadata.num_rows} rows")
        for batch in pf.iter_batches(batch_size=64, columns=["audio", "emotion", "speaker", "text"]):
            d = batch.to_pydict()
            n = len(d["emotion"])
            for i in range(n):
                speaker = str(d["speaker"][i]).strip()
                emo_raw = str(d["emotion"][i]).strip().lower()
                emo = EMO_ALIAS.get(emo_raw)
                if emo is None:
                    continue
                counts.setdefault((speaker, emo), 0)
                if counts[(speaker, emo)] >= args.per_speaker_emotion:
                    continue

                au = d["audio"][i]
                if not au:
                    continue
                # In parquet, audio is struct<bytes: binary, path: string>; decode from bytes
                raw = au.get("bytes") if isinstance(au, dict) else None
                if raw is None:
                    continue
                try:
                    arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
                except Exception as ex:
                    log(f"   Decode failed {speaker}/{emo}: {ex}")
                    continue
                arr = arr.mean(axis=1)
                sr = int(sr)
                dur = len(arr) / sr
                if dur < args.min_sec:
                    continue
                if args.max_sec > 0 and dur > args.max_sec:
                    # Take the middle segment, preserving the region with the strongest emotional expression
                    c = len(arr) // 2
                    half = int(args.max_sec * sr / 2)
                    arr = arr[max(0, c - half): c + half]
                try:
                    e = emb_of(arr, sr)
                except Exception as ex:
                    log(f"   Skipping {speaker}/{emo}: {ex}")
                    continue
                store.setdefault(speaker, {}).setdefault(emo, []).append(e)
                counts[(speaker, emo)] += 1
                meta.append({"speaker": speaker, "emotion": emo, "dur": round(dur, 2),
                             "text": (d["text"][i] or "")[:40], "shard": os.path.basename(f)})

                if args.save_wav:
                    wavdir = os.path.join(args.out, "samples", speaker)
                    os.makedirs(wavdir, exist_ok=True)
                    p = os.path.join(wavdir, f"{emo}_{counts[(speaker,emo)]:03d}.wav")
                    sf.write(p, arr, sr)

        log("  Current sample counts:", {f"{k[0]}/{k[1]}": v for k, v in sorted(counts.items())[:6]}, "...")

    log(f"\nCollection complete: {len(meta)} samples, {len(store)} speakers, elapsed {time.time()-t0:.1f}s")
    for spk_id in sorted(store):
        row = {e: len(store[spk_id].get(e, [])) for e in EMOTIONS}
        log(f"  {spk_id}: {row}")

    # ---------- 2. Compute τ from paired differences ----------
    log("\n==== Paired-difference τ computation ====")
    taus = {}
    diag = {}
    for emo in EMOTIONS:
        if emo == "neutral":
            continue
        diffs = []
        per_speaker = {}
        for s in sorted(store):
            xs = store[s]
            if "neutral" not in xs or not xs["neutral"]:
                continue
            if emo not in xs or not xs[emo]:
                continue
            mean_n = np.mean(np.stack(xs["neutral"]), axis=0)
            mean_e = np.mean(np.stack(xs[emo]), axis=0)
            diffs.append(mean_e - mean_n)
            per_speaker[s] = {"n_emo": len(xs[emo]), "n_neu": len(xs["neutral"]),
                              "diff_norm": float(np.linalg.norm(mean_e - mean_n))}
        if not diffs:
            log(f"  {emo}: no paired samples")
            continue
        D = np.stack(diffs)
        tau = D.mean(axis=0)
        # Diagnostic: cross-speaker consistency = norm of mean vector / mean norm of individual difference vectors (closer to 1 = more consistent)
        consistency = float(np.linalg.norm(tau) / (np.linalg.norm(D, axis=1).mean() + 1e-9))
        taus[emo] = tau
        diag[emo] = {
            "n_speakers": len(diffs),
            "tau_norm": float(np.linalg.norm(tau)),
            "mean_diff_norm": float(np.linalg.norm(D, axis=1).mean()),
            "std_diff_norm": float(np.linalg.norm(D, axis=1).std()),
            "consistency": consistency,
            "per_speaker": per_speaker,
        }
        log(f"  {emo:<12} n_spk={len(diffs)}  |τ|={np.linalg.norm(tau):8.3f}  "
            f"mean|Δ|={diag[emo]['mean_diff_norm']:8.3f}  consistency={consistency:.4f}")

    # ---------- 3. Orthogonality and normalization ----------
    log("\n==== Cosine similarity between τ vectors (should be low, indicating distinguishable directions) ====")
    names = [e for e in EMOTIONS if e != "neutral" and e in taus]
    for i in range(len(names)):
        row = []
        for j in range(len(names)):
            a, b = taus[names[i]], taus[names[j]]
            c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            row.append(f"{c:+.3f}")
        log(f"  {names[i]:<12} " + " ".join(row))

    # ---------- 4. Save ----------
    os.makedirs(args.out, exist_ok=True)
    # 4a. npz (for numerical analysis)
    np.savez(os.path.join(args.out, "tau_vectors.npz"), **{k: v for k, v in taus.items()})
    # 4b. Normalized float32 binary (read directly by C#, mmap-style) + json metadata
    manifest = {
        "format": "soltts-emotion-tau",
        "version": 1,
        "dim": int(len(next(iter(taus.values())))),
        "source_dataset": "AIDC-AI/CSEMOTIONS (ATH-MaaS/CSEMOTIONS)",
        "method": "paired-difference: tau_e = mean_s(mean_i x_e^{s,i} - mean_j x_n^{s,j})",
        "samples_per_speaker_emotion": args.per_speaker_emotion,
        "n_samples_total": len(meta),
        "emotions": {},
    }
    for e, tau in taus.items():
        n = tau / (np.linalg.norm(tau) + 1e-9)
        with open(os.path.join(args.out, f"tau_{e}.f32"), "wb") as fh:
            fh.write(n.astype("<f4").tobytes())
        manifest["emotions"][e] = {
            "file": f"tau_{e}.f32",
            "dim": int(tau.size),
            "tau_norm": diag[e]["tau_norm"],
            "consistency": diag[e]["consistency"],
            "n_speakers": diag[e]["n_speakers"],
            "normalized": True,
        }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "diagnostics.json"), "w", encoding="utf-8") as fh:
        json.dump(diag, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "samples_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)

    log(f"\nSaved to {args.out}")
    log("  tau_vectors.npz / tau_<emotion>.f32 / manifest.json / diagnostics.json")
    log(f"Total elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
