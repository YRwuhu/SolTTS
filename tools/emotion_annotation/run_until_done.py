"""Keep labeling until the corpus is complete, surviving intermittent model permissions.

The account's audio-model access flips between available and `403 Access denied` /
`403 quota exhausted` within minutes. Restarting by hand each time is wasteful, so this
driver probes a model, runs a batch, and backs off when the batch aborts.

It never fabricates labels: an aborted batch simply leaves the remaining clips pending.
Every attempt appends to the normal labels.jsonl / failures.jsonl, so progress is durable.

Usage:
    python -I run_until_done.py --models qwen3.5-omni-plus,qwen3-omni-flash,qwen3.8-omni-flash
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qwen_client  # noqa: E402


def count_labels(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def probe(model: str, key: str, audio: str, timeout: float) -> tuple[bool, str]:
    try:
        client = qwen_client.OmniClient(key, model=model, timeout=timeout)
        client.analyze_audio(audio, '只输出JSON：{"ok":true}')
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 - we only need a yes/no plus a reason
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3.5-omni-plus,qwen3.5-omni-flash,qwen3-omni-flash,qwen3.8-omni-flash")
    ap.add_argument("--key-file", default=r"F:\CNBackups\ATRI\apikey.txt")
    ap.add_argument("--audio-dir", default=r"F:\CNBackups\ATRI\raw")
    ap.add_argument("--labels", default=r"F:\CNBackups\ATRI\emotion_annotation\labels.jsonl")
    ap.add_argument("--out", default=r"F:\CNBackups\ATRI\emotion_annotation")
    ap.add_argument("--probe-audio", default=r"F:\CNBackups\ATRI\raw\ATR_b999_001.wav")
    ap.add_argument("--total", type=int, default=2222)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-interval", type=float, default=1.0)
    ap.add_argument("--max-rounds", type=int, default=0, help="0 = until complete")
    ap.add_argument("--idle-wait", type=float, default=120.0, help="seconds between probes when all models are denied")
    ap.add_argument("--max-idle-wait", type=float, default=900.0)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    key = qwen_client.read_api_key(args.key_file)
    models: List[str] = [m.strip() for m in args.models.split(",") if m.strip()]
    log_path = os.path.join(args.out, "driver.log")
    os.makedirs(args.out, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    wait = args.idle_wait
    rounds = 0
    while True:
        done = count_labels(args.labels)
        log(f"progress {done}/{args.total} ({100.0 * done / args.total:.1f}%)")
        if done >= args.total:
            log("all clips labeled; nothing left to do")
            return 0
        rounds += 1
        if args.max_rounds and rounds > args.max_rounds:
            log(f"reached max rounds ({args.max_rounds}); stopping with {done} labeled")
            return 0

        model = None
        last_err = ""
        for m in models:
            ok, reason = probe(m, key, args.probe_audio, args.timeout)
            if ok:
                model = m
                log(f"using model {m}")
                break
            last_err = reason
            log(f"probe {m}: {reason[:110]}")

        if model is None:
            log(f"no model available; sleeping {wait:.0f}s")
            time.sleep(wait)
            wait = min(args.max_idle_wait, wait * 1.5)
            continue
        wait = args.idle_wait  # reset after a success

        cmd = [
            sys.executable, "-I", os.path.join(HERE, "annotate.py"),
            "--model", model,
            "--workers", str(args.workers),
            "--min-interval", str(args.min_interval),
            "--out", args.out,
            "--audio-dir", args.audio_dir,
            "--key-file", args.key_file,
        ]
        log(f"starting batch with {model}")
        before = done
        try:
            proc = subprocess.run(cmd, cwd=HERE, capture_output=False)
            rc = proc.returncode
        except KeyboardInterrupt:
            log("interrupted by user")
            return 130
        after = count_labels(args.labels)
        log(f"batch exited rc={rc} labeled {before} -> {after} (+{after - before})")
        if after == before:
            # No progress at all: the model died mid-batch. Back off before retrying.
            time.sleep(min(args.max_idle_wait, 30.0))


if __name__ == "__main__":
    sys.exit(main())