"""Batch-label ATRI WAV clips with the 19-emotion schema via qwen3.8-omni-flash.

Guarantees:
  * resumable: one JSONL line per completed clip, keyed by filename+sha256+prompt_version
  * idempotent: rerunning skips clips whose stored hash and prompt version match
  * atomic-ish: each finished line is flushed and fsynced before moving on
  * auditable: records duration, hash, model, schema/prompt version, usage, review reasons
  * no secrets on disk: the API key is read in-process and never written or logged

Usage (run from the project tools directory, isolated interpreter):
    python -I annotate.py --audio-dir F:\\CNBackups\\ATRI\\raw \
        --out F:\\CNBackups\\ATRI\\emotion_annotation \
        --key-file F:\\CNBackups\\ATRI\\apikey.txt --workers 4
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schema import EmotionSchema, extract_json_object, validate_reply  # noqa: E402
import qwen_client  # noqa: E402

WAV_RE = re.compile(r"^ATR_b(\d+)_(\d+)([a-z_]*)\.wav$", re.I)


# ----------------------------------------------------------------------------
# audio inventory
# ----------------------------------------------------------------------------
@dataclass
class Clip:
    name: str
    path: str
    chapter: str
    duration: float
    sha256: str = ""
    transcript: str = ""

    @property
    def has_transcript(self) -> bool:
        return bool(self.transcript.strip())

    @property
    def suffix(self) -> str:
        m = WAV_RE.match(self.name)
        return m.group(3) if m else ""


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        return w.getnframes() / float(rate) if rate else 0.0


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_transcripts(path: Optional[str]) -> Dict[str, str]:
    """Parse esd.list (file|speaker|language|text) into a filename -> text mapping."""
    out: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 4 and parts[3].strip():
                out[os.path.basename(parts[0])] = parts[3].strip()
    return out


def scan(audio_dir: str, transcript_path: Optional[str]) -> List[Clip]:
    transcripts = load_transcripts(transcript_path)
    clips: List[Clip] = []
    for name in sorted(os.listdir(audio_dir)):
        if not name.lower().endswith(".wav"):
            continue
        path = os.path.join(audio_dir, name)
        if not os.path.isfile(path):
            continue
        m = WAV_RE.match(name)
        clips.append(
            Clip(
                name=name,
                path=path,
                chapter=m.group(1) if m else "unknown",
                duration=wav_duration(path),
                transcript=transcripts.get(name, ""),
            )
        )
    return clips


# ----------------------------------------------------------------------------
# state
# ----------------------------------------------------------------------------
class LabelStore:
    """Append-only JSONL store with an in-memory index for resume."""

    def __init__(self, out_dir: str, prompt_version: str):
        self.dir = out_dir
        self.prompt_version = prompt_version
        os.makedirs(out_dir, exist_ok=True)
        self.labels_path = os.path.join(out_dir, "labels.jsonl")
        self.failures_path = os.path.join(out_dir, "failures.jsonl")
        self.state_path = os.path.join(out_dir, "state.json")
        self._lock = threading.Lock()
        self._labels = open(self.labels_path, "a", encoding="utf-8")
        self._failures = open(self.failures_path, "a", encoding="utf-8")
        self.index: Dict[str, dict] = {}
        self.fail_counts: Dict[str, int] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    name = rec.get("file")
                    if name:
                        self.index[name] = rec
        if os.path.exists(self.failures_path):
            with open(self.failures_path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    n = rec.get("file")
                    if n:
                        self.fail_counts[n] = self.fail_counts.get(n, 0) + 1

    def is_done(self, clip: Clip) -> bool:
        rec = self.index.get(clip.name)
        if not rec:
            return False
        return (
            rec.get("sha256") == clip.sha256
            and rec.get("prompt_version") == self.prompt_version
            and rec.get("status") == "done"
        )

    def append(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            self._labels.write(line + "\n")
            self._labels.flush()
            os.fsync(self._labels.fileno())
            self.index[rec["file"]] = rec

    def append_failure(self, rec: dict, count: bool = True) -> None:
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            self._failures.write(line + "\n")
            self._failures.flush()
            os.fsync(self._failures.fileno())
            if count:
                self.fail_counts[rec["file"]] = self.fail_counts.get(rec["file"], 0) + 1

    def close(self) -> None:
        self._labels.close()
        self._failures.close()


# ----------------------------------------------------------------------------
# labeling
# ----------------------------------------------------------------------------
@dataclass
class Counters:
    done: int = 0
    skipped: int = 0
    failed: int = 0
    review: int = 0
    disputed: int = 0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_usage(self, u: "qwen_client.Usage") -> None:
        with self.lock:
            self.calls += 1
            self.prompt_tokens += u.prompt_tokens
            self.completion_tokens += u.completion_tokens
            self.total_tokens += u.total_tokens


SECOND_OPINION_SUFFIX = (
    "\n\n这是独立复核。请不要受任何先前判断影响，重新只听一遍这段音频，"
    "严格按上面的 19 类定义给出唯一主标签。"
)


# Errors that mean "the account/model cannot serve requests right now", not "this clip failed".
_ACCOUNT_ERROR_MARKERS = (
    "quota exhausted",
    "insufficient balance",
    "invalid api key",
    "unauthorized",
    "incorrect api key",
    "model not found",
    "does not exist",
    "access denied",
    "account",
)


def _is_account_level_error(exc: BaseException) -> bool:
    """True for quota/auth/model failures that would fail every remaining clip identically."""
    status = getattr(exc, "status", None)
    if status in (401, 402, 403, 404):
        return True
    text = str(exc).lower()
    return any(m in text for m in _ACCOUNT_ERROR_MARKERS)


def label_one(
    client: "qwen_client.OmniClient",
    schema: EmotionSchema,
    clip: Clip,
    schema_prompt: str,
    *,
    use_transcript: bool,
    review_threshold: float,
    recheck: bool,
    max_attempts: int,
) -> dict:
    started = time.time()
    prompt = schema.build_prompt(clip.transcript if use_transcript else None)
    first_text, usage1 = qwen_client.with_retries(
        lambda: client.analyze_audio(clip.path, prompt, json_mode=True),
        attempts=max_attempts,
        on_retry=lambda n, d, e: print(f"  retry {n} in {d:.1f}s for {clip.name}: {e}", flush=True),
    )
    first = validate_reply(extract_json_object(first_text), schema)

    second = None
    usage2 = qwen_client.Usage()
    reasons: List[str] = []

    low_conf = first["confidence"] < review_threshold
    if first["alternate"] and first["confidence"] < review_threshold + 0.15:
        reasons.append("close_alternate")

    if recheck and low_conf:
        try:
            text2, usage2 = qwen_client.with_retries(
                lambda: client.analyze_audio(clip.path, prompt + SECOND_OPINION_SUFFIX, json_mode=True),
                attempts=max_attempts,
                on_retry=lambda n, d, e: print(f"  retry {n} in {d:.1f}s for {clip.name}: {e}", flush=True),
            )
            second = validate_reply(extract_json_object(text2), schema)
        except Exception as exc:  # a failed recheck must not lose the primary label
            reasons.append(f"recheck_failed:{type(exc).__name__}")

    if second:
        if second["label"] != first["label"]:
            reasons.append("label_disagreement")
            second["agrees"] = False
        else:
            second["agrees"] = True

    if low_conf:
        reasons.append("low_confidence")
    if not first["audible_speech"]:
        reasons.append("no_audible_speech")
    if clip.duration < 1.0:
        reasons.append("very_short")
    if clip.duration > 12.0:
        reasons.append("long_clip")
    if reasons:
        reasons = list(dict.fromkeys(reasons))

    return {
        "file": clip.name,
        "path": clip.path,
        "chapter": clip.chapter,
        "suffix": clip.suffix,
        "duration_sec": round(clip.duration, 3),
        "sha256": clip.sha256,
        "has_transcript": clip.has_transcript,
        "label": first["label"],
        "label_zh": first["label_zh"],
        "confidence": round(first["confidence"], 4),
        "evidence": first["evidence"],
        "alternate": first["alternate"],
        "transcript_used": bool(use_transcript and clip.has_transcript),
        "transcript": (clip.transcript[:200] if clip.transcript else None),
        "audible_speech": first["audible_speech"],
        "second_opinion": second,
        "review": bool(reasons),
        "review_reasons": reasons,
        "status": "done",
        "model": client.model,
        "schema_version": schema.version,
        "prompt_version": schema.prompt_version,
        "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - started, 3),
        "usage": {
            "first": usage1.as_dict(),
            "second": usage2.as_dict(),
        },
    }


# ----------------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------------
def stratified_pick(clips: List[Clip], count: int, seed: int) -> List[Clip]:
    """Representative sample: every chapter, several duration buckets, edge cases."""
    rng = random.Random(seed)
    picked: List[Clip] = []
    seen = set()

    def take(c: Clip) -> None:
        if c.name not in seen and len(picked) < count:
            seen.add(c.name)
            picked.append(c)

    # 1) edge cases: very short, long, special suffixes, no transcript
    edge_groups = {
        "veryshort": [c for c in clips if c.duration < 0.8],
        "long": [c for c in clips if c.duration > 12.0],
        "suffix": [c for c in clips if c.suffix],
        "notranscript": [c for c in clips if not c.has_transcript],
    }
    for _, group in edge_groups.items():
        for c in rng.sample(group, min(2, len(group))):
            take(c)

    # 2) one or two per chapter across the duration range
    by_chapter: Dict[str, List[Clip]] = {}
    for c in clips:
        by_chapter.setdefault(c.chapter, []).append(c)
    chapters = sorted(by_chapter)
    rng.shuffle(chapters)
    for ch in chapters:
        group = sorted(by_chapter[ch], key=lambda c: c.duration)
        if not group:
            continue
        take(group[len(group) // 2])
        if len(picked) < count:
            take(group[rng.randrange(len(group))])

    # 3) fill by duration stratification
    ordered = sorted(clips, key=lambda c: c.duration)
    while len(picked) < count and len(seen) < len(ordered):
        take(ordered[rng.randrange(len(ordered))])
    return picked[:count]


# ----------------------------------------------------------------------------
# reports
# ----------------------------------------------------------------------------
def write_review_queue(store: LabelStore, path: str) -> int:
    rows = [
        r
        for r in store.index.values()
        if r.get("review") and r.get("status") == "done"
    ]
    rows.sort(key=lambda r: (r.get("confidence", 1.0), r["file"]))
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "file", "chapter", "duration_sec", "label", "label_zh", "confidence",
            "alternate", "second_label", "second_confidence", "review_reasons",
            "evidence", "has_transcript",
        ])
        for r in rows:
            s = r.get("second_opinion") or {}
            w.writerow([
                r["file"], r.get("chapter", ""), r.get("duration_sec", ""),
                r["label"], r.get("label_zh", ""), r.get("confidence", ""),
                r.get("alternate") or "", s.get("label", ""), s.get("confidence", ""),
                ";".join(r.get("review_reasons") or []), r.get("evidence", ""),
                r.get("has_transcript", ""),
            ])
    return len(rows)


def write_summary(store: LabelStore, schema: EmotionSchema, clips: List[Clip], out_dir: str, counters: Counters) -> dict:
    total_expected = len(clips)
    done = [r for r in store.index.values() if r.get("status") == "done"]
    by_label: Dict[str, int] = {e.id: 0 for e in schema.emotions}
    by_label_transcript: Dict[str, int] = {e.id: 0 for e in schema.emotions}
    by_chapter: Dict[str, Dict[str, int]] = {}
    dur = {"min": None, "max": None, "mean": 0.0}
    for r in done:
        lab = r.get("label")
        if lab in by_label:
            by_label[lab] += 1
            if r.get("has_transcript"):
                by_label_transcript[lab] += 1
        ch = by_chapter.setdefault(r.get("chapter", "unknown"), {})
        ch[lab] = ch.get(lab, 0) + 1
        d = r.get("duration_sec") or 0
        dur["min"] = d if dur["min"] is None else min(dur["min"], d)
        dur["max"] = d if dur["max"] is None else max(dur["max"], d)

    if done:
        dur["mean"] = round(sum(r.get("duration_sec") or 0 for r in done) / len(done), 3)

    agreement = [
        r for r in done if r.get("second_opinion") is not None
    ]
    agree_n = sum(1 for r in agreement if r["second_opinion"].get("agrees"))

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema_version": schema.version,
        "prompt_version": schema.prompt_version,
        "model": next((r.get("model") for r in done), None),
        "counts": {
            "expected": total_expected,
            "done": len(done),
            "remaining": max(0, total_expected - len(done)),
            "review": sum(1 for r in done if r.get("review")),
            "disputed": sum(1 for r in done if "label_disagreement" in (r.get("review_reasons") or [])),
            "failed_clips": len(store.fail_counts),
        },
        "label_distribution": by_label,
        "label_distribution_with_transcript": by_label_transcript,
        "chapters": {k: dict(sorted(v.items())) for k, v in sorted(by_chapter.items())},
        "duration": dur,
        "second_opinion": {
            "count": len(agreement),
            "agree": agree_n,
            "agreement_rate": round(agree_n / len(agreement), 4) if agreement else None,
        },
        "usage_run": {
            "requests": counters.calls,
            "prompt_tokens": counters.prompt_tokens,
            "completion_tokens": counters.completion_tokens,
            "total_tokens": counters.total_tokens,
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    write_summary_md(summary, schema, os.path.join(out_dir, "summary.md"))
    return summary


def write_summary_md(s: dict, schema: EmotionSchema, path: str) -> None:
    c = s["counts"]
    lines = [
        "# ATRI 19 类情感标注报告",
        "",
        f"- 生成时间：{s['generated_at']}",
        f"- 模型：`{s['model']}`　schema `{s['schema_version']}`　prompt `{s['prompt_version']}`",
        f"- 计划 {c['expected']} 条；已完成 {c['done']}；剩余 {c['remaining']}",
        f"- 需复核 {c['review']}；复标分歧 {c['disputed']}；失败条目 {c['failed_clips']}",
        f"- 时长范围 {s['duration']['min']}–{s['duration']['max']} 秒，均值 {s['duration']['mean']} 秒",
        "",
        "## 类别分布",
        "",
        "| 类别 | id | 全部 | 有转写 | 占比 |",
        "|---|---|---:|---:|---:|",
    ]
    total = c["done"] or 1
    for e in schema.emotions:
        n = s["label_distribution"].get(e.id, 0)
        nt = s["label_distribution_with_transcript"].get(e.id, 0)
        pct = f"{100.0 * n / total:.1f}%"
        lines.append(f"| {e.zh} | `{e.id}` | {n} | {nt} | {pct} |")
    so = s["second_opinion"]
    lines += [
        "",
        f"## 复标一致性：{so['agree']}/{so['count']}（{so['agreement_rate']}）" if so["count"] else "## 复标一致性：本轮尚未产生复标样本",
        "",
        "## 本轮 API 用量",
        "",
        f"- 请求数：{s['usage_run']['requests']}",
        f"- prompt tokens：{s['usage_run']['prompt_tokens']}",
        f"- completion tokens：{s['usage_run']['completion_tokens']}",
        f"- total tokens：{s['usage_run']['total_tokens']}",
        "",
        "> 费用需按服务商当前单价单独核算；模型自报 confidence 是主观分数，不是校准概率。",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Label ATRI clips with 19 emotions via qwen3.8-omni-flash")
    ap.add_argument("--audio-dir", default=r"F:\CNBackups\ATRI\raw")
    ap.add_argument("--out", default=r"F:\CNBackups\ATRI\emotion_annotation")
    ap.add_argument("--key-file", default=r"F:\CNBackups\ATRI\apikey.txt")
    ap.add_argument("--transcripts", default=r"F:\CNBackups\ATRI\esd.list")
    ap.add_argument("--emotions", default=None, help="emotions.json path (default: alongside this script)")
    ap.add_argument("--base-url", default=qwen_client.DEFAULT_BASE_URL)
    ap.add_argument("--model", default=qwen_client.DEFAULT_MODEL)
    ap.add_argument("--use-transcript", dest="use_transcript", action="store_true", default=True,
                    help="Feed esd.list transcript text to the model as auxiliary context")
    ap.add_argument("--no-transcript", dest="use_transcript", action="store_false")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-interval", type=float, default=0.0,
                    help="minimum seconds between request starts, shared across workers (RPM pacing)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips")
    ap.add_argument("--select", choices=["all", "stratified", "first"], default="all")
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--review-threshold", type=float, default=0.55)
    ap.add_argument("--recheck", dest="recheck", action="store_true", default=True)
    ap.add_argument("--no-recheck", dest="recheck", action="store_false")
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hash-workers", type=int, default=8)
    args = ap.parse_args()

    schema = EmotionSchema(args.emotions or os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotions.json"))
    clips = scan(args.audio_dir, args.transcripts)
    print(f"scanned {len(clips)} wav files in {args.audio_dir}", flush=True)
    if not clips:
        print("no audio found", file=sys.stderr)
        return 2

    # hashes are expensive-ish; compute in parallel once per run
    todo_hash = [c for c in clips if not c.sha256]
    if todo_hash:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(1, args.hash_workers)) as ex:
            for c, h in zip(todo_hash, ex.map(lambda c: sha256_file(c.path), todo_hash)):
                c.sha256 = h

    store = LabelStore(args.out, schema.prompt_version)
    pending = [c for c in clips if not store.is_done(c)]
    print(f"already done: {len(clips) - len(pending)}; pending: {len(pending)}", flush=True)

    if args.select == "stratified" and args.limit:
        selected = stratified_pick(pending, args.limit, args.seed)
    elif args.limit:
        selected = pending[: args.limit]
    else:
        selected = pending

    # retry clips that failed in earlier runs, but cap total attempts across runs
    selected = [c for c in selected if store.fail_counts.get(c.name, 0) < args.max_attempts]

    print(f"this run: {len(selected)} clips ({args.model} @ {args.base_url})", flush=True)
    if args.dry_run:
        for c in selected:
            print(f"  {c.name}  {c.duration:.2f}s  ch{c.chapter}  tr={c.has_transcript}", flush=True)
        print("dry run: no API calls made", flush=True)
        store.close()
        return 0

    key = qwen_client.read_api_key(args.key_file)
    client = qwen_client.OmniClient(
        key, base_url=args.base_url, model=args.model, timeout=args.timeout
    )
    counters = Counters()
    pace_lock = threading.Lock()
    last_start = [0.0]

    def pace():
        if args.min_interval <= 0:
            return
        with pace_lock:
            wait = last_start[0] + args.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            last_start[0] = time.time()

    review_csv = os.path.join(args.out, "review_queue.csv")
    lock_print = threading.Lock()
    fatal: list[Optional[str]] = [None]

    def work(clip: Clip) -> None:
        if fatal[0] is not None:
            return
        try:
            pace()
            rec = label_one(
                client, schema, clip, None,
                use_transcript=args.use_transcript,
                review_threshold=args.review_threshold,
                recheck=args.recheck,
                max_attempts=args.max_attempts,
            )
        except Exception as exc:
            # Account-level failures (quota, auth, model unavailable) are not this clip's fault.
            # Record them for audit, do NOT count them against the clip, and stop the whole batch
            # so the remaining clips are not burned through as false per-clip failures.
            is_fatal = _is_account_level_error(exc)
            store.append_failure(
                {
                    "file": clip.name,
                    "sha256": clip.sha256,
                    "prompt_version": schema.prompt_version,
                    "error": f"{type(exc).__name__}: {exc}",
                    "account_level": is_fatal,
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
                count=not is_fatal,
            )
            if is_fatal:
                with counters.lock:
                    if fatal[0] is None:
                        fatal[0] = str(exc)
                with lock_print:
                    print(f"\nABORT: account-level API failure, stopping batch: {exc}", flush=True)
                return
            with counters.lock:
                counters.failed += 1
            with lock_print:
                print(f"FAIL {clip.name}: {type(exc).__name__}: {exc}", flush=True)
            return

        store.append(rec)
        u1 = rec["usage"]["first"]
        u2 = rec["usage"]["second"]
        counters.add_usage(qwen_client.Usage(u1["prompt_tokens"], u1["completion_tokens"], u1["total_tokens"]))
        if u2["total_tokens"]:
            counters.add_usage(qwen_client.Usage(u2["prompt_tokens"], u2["completion_tokens"], u2["total_tokens"]))
        with counters.lock:
            counters.done += 1
            if rec["review"]:
                counters.review += 1
            if "label_disagreement" in rec["review_reasons"]:
                counters.disputed += 1
            n = counters.done
        with lock_print:
            marker = "REVIEW" if rec["review"] else ""
            print(
                f"[{n}/{len(selected)}] {clip.name} -> {rec['label']}({rec['label_zh']}) "
                f"conf={rec['confidence']:.2f} {marker}",
                flush=True,
            )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = [ex.submit(work, c) for c in selected]
            for _ in as_completed(futures):
                pass
    except KeyboardInterrupt:
        print("\ninterrupted; completed lines are already on disk", flush=True)
    finally:
        elapsed = time.time() - started
        review_n = write_review_queue(store, review_csv)
        summary = write_summary(store, schema, clips, args.out, counters)
        store.close()
        print(
            f"\nrun complete in {elapsed:.1f}s: done={counters.done} failed={counters.failed} "
            f"review={counters.review} disputed={counters.disputed}",
            flush=True,
        )
        print(
            f"usage: requests={counters.calls} total_tokens={counters.total_tokens} "
            f"(prompt={counters.prompt_tokens}, completion={counters.completion_tokens})",
            flush=True,
        )
        print(f"labels: {store.labels_path}\nreview queue ({review_n}): {review_csv}\nsummary: {args.out}\\summary.md", flush=True)
        print(f"progress: {summary['counts']['done']}/{summary['counts']['expected']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())