"""Minimal OpenAI-compatible client for qwen3.8-omni-flash audio understanding.

Only the pieces this pipeline needs: stream a base64 WAV plus a text instruction,
concatenate `delta.content`, and return the text. No third-party SDK required.

The API key is read from a file path given by the caller and is never logged,
never written into results, and never included in exception messages.
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests

DEFAULT_BASE_URL = "https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1"
ALT_BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.8-omni-flash"


class ApiError(RuntimeError):
    """Non-retryable or exhausted API failure. Message never contains the key."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def read_api_key(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as fh:
        key = fh.read().strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    if len(key) < 8:
        raise ValueError(f"API key file looks invalid (too short): {path}")
    return key


def encode_wav_data_uri(path: str) -> str:
    with open(path, "rb") as fh:
        raw = fh.read()
    return "data:;base64," + base64.b64encode(raw).decode("ascii")


class OmniClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 180.0,
        disable_thinking: bool = True,
        extra_body: Optional[dict] = None,
        session: Optional[requests.Session] = None,
    ):
        self._key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.disable_thinking = disable_thinking
        self.extra_body = dict(extra_body or {})
        self._session = session or requests.Session()
        self.request_count = 0

    # ------------------------------------------------------------------
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def _payload(self, audio_path: str, prompt: str, json_mode: bool) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encode_wav_data_uri(audio_path),
                                "format": "wav",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "modalities": ["text"],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.1,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.disable_thinking:
            # Compatible-mode spelling used by Qwen models; harmless if the
            # endpoint ignores it, and configurable if a deployment rejects it.
            payload["enable_thinking"] = False
        payload.update(self.extra_body)
        return payload

    # ------------------------------------------------------------------
    def analyze_audio(self, audio_path: str, prompt: str, json_mode: bool = True) -> tuple[str, Usage]:
        """Send one audio clip and return (concatenated_text, usage)."""
        payload = self._payload(audio_path, prompt, json_mode)
        url = f"{self.base_url}/chat/completions"
        self.request_count += 1
        try:
            resp = self._session.post(
                url, headers=self._headers(), json=payload, stream=True, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise ApiError(f"request failed: {type(exc).__name__}: {exc}") from None

        if resp.status_code != 200:
            body = _safe_error_body(resp)
            raise ApiError(f"HTTP {resp.status_code}: {body}", status=resp.status_code)

        text_parts: list[str] = []
        usage = Usage()
        try:
            for line in resp.iter_lines(decode_unicode=False):
                if not line:
                    continue
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    chunk = json.loads(data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if chunk.get("usage"):
                    u = chunk["usage"] or {}
                    usage = Usage(
                        int(u.get("prompt_tokens") or 0),
                        int(u.get("completion_tokens") or 0),
                        int(u.get("total_tokens") or 0),
                    )
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if isinstance(piece, str):
                        text_parts.append(piece)
                    elif isinstance(piece, list):
                        for item in piece:
                            if isinstance(item, dict) and isinstance(item.get("text"), str):
                                text_parts.append(item["text"])
                    if choice.get("finish_reason"):
                        pass
        finally:
            resp.close()

        text = "".join(text_parts).strip()
        if not text:
            raise ApiError("empty model output")
        return text, usage


def _safe_error_body(resp: requests.Response, limit: int = 600) -> str:
    """Extract an error message without ever echoing credentials."""
    try:
        body = resp.text
    except Exception:
        return "<unreadable body>"
    try:
        obj = json.loads(body)
        if isinstance(obj, dict):
            err = obj.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or err.get("code") or ""
                return str(msg)[:limit] or "<no message>"
            if isinstance(err, str):
                return err[:limit]
            msg = obj.get("message")
            if msg:
                return str(msg)[:limit]
    except json.JSONDecodeError:
        pass
    body = " ".join(body.split())
    return body[:limit]


def with_retries(fn, *, attempts: int = 5, base_delay: float = 2.0, max_delay: float = 60.0, on_retry=None):
    """Retry transient failures (429/5xx/network) with jittered exponential backoff."""
    import random

    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            return fn()
        except ApiError as exc:
            last = exc
            retryable = exc.status is None or exc.status == 429 or (exc.status or 0) >= 500
            if not retryable or i == attempts - 1:
                raise
        except (requests.RequestException, OSError) as exc:
            last = exc
            if i == attempts - 1:
                raise
        delay = min(max_delay, base_delay * (2 ** i)) * (0.7 + 0.6 * random.random())
        if on_retry:
            on_retry(i + 1, delay, last)
        time.sleep(delay)
    if last:
        raise last
    raise RuntimeError("retry loop exited without a result")