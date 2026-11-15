"""Shared schema for the ATRI 19-emotion annotation pipeline.

Single source of truth for emotion ids, Chinese labels and aliases. The C# runtime
consumes a manifest derived from the same ids, so do not duplicate lists elsewhere.

Safe to import with `python -I`: the caller must add this directory to sys.path
explicitly (isolated mode strips the script directory).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotions.json")

# Runtime ids in the exact order requested by the user.
EXPECTED_IDS = [
    "happy", "calm", "disgust", "shy", "fearful", "angry", "serious",
    "nervous", "flustered", "confused", "excited", "helpless", "worried",
    "surprised", "crying", "lovestruck", "embarrassed", "confident", "playful",
]


@dataclass(frozen=True)
class Emotion:
    id: str
    zh: str
    en: str
    definition: str
    boundary: str
    confusable_with: List[str]


class EmotionSchema:
    def __init__(self, path: str = DEFAULT_CONFIG):
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.version: str = raw["version"]
        self.prompt_version: str = raw["prompt_version"]
        self.scheme: str = raw["scheme"]
        self.baseline: str = raw["baseline"]
        self.notes: List[str] = raw.get("notes", [])
        self.emotions: List[Emotion] = [
            Emotion(
                id=e["id"],
                zh=e["zh"],
                en=e["en"],
                definition=e["definition"],
                boundary=e["boundary"],
                confusable_with=list(e.get("confusable_with", [])),
            )
            for e in raw["emotions"]
        ]
        self.by_id: Dict[str, Emotion] = {e.id: e for e in self.emotions}
        self.alias: Dict[str, str] = dict(raw.get("aliases", {}))
        for e in self.emotions:
            self.alias.setdefault(e.id, e.id)
            self.alias.setdefault(e.zh, e.id)

        ids = [e.id for e in self.emotions]
        if ids != EXPECTED_IDS:
            raise ValueError(f"emotions.json ids mismatch: {ids} != {EXPECTED_IDS}")
        if self.baseline not in self.by_id:
            raise ValueError(f"baseline {self.baseline!r} is not a defined emotion")
        unknown = sorted({v for v in self.alias.values() if v not in self.by_id})
        if unknown:
            raise ValueError(f"aliases point at unknown ids: {unknown}")

    # ---- normalization -------------------------------------------------
    def normalize(self, value: Optional[str]) -> Optional[str]:
        """Map any accepted spelling (id / Chinese / legacy alias) to a runtime id."""
        if value is None:
            return None
        key = value.strip()
        if not key:
            return None
        if key in self.alias:
            return self.alias[key]
        low = key.lower()
        if low in self.alias:
            return self.alias[low]
        for e in self.emotions:
            if e.en.lower() == low:
                return e.id
        return None

    def zh(self, emotion_id: str) -> str:
        return self.by_id[emotion_id].zh

    def tau_ids(self) -> List[str]:
        """Emotions that need a direction vector (everything except the baseline)."""
        return [e.id for e in self.emotions if e.id != self.baseline]

    # ---- prompt --------------------------------------------------------
    def build_prompt(self, transcript: Optional[str] = None) -> str:
        has_transcript = bool(transcript and transcript.strip())
        lines = [
            "你是语音情感标注员。你会收到一段很短的角色语音片段（日文动画语音，48kHz 单声道）。",
            "以声音证据为主：音高走势、音量、语速与节奏、发声紧张度、气息/气声、笑声、哭腔、",
            "抽泣、叹气、啧声、句尾收束方式。",
        ]
        if has_transcript:
            lines += [
                "同时提供台词语音识别文本（日文，可能有识别错误）。台词可用于判断说话人的意图和",
                "语境（例如疑问、撒娇、指责、自嘲），但**情绪强弱必须以声音为准**；台词写着",
                '"太好了"却用平淡语气说出，应判为平静而不是高兴。台词不可用时忽略它。',
            ]
        else:
            lines += ["本条没有可用台词，只依据声音判断。"]

        lines += [
            "",
            "从下面 19 个标签中选出唯一的主标签，必须使用英文 id：",
        ]
        for e in self.emotions:
            lines.append(f"- {e.id}（{e.zh}）: {e.definition} 边界：{e.boundary}")
        lines += [
            "",
            "要求：",
            "1. 只输出一个 JSON 对象，不要 Markdown、不要解释文字。",
            "2. 字段：",
            '   {"label": "<英文id>", "confidence": <0到1的小数>, '
            '"evidence": "<40字以内的主要依据>", "alternate": "<次选英文id或null>", '
            '"audible_speech": <true/false>}',
            "3. evidence 优先描述听到的声音；若台词影响了判断，可补充一句台词依据。",
            "4. confidence 是你的主观把握，不是校准概率；听不清就给低分，不要硬猜。",
            "5. 若片段没有可辨认的说话声（纯音效、纯呼吸、静音），audible_speech 设为 false，",
            "   仍给出最接近的标签并在 evidence 中说明。",
            "6. 哭泣必须有可听见的哭腔/哽咽/抽泣证据；厌恶必须有可听见的排斥/嫌恶证据。",
            "7. 台词与声音冲突时以声音为准，并在 evidence 中说明冲突。",
            "8. 不要输出多个主标签。",
        ]
        if has_transcript:
            lines += [
                "",
                "本条台词语音识别结果（日文，仅作辅助参考）：",
                transcript.strip()[:300],
            ]
        return "\n".join(lines)


_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def extract_json_object(text: str) -> dict:
    """Parse the first JSON object in a model reply, tolerating fences/preamble."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJ.search(text)
    if not m:
        raise ValueError("no JSON object found in reply")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("reply JSON is not an object")
    return obj


def validate_reply(obj: dict, schema: EmotionSchema) -> dict:
    """Validate and normalize a model reply. Raises ValueError on unusable output."""
    raw_label = obj.get("label")
    label = schema.normalize(str(raw_label)) if raw_label is not None else None
    if not label:
        raise ValueError(f"invalid label: {raw_label!r}")

    conf = obj.get("confidence")
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        raise ValueError(f"invalid confidence: {conf!r}")
    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"confidence out of range: {conf}")

    evidence = obj.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("missing evidence")
    evidence = evidence.strip()[:200]

    alt_raw = obj.get("alternate")
    alternate = None
    if alt_raw not in (None, "", "null"):
        alternate = schema.normalize(str(alt_raw))
        if alternate == label:
            alternate = None

    speech = obj.get("audible_speech", True)
    if not isinstance(speech, bool):
        if isinstance(speech, str) and speech.lower() in ("true", "false"):
            speech = speech.lower() == "true"
        else:
            speech = True

    return {
        "label": label,
        "label_zh": schema.zh(label),
        "confidence": conf,
        "evidence": evidence,
        "alternate": alternate,
        "audible_speech": speech,
    }


def iter_ids() -> Iterable[str]:
    return iter(EXPECTED_IDS)