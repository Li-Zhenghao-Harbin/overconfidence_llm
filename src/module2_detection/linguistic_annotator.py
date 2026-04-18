"""Module 2 — Rule-based assertiveness annotation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from src.module2_detection.assertiveness_dl import AssertivenessPredictor

logger = logging.getLogger(__name__)


@dataclass
class AnnotationRecord:
    sample_id: str
    task_id: str
    model: str
    condition: str
    explanation: str
    assertiveness_level: int
    annotator_id: str
    annotation_note: str = ""
    round_number: int = 0


STRONG_ASSERTIVENESS_PATTERNS = [
    r"\bwill correctly\b",
    r"\bperfectly handles?\b",
    r"\bguaranteed to\b",
    r"\bthis is the correct solution\b",
    r"\bhandles? all (edge )?cases\b",
    r"\bno (issues?|problems?|errors?)\b",
    r"\bcompletely (correct|accurate|handles?)\b",
    r"一定(正确|没问题)",
    r"(完全|绝对)(正确|没问题)",
    r"(保证|确保).*(正确|通过|可用)",
    r"(覆盖|处理).*(所有|全部).*(情况|边界)",
]

MODERATE_ASSERTIVENESS_PATTERNS = [
    r"\bshould work\b",
    r"\bhandles? most\b",
    r"\bgenerally correct\b",
    r"\byou (may|might) want to\b",
    r"\bfor most inputs?\b",
    r"(应该|大概率|通常)可以",
    r"(大部分|多数).*(情况|输入).*(正确|可用)",
    r"(建议|可以)进一步(测试|验证|检查)",
]

TENTATIVE_PATTERNS = [
    r"\bmight not handle\b",
    r"\bi'?m not (sure|confident)\b",
    r"\bconsider testing\b",
    r"\bthere could be issues?\b",
    r"\bnot confident\b",
    r"\bpotential issue\b",
    r"\bworth (testing|verifying|checking)\b",
    r"(不确定|没把握|无法保证)",
    r"(可能|也许|或许).*(有问题|失败|不通过)",
    r"(建议|最好)再(测试|验证|检查)",
]


class LinguisticAnnotator:
    def __init__(self, config: dict):
        self.config = config
        self.kappa_threshold = config["annotation"]["kappa_threshold"]
        self.rubric_levels = config["annotation"]["rubric_levels"]
        self.default_level = int((config.get("annotation") or {}).get("default_assertiveness_level", 2))
        self._dl = AssertivenessPredictor(config)

    def annotate_batch(self, results: list[dict]) -> list[AnnotationRecord]:
        out: list[AnnotationRecord] = []
        for r in results:
            level, note = self.auto_annotate(r.get("explanation", ""))
            annotator_id = "dl_cnn" if note == "dl_cnn" else "auto_regex"
            out.append(
                AnnotationRecord(
                    sample_id=r["sample_id"],
                    task_id=r["task_id"],
                    model=r["model"],
                    condition=r.get("condition", "C0"),
                    explanation=r.get("explanation", ""),
                    assertiveness_level=level,
                    annotator_id=annotator_id,
                    annotation_note=note,
                )
            )
        ann_dir = (self.config.get("outputs") or {}).get("annotations_dir", "data/annotations")
        path = Path(ann_dir) / "auto_annotations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for a in out:
                f.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")
        logger.info("Wrote %d auto annotations to %s", len(out), path)
        return out

    def auto_annotate(self, explanation: str) -> tuple[int, str]:
        if self._dl.available:
            try:
                level = self._dl.predict_level(explanation or "")
                return level, "dl_cnn"
            except Exception as e:  # noqa: BLE001
                logger.warning("assertiveness DL inference failed; fallback to regex: %s", e)

        text = explanation.lower()
        matched: list[str] = []

        def any_pat(pats: list[str]) -> bool:
            found = False
            for p in pats:
                if re.search(p, text, re.I):
                    matched.append(p)
                    found = True
            return found

        has_tent = any_pat(TENTATIVE_PATTERNS)
        has_strong = any_pat(STRONG_ASSERTIVENESS_PATTERNS)
        has_mod = any_pat(MODERATE_ASSERTIVENESS_PATTERNS)

        if has_tent:
            return 1, "tentative:" + ",".join(matched[:5])
        if has_strong:
            return 3, "strong:" + ",".join(matched[:5])
        if has_mod:
            return 2, "moderate:" + ",".join(matched[:5])
        lvl = 2 if self.default_level not in (1, 2, 3) else self.default_level
        return lvl, f"default_level_{lvl}"
