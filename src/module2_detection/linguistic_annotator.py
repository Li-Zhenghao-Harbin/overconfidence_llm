"""Module 2 — Rule-based assertiveness annotation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

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
]

MODERATE_ASSERTIVENESS_PATTERNS = [
    r"\bshould work\b",
    r"\bhandles? most\b",
    r"\bgenerally correct\b",
    r"\byou (may|might) want to\b",
    r"\bfor most inputs?\b",
]

TENTATIVE_PATTERNS = [
    r"\bmight not handle\b",
    r"\bi'?m not (sure|confident)\b",
    r"\bconsider testing\b",
    r"\bthere could be issues?\b",
    r"\bnot confident\b",
    r"\bpotential issue\b",
    r"\bworth (testing|verifying|checking)\b",
]


class LinguisticAnnotator:
    def __init__(self, config: dict):
        self.config = config
        self.kappa_threshold = config["annotation"]["kappa_threshold"]
        self.rubric_levels = config["annotation"]["rubric_levels"]

    def annotate_batch(self, results: list[dict]) -> list[AnnotationRecord]:
        out: list[AnnotationRecord] = []
        for r in results:
            level, note = self.auto_annotate(r.get("explanation", ""))
            out.append(
                AnnotationRecord(
                    sample_id=r["sample_id"],
                    task_id=r["task_id"],
                    model=r["model"],
                    condition=r.get("condition", "C0"),
                    explanation=r.get("explanation", ""),
                    assertiveness_level=level,
                    annotator_id="auto_regex",
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
        return 2, "default_moderate"
