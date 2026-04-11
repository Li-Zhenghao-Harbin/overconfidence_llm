"""
Module 2 — Linguistic Annotator
Annotates LLM-generated natural language explanations with a 3-level assertiveness rubric.

Rubric:
  3 — Strongly Assertive:    "will correctly", "perfectly handles", "guaranteed to"
  2 — Moderately Assertive:  "should work", "handles most cases", minor hedging
  1 — Tentative/Calibrated:  "might not handle", "consider testing", explicit uncertainty

Inter-annotator agreement measured by Cohen's Kappa (target κ ≥ 0.7).

Output format (one record per annotated sample):
{
    "sample_id": "...",
    "task_id": "...",
    "model": "gpt-4o",
    "condition": "C0",
    "explanation": "This code correctly handles all edge cases.",
    "assertiveness_level": 3,
    "annotator_id": "A1",
    "annotation_note": "Uses 'correctly' + 'all edge cases'"
}
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AnnotationRecord:
    sample_id: str
    task_id: str
    model: str
    condition: str
    explanation: str
    assertiveness_level: int          # 1 | 2 | 3
    annotator_id: str
    annotation_note: str = ""
    round_number: int = 0             # 0 = initial, 1+ = refinement rounds


# ─── Linguistic Indicator Lists ───────────────────────────────────────────────

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
    """
    Two-mode annotator:
      1. Rule-based auto-annotation using regex pattern matching (fast, scalable).
      2. Human annotation interface (CLI prompt or file import for ground truth).

    In the experiment, auto-annotation is used for initial screening; two human
    annotators independently label a stratified subsample and inter-annotator
    agreement is computed to validate the rubric.
    """

    def __init__(self, config: dict):
        self.config = config
        self.kappa_threshold = config["annotation"]["kappa_threshold"]
        self.rubric_levels = config["annotation"]["rubric_levels"]

    def annotate_batch(self, results: list[dict]) -> list[AnnotationRecord]:
        """
        Auto-annotate a batch of LLM output records.

        Args:
            results: List of dicts, each containing at least:
                     sample_id, task_id, model, condition, explanation

        Returns:
            List of AnnotationRecord with assertiveness_level filled by rule-based system.
        """
        # TODO: Iterate results, call auto_annotate() per record
        # TODO: Log distribution across levels 1/2/3
        # TODO: Persist to data/annotations/auto_annotations.jsonl
        raise NotImplementedError

    def auto_annotate(self, explanation: str) -> tuple[int, str]:
        """
        Classify a single explanation string using regex-based pattern matching.

        Algorithm:
          1. Check STRONG patterns → level 3
          2. Check TENTATIVE patterns → level 1 (takes precedence over moderate if both match)
          3. Check MODERATE patterns → level 2
          4. Default → level 2 (moderately assertive) if no pattern matches

        Returns:
            (assertiveness_level, note_string_describing_matched_patterns)
        """
        # TODO: Compile and apply STRONG, TENTATIVE, MODERATE pattern lists
        # TODO: Handle mixed signals (e.g., strong + tentative patterns co-present)
        # TODO: Return level and note
        raise NotImplementedError

    def load_human_annotations(self, filepath: Path) -> list[AnnotationRecord]:
        """
        Load human annotation records from a CSV/JSONL file for agreement analysis.

        Expected CSV columns: sample_id, annotator_id, assertiveness_level, note
        """
        # TODO: Parse file, map to AnnotationRecord objects
        raise NotImplementedError

    def compute_cohen_kappa(
        self,
        annotations_a: list[AnnotationRecord],
        annotations_b: list[AnnotationRecord],
    ) -> float:
        """
        Compute Cohen's Kappa between two annotators on the same sample set.

        Steps:
          1. Align records by sample_id
          2. Build observed agreement and expected agreement matrices
          3. κ = (p_o - p_e) / (1 - p_e)

        Raises:
            ValueError if sample sets do not match.
        """
        # TODO: Align by sample_id
        # TODO: Compute κ using sklearn.metrics.cohen_kappa_score or manual calculation
        raise NotImplementedError

    def resolve_disagreements(
        self,
        annotations_a: list[AnnotationRecord],
        annotations_b: list[AnnotationRecord],
        strategy: str = "discussion",
    ) -> list[AnnotationRecord]:
        """
        Merge two annotation sets, resolving disagreements.

        Strategies:
          - "average":    Floor-average of the two levels (for adjacent disagreements).
          - "discussion": Flag disagreements (|level_a - level_b| > 1) for manual review.

        Returns merged list of AnnotationRecord (one per sample_id).
        """
        # TODO: Align, detect disagreements, apply strategy
        raise NotImplementedError

    def export_annotation_template(self, results: list[dict], output_path: Path) -> None:
        """
        Export a blank CSV template for human annotators to fill in assertiveness levels.

        Columns: sample_id, task_id, model, condition, explanation,
                 assertiveness_level (blank), annotation_note (blank)
        """
        # TODO: Write CSV with explanation pre-filled, level column blank
        raise NotImplementedError
