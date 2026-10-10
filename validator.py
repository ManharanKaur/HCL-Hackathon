"""Output Validator module for retail operations copilot.

Corresponds to the diagram node: 'Output Validator'.
Validates output records from both Response Composer and Safe Refusal
before they are displayed in the UI or written to evaluation / predictions.csv.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from labels_and_citations import CATEGORIES, PRIORITIES, citations_for_markdown
from reference_fields import parse_reference_field


ROOT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = ROOT_DIR / "knowledge_base"
DATA_DIR = ROOT_DIR / "data"

PREDICTION_SCHEMA_COLUMNS = [
    "test_id",
    "category",
    "priority",
    "confidence",
    "answer",
    "citations",
    "data_used",
    "escalated",
    "escalation_team",
    "pii_masked",
    "safety_refusal",
    "notes",
]


class OutputValidator:
    """Validates and enforces schema compliance for copilot responses."""

    def __init__(self):
        self._allowed_citations = self._load_real_policy_citations()
        self._allowed_data_files = {p.name for p in DATA_DIR.glob("*.csv")}

    def _load_real_policy_citations(self) -> set[str]:
        """Read headings on disk to determine legitimate citations."""
        citations = set()
        for md_path in KNOWLEDGE_BASE_DIR.glob("*.md"):
            try:
                content = md_path.read_text(encoding="utf-8")
                citations.update(citations_for_markdown(md_path.name, content))
            except Exception:
                continue
        return citations

    def validate(self, record: dict[str, Any], test_id: str = "") -> dict[str, Any]:
        """Validate, filter, and normalize fields to guarantee schema compliance."""
        if not isinstance(record, dict):
            record = {}

        # 1. test_id
        tid = record.get("test_id") or test_id or ""
        valid_tid = str(tid).strip()

        # 2. category
        cat = record.get("category", "")
        valid_cat = cat if cat in CATEGORIES else "Other or Security"

        # 3. priority
        prio = record.get("priority", "")
        valid_prio = prio if prio in PRIORITIES else "P4"

        # 4. confidence
        conf = record.get("confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(conf):
            valid_conf = 0.0
        else:
            valid_conf = float(max(0.0, min(1.0, conf)))

        # 5. answer
        ans = record.get("answer", "")
        valid_ans = str(ans).strip() if ans else "No answer provided."

        # 6. citations (filter out hallucinated/unmatched citations)
        raw_citations = parse_reference_field(record.get("citations", ""))
        valid_citations_list = [c for c in raw_citations if c in self._allowed_citations]
        valid_citations = ";".join(dict.fromkeys(valid_citations_list))

        # 7. data_used (filter out non-existent data files)
        raw_data = parse_reference_field(record.get("data_used", ""))
        valid_data_list = [d for d in raw_data if d in self._allowed_data_files]
        valid_data = ";".join(dict.fromkeys(valid_data_list))

        # 8. Yes/No flags
        def _clean_yes_no(val: Any, default: str = "No") -> str:
            if val is True or (isinstance(val, str) and val.strip().casefold() == "yes"):
                return "Yes"
            if val is False or (isinstance(val, str) and val.strip().casefold() == "no"):
                return "No"
            return default

        valid_escalated = _clean_yes_no(record.get("escalated"), default="No")
        valid_pii_masked = _clean_yes_no(record.get("pii_masked"), default="No")
        valid_refusal = _clean_yes_no(record.get("safety_refusal"), default="No")

        # 9. escalation_team
        team = record.get("escalation_team", "")
        valid_team = str(team).strip() if valid_escalated == "Yes" else ""

        # 10. notes
        notes = record.get("notes", "")
        valid_notes = str(notes).strip()

        validated_record = {
            "test_id": valid_tid,
            "category": valid_cat,
            "priority": valid_prio,
            "confidence": valid_conf,
            "answer": valid_ans,
            "citations": valid_citations,
            "data_used": valid_data,
            "escalated": valid_escalated,
            "escalation_team": valid_team,
            "pii_masked": valid_pii_masked,
            "safety_refusal": valid_refusal,
            "notes": valid_notes,
        }
        return validated_record


def validate_response(record: dict[str, Any], test_id: str = "") -> dict[str, Any]:
    """Module-level entry point delegating to OutputValidator."""
    return OutputValidator().validate(record, test_id=test_id)
