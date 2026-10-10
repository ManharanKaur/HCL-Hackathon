"""LLM-assisted classification with deterministic high-risk overrides."""

import json
import math
import re

from llm_client import chat_completion
from safety import check_request


DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"
LOW_CONFIDENCE_THRESHOLD = 0.60

CATEGORIES = {
    "POS and Payments",
    "Inventory and Replenishment",
    "Pricing and Promotions",
    "Returns and Refunds",
    "Vendor and Supply Chain",
    "Product Quality and Safety",
    "Customer Service",
    "Other or Security",
}
PRIORITIES = {"P1", "P2", "P3", "P4"}

# These rules can raise priority after the model responds. A model answer must
# never lower an immediate safety or store-wide payment outage to a routine case.
IMMEDIATE_SAFETY_PATTERN = re.compile(
    r"\b(injur(?:y|ed)|recall|contamination|unsafe(?:\s+\w+){0,2}|"
    r"customer safety|safety incident|fire|smoke|electrocution)\b|"
    r"\b(demo|fitness|treadmill|exercise bike)\b.{0,40}\b(malfunction|unstable)\b",
    re.IGNORECASE,
)
BROAD_REGISTER_OUTAGE_PATTERN = re.compile(
    r"\b(all|every|multiple|several)\s+(?:of\s+the\s+)?registers?\b|"
    r"\b(store[- ]wide|entire store)\b.{0,50}\b(pos|payment|card|register)\b|"
    r"\b(pos|payment|card)\b.{0,50}\b(store[- ]wide|entire store|all registers)\b",
    re.IGNORECASE,
)


def _fallback_result(reason: str) -> dict:
    """Build a valid low-confidence result when the model cannot be used."""
    return {
        "category": "Other or Security",
        "priority": "P4",
        "confidence": 0.0,
        "rationale": reason,
        "needs_human_review": True,
        "review_reason": reason,
    }


def _parse_model_response(content: str) -> dict:
    """Parse and strictly validate the four fields requested from the model."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The model returned an empty response.")

    try:
        result = json.loads(content.strip())
    except json.JSONDecodeError as error:
        raise ValueError("The model response was not valid JSON.") from error

    expected_fields = {"category", "priority", "confidence", "reason"}
    if not isinstance(result, dict) or set(result) != expected_fields:
        raise ValueError("The JSON response did not contain exactly the required fields.")

    if result["category"] not in CATEGORIES:
        raise ValueError("The response used an unsupported category.")
    if result["priority"] not in PRIORITIES:
        raise ValueError("The response used an unsupported priority.")

    confidence = result["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("Confidence must be a number from 0 to 1.")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Confidence must be a finite number from 0 to 1.")

    if not isinstance(result["reason"], str) or not result["reason"].strip():
        raise ValueError("The response needs a short reason.")

    # Normalize confidence to a regular float for callers and CSV output.
    result["confidence"] = float(confidence)
    result["reason"] = result["reason"].strip()
    return result


def _apply_priority_overrides(question: str, result: dict) -> dict:
    """Apply non-downgradable safety and broad-outage rules after model output."""
    if IMMEDIATE_SAFETY_PATTERN.search(question):
        result["category"] = "Product Quality and Safety"
        result["priority"] = "P1"
        result["rationale"] = (
            "Immediate safety risk detected; human escalation is required. "
            + result["rationale"]
        )
        result["needs_human_review"] = True
        result["review_reason"] = "Immediate safety risk requires human review."
    elif BROAD_REGISTER_OUTAGE_PATTERN.search(question):
        result["category"] = "POS and Payments"
        result["priority"] = "P1"
        result["rationale"] = (
            "A broad register or payment outage requires immediate escalation. "
            + result["rationale"]
        )
        result["needs_human_review"] = True
        result["review_reason"] = "Store-wide or multiple-register outage requires review."

    return result


def classify_request(question: str) -> dict:
    """Return category, priority, confidence, and rationale for one question.

    Safety screening and masking happen before the LLM call. Confidence is the
    model's estimate, not a calibrated probability; scores below 0.60 need a
    person to review the result.
    """
    # Do not send unsafe or unmasked input to the model. `check_request` returns
    # generic text for unsafe requests and masks recognizable PII in safe text.
    safety_result = check_request(question or "")
    safe_question = safety_result["cleaned_text"]

    if not safety_result["allowed"]:
        result = {
            "category": "Other or Security",
            "priority": "P1",
            "confidence": 1.0,
            "rationale": "Restricted-data or safety-bypass request refused and sent for security review.",
            "needs_human_review": True,
            "review_reason": "Security escalation is required for this request.",
        }
        return result

    messages = [
        {
            "role": "system",
            "content": (
                "Classify the employee's retail operations question. Return only one valid JSON object "
                "with exactly these fields: category, priority, confidence, reason. "
                "Use category exactly one of: POS and Payments; Inventory and Replenishment; "
                "Pricing and Promotions; Returns and Refunds; Vendor and Supply Chain; "
                "Product Quality and Safety; Customer Service; Other or Security. "
                "Use priority exactly one of P1, P2, P3, P4. Confidence must be a number from 0 to 1. "
                "P1 is immediate human escalation, P2 is high impact, P3 is routine, and P4 is a general inquiry. "
                "The reason must be a short string. Do not add markdown or any text outside the JSON object."
            ),
        },
        {"role": "user", "content": safe_question},
    ]

    try:
        # Do not pass response_format: the selected free model does not support it.
        model_response = chat_completion(
            messages,
            model=DEFAULT_MODEL,
            temperature=0.0,
        )
        parsed = _parse_model_response(model_response.get("content", ""))
    except Exception:
        # Keep provider details and any input text out of error messages.
        result = _fallback_result(
            "Classification could not be verified because the model call or response failed."
        )
        return _apply_priority_overrides(safe_question, result)

    result = {
        "category": parsed["category"],
        "priority": parsed["priority"],
        "confidence": parsed["confidence"],
        "rationale": parsed["reason"],
        "needs_human_review": False,
        "review_reason": "",
    }

    # A low estimate signals uncertainty, even when the JSON is valid.
    if result["confidence"] < LOW_CONFIDENCE_THRESHOLD:
        result["needs_human_review"] = True
        result["review_reason"] = "Model confidence is below 0.60."

    return _apply_priority_overrides(safe_question, result)
