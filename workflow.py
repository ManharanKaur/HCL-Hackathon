"""Deterministic escalation and review routing using the supplied matrix."""

import re
from pathlib import Path

import pandas as pd


MATRIX_PATH = Path(__file__).resolve().parent / "data" / "escalation_matrix.csv"
PRIORITY_RANK = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

URGENT_SAFETY_PATTERN = re.compile(
    r"\b(injur(?:y|ed)|recall|contamination|unsafe|electrocution|fire|smoke|"
    r"malfunction(?:ed|ing)?\s+(?:demo|fitness|treadmill|exercise))\b",
    re.IGNORECASE,
)
BROAD_PAYMENT_PATTERN = re.compile(
    r"\b(all|every|multiple|several)\s+(?:of\s+the\s+)?registers?\b|"
    r"\b(store[- ]wide|entire store)\b.{0,60}\b(pos|payment|card|register)\b|"
    r"\b(pos|payment|card)\b.{0,60}\b(store[- ]wide|entire store|all registers)\b",
    re.IGNORECASE,
)
BROAD_PRICING_PATTERN = re.compile(
    r"\b(department|multiple customers|many customers|many skus|several skus|store[- ]wide)\b",
    re.IGNORECASE,
)


def _is_yes(value: object) -> bool:
    """Accept the project's Yes/No strings and ordinary booleans."""
    return value is True or (isinstance(value, str) and value.strip().casefold() == "yes")


def _best_matrix_route(category: str, priority: str, matrix: pd.DataFrame) -> pd.Series | None:
    """Choose a matrix row with the exact category and priority."""
    candidates = matrix[
        matrix["category"].eq(category) & matrix["priority"].eq(priority)
    ]
    if candidates.empty:
        return None
    # If several rows share the pair, prefer the route requiring approval because
    # it is safer when the input does not identify a more specific trigger.
    if candidates["human_approval_required"].eq("Yes").any():
        return candidates[candidates["human_approval_required"].eq("Yes")].iloc[0]
    return candidates.iloc[0]


def route_case(triage: dict, safety: dict, question: str = "") -> dict:
    """Select an escalation route and human-review flags from fixed rules.

    The optional question supports deterministic impact checks. It should be
    the safety-screened text when available. This function only recommends a
    route; it never submits a ticket or performs a store action.
    """
    if not MATRIX_PATH.is_file():
        return {
            "escalated": "Yes",
            "escalation_team": "",
            "channel": "",
            "sla_minutes": None,
            "human_approval_required": "Yes",
            "needs_human_review": True,
            "route_found": False,
            "reason": "Escalation matrix is unavailable; routing information is unavailable and a human must review.",
            "category": triage.get("category", "Other or Security"),
            "priority": triage.get("priority", "P4"),
        }

    matrix = pd.read_csv(MATRIX_PATH, dtype=str, keep_default_na=False)
    category = triage.get("category", "Other or Security")
    priority = triage.get("priority", "P4")
    confidence = pd.to_numeric(triage.get("confidence"), errors="coerce")
    request_text = (safety.get("cleaned_text") or question or "").casefold()

    safety_refusal = _is_yes(safety.get("safety_refusal"))
    security_escalation = _is_yes(safety.get("security_escalation"))
    approval_required = False
    impact_reason = ""

    # Sensitive-data requests always take the Security route, even when model
    # triage chose a different category.
    if safety_refusal or security_escalation:
        category = "Other or Security"
        priority = "P1"
        approval_required = True
        impact_reason = "Safety screening requires a Security review."

    # Raise payment outages and immediate safety risks above the model's choice.
    elif URGENT_SAFETY_PATTERN.search(request_text):
        category = "Product Quality and Safety"
        priority = "P1"
        approval_required = True
        impact_reason = "A safety risk requires immediate human review."
    elif BROAD_PAYMENT_PATTERN.search(request_text):
        category = "POS and Payments"
        priority = "P1"
        approval_required = True
        impact_reason = "Broad register or payment impact requires immediate human escalation."
    elif category == "Pricing and Promotions" and BROAD_PRICING_PATTERN.search(request_text):
        priority = "P2" if PRIORITY_RANK.get(priority, 4) > PRIORITY_RANK["P2"] else priority
        approval_required = True
        impact_reason = "A broad pricing mismatch requires manager and pricing-team review."

    # Policy flags approval-required actions, including returns, vendor claims,
    # product quarantine, and all matrix routes marked for human approval.
    if category == "Returns and Refunds" and re.search(
        r"no[- ]receipt|without receipt|refund.{0,30}(?:inr|rs\.?|₹)?\s*\d{4,}|(?:inr|rs\.?|₹)\s*\d{4,}",
        request_text,
    ):
        approval_required = True
        if PRIORITY_RANK.get(priority, 4) > PRIORITY_RANK["P2"]:
            priority = "P2"
        impact_reason = impact_reason or "Return policy requires manager approval for this request."

    matrix_route = _best_matrix_route(category, priority, matrix)
    if matrix_route is None:
        return {
            "escalated": "Yes",
            "escalation_team": "",
            "channel": "",
            "sla_minutes": None,
            "human_approval_required": "Yes",
            "needs_human_review": True,
            "route_found": False,
            "reason": (
                f"No escalation-matrix route matches {category} {priority}; "
                "routing information is unavailable and a human must review."
            ),
            "category": category,
            "priority": priority,
        }

    route_requires_approval = _is_yes(matrix_route["human_approval_required"])
    low_confidence = pd.notna(confidence) and confidence < 0.60
    triage_review = _is_yes(triage.get("needs_human_review"))
    priority_one = priority == "P1"
    human_review = (
        safety_refusal
        or security_escalation
        or approval_required
        or route_requires_approval
        or low_confidence
        or triage_review
        or priority_one
    )

    reason_parts = []
    if impact_reason:
        reason_parts.append(impact_reason)
    elif low_confidence:
        reason_parts.append("Model confidence is below 0.60; a human should verify the classification.")
    elif route_requires_approval:
        reason_parts.append("The escalation matrix requires human approval for this route.")
    elif priority_one:
        reason_parts.append("P1 requires immediate human escalation.")
    else:
        reason_parts.append(f"Route selected for: {matrix_route['trigger']}.")

    return {
        # A routine matrix destination is a suggested queue, not automatically
        # an escalation. Escalate only when the deterministic review rules fire.
        "escalated": "Yes" if human_review else "No",
        "escalation_team": matrix_route["team"],
        "channel": matrix_route["channel"],
        "sla_minutes": int(matrix_route["sla_minutes"]),
        "human_approval_required": "Yes" if route_requires_approval or approval_required else "No",
        "needs_human_review": bool(human_review),
        "route_found": True,
        "reason": " ".join(reason_parts),
        "category": category,
        "priority": priority,
    }
