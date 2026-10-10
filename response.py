"""Deterministic response composition and validation for the copilot."""

import re
from pathlib import Path

from safety import mask_sensitive_data


ROOT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = ROOT_DIR / "knowledge_base"
DATA_DIR = ROOT_DIR / "data"

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
YES_NO = {"Yes", "No"}


def _is_yes(value: object) -> bool:
    """Read a Yes/No flag or boolean without treating other text as true."""
    return value is True or (isinstance(value, str) and value.strip().casefold() == "yes")


def _heading_anchor(heading: str) -> str:
    """Create the same simple anchor used for headings in the policy index."""
    anchor = heading.casefold().replace("&", " and ").replace("`", "")
    anchor = re.sub(r"[^a-z0-9\s-]", "", anchor)
    return re.sub(r"[\s-]+", "-", anchor).strip("-")


def _valid_citation(citation: str) -> bool:
    """Check that a citation points to a real Markdown heading in this project."""
    if not isinstance(citation, str) or citation.count("#") != 1:
        return False
    filename, anchor = citation.split("#", 1)
    if Path(filename).name != filename or not filename.endswith(".md"):
        return False

    policy_path = KNOWLEDGE_BASE_DIR / filename
    if not policy_path.is_file():
        return False

    heading_pattern = re.compile(r"(?m)^#{1,6}[ \t]+(.+?)\s*#*\s*$")
    markdown = policy_path.read_text(encoding="utf-8")
    known_anchors = {
        _heading_anchor(match.group(1).strip())
        for match in heading_pattern.finditer(markdown)
    }
    return anchor in known_anchors


def _safe_text(value: object) -> str:
    """Normalize text and mask any recognizable personal/payment information."""
    if not isinstance(value, str):
        return ""
    normalized = " ".join(value.split())
    masked, _ = mask_sensitive_data(normalized)
    return masked


def _first_policy_sentence(passage: dict) -> str:
    """Take a short policy sentence without rewriting its wording."""
    text = passage.get("text", "")
    heading = passage.get("heading", "")
    if not isinstance(text, str):
        return ""

    lines = []
    for line in text.splitlines():
        cleaned = line.strip()
        cleaned = re.sub(r"^#{1,6}\s*", "", cleaned)
        cleaned = re.sub(r"^[-*+]\s+", "", cleaned)
        if not cleaned or cleaned == heading:
            continue
        if cleaned.casefold().startswith(("source id:", "applies to:")):
            continue
        lines.append(cleaned)

    if not lines:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", lines[0], maxsplit=1)[0]
    return _safe_text(sentence)


def _validate_inputs(triage: dict, policy: dict, data: dict, routing: dict, safety: dict) -> tuple[float, str, str]:
    """Check required incoming fields and return normalized confidence/labels."""
    if not all(isinstance(value, dict) for value in (triage, policy, data, routing, safety)):
        raise ValueError("triage, policy, data, routing, and safety must each be dictionaries.")

    category = triage.get("category")
    priority = triage.get("priority")
    if category not in CATEGORIES:
        raise ValueError("triage.category must use one of the supplied category labels.")
    if priority not in PRIORITIES:
        raise ValueError("triage.priority must be P1, P2, P3, or P4.")

    confidence = triage.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("triage.confidence must be a number from 0 to 1.")

    if not isinstance(policy.get("passages", []), list):
        raise ValueError("policy.passages must be a list.")
    if not isinstance(data.get("facts", []), list) or not isinstance(data.get("data_used", []), list):
        raise ValueError("data.facts and data.data_used must be lists.")

    if safety.get("pii_masked", "No") not in YES_NO:
        raise ValueError("safety.pii_masked must be Yes or No.")
    if safety.get("safety_refusal", "No") not in YES_NO:
        raise ValueError("safety.safety_refusal must be Yes or No.")
    if routing.get("escalated", "No") not in YES_NO:
        raise ValueError("routing.escalated must be Yes or No.")
    if routing.get("human_approval_required", "No") not in YES_NO:
        raise ValueError("routing.human_approval_required must be Yes or No.")

    return float(confidence), category, priority


def build_response(
    triage: dict,
    policy: dict,
    data: dict,
    routing: dict,
    safety: dict,
) -> dict:
    """Build a grounded answer and prediction-schema fields without generation.

    Only validated category/priority labels, stored policy citations, and
    locally calculated data results are used. The function recommends actions
    but never claims a refund, inventory update, or vendor action was completed.
    """
    confidence, category, priority = _validate_inputs(triage, policy, data, routing, safety)
    routed_category = routing.get("category", category)
    routed_priority = routing.get("priority", priority)
    if routed_category not in CATEGORIES or routed_priority not in PRIORITIES:
        raise ValueError("workflow category and priority must use the supplied labels.")
    category = routed_category
    priority = routed_priority

    citations = []
    policy_sentences = []
    rejected_citation = False
    for passage in policy.get("passages", []):
        if not isinstance(passage, dict):
            rejected_citation = True
            continue
        citation = passage.get("citation", "")
        if not _valid_citation(citation):
            rejected_citation = True
            continue
        if citation not in citations:
            citations.append(citation)
        sentence = _first_policy_sentence(passage)
        if sentence and sentence not in policy_sentences:
            policy_sentences.append(sentence)

    safe_data_facts = [_safe_text(fact) for fact in data.get("facts", []) if isinstance(fact, str)]
    safe_data_facts = [fact for fact in safe_data_facts if fact]
    data_answer = _safe_text(data.get("answer", ""))
    data_notes = _safe_text(data.get("notes", ""))

    safety_refusal = _is_yes(safety.get("safety_refusal"))
    pii_masked = _is_yes(safety.get("pii_masked"))
    security_escalation = _is_yes(safety.get("security_escalation"))
    low_confidence = confidence < 0.60
    route_found = routing.get("route_found", True) is not False
    needs_human_review = (
        _is_yes(routing.get("needs_human_review"))
        or _is_yes(routing.get("human_approval_required"))
        or security_escalation
        or safety_refusal
        or priority == "P1"
        or low_confidence
        or not route_found
    )
    escalated = (
        "Yes"
        if _is_yes(routing.get("escalated")) or needs_human_review
        else "No"
    )
    human_approval_required = (
        "Yes"
        if (
            _is_yes(routing.get("human_approval_required"))
            or safety_refusal
            or security_escalation
            or priority == "P1"
            or low_confidence
        )
        else "No"
    )

    answer_parts = []
    if safety_refusal or security_escalation:
        refusal = _safe_text(safety.get("refusal_message", ""))
        answer_parts.append(
            refusal
            or "I can’t provide restricted information. Use an approved, access-controlled process."
        )
    else:
        answer_parts.append(f"Category: {category}. Priority: {priority}.")

        # Facts are copied from local calculations; no new numbers are created here.
        if data_answer:
            answer_parts.append("Data result: " + data_answer)
        if safe_data_facts:
            answer_parts.append("Confirmed facts: " + "; ".join(safe_data_facts[:5]) + ".")

        for sentence in policy_sentences[:2]:
            answer_parts.append("Policy guidance: " + sentence)

        if not policy_sentences and not safe_data_facts and not data_answer:
            answer_parts.append("I could not confirm an answer from the supplied policy or data files.")

        if data_notes:
            # The note is explicitly labeled as unconfirmed when it suggests a cause.
            possible_cause = any(
                phrase in data_notes.casefold()
                for phrase in ("possible", "likely", "cause", "should be reviewed", "signal for review")
            )
            prefix = "Possible cause or next check (not confirmed): " if possible_cause else "Data note: "
            answer_parts.append(prefix + data_notes)

    if not route_found:
        answer_parts.append("Routing information is unavailable; a human must review this case.")
    else:
        team = _safe_text(routing.get("escalation_team", ""))
        channel = _safe_text(routing.get("channel", ""))
        sla_minutes = routing.get("sla_minutes")
        if escalated == "Yes" and team:
            route_text = f"Escalation route: {team}"
            if channel:
                route_text += f" via {channel}"
            if isinstance(sla_minutes, int) and not isinstance(sla_minutes, bool):
                route_text += f" (SLA {sla_minutes} minutes)"
            answer_parts.append(route_text + ".")

    if human_approval_required == "Yes":
        answer_parts.append("Human approval is required before any action is taken.")
    answer_parts.append("Recommendations only; no refund, inventory change, or vendor action has been executed.")

    answer = _safe_text(" ".join(answer_parts))
    if re.search(r"\*+|\[REDACTED(?:_CARD)?\]", answer):
        pii_masked = True

    known_data_files = {path.name for path in DATA_DIR.glob("*.csv")}
    data_used = []
    for filename in ([] if safety_refusal or security_escalation else data.get("data_used", [])):
        if isinstance(filename, str) and Path(filename).name == filename and filename in known_data_files:
            if filename not in data_used:
                data_used.append(filename)

    notes_parts = []
    route_reason = _safe_text(routing.get("reason", ""))
    if route_reason:
        notes_parts.append(route_reason)
    if low_confidence:
        notes_parts.append("Model confidence is below 0.60 and is an estimate, not a calibrated probability.")
    if rejected_citation:
        notes_parts.append("One or more unsupported citations were omitted.")
    if data_notes and data_notes not in " ".join(notes_parts):
        notes_parts.append(data_notes)
    notes_text = _safe_text(" ".join(notes_parts))
    if re.search(r"\*+|\[REDACTED(?:_CARD)?\]", answer + " " + notes_text):
        pii_masked = True

    response = {
        "category": category,
        "priority": priority,
        "confidence": confidence,
        "answer": answer,
        "citations": ";".join(citations),
        "data_used": ";".join(data_used),
        "escalated": escalated,
        "escalation_team": _safe_text(routing.get("escalation_team", "")) if escalated == "Yes" else "",
        "pii_masked": "Yes" if pii_masked else "No",
        "safety_refusal": "Yes" if safety_refusal or security_escalation else "No",
        "notes": notes_text,
    }

    # The final record uses exactly the evaluation schema's response fields.
    required_fields = {
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
    }
    if set(response) != required_fields:
        raise ValueError("The response does not match the supplied prediction schema.")
    return response
