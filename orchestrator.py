"""Connect the copilot modules while keeping safety as the first gate."""

from typing import Any

from labels_and_citations import CATEGORIES, PRIORITIES


def _empty_evidence_note(message: str) -> dict:
    """Return evidence in the same shape as the retriever and data agent."""
    return {"passages": [], "answer": "", "facts": [], "data_used": [], "notes": message}


def _needs_business_data(question: str) -> bool:
    """Use CSVs only for questions that appear to ask for business facts."""
    data_terms = (
        "how many", "count", "quantity", "stock", "inventory", "on hand",
        "available", "stock-out", "stockout", "receipt", "shipment", "vendor",
        "supplier", "support case", "recent case", "last 24 hours", "complaint",
        "quality issue", "sku", "product", "store",
    )
    lowered_question = question.casefold()
    return any(term in lowered_question for term in data_terms)


def _safe_fallback_record(test_id: str, message: str, safety: dict | None = None) -> dict:
    """Build the required prediction fields if a later module cannot run."""
    safety = safety if isinstance(safety, dict) else {}
    refused = safety.get("safety_refusal") == "Yes" or safety.get("security_escalation") == "Yes"
    masked = safety.get("pii_masked") == "Yes"
    return {
        "test_id": test_id,
        "category": "Other or Security",
        "priority": "P1" if refused else "P4",
        "confidence": 0.0,
        "answer": (
            safety.get("refusal_message")
            if refused and isinstance(safety.get("refusal_message"), str)
            else message
        ),
        "citations": "",
        "data_used": "",
        "escalated": "Yes" if refused or message else "No",
        "escalation_team": "Security" if refused else "",
        "pii_masked": "Yes" if masked else "No",
        "safety_refusal": "Yes" if refused else "No",
        "notes": "Request was not processed further." if refused else message,
    }


def _valid_triage(result: Any) -> bool:
    """Check that triage returned labels and a usable confidence score."""
    if not isinstance(result, dict):
        return False
    confidence = result.get("confidence")
    return (
        result.get("category") in CATEGORIES
        and result.get("priority") in PRIORITIES
        and isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and 0 <= confidence <= 1
    )


def run_copilot(question: str, test_id: str = "") -> dict:
    """Run safety, triage, evidence, routing, and response in that order.

    The orchestrator screens the raw question once; every later module receives
    only the cleaned text and existing safety result. Errors are converted into
    short, safe fallbacks; exception details and question text are never returned.
    """
    safe_test_id = test_id if isinstance(test_id, str) else ""

    # Safety is deliberately imported and called before any LLM-backed module.
    try:
        from safety import check_request

        safety = check_request(question if isinstance(question, str) else "")
    except Exception:
        return _safe_fallback_record(
            safe_test_id,
            "Safety screening is unavailable, so this request was not processed.",
        )

    if not isinstance(safety, dict):
        return _safe_fallback_record(
            safe_test_id,
            "Safety screening returned an invalid result, so this request was not processed.",
        )

    safe_question = safety.get("cleaned_text", "")
    if not isinstance(safe_question, str):
        safe_question = ""

    # Do not call triage, policy retrieval, or the data agent for refused input.
    if safety.get("allowed") is not True or safety.get("safety_refusal") == "Yes":
        triage = {
            "category": "Other or Security",
            "priority": "P1",
            "confidence": 1.0,
            "rationale": "Safety screening requires refusal and security review.",
            "needs_human_review": True,
        }
        policy = {"passages": []}
        data = {"answer": "", "facts": [], "data_used": [], "notes": ""}
        try:
            from workflow import route_case
            from response import build_response

            routing = route_case(triage, safety, safe_question)
            result = build_response(triage, policy, data, routing, safety)
            return {"test_id": safe_test_id, **result}
        except Exception:
            return _safe_fallback_record(
                safe_test_id,
                "I can’t provide restricted information. Use an approved, access-controlled process.",
                safety,
            )

    # Triage can fail because an LLM service is unavailable or its JSON is invalid.
    try:
        from triage import _classify_screened_request

        triage = _classify_screened_request(safety)
    except Exception:
        triage = None

    if not _valid_triage(triage):
        triage = {
            "category": "Other or Security",
            "priority": "P4",
            "confidence": 0.0,
            "rationale": "Triage was unavailable or returned invalid labels; human review is needed.",
            "needs_human_review": True,
            "review_reason": "Triage error or invalid result.",
        }

    # Policy evidence is useful for most employee questions. An empty result is
    # valid; retrieval failures simply leave the answer without policy citations.
    try:
        from knowledge_retriever import search_policy

        policy = search_policy(safe_question, category=triage["category"], top_k=3)
        if not isinstance(policy, dict) or not isinstance(policy.get("passages", []), list):
            policy = _empty_evidence_note("Policy retrieval returned an invalid result.")
        else:
            policy.setdefault("passages", [])
    except Exception:
        policy = _empty_evidence_note("Policy files or retrieval service were unavailable.")

    # Only query the CSV agent when the question appears to ask for store data.
    if _needs_business_data(safe_question):
        try:
            from data_agent import _answer_screened_data_question

            data = _answer_screened_data_question(safety)
            if not isinstance(data, dict):
                data = {"answer": "", "facts": [], "data_used": [], "notes": "The data agent returned an invalid result."}
            data.setdefault("answer", "")
            data.setdefault("facts", [])
            data.setdefault("data_used", [])
            data.setdefault("notes", "")
        except Exception:
            data = {"answer": "", "facts": [], "data_used": [], "notes": "Business data files or query service were unavailable."}
    else:
        data = {"answer": "", "facts": [], "data_used": [], "notes": ""}

    # The response builder surfaces data notes, so include a safe retrieval
    # status there instead of silently hiding a policy-index failure.
    policy_note = policy.get("notes", "") if isinstance(policy, dict) else ""
    if isinstance(policy_note, str) and policy_note.strip():
        existing_note = data.get("notes", "")
        data["notes"] = " ".join(
            part for part in (existing_note, policy_note.strip()) if isinstance(part, str) and part.strip()
        )

    # Workflow adds deterministic impact and approval rules to model triage.
    try:
        from workflow import route_case

        routing = route_case(triage, safety, safe_question)
        if not isinstance(routing, dict):
            raise ValueError("Invalid workflow result")
    except Exception:
        routing = {
            "escalated": "Yes",
            "escalation_team": "",
            "channel": "",
            "sla_minutes": None,
            "human_approval_required": "Yes",
            "needs_human_review": True,
            "route_found": False,
            "reason": "Workflow or escalation data was unavailable; routing information is unavailable.",
            "category": triage["category"],
            "priority": triage["priority"],
        }

    # Response creation validates labels and citations. If that validation fails,
    # return a safe schema-shaped fallback instead of leaking exception details.
    try:
        from response import build_response

        response = build_response(triage, policy, data, routing, safety)
        if not isinstance(response, dict):
            raise ValueError("Invalid response result")
        return {"test_id": safe_test_id, **response}
    except Exception:
        return _safe_fallback_record(
            safe_test_id,
            "The request could not be completed safely. A human should review it.",
            safety,
        )
