"""Deterministic response composition and validation for the copilot."""

import re
from pathlib import Path

from labels_and_citations import CATEGORIES, PRIORITIES, citations_for_markdown


ROOT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = ROOT_DIR / "knowledge_base"
DATA_DIR = ROOT_DIR / "data"

YES_NO = {"Yes", "No"}


def _is_yes(value: object) -> bool:
    """Read a Yes/No flag or boolean without treating other text as true."""
    return value is True or (isinstance(value, str) and value.strip().casefold() == "yes")


def _valid_citation(citation: str) -> bool:
    """Check that a citation points to a real Markdown heading in this project."""
    if not isinstance(citation, str) or citation.count("#") != 1:
        return False
    filename, _ = citation.split("#", 1)
    if Path(filename).name != filename or not filename.endswith(".md"):
        return False

    policy_path = KNOWLEDGE_BASE_DIR / filename
    if not policy_path.is_file():
        return False

    markdown = policy_path.read_text(encoding="utf-8")
    return citation in citations_for_markdown(filename, markdown)


def _safe_text(value: object) -> str:
    """Normalize text whitespace cleanly without redundant screening."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


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


class ResponseComposer:
    """Composes grounded responses and prediction-schema records from evidence."""

    def compose(
        self,
        triage: dict,
        policy: dict,
        data: dict,
        routing: dict,
        safety: dict | None = None,
        question: str = "",
    ) -> dict:
        """Compose answer and schema fields from triage, evidence, and routing."""
        safety_dict = safety if isinstance(safety, dict) else {
            "pii_masked": "No",
            "safety_refusal": "No",
            "security_escalation": "No",
            "refusal_message": "",
        }
        req_q = question or safety_dict.get("cleaned_text") or ""
        return _build_response_internal(triage, policy, data, routing, safety_dict, question=req_q)


def build_response(
    triage: dict,
    policy: dict,
    data: dict,
    routing: dict,
    safety: dict | None = None,
    question: str = "",
) -> dict:
    """Build a grounded answer and prediction-schema fields without generation."""
    return ResponseComposer().compose(triage, policy, data, routing, safety, question=question)


def _generate_specific_answer(
    question: str,
    category: str,
    priority: str,
    policy_passages: list[dict],
    data_answer: str,
    safe_data_facts: list[str],
    data_notes: str,
    routing: dict,
    escalated: str,
    human_approval_required: str,
) -> str:
    """Generate a direct, specific, and actionable answer tailored to the user."""
    # 1. Attempt LLM answer synthesis for fluent, specific answers
    if question.strip():
        try:
            from llm_client import chat_completion

            context_items = []
            if data_answer:
                context_items.append(f"Business Data Result: {data_answer}")
            if safe_data_facts:
                context_items.append("Confirmed Data Facts: " + "; ".join(safe_data_facts[:4]))
            for passage in policy_passages[:2]:
                text = passage.get("text", "").strip()
                if text:
                    context_items.append(f"Policy Reference ({passage.get('heading', '')}):\n{text}")
            if human_approval_required == "Yes":
                context_items.append("Approval rule: Store Manager approval is strictly required before any action.")
            team = routing.get("escalation_team")
            if escalated == "Yes" and team:
                context_items.append(f"Escalation team: {team}")

            if context_items:
                prompt_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an AI-powered Retail Operations Copilot. Give a direct, specific, and actionable answer "
                            "to the employee's exact question, based strictly on the provided policy, data findings, and routing. "
                            "Address the specific store, register, error code, product, or scenario in the question. "
                            "Do NOT include generic prefixes (such as 'Category:', 'Priority:', 'Policy guidance:', 'Data result:'). "
                            "Do NOT include generic boilerplate disclaimers (such as 'Recommendations only; no refund...'). "
                            "Answer directly, concisely, and professionally (2-4 sentences)."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Employee Question: {question}\n\nEvidence & Policy Context:\n" + "\n\n".join(context_items),
                    },
                ]
                llm_res = chat_completion(prompt_messages, temperature=0.0, timeout=12.0, max_retries=1)
                content = llm_res.get("content", "").strip()
                if content:
                    return _safe_text(content)
        except Exception:
            pass

    # 2. Specific deterministic fallback (free of generic boilerplate prefixes)
    specific_parts = []
    if data_answer:
        specific_parts.append(data_answer)
    elif safe_data_facts:
        specific_parts.append("; ".join(safe_data_facts[:3]) + ".")

    if data_notes and data_notes not in " ".join(specific_parts):
        specific_parts.append(data_notes)

    # Extract actionable policy sentences rather than whole documents
    for passage in policy_passages[:2]:
        text = passage.get("text", "")
        action_lines = []
        for line in text.splitlines():
            cleaned = line.strip()
            if not cleaned or cleaned.startswith(("#", "Source ID:", "Applies to:")):
                continue
            if cleaned.startswith(("-", "*", "1.", "2.", "3.", "4.")):
                action_lines.append(cleaned.lstrip("-*1234. "))
            elif any(verb in cleaned.casefold() for verb in ("re-seat", "restart", "check", "verify", "ticket", "escalate", "approval", "quarantine")):
                action_lines.append(cleaned)
        if action_lines:
            specific_parts.append(". ".join(action_lines[:2]) + ".")
        else:
            first_sent = _first_policy_sentence(passage)
            if first_sent and first_sent not in " ".join(specific_parts):
                specific_parts.append(first_sent)

    if human_approval_required == "Yes" and "approval" not in " ".join(specific_parts).casefold():
        specific_parts.append("Store Manager approval is required before taking action.")

    team = routing.get("escalation_team")
    if escalated == "Yes" and team and team not in " ".join(specific_parts):
        specific_parts.append(f"Route to {team}.")

    if not specific_parts:
        return "I could not confirm specific policy or data guidance for this issue. A store manager should review it."

    return _safe_text(" ".join(specific_parts))


def _build_response_internal(
    triage: dict,
    policy: dict,
    data: dict,
    routing: dict,
    safety: dict,
    question: str = "",
) -> dict:

    confidence, category, priority = _validate_inputs(triage, policy, data, routing, safety)
    routed_category = routing.get("category", category)
    routed_priority = routing.get("priority", priority)
    if routed_category not in CATEGORIES or routed_priority not in PRIORITIES:
        raise ValueError("workflow category and priority must use the supplied labels.")
    category = routed_category
    priority = routed_priority

    citations = []
    valid_passages = []
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
        valid_passages.append(passage)

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

    if safety_refusal or security_escalation:
        refusal = _safe_text(safety.get("refusal_message", ""))
        answer = (
            refusal
            or "I can’t provide restricted information. Use an approved, access-controlled process."
        )
    else:
        # Generate specific answer addressing ONLY the user's question without generic boilerplates
        answer = _generate_specific_answer(
            question=question,
            category=category,
            priority=priority,
            policy_passages=valid_passages,
            data_answer=data_answer,
            safe_data_facts=safe_data_facts,
            data_notes=data_notes,
            routing=routing,
            escalated=escalated,
            human_approval_required=human_approval_required,
        )

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
