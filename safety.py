"""Safety checks for employee questions before they reach an LLM or agent."""

import re
from typing import Any


# Patterns cover common contact details and payment identifiers. These masks are
# applied before the question is returned to another module.
EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
)
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?91[\s.-]?)?[6-9]\d{9}(?!\w)")
CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
AADHAAR_PATTERN = re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)")
LABELED_SECRET_PATTERN = re.compile(
    r"(?i)\b(cvv|cvc|pin|otp|password|passcode|token|account number)"
    r"(\s*(?:is|:|=)\s*)[^\s,;]+"
)

# Words that indicate a request to retrieve, reveal, or otherwise access
# information that policy keeps private. The patterns accept common word order.
RESTRICTED_DATA_PATTERNS = (
    re.compile(
        r"\b(show|reveal|display|give|send|share|list|export|print|retrieve|"
        r"look\s+up|access|provide)\b.{0,100}\b(customer|card|payment|"
        r"bank|credential|password|supplier|vendor|contract|private|personal)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(customer|payment|card|bank|supplier|vendor|contract|private|"
        r"personal)\b.{0,100}\b(numbers?|details?|records?|data|credentials?|"
        r"prices?|terms?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(store|save|record|attach|add|log)\b.{0,80}\b(card number|"
        r"cvv|payment details|password|credentials?)\b",
        re.IGNORECASE,
    ),
)

BYPASS_PATTERNS = (
    re.compile(r"\bignore\b.{0,50}\b(previous|prior|system|safety|security|rules?)\b", re.I),
    re.compile(r"\b(bypass|override|disable|ignore)\b.{0,40}\b(safety|policy|rules?|guardrails?)\b", re.I),
    re.compile(r"\b(reveal|print|show|repeat)\b.{0,50}\b(hidden|system|developer)\b.{0,30}\b(prompt|instructions?)\b", re.I),
    re.compile(r"\b(access|read|open|print|show)\b.{0,50}\b(restricted|private|secret)\b.{0,30}\b(files?|data|records?)\b", re.I),
)


def mask_sensitive_data(text: str) -> tuple[str, bool]:
    """Replace recognizable personal and payment details with safe markers.

    Phone numbers keep only their final four digits, matching the responsible
    AI policy. Payment card numbers and government identifiers are fully
    replaced so their original values cannot flow to later processing.
    """
    masked_text = text
    found_sensitive_data = False

    def mask_email(match: re.Match[str]) -> str:
        local_part, domain = match.group(0).split("@", 1)
        if len(local_part) <= 2:
            masked_local = local_part[0] + "*" * max(1, len(local_part) - 1)
        else:
            masked_local = local_part[0] + "*" * (len(local_part) - 2) + local_part[-1]
        return f"{masked_local}@{domain}"

    def mask_phone(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        return "******" + digits[-4:]

    def replace_pattern(pattern: re.Pattern[str], replacement: Any) -> None:
        nonlocal masked_text, found_sensitive_data
        masked_text, count = pattern.subn(replacement, masked_text)
        found_sensitive_data = found_sensitive_data or count > 0

    replace_pattern(EMAIL_PATTERN, mask_email)
    replace_pattern(PHONE_PATTERN, mask_phone)
    replace_pattern(AADHAAR_PATTERN, "[REDACTED]")

    def mask_card(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        # Do not mistake a 12-digit identifier already covered as Aadhaar for a card.
        return "[REDACTED_CARD]" if len(digits) >= 13 else match.group(0)

    replace_pattern(CARD_PATTERN, mask_card)

    def mask_labeled_secret(match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group(2)}[REDACTED]"

    replace_pattern(LABELED_SECRET_PATTERN, mask_labeled_secret)
    return masked_text, found_sensitive_data


def check_request(question: str) -> dict:
    """Check and sanitize a question before any LLM or downstream processing.

    Unsafe access and bypass requests receive a generic refusal, rather than
    an echoed copy of the question. Raw input is never logged or returned.
    """
    safe_text, pii_was_masked = mask_sensitive_data(question or "")
    asks_for_restricted_data = any(
        pattern.search(safe_text) for pattern in RESTRICTED_DATA_PATTERNS
    )
    attempts_to_bypass_rules = any(
        pattern.search(safe_text) for pattern in BYPASS_PATTERNS
    )
    unsafe_request = asks_for_restricted_data or attempts_to_bypass_rules

    if unsafe_request:
        # Keep the original request out of the response passed to later modules.
        return {
            "allowed": False,
            "cleaned_text": "Restricted request withheld. Use an approved, access-controlled process.",
            "pii_masked": "Yes" if pii_was_masked else "No",
            "safety_refusal": "Yes",
            "security_escalation": "Yes",
            "refusal_message": "I can’t provide restricted information. Use an approved, access-controlled process for an authorized business need.",
            "reason": "The request seeks restricted information or attempts to bypass safety rules.",
        }

    return {
        "allowed": True,
        "cleaned_text": safe_text,
        "pii_masked": "Yes" if pii_was_masked else "No",
        "safety_refusal": "No",
        "security_escalation": "No",
        "refusal_message": "",
        "reason": "No restricted-data access or safety-bypass request was detected.",
    }
