"""Safety and privacy checks that run before any model or data request."""

import re
from functools import lru_cache
from typing import Any


# Use Presidio for personal and payment data recognition. Keeping the entity
# names here makes it clear which types of information are anonymized.
PII_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "LOCATION",
    "IBAN_CODE",
    "US_SSN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "US_BANK_NUMBER",
    "IP_ADDRESS",
]

# These are credential secrets rather than personal-identity entities. Keep a
# small deterministic mask for them because Presidio's default recognizers do
# not identify every store-specific PIN, OTP, or password label.
CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(cvv|cvc|pin|otp|password|passcode|token)"
    r"(\s*(?:is|:|=)\s*)[^\s,;]+"
)

# These rules identify unsafe intent, not PII. They remain deterministic and
# run after Presidio has anonymized the text and before any LLM call.
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


class PresidioUnavailableError(RuntimeError):
    """Signals that PII could not be safely checked or anonymized."""


@lru_cache(maxsize=1)
def _get_presidio_engines():
    """Create the analyzer and anonymizer once, then reuse them for all calls."""
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
    except Exception:
        # Keep dependency/model error details out of messages and logs.
        raise PresidioUnavailableError("Presidio privacy screening is unavailable.") from None

    try:
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
    except Exception:
        # AnalyzerEngine loads the configured spaCy model here. If that model
        # is missing or broken, callers must block instead of sending raw text.
        raise PresidioUnavailableError("Presidio privacy screening is unavailable.") from None

    return analyzer, anonymizer


def mask_sensitive_data(text: str) -> tuple[str, bool]:
    """Return text with Presidio-detected PII replaced by entity placeholders.

    For example, a detected phone number becomes ``<PHONE_NUMBER>``. If
    Presidio cannot analyze or anonymize the text, this function raises a
    generic error; its caller must not continue with the original text.
    """
    if not isinstance(text, str) or not text:
        return "", False

    try:
        from presidio_anonymizer.entities import OperatorConfig

        analyzer, anonymizer = _get_presidio_engines()
        findings = analyzer.analyze(
            text=text,
            entities=PII_ENTITIES,
            language="en",
        )

        # Tell Presidio's anonymizer to replace each recognized value with a
        # visible tag naming the entity type, without retaining any characters.
        replacement_operators = {
            finding.entity_type: OperatorConfig(
                "replace",
                {"new_value": f"<{finding.entity_type}>"},
            )
            for finding in findings
        }
        if findings:
            anonymized = anonymizer.anonymize(
                text=text,
                analyzer_results=findings,
                operators=replacement_operators,
            ).text
            was_masked = True
        else:
            anonymized = text
            was_masked = False

        # Mask labelled credentials separately; these are secrets, not PII
        # entity detection, so this rule does not replace Presidio recognition.
        anonymized, credential_count = CREDENTIAL_PATTERN.subn(
            r"\1\2<CREDENTIAL>", anonymized
        )
        return anonymized, was_masked or credential_count > 0
    except PresidioUnavailableError:
        raise
    except Exception:
        # Do not include the exception text because third-party errors may
        # include snippets of the text that was being analyzed.
        raise PresidioUnavailableError("Presidio privacy screening is unavailable.") from None


def _blocked_for_screening_failure() -> dict[str, Any]:
    """Return a generic fail-closed result when PII protection is unavailable."""
    return {
        "allowed": False,
        "cleaned_text": "Request withheld because personal-data screening is unavailable.",
        "pii_masked": "No",
        "safety_refusal": "Yes",
        "security_escalation": "No",
        "refusal_message": (
            "I can’t process this request because privacy screening is unavailable. "
            "Please retry later or ask a human reviewer for help."
        ),
        "reason": "Privacy screening is unavailable; the request is blocked for human review.",
        "blocked": True,
        "review_needed": True,
    }


def check_request(question: str) -> dict[str, Any]:
    """Check and sanitize an employee question before downstream processing.

    Unsafe access and bypass requests receive a generic refusal. Raw input and
    third-party exception details are never returned or logged.
    """
    try:
        safe_text, pii_was_masked = mask_sensitive_data(question or "")
    except PresidioUnavailableError:
        return _blocked_for_screening_failure()

    asks_for_restricted_data = any(
        pattern.search(safe_text) for pattern in RESTRICTED_DATA_PATTERNS
    )
    attempts_to_bypass_rules = any(
        pattern.search(safe_text) for pattern in BYPASS_PATTERNS
    )
    unsafe_request = asks_for_restricted_data or attempts_to_bypass_rules

    if unsafe_request:
        # Do not pass even the anonymized unsafe request downstream.
        return {
            "allowed": False,
            "cleaned_text": "Restricted request withheld. Use an approved, access-controlled process.",
            "pii_masked": "Yes" if pii_was_masked else "No",
            "safety_refusal": "Yes",
            "security_escalation": "Yes",
            "refusal_message": "I can’t provide restricted information. Use an approved, access-controlled process for an authorized business need.",
            "reason": "The request seeks restricted information or attempts to bypass safety rules.",
            "blocked": True,
            "review_needed": True,
        }

    return {
        "allowed": True,
        "cleaned_text": safe_text,
        "pii_masked": "Yes" if pii_was_masked else "No",
        "safety_refusal": "No",
        "security_escalation": "No",
        "refusal_message": "",
        "reason": "No restricted-data access or safety-bypass request was detected.",
        "blocked": False,
        "review_needed": False,
    }
