"""Safety and privacy checks that run before any model or data request.

This module is the SINGLE SOURCE OF TRUTH for:
1. Input normalization
2. PII / Sensitive data masking
3. Safety Gate evaluation (restricted data & prompt bypass detection)
4. Safe refusal and security escalation record generation
"""

import re
import unicodedata
from functools import lru_cache
from typing import Any


# Entity types recognized and anonymized
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

CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(cvv|cvc|pin|otp|password|passcode|token)"
    r"(\s*(?:is|:|=)\s*)[^\s,;]+"
)

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

FALLBACK_PII_PATTERNS = {
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{15,16}\b"),
    "PHONE_NUMBER": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{10}\b"),
    "EMAIL_ADDRESS": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "IP_ADDRESS": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
}


class PresidioUnavailableError(RuntimeError):
    """Signals that PII could not be safely checked or anonymized."""


class InputNormalizer:
    """Normalizes raw user input into clean, standardized text."""

    @staticmethod
    def normalize(text: object) -> str:
        """Strip non-printable characters, normalize whitespace and unicode."""
        if not isinstance(text, str):
            return ""
        # Normalize unicode characters (e.g. smart quotes, accented chars)
        normalized = unicodedata.normalize("NFKC", text)
        # Strip control characters (except common whitespace)
        normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", normalized)
        # Collapse whitespace into single spaces and strip edges
        normalized = " ".join(normalized.split())
        return normalized


@lru_cache(maxsize=1)
def _get_presidio_engines():
    """Create the analyzer and anonymizer once, preferring en_core_web_sm."""
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        anonymizer = AnonymizerEngine()
        return analyzer, anonymizer
    except Exception:
        pass

    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        return analyzer, anonymizer
    except Exception:
        raise PresidioUnavailableError("Presidio privacy screening is unavailable.") from None


def _regex_mask_fallback(text: str) -> tuple[str, bool]:
    """Fallback regex masker for when Presidio engine is unavailable."""
    anonymized = text
    masked_any = False
    for entity, pattern in FALLBACK_PII_PATTERNS.items():
        anonymized, count = pattern.subn(f"<{entity}>", anonymized)
        if count > 0:
            masked_any = True
    anonymized, cred_count = CREDENTIAL_PATTERN.subn(r"\1\2<CREDENTIAL>", anonymized)
    if cred_count > 0:
        masked_any = True
    return anonymized, masked_any


def mask_sensitive_data(text: str) -> tuple[str, bool]:
    """Return text with detected PII replaced by entity placeholders.

    Uses Presidio if available; falls back to robust regexes if Presidio fails.
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

        anonymized, credential_count = CREDENTIAL_PATTERN.subn(
            r"\1\2<CREDENTIAL>", anonymized
        )
        return anonymized, was_masked or credential_count > 0
    except Exception:
        # Graceful fallback to regex patterns so screening never halts completely
        return _regex_mask_fallback(text)


class SafeRefusalBuilder:
    """Constructs schema-compliant refusal and security escalation records."""

    @staticmethod
    def build(safety_result: dict[str, Any], test_id: str = "") -> dict[str, Any]:
        """Build the required prediction-schema fields for unsafe requests."""
        refusal_msg = safety_result.get("refusal_message") or (
            "I can’t provide restricted information. Use an approved, access-controlled process."
        )
        return {
            "test_id": test_id,
            "category": "Other or Security",
            "priority": "P1",
            "confidence": 1.0,
            "answer": refusal_msg,
            "citations": "",
            "data_used": "",
            "escalated": "Yes",
            "escalation_team": "Security",
            "pii_masked": safety_result.get("pii_masked", "No"),
            "safety_refusal": "Yes",
            "notes": safety_result.get("reason", "Security escalation: restricted request refused."),
        }


class SafetyGate:
    """Single gate responsible for input normalization and safety screening."""

    def __init__(self, normalizer: InputNormalizer | None = None):
        self.normalizer = normalizer or InputNormalizer()

    def evaluate(self, raw_question: str) -> dict[str, Any]:
        """Evaluate raw input through normalization, PII masking, and safety rules."""
        normalized_text = self.normalizer.normalize(raw_question)
        safe_text, pii_was_masked = mask_sensitive_data(normalized_text)

        asks_for_restricted_data = any(
            pattern.search(safe_text) for pattern in RESTRICTED_DATA_PATTERNS
        )
        attempts_to_bypass_rules = any(
            pattern.search(safe_text) for pattern in BYPASS_PATTERNS
        )
        unsafe_request = asks_for_restricted_data or attempts_to_bypass_rules

        if unsafe_request:
            return {
                "allowed": False,
                "is_safe": False,
                "normalized_text": normalized_text,
                "cleaned_text": "Restricted request withheld. Use an approved, access-controlled process.",
                "pii_masked": "Yes" if pii_was_masked else "No",
                "safety_refusal": "Yes",
                "security_escalation": "Yes",
                "refusal_message": (
                    "I can’t provide restricted information. Use an approved, "
                    "access-controlled process for an authorized business need."
                ),
                "reason": "The request seeks restricted information or attempts to bypass safety rules.",
                "blocked": True,
                "review_needed": True,
            }

        return {
            "allowed": True,
            "is_safe": True,
            "normalized_text": normalized_text,
            "cleaned_text": safe_text,
            "pii_masked": "Yes" if pii_was_masked else "No",
            "safety_refusal": "No",
            "security_escalation": "No",
            "refusal_message": "",
            "reason": "No restricted-data access or safety-bypass request was detected.",
            "blocked": False,
            "review_needed": False,
        }


def check_request(question: str) -> dict[str, Any]:
    """Compatibility entry point delegating to SafetyGate."""
    return SafetyGate().evaluate(question or "")

