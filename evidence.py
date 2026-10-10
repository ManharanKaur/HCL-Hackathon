"""Evidence routing and bundling module for retail copilot.

Corresponds to the diagram nodes:
- 'What evidence is needed?' (EvidenceRouter)
- 'Evidence Bundle' (EvidenceBundle)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BUSINESS_DATA_TERMS = (
    "how many", "count", "quantity", "stock", "inventory", "on hand",
    "available", "stock-out", "stockout", "receipt", "shipment", "vendor",
    "supplier", "support case", "recent case", "last 24 hours", "last 30 days",
    "complaint", "quality issue", "sku", "product", "store",
)

# Operational categories that almost always require reference to policy SOPs
POLICY_RELEVANT_CATEGORIES = {
    "POS and Payments",
    "Pricing and Promotions",
    "Returns and Refunds",
    "Product Quality and Safety",
    "Customer Service",
    "Other or Security",
}


class EvidenceRouter:
    """Decides what evidence is needed (Policy, Business data, or Both)."""

    @staticmethod
    def needs_business_data(question: str) -> bool:
        """Check if question asks for metrics, inventory, counts, or CSV records."""
        lowered = (question or "").casefold()
        return any(term in lowered for term in BUSINESS_DATA_TERMS)

    @staticmethod
    def needs_policy(category: str, question: str) -> bool:
        """Check if question requires SOP policy runbooks or procedures."""
        lowered = (question or "").casefold()
        policy_signals = (
            "how to", "how can", "what should", "procedure", "policy", "sop",
            "guide", "runbook", "rule", "process", "approval", "error", "scan",
            "customer wants", "return", "refund", "ticket", "why is", "root cause",
        )
        return (
            any(sig in lowered for sig in policy_signals)
            or category in POLICY_RELEVANT_CATEGORIES
            or not EvidenceRouter.needs_business_data(question)
        )

    @classmethod
    def determine_evidence_needed(cls, category: str, question: str) -> str:
        """Return 'policy', 'business_data', or 'both' according to diagram."""
        data_needed = cls.needs_business_data(question)
        policy_needed = cls.needs_policy(category, question)

        if policy_needed and data_needed:
            return "both"
        elif data_needed:
            return "business_data"
        return "policy"


@dataclass
class EvidenceBundle:
    """Aggregates triage outputs and gathered evidence from agents."""

    question: str = ""
    category: str = "Other or Security"
    priority: str = "P4"
    confidence: float = 0.0
    rationale: str = ""
    policy: dict[str, Any] = field(default_factory=lambda: {"passages": [], "notes": ""})
    data: dict[str, Any] = field(default_factory=lambda: {
        "answer": "",
        "facts": [],
        "data_used": [],
        "notes": "",
    })

    @property
    def triage_dict(self) -> dict[str, Any]:
        """Return triage output shape."""
        return {
            "category": self.category,
            "priority": self.priority,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }

    @property
    def passages(self) -> list[dict[str, Any]]:
        return self.policy.get("passages", [])

    @property
    def data_facts(self) -> list[str]:
        return self.data.get("facts", [])

    @property
    def data_used(self) -> list[str]:
        return self.data.get("data_used", [])

    @property
    def data_answer(self) -> str:
        return self.data.get("answer", "")
