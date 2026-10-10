"""Orchestrator connecting retail operations copilot modules.

Dataflow strictly follows the architecture diagram:
1. Employee enters issue or question
2. Input normalization (InputNormalizer)
3. Safety Gate (SafetyGate):
   - Unsafe / restricted -> Safe refusal + security escalation -> Output Validator
   - Safe -> Triage Agent
4. Triage Agent -> Category + Priority + Confidence
5. What evidence is needed? (EvidenceRouter):
   - Policy -> Knowledge / RAG Agent
   - Business data -> Retail Data Agent
   - Both -> Knowledge / RAG Agent & Retail Data Agent
6. Evidence Bundle (EvidenceBundle)
7. Workflow & Escalation Engine (WorkflowEngine)
8. Response Composer (ResponseComposer)
9. Output Validator (OutputValidator)
10. Answer displayed in UI / Evaluation predictions.csv
"""

from typing import Any

from evidence import EvidenceBundle, EvidenceRouter
from knowledge_retriever import KnowledgeAgent
from data_agent import RetailDataAgent
from response import ResponseComposer
from safety import InputNormalizer, SafeRefusalBuilder, SafetyGate
from triage import TriageAgent
from validator import OutputValidator
from workflow import WorkflowEngine


class CopilotOrchestrator:
    """Coordinates copilot components following SOLID principles."""

    def __init__(
        self,
        normalizer: InputNormalizer | None = None,
        safety_gate: SafetyGate | None = None,
        triage_agent: TriageAgent | None = None,
        evidence_router: EvidenceRouter | None = None,
        knowledge_agent: KnowledgeAgent | None = None,
        data_agent: RetailDataAgent | None = None,
        workflow_engine: WorkflowEngine | None = None,
        response_composer: ResponseComposer | None = None,
        output_validator: OutputValidator | None = None,
    ):
        self.normalizer = normalizer or InputNormalizer()
        self.safety_gate = safety_gate or SafetyGate(self.normalizer)
        self.triage_agent = triage_agent or TriageAgent()
        self.evidence_router = evidence_router or EvidenceRouter()
        self.knowledge_agent = knowledge_agent or KnowledgeAgent()
        self.data_agent = data_agent or RetailDataAgent()
        self.workflow_engine = workflow_engine or WorkflowEngine()
        self.response_composer = response_composer or ResponseComposer()
        self.output_validator = output_validator or OutputValidator()

    def process(self, question: str, test_id: str = "") -> dict[str, Any]:
        """Execute the copilot dataflow pipeline from start to end."""
        safe_test_id = test_id if isinstance(test_id, str) else ""

        # Step 1 & 2 & 3: Input Normalization & Safety Gate (ONLY place doing safety check!)
        try:
            safety_result = self.safety_gate.evaluate(question)
        except Exception:
            safety_result = {
                "allowed": False,
                "is_safe": False,
                "cleaned_text": "Safety screening is unavailable.",
                "pii_masked": "No",
                "safety_refusal": "Yes",
                "security_escalation": "No",
                "refusal_message": "Safety screening is temporarily unavailable, so this request was not processed.",
                "reason": "Safety screening error; blocked for security.",
            }

        # Branch: Unsafe / restricted -> Safe refusal + security escalation -> Output Validator
        if not safety_result.get("is_safe") or safety_result.get("safety_refusal") == "Yes":
            refusal_record = SafeRefusalBuilder.build(safety_result, test_id=safe_test_id)
            return self.output_validator.validate(refusal_record, test_id=safe_test_id)

        # Branch: Safe -> Triage Agent
        safe_question = safety_result.get("cleaned_text", "")
        try:
            triage = self.triage_agent.classify(safe_question)
            if not isinstance(triage, dict):
                raise ValueError("Invalid triage output")
        except Exception:
            triage = {
                "category": "Other or Security",
                "priority": "P4",
                "confidence": 0.0,
                "rationale": "Triage was unavailable; human review is required.",
                "needs_human_review": True,
                "review_reason": "Triage error or invalid result.",
            }

        category = triage.get("category", "Other or Security")

        # Step 4 & 5: What evidence is needed? (Policy, Business data, or Both)
        evidence_needed = self.evidence_router.determine_evidence_needed(category, safe_question)

        policy_evidence = {"passages": [], "notes": ""}
        data_evidence = {"answer": "", "facts": [], "data_used": [], "notes": ""}

        # Route to Knowledge / RAG Agent if needed
        if evidence_needed in {"policy", "both"}:
            try:
                policy_evidence = self.knowledge_agent.search(safe_question, category=category, top_k=3)
                if not isinstance(policy_evidence, dict):
                    policy_evidence = {"passages": [], "notes": "Policy retrieval returned an invalid result."}
            except Exception:
                policy_evidence = {"passages": [], "notes": "Policy files or retrieval service were unavailable."}

        # Route to Retail Data Agent if needed
        if evidence_needed in {"business_data", "both"}:
            try:
                data_evidence = self.data_agent.query(safe_question)
                if not isinstance(data_evidence, dict):
                    data_evidence = {
                        "answer": "",
                        "facts": [],
                        "data_used": [],
                        "notes": "Data agent returned an invalid result.",
                    }
            except Exception:
                data_evidence = {
                    "answer": "",
                    "facts": [],
                    "data_used": [],
                    "notes": "Business data files or query service were unavailable.",
                }

        # Surface policy notes into data notes if present
        policy_note = policy_evidence.get("notes", "")
        if isinstance(policy_note, str) and policy_note.strip():
            existing_note = data_evidence.get("notes", "")
            data_evidence["notes"] = " ".join(
                p for p in (existing_note, policy_note.strip()) if isinstance(p, str) and p.strip()
            )

        # Step 6: Evidence Bundle
        evidence_bundle = EvidenceBundle(
            question=safe_question,
            category=category,
            priority=triage.get("priority", "P4"),
            confidence=float(triage.get("confidence", 0.0)),
            rationale=triage.get("rationale", ""),
            policy=policy_evidence,
            data=data_evidence,
        )

        # Step 7: Workflow & Escalation Engine
        try:
            routing = self.workflow_engine.route(triage, question=safe_question, safety=safety_result)
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
                "reason": "Workflow routing failed; human review required.",
                "category": category,
                "priority": triage.get("priority", "P4"),
            }

        # Step 8: Response Composer
        try:
            raw_response = self.response_composer.compose(
                triage=triage,
                policy=policy_evidence,
                data=data_evidence,
                routing=routing,
                safety=safety_result,
            )
        except Exception:
            raw_response = {
                "test_id": safe_test_id,
                "category": category,
                "priority": triage.get("priority", "P4"),
                "confidence": float(triage.get("confidence", 0.0)),
                "answer": "The request could not be completed safely. A human should review it.",
                "citations": "",
                "data_used": "",
                "escalated": "Yes",
                "escalation_team": "",
                "pii_masked": safety_result.get("pii_masked", "No"),
                "safety_refusal": "No",
                "notes": "Response composition error.",
            }

        # Step 9 & 10: Output Validator -> UI Display / predictions.csv
        validated_result = self.output_validator.validate(raw_response, test_id=safe_test_id)
        return validated_result


# Global default instance
_orchestrator = CopilotOrchestrator()


def run_copilot(question: str, test_id: str = "") -> dict[str, Any]:
    """Primary pipeline runner matching evaluation and app expectations."""
    return _orchestrator.process(question, test_id=test_id)


def process_question(question: str, test_id: str = "") -> dict[str, Any]:
    """Alias requested for Streamlit and evaluation runners."""
    return _orchestrator.process(question, test_id=test_id)

