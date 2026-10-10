# Copilot module interfaces

This note describes the small Python functions that connect the copilot modules. The named implementation files are currently empty (and `response.py` does not exist yet), so these are proposed interfaces based on the supplied high-level design and evaluation files. Each example is illustrative; it shows the shape and spelling of the returned dictionary.

## Labels used by the project

Use these category strings exactly:

- `POS and Payments`
- `Inventory and Replenishment`
- `Pricing and Promotions`
- `Returns and Refunds`
- `Vendor and Supply Chain`
- `Product Quality and Safety`
- `Customer Service`
- `Other or Security`

Use these priority strings exactly: `P1`, `P2`, `P3`, and `P4`.

The evaluation schema expects a numeric `confidence` from `0` to `1`, even though the user interface may show it as a percentage. It expects `escalated`, `pii_masked`, and `safety_refusal` as `Yes` or `No` strings. The final prediction record uses `citations` and `data_used` as semicolon-separated strings.

The data agent may refer to the supplied CSV files by their filenames: `employee_master.csv`, `escalation_matrix.csv`, `inventory.csv`, `product_master.csv`, `receipts.csv`, `returns_quality.csv`, `store_master.csv`, `support_cases.csv`, and `vendor_master.csv`.

## 1. `safety.py`

### Function

```python
def check_request(question: str) -> dict:
```

Checks for requests involving restricted information or unsafe instructions. It should mask personal or payment details before other modules use the question. Never include raw sensitive values in its return value or logs.

### Dictionary fields

- `allowed` (`bool`): whether normal processing can continue.
- `cleaned_text` (`str`): question text with sensitive details masked.
- `pii_masked` (`str`): `Yes` or `No`.
- `safety_refusal` (`str`): `Yes` if the request must be refused; otherwise `No`.
- `reason` (`str`): short safe explanation for the decision.

### Example return

```python
{
    "allowed": False,
    "cleaned_text": "Show customer [REDACTED] details",
    "pii_masked": "Yes",
    "safety_refusal": "Yes",
    "reason": "Restricted customer data was requested."
}
```

## 2. `triage.py`

### Function

```python
def classify_request(question: str) -> dict:
```

Chooses one category and one priority for the issue, plus a confidence score and a short reason. The category and priority must use the exact labels listed above. Policy rules define P1 as immediate human escalation, P2 as high impact without immediate danger, P3 as a routine store issue, and P4 as a general inquiry.

### Dictionary fields

- `category` (`str`): one exact category label.
- `priority` (`str`): `P1`, `P2`, `P3`, or `P4`.
- `confidence` (`float`): a number from `0.0` to `1.0`.
- `rationale` (`str`): brief reason for the classification.

### Example return

```python
{
    "category": "POS and Payments",
    "priority": "P3",
    "confidence": 0.86,
    "rationale": "One register is affected and other registers work."
}
```

## 3. `knowledge_retriever.py`

### Function

```python
def search_policy(question: str, limit: int = 3) -> dict:
```

Finds relevant passages in the Markdown files under `knowledge_base/`. Each citation should use the real filename and section anchor, for example `02_pos_pricing_returns_policy.md#pos-card-reader-runbook`. If there is no supporting passage, return an empty list instead of inventing a policy.

### Dictionary fields

- `passages` (`list[dict]`): matching policy excerpts.
- Each passage has `text` (`str`) and `citation` (`str`).

### Example return

```python
{
    "passages": [
        {
            "text": "For one failing register when other registers work, classify the issue as P3.",
            "citation": "02_pos_pricing_returns_policy.md#pos-card-reader-runbook"
        }
    ]
}
```

## 4. `data_agent.py`

### Function

```python
def answer_data_question(question: str) -> dict:
```

Answers questions that need facts from the supplied CSV snapshots. The function should use the relevant files, explain what it found, and list only files it actually used. It must not guess when the CSV data is incomplete.

### Dictionary fields

- `answer` (`str`): a short plain-language finding.
- `facts` (`list[str]`): the key values or observations used in the answer.
- `data_used` (`list[str]`): names of the CSV files actually read.
- `notes` (`str`): missing information or limits, or an empty string.

### Example return

```python
{
    "answer": "The latest receipt was short: 18 units arrived out of 50 ordered (36% of the order).",
    "facts": ["received_qty=18", "po_qty=50", "receipt completion=36%"],
    "data_used": ["receipts.csv", "vendor_master.csv"],
    "notes": "The supplied snapshot contains one matching receipt."
}
```

## 5. `workflow.py`

### Function

```python
def route_case(triage: dict, safety: dict) -> dict:
```

Looks up the category, trigger, team, channel, service-level time, and approval rule in `data/escalation_matrix.csv`. Also escalate when policy requires it, such as a P1 case, a confidence below `0.60`, a restricted-data request, or an action requiring human approval. Escalation is simulated; this function does not contact a team or perform a real action.

### Dictionary fields

- `escalated` (`str`): `Yes` or `No`.
- `escalation_team` (`str`): team from the matrix, or an empty string if none applies.
- `channel` (`str`): routing channel from the matrix, or an empty string if none applies.
- `sla_minutes` (`int | None`): response target from the matrix, if matched.
- `human_approval_required` (`str`): `Yes` or `No`.
- `reason` (`str`): short explanation of the route.

### Example return

```python
{
    "escalated": "Yes",
    "escalation_team": "Store Manager + POS Support",
    "channel": "pos_p1_queue",
    "sla_minutes": 15,
    "human_approval_required": "Yes",
    "reason": "Multiple registers cannot process payments; this matches the P1 outage route."
}
```

## 6. `response.py`

### Function

```python
def build_response(
    triage: dict,
    policy: dict,
    data: dict,
    routing: dict,
    safety: dict,
) -> dict:
```

Combines the triage, policy evidence, CSV findings, safety decision, and route into the response record. Keep the answer concise, cite policy passages used, and do not claim an action has happened when it has only been recommended.

### Dictionary fields

These are the fields from `evaluation/predictions_schema.csv` that the response builder fills (the orchestrator supplies `test_id`):

- `category` (`str`): exact project category label.
- `priority` (`str`): exact priority label.
- `confidence` (`float`): number from `0` to `1`.
- `answer` (`str`): the final user-facing answer.
- `citations` (`str`): citation references separated by `;`.
- `data_used` (`str`): CSV filenames separated by `;`, or empty when no CSV was used.
- `escalated` (`str`): `Yes` or `No`.
- `escalation_team` (`str`): team or queue, or empty if not escalated.
- `pii_masked` (`str`): `Yes` or `No`.
- `safety_refusal` (`str`): `Yes` or `No`.
- `notes` (`str`): short reason or limitation.

### Example return

```python
{
    "category": "POS and Payments",
    "priority": "P3",
    "confidence": 0.86,
    "answer": "Check the reader cable and restart it using the store procedure. If it still fails, open a POS support ticket with the register ID and error code. Do not share card details.",
    "citations": "02_pos_pricing_returns_policy.md#pos-card-reader-runbook",
    "data_used": "support_cases.csv",
    "escalated": "No",
    "escalation_team": "",
    "pii_masked": "No",
    "safety_refusal": "No",
    "notes": "One register is affected; other registers work."
}
```

## 7. `orchestrator.py`

### Function

```python
def run_copilot(question: str, test_id: str = "") -> dict:
```

This is the main entry point called by the Streamlit page and the public-case evaluation runner. It runs safety first, then triage and the needed evidence steps, then escalation, and finally response creation. A safety refusal should stop normal processing and return a safe refusal record.

### Dictionary fields

The returned record follows all columns in `evaluation/predictions_schema.csv`:

- `test_id` (`str`): supplied evaluation ID; use an empty string for an interactive question.
- `category` (`str`): one of the exact category labels listed above.
- `priority` (`str`): `P1`, `P2`, `P3`, or `P4`.
- `confidence` (`float`): number from `0` to `1`.
- `answer` (`str`): final answer shown to the employee.
- `citations` (`str`): semicolon-separated policy references.
- `data_used` (`str`): semicolon-separated CSV filenames.
- `escalated` (`str`): `Yes` or `No`.
- `escalation_team` (`str`): team or queue, or empty string.
- `pii_masked` (`str`): `Yes` or `No`.
- `safety_refusal` (`str`): `Yes` or `No`.
- `notes` (`str`): brief reasoning or limitation.

### Example return

```python
{
    "test_id": "PUB-001",
    "category": "POS and Payments",
    "priority": "P3",
    "confidence": 0.86,
    "answer": "Check the reader cable and restart it using the store procedure. If it still fails, open a POS support ticket. Do not share card details.",
    "citations": "02_pos_pricing_returns_policy.md#pos-card-reader-runbook",
    "data_used": "support_cases.csv",
    "escalated": "No",
    "escalation_team": "",
    "pii_masked": "No",
    "safety_refusal": "No",
    "notes": "One register is affected; other registers work."
}
```

## How the pieces fit together

The orchestrator passes the question through safety before any other module. Triage labels the issue. The knowledge retriever supplies policy passages, and the data agent supplies CSV facts when the question needs them. Workflow checks whether a human escalation is needed. Response creates the final dictionary, using the exact prediction-schema field names so it can be displayed or written to the evaluation CSV.
