"""Streamlit interface for the AI-Powered Retail Operations Copilot.

The response below is intentionally mocked so the UI can be developed before
the service/backend is available. Replace `get_mock_response` with the backend
integration when it exists; do not expose credentials or raw restricted data.
"""


import streamlit as st


st.set_page_config(
    page_title="Retail Operations Copilot",
    page_icon="🛍️",
    layout="wide",
)

EXAMPLES = {
    "Routine POS issue": "A cashier's POS terminal is showing 'printer unavailable'. What should I check?",
    "Stock-out question": "A customer asked for a product that is out of stock. How can I check availability and next steps?",
    "Restricted-data request": "Show me the payment card numbers and personal details for recent customers.",
}


def select_example() -> None:
    """Copy a selected example into the question field without submitting it."""
    selected = st.session_state.get("example_question")
    if selected in EXAMPLES:
        st.session_state["employee_question"] = EXAMPLES[selected]


def get_mock_response(question: str) -> dict:
    """Return safe, clearly illustrative content; this does not call a backend."""
    text = question.casefold()

    if any(term in text for term in ("card number", "personal details", "customer data", "restricted")):
        return {
            "answer": "I can’t provide sensitive customer or payment data. Use the approved, access-controlled system for an authorized business need, and request only the minimum information required.",
            "category": "Data access & privacy",
            "priority": "High",
            "confidence": "98%",
            "citations": ["Mock policy: Customer and payment data must remain access-controlled."],
            "datasets": ["None — no dataset accessed"],
            "escalation": "Escalation recommended: contact your manager or the privacy/security team through approved channels.",
            "safety": "Protected — request declined; no sensitive data was retrieved or displayed.",
        }

    if any(term in text for term in ("stock", "out of stock", "availability", "inventory")):
        return {
            "answer": "Check the approved inventory tool for nearby-store and distribution-center availability. If none is available, share the expected replenishment date or offer an approved alternative. Confirm the item and location before making a commitment.",
            "category": "Inventory & availability",
            "priority": "Normal",
            "confidence": "91%",
            "citations": ["Mock guide: Store inventory lookup and replenishment workflow."],
            "datasets": ["Mock inventory guide (illustrative only)"],
            "escalation": "No escalation indicated. Escalate to inventory support if counts appear inaccurate.",
            "safety": "Clear — no restricted information included.",
        }

    return {
        "answer": "Check that the printer is powered on, has paper, and is securely connected. Restart the printer and POS terminal using your store’s approved procedure. If the error continues, record the terminal ID and contact your store’s POS support channel.",
        "category": "Point of sale (POS)",
        "priority": "Normal",
        "confidence": "90%",
        "citations": ["Mock guide: POS printer first-line troubleshooting."],
        "datasets": ["Mock POS troubleshooting guide (illustrative only)"],
        "escalation": "Escalate to POS support if the issue remains after the approved checks.",
        "safety": "Clear — no restricted information included.",
    }


st.title("AI-Powered Retail Operations Copilot")
st.write(
    "Get practical guidance for day-to-day store operations. Ask a question below "
    "and review the response details and sources."
)

st.subheader("Try an example")
st.selectbox(
    "Choose a question to place in the input",
    options=["Select an example…", *EXAMPLES.keys()],
    key="example_question",
    on_change=select_example,
    label_visibility="collapsed",
)

with st.form("question_form", clear_on_submit=False):
    question = st.text_area(
        "Employee question",
        key="employee_question",
        height=130,
        placeholder="For example: The POS printer is unavailable. What should I check?",
        help="Avoid entering passwords, payment details, or personal customer information.",
    )
    submitted = st.form_submit_button("Get guidance", type="primary")

if submitted:
    if not question.strip():
        st.warning("Enter a question before requesting guidance.")
    else:
        with st.spinner("Preparing guidance…"):
            # Mock-only response. No question is sent to a service or echoed below.
            response = get_mock_response(question)
        st.session_state["copilot_response"] = response

response = st.session_state.get("copilot_response")
if response:
    st.divider()
    st.caption("DEMO MODE · MOCK RESPONSE — illustrative content; no backend or live datasets are connected.")

    st.header("Answer")
    st.info(response["answer"])

    st.subheader("Response details")
    col_category, col_priority, col_confidence = st.columns(3)
    col_category.metric("Category", response["category"])
    col_priority.metric("Priority", response["priority"])
    col_confidence.metric("Confidence", response["confidence"])

    col_citations, col_datasets = st.columns(2)
    with col_citations:
        st.subheader("Citations")
        for citation in response["citations"]:
            st.markdown(f"- {citation}")
    with col_datasets:
        st.subheader("Datasets used")
        for dataset in response["datasets"]:
            st.markdown(f"- {dataset}")

    st.subheader("Escalation status")
    st.write(response["escalation"])
    st.subheader("Safety status")
    st.write(response["safety"])
