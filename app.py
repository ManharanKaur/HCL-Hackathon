"""Streamlit interface for the AI-Powered Retail Operations Copilot."""

import streamlit as st

from reference_fields import parse_reference_field

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
    """Copy the selected example into the form without submitting it."""
    selected = st.session_state.get("example_question")
    if selected in EXAMPLES:
        st.session_state["employee_question"] = EXAMPLES[selected]


def get_response(text: str) -> dict:
    """Call the orchestrator entry point, preferring its public UI method."""
    import orchestrator

    # process_question is the requested UI entry point. Older project versions
    # expose the same pipeline as run_copilot, so use it when needed.
    process = getattr(orchestrator, "process_question", None)
    if not callable(process):
        process = getattr(orchestrator, "run_copilot", None)
    if not callable(process):
        raise RuntimeError("The orchestrator entry point is unavailable.")

    result = process(text)
    if not isinstance(result, dict):
        raise RuntimeError("The orchestrator returned an unexpected result.")
    return result


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
        try:
            with st.spinner("Preparing guidance…"):
                # The backend safety module masks or refuses sensitive requests.
                st.session_state["copilot_response"] = get_response(question.strip())
            st.session_state.pop("copilot_error", None)
        except Exception:
            # Do not show exception text: it can contain internal details or input.
            st.session_state.pop("copilot_response", None)
            st.session_state["copilot_error"] = (
                "Guidance is temporarily unavailable. The backend or an AI service "
                "may be unavailable; please try again later."
            )

if st.session_state.get("copilot_error"):
    st.error(st.session_state["copilot_error"])

response = st.session_state.get("copilot_response")
if isinstance(response, dict):
    st.divider()
    st.header("Answer")
    answer = response.get("answer")
    st.info(answer if isinstance(answer, str) and answer.strip() else "No answer was returned.")

    st.subheader("Response details")
    col_category, col_priority, col_confidence = st.columns(3)
    col_category.metric("Category", response.get("category") or "Not provided")
    col_priority.metric("Priority", response.get("priority") or "Not provided")

    confidence = response.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        col_confidence.metric("Confidence", f"{confidence:.0%}")
    else:
        col_confidence.metric("Confidence", "Not provided")

    col_citations, col_datasets = st.columns(2)
    with col_citations:
        st.subheader("Citations")
        citations = parse_reference_field(response.get("citations"))
        if citations:
            for citation in citations:
                st.markdown(f"- {citation}")
        else:
            st.caption("No policy citations were returned.")

    with col_datasets:
        st.subheader("Datasets used")
        datasets = parse_reference_field(response.get("data_used"))
        if datasets:
            for dataset in datasets:
                st.markdown(f"- {dataset}")
        else:
            st.caption("No datasets were used or reported.")

    st.subheader("Escalation status")
    escalated = response.get("escalated") or "Not provided"
    team = response.get("escalation_team")
    escalation_text = str(escalated)
    if team:
        escalation_text += f" — {team}"
    st.write(escalation_text)

    st.subheader("Safety status")
    refusal = response.get("safety_refusal") or "Not provided"
    pii_masked = response.get("pii_masked") or "Not provided"
    st.write(f"Safety refusal: {refusal} · Personal data masked: {pii_masked}")

    notes = response.get("notes")
    if isinstance(notes, str) and notes.strip():
        st.caption(f"Notes: {notes}")
