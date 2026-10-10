# AI-Powered Retail Operations Copilot

A small Streamlit app that helps store employees with retail operations questions. It checks requests for sensitive information, classifies the case, retrieves policy passages, and uses local CSV snapshots for supported business questions.

## Setup

Use Python 3.10 or newer. From the project root, create and activate a virtual environment, then install the packages:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

On macOS or Linux, activate it with `source .venv/bin/activate`.

The spaCy model is downloaded once for Presidio's name recognition. If that model or Presidio cannot start, safety screening blocks requests instead of sending unscanned text to a model.

The app and policy search need an OpenRouter credential. Set the environment variable named `OPENROUTER_API_KEY`, or add it to a local `.env` file in the project root. The local `.env` file is ignored by Git. Never paste a real key into this README, source code, screenshots, or logs.

The LLM client uses Python's built-in `urllib` for HTTP requests. `certifi` is an optional certificate bundle used by the client when available. No separate HTTP or OpenAI SDK is required.

## Run the Streamlit app

From the project root, with the virtual environment active and the API key configured, run:

```powershell
streamlit run app.py
```

Choose an example or type an employee question, then select **Get guidance**. Examples only fill the input; they do not submit it.

## Build and search the policy index

The first call to `search_policy` reads the Markdown files in `knowledge_base/`, creates embeddings, and saves a persistent ChromaDB collection under `chroma_db/`. Later searches reuse the index and update sections whose source text changed. The embedding model is separate from the chat model.

Run this from the project root to build or refresh the index and search it:

```powershell
python -c "from knowledge_retriever import search_policy; print(search_policy('What should I check when one card reader fails?', category='POS and Payments', top_k=3))"
```

The index is local generated data and is excluded from Git. A successful search requires the API key because policy sections and each search query need embeddings.

## Run public-case evaluation

Run:

```powershell
python evaluate.py
```

The script reads `evaluation/public_test_cases.csv` and `evaluation/predictions_schema.csv`, calls the same orchestrator entry point selected by the Streamlit app, checks citations against real Markdown sections, and writes `outputs/predictions.csv` in schema column order. It reports category and priority accuracy and expected-source coverage because those expected labels are supplied. Some data statements are also checked against local CSV calculations when they can be compared directly. The output folder is excluded from Git.

The current orchestrator function is `run_copilot(question, test_id="")`. The app and evaluator prefer `process_question(question)` if a project version provides it, and otherwise use `run_copilot`.

## Main module interfaces

- `safety.check_request(question)` screens and masks the question before downstream processing.
- `triage.classify_request(question)` returns an allowed category, priority, confidence, and rationale.
- `knowledge_retriever.search_policy(query, category=None, top_k=3)` returns policy passages and citations.
- `data_agent.answer_data_question(question)` returns locally calculated facts and the CSV files used.
- `workflow.route_case(triage, safety, question="")` selects a matrix route and review flags.
- `response.build_response(triage, policy, data, routing, safety)` creates the prediction fields.
- `orchestrator.run_copilot(question, test_id="")` connects the steps and adds `test_id`.

See [interfaces.md](interfaces.md) for the dictionary fields and examples. The response reports simulated routing only; it does not execute refunds, inventory changes, or vendor actions.
