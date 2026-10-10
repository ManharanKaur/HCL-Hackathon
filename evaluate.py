"""Run the public retail cases and save predictions in the supplied schema.

Run from the project root with:

    python evaluate.py

The script calls the same orchestrator entry point selected by app.py. It may
need working model and embedding service credentials to return live answers.
It writes predictions.csv and prints only metrics supported by public labels.
"""

import csv
from pathlib import Path

import pandas as pd

from labels_and_citations import citations_for_markdown
from reference_fields import parse_reference_field


ROOT = Path(__file__).resolve().parent
EVALUATION_DIR = ROOT / "evaluation"
CASES_PATH = EVALUATION_DIR / "public_test_cases.csv"
SCHEMA_PATH = EVALUATION_DIR / "predictions_schema.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "predictions.csv"
DATA_DIR = ROOT / "data"
POLICY_DIR = ROOT / "knowledge_base"


def read_schema_columns() -> list[str]:
    """Read the prediction field names in their supplied order."""
    with SCHEMA_PATH.open("r", encoding="utf-8-sig", newline="") as schema_file:
        schema_rows = csv.DictReader(schema_file)
        return [row["column_name"] for row in schema_rows]


def read_real_policy_sections() -> set[str]:
    """Build allowed citations from headings that actually exist on disk."""
    real_sections = set()

    for policy_file in POLICY_DIR.glob("*.md"):
        markdown = policy_file.read_text(encoding="utf-8")
        real_sections.update(citations_for_markdown(policy_file.name, markdown))
    return real_sections


def load_processor():
    """Resolve the same public function as app.py, with its compatibility path."""
    import orchestrator

    # Newer versions may export process_question. The current app falls back to
    # run_copilot when that name is not present; keep this runner in sync.
    process = getattr(orchestrator, "process_question", None)
    if not callable(process):
        process = getattr(orchestrator, "run_copilot", None)
    if not callable(process):
        raise RuntimeError("No supported question-processing function was found.")
    return process


def safe_error_row(test_id: str, columns: list[str]) -> dict:
    """Create a schema-shaped row without exposing exception details or input."""
    safe_defaults = {
        "test_id": test_id,
        "category": "Other or Security",
        "priority": "P4",
        "confidence": 0.0,
        "answer": "The copilot could not complete this case because a backend service was unavailable.",
        "citations": "",
        "data_used": "",
        "escalated": "Yes",
        "escalation_team": "",
        "pii_masked": "No",
        "safety_refusal": "No",
        "notes": "Backend or API error; result requires human review.",
    }
    return {column: safe_defaults.get(column, "") for column in columns}


def validate_data_sources(value: object) -> tuple[str, int, int]:
    """Keep only data source names that refer to existing local CSV files."""
    known_files = {path.name for path in DATA_DIR.glob("*.csv")}
    reported = parse_reference_field(value)
    valid = [name for name in reported if Path(name).name == name and name in known_files]
    return ";".join(dict.fromkeys(valid)), len(valid), len(reported)


def calculate_reference_facts() -> dict[str, object]:
    """Independently derive a few public-case facts from the local CSV snapshots.

    These checks establish what the supplied data supports. They do not assume
    that an answer is correct just because it mentions the same case.
    """
    facts: dict[str, object] = {}
    try:
        products = pd.read_csv(DATA_DIR / "product_master.csv", dtype=str)
        inventory = pd.read_csv(DATA_DIR / "inventory.csv", dtype=str)
        receipts = pd.read_csv(DATA_DIR / "receipts.csv", dtype=str)
        support = pd.read_csv(DATA_DIR / "support_cases.csv", dtype=str)
        returns = pd.read_csv(DATA_DIR / "returns_quality.csv", dtype=str)
        vendors = pd.read_csv(DATA_DIR / "vendor_master.csv", dtype=str)

        # English Willow Cricket Bat / ST002 (PUB-003).
        cricket_row = products[products["name"].eq("English Willow Cricket Bat")]
        if not cricket_row.empty:
            cricket_sku = cricket_row.iloc[0]["sku"]
            cricket_stock = inventory[
                inventory["sku"].eq(cricket_sku) & inventory["store_id"].eq("ST002")
            ]
            cricket_receipts = receipts[
                receipts["sku"].eq(cricket_sku) & receipts["store_id"].eq("ST002")
            ]
            if not cricket_receipts.empty:
                latest_receipt = cricket_receipts.sort_values("receipt_date", ascending=False).iloc[0]
                ordered = int(latest_receipt["po_qty"])
                received = int(latest_receipt["received_qty"])
                facts["PUB-003"] = {
                    "sku": cricket_sku,
                    "store_stock": int(cricket_stock.iloc[0]["on_hand"]) if not cricket_stock.empty else None,
                    "latest_ordered": ordered,
                    "latest_received": received,
                    "latest_fill_percent": round(received / ordered * 100, 1) if ordered else None,
                }

        # The fixed evaluation anchor is 2026-10-10 12:00 local time (PUB-007).
        created = pd.to_datetime(support["created_ts"], errors="coerce")
        window_start = pd.Timestamp("2026-10-09 12:00:00")
        window_end = pd.Timestamp("2026-10-10 12:00:00")
        recent = support[created.between(window_start, window_end, inclusive="both")].copy()
        unresolved = recent[
            recent["priority_label"].isin(["P1", "P2"])
            & recent["status"].str.casefold().isin(["open", "in progress", "pending"])
        ]
        facts["PUB-007"] = {
            "case_count": int(len(unresolved)),
            "by_store": unresolved.groupby("store_id").size().astype(int).to_dict(),
            "by_store_and_category": unresolved.groupby(["store_id", "category_label"]).size().astype(int).to_dict(),
        }

        # Sports quality complaint vendor mapping for the last 30 days ending
        # at the supplied fixed evaluation time.
        thirty_day_start = pd.Timestamp("2026-09-10 12:00:00")
        fixed_end = pd.Timestamp("2026-10-10 12:00:00")
        support_dates = pd.to_datetime(support["created_ts"], errors="coerce")
        support = support[support_dates.between(thirty_day_start, fixed_end, inclusive="both")]
        return_dates = pd.to_datetime(returns["created_ts"], errors="coerce")
        returns = returns[return_dates.between(thirty_day_start, fixed_end, inclusive="both")]
        quality = support[
            support["category_label"].eq("Product Quality and Safety")
        ]
        quality = quality.merge(products[["sku", "vendor_id"]], on="sku", how="left")
        quality_returns = returns[
            returns["reason"].str.contains(
                r"quality|seam|split|crack|slipped|malfunction|unsafe|damage|defect|broken|warped|torn|lost air|faulty",
                case=False,
                regex=True,
                na=False,
            )
        ].merge(products[["sku", "vendor_id"]], on="sku", how="left")
        linked_quality = pd.concat(
            [quality[["vendor_id"]], quality_returns[["vendor_id"]]],
            ignore_index=True,
        )
        linked_quality = linked_quality[linked_quality["vendor_id"].notna()]
        complaint_counts = linked_quality.groupby("vendor_id").size().sort_values(ascending=False)
        facts["PUB-010"] = {
            "complaint_counts": complaint_counts.astype(int).to_dict(),
            "vendor_names": vendors.set_index("vendor_id")["name"].to_dict(),
        }

        # Tennis Ball Pack store and warehouse availability / receipt (PUB-011).
        tennis = products[products["name"].eq("Tennis Ball Pack")]
        if not tennis.empty:
            tennis_sku = tennis.iloc[0]["sku"]
            tennis_stock = inventory[
                inventory["sku"].eq(tennis_sku)
                & inventory["store_id"].isin(["ST005", "WH001"])
            ]
            warehouse_receipt = receipts[
                receipts["sku"].eq(tennis_sku) & receipts["store_id"].eq("WH001")
            ]
            facts["PUB-011"] = {
                "sku": tennis_sku,
                "stock_by_location": tennis_stock.set_index("store_id")["on_hand"].astype(int).to_dict(),
                "warehouse_receipt_qty": (
                    int(warehouse_receipt.sort_values("receipt_date", ascending=False).iloc[0]["received_qty"])
                    if not warehouse_receipt.empty else None
                ),
            }
    except (OSError, KeyError, ValueError, pd.errors.ParserError):
        # Missing or malformed snapshots mean these optional checks are skipped.
        return {}
    return facts


def check_local_data_claims(case_id: str, answer: str, reference_facts: dict[str, object]) -> str:
    """Check explicit, simple data claims against independently read CSV facts."""
    lowered = answer.casefold()
    case_facts = reference_facts.get(case_id)
    if not isinstance(case_facts, dict):
        return "Not checked (required local CSV facts were unavailable)."

    if case_id == "PUB-003":
        received = case_facts.get("latest_received")
        ordered = case_facts.get("latest_ordered")
        if isinstance(received, int) and isinstance(ordered, int):
            if "short shipment" in lowered or "shortfall" in lowered or "received" in lowered:
                if received < ordered and str(received) in answer and str(ordered) in answer:
                    return "Verified: receipt quantities match the local receipts CSV."
                return "Not verified: the explicit receipt quantities do not match the local CSV facts."
            if "fill rate" in lowered or "fill" in lowered:
                percent = case_facts.get("latest_fill_percent")
                if percent is not None and str(int(percent)) in answer:
                    return "Verified: latest receipt fill percentage matches local CSV calculation."
                return "Not verified: the stated fill percentage does not match the local calculation."

    if case_id == "PUB-011":
        stocks = case_facts.get("stock_by_location", {})
        if "stock" in lowered or "zero" in lowered or "warehouse" in lowered:
            if isinstance(stocks, dict) and stocks.get("ST005") == 0 and stocks.get("WH001") == 0:
                if "0" in answer or "zero" in lowered or "both" in lowered:
                    return "Verified: store and warehouse stock are zero in the local inventory CSV."
                return "Not verified: the answer discusses stock but omits the local zero-stock finding."

    if case_id == "PUB-007":
        expected_count = case_facts.get("case_count")
        if any(word in lowered for word in ("incident", "case", "p1", "p2", "last 24 hours")):
            # Count claims are checked only when the answer states the exact
            # independently calculated total. Otherwise mark as not verified.
            if isinstance(expected_count, int) and str(expected_count) in answer:
                return "Verified: stated total matches the fixed-window local CSV calculation."
            return "Not verified: no matching fixed-window total could be confirmed in the answer."

    if case_id == "PUB-010":
        counts = case_facts.get("complaint_counts", {})
        vendor_names = case_facts.get("vendor_names", {})
        if isinstance(counts, dict) and counts and isinstance(vendor_names, dict):
            highest_vendor = max(counts, key=counts.get)
            expected_name = vendor_names.get(highest_vendor, highest_vendor)
            named_vendors = [
                name for name in vendor_names.values()
                if isinstance(name, str) and name and name.casefold() in lowered
            ]
            if not named_vendors:
                return "Not checked (the answer did not name a vendor)."
            if expected_name in named_vendors:
                return "Verified: the named vendor occurs in the locally calculated complaint ranking."
            return "Not verified: the named vendor does not match the local complaint ranking."

    return "Not checked (answer did not make a directly comparable claim)."


def main() -> None:
    """Run every public case, save schema-ordered predictions, and report metrics."""
    if not CASES_PATH.is_file() or not SCHEMA_PATH.is_file():
        raise FileNotFoundError("Public cases or prediction schema file is missing.")

    columns = read_schema_columns()
    if not columns:
        raise ValueError("The prediction schema contains no columns.")

    with CASES_PATH.open("r", encoding="utf-8-sig", newline="") as cases_file:
        cases = list(csv.DictReader(cases_file))
    if not cases:
        raise ValueError("No public test cases were found.")

    try:
        process_question = load_processor()
    except Exception:
        # Keep one output row per case even when imports or services are down.
        process_question = None

    allowed_citations = read_real_policy_sections()
    local_reference_facts = calculate_reference_facts()
    prediction_rows = []
    citation_checks = []
    valid_source_count = 0
    reported_source_count = 0
    data_answer_checks = []

    for case in cases:
        case_id = case.get("test_id", "")
        if process_question is None:
            raw_result = safe_error_row(case_id, columns)
        else:
            try:
                raw_result = process_question(case.get("prompt", ""))
                if not isinstance(raw_result, dict):
                    raw_result = safe_error_row(case_id, columns)
            except Exception:
                # Do not put input text, API keys, or tracebacks in output files.
                raw_result = safe_error_row(case_id, columns)

        row = {column: raw_result.get(column, "") for column in columns}
        # The supplied test ID is authoritative for matching expected labels.
        if "test_id" in row:
            row["test_id"] = case_id

        reported_citations = parse_reference_field(row.get("citations", ""))
        valid_citations = [citation for citation in reported_citations if citation in allowed_citations]
        citation_checks.append((case, set(valid_citations)))
        row["citations"] = ";".join(dict.fromkeys(valid_citations))

        clean_sources, valid_count, reported_count = validate_data_sources(row.get("data_used", ""))
        row["data_used"] = clean_sources
        valid_source_count += valid_count
        reported_source_count += reported_count
        data_answer_checks.append(
            check_local_data_claims(case_id, str(row.get("answer", "")), local_reference_facts)
        )
        prediction_rows.append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # DictWriter receives the exact column order read from the schema file.
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(prediction_rows)

    # Only category and priority have explicit expected labels in the cases.
    category_matches = [
        row.get("category") == case.get("expected_category")
        for row, case in zip(prediction_rows, cases)
        if case.get("expected_category")
    ]
    priority_matches = [
        row.get("priority") == case.get("expected_priority")
        for row, case in zip(prediction_rows, cases)
        if case.get("expected_priority")
    ]

    expected_source_checks = []
    for case, citations in citation_checks:
        expected_sources = set(parse_reference_field(case.get("expected_sources", "")))
        if expected_sources:
            expected_source_checks.append(expected_sources.issubset(citations))

    print(f"Wrote {len(prediction_rows)} predictions to {OUTPUT_PATH}")
    if category_matches:
        print(f"Category accuracy: {sum(category_matches)}/{len(category_matches)} ({sum(category_matches) / len(category_matches):.1%})")
    if priority_matches:
        print(f"Priority accuracy: {sum(priority_matches)}/{len(priority_matches)} ({sum(priority_matches) / len(priority_matches):.1%})")
    if expected_source_checks:
        print(f"Complete expected-source coverage: {sum(expected_source_checks)}/{len(expected_source_checks)} ({sum(expected_source_checks) / len(expected_source_checks):.1%})")
    if reported_source_count:
        print(f"Reported data-source filenames validated: {valid_source_count}/{reported_source_count}")
    else:
        print("Reported data-source filenames validated: none were reported")

    checked_answers = [item for item in data_answer_checks if not item.startswith("Not checked")]
    if checked_answers:
        verified = sum(item.startswith("Verified") for item in checked_answers)
        print(f"Direct data-claim checks: {verified}/{len(checked_answers)} verified; see check detail below")
        for case, result in zip(cases, data_answer_checks):
            if not result.startswith("Not checked"):
                print(f"  {case.get('test_id')}: {result}")
    else:
        print("Direct data-claim checks: no comparable numeric or named claims were made in returned answers.")


if __name__ == "__main__":
    main()
