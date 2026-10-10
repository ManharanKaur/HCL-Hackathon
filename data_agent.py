"""Pandas-based analysis of the supplied, local retail CSV snapshots."""

import json
import re
from pathlib import Path

import pandas as pd

from llm_client import chat_completion
from diagnostics import log_failure
from labels_and_citations import PRIORITIES


DATA_DIR = Path(__file__).resolve().parent / "data"
# The supplied CSV timestamps are local/naive; this is the documented IST anchor.
EVALUATION_TIME = pd.Timestamp("2026-10-10 12:00:00")
ALLOWED_INTENTS = {
    "inventory_lookup",
    "stockout_root_cause",
    "recent_support_cases",
    "quality_complaints_by_vendor",
    "product_quality_by_sku",
    "unsupported",
}
ALLOWED_TIME_WINDOWS = {"last_24_hours", "last_30_days", "none"}
ALLOWED_STATUSES = {"unresolved", "open", "in progress", "closed", "all"}
ALLOWED_GROUP_FIELDS = {"store", "category", "status", "priority", "vendor", "sku", "reason"}

INTENT_FIELDS = {
    "intent",
    "store_id",
    "product_name",
    "sku",
    "time_window",
    "priorities",
    "status",
    "group_by",
}


def _empty_result(answer: str, notes: str) -> dict:
    """Create the same return shape for safe refusals and unsupported questions."""
    return {"answer": answer, "facts": [], "data_used": [], "notes": notes}


def _extract_query(question: str) -> dict:
    """Ask the model to label intent and extract filters, never to do analysis."""
    messages = [
        {
            "role": "system",
            "content": (
                "Identify the supported retail CSV query intent and extract only filter values from the question. "
                "Return only one JSON object with exactly these fields: intent, store_id, product_name, sku, "
                "time_window, priorities, status, group_by. Do not calculate, write SQL or Python, or request "
                "data from any file. Allowed intent values: inventory_lookup, stockout_root_cause, "
                "recent_support_cases, quality_complaints_by_vendor, product_quality_by_sku, unsupported. "
                "Use stockout_root_cause for a product availability/root-cause question, inventory_lookup for a direct stock check, "
                "recent_support_cases for incident counts or filters, quality_complaints_by_vendor for a vendor trend, "
                "and product_quality_by_sku for a product quality issue. Use unsupported when the CSVs cannot answer it. "
                "Use null for unknown store_id, product_name, sku, status; use time_window only as "
                "last_24_hours, last_30_days, or none. priorities must be a list using P1, P2, P3, P4, "
                "or an empty list. group_by must be a list using store, category, status, priority, vendor, "
                "sku, reason, or an empty list."
            ),
        },
        {"role": "user", "content": question},
    ]
    # No CSV rows or employee/customer fields are included in this call.
    response = chat_completion(messages, temperature=0.0)
    content = response.get("content", "")
    if not isinstance(content, str):
        raise ValueError("Intent response was not text.")

    try:
        result = json.loads(content.strip())
    except json.JSONDecodeError as error:
        raise ValueError("Intent response was not valid JSON.") from error

    if not isinstance(result, dict) or set(result) != INTENT_FIELDS:
        raise ValueError("Intent response did not contain the required fields.")
    if result["intent"] not in ALLOWED_INTENTS:
        raise ValueError("Intent is outside the supported query allowlist.")

    store_id = result["store_id"]
    if store_id is not None:
        if not isinstance(store_id, str) or not re.fullmatch(r"(?:ST|WH)\d{3}", store_id.upper()):
            raise ValueError("Store identifier is not in the supported format.")
        result["store_id"] = store_id.upper()

    for value_name in ("product_name", "sku"):
        value = result[value_name]
        if value is not None and (not isinstance(value, str) or len(value) > 120):
            raise ValueError(f"{value_name} must be a short string or null.")
        if value_name == "sku" and value is not None and not re.fullmatch(r"SKU\d{3,5}", value.upper()):
            raise ValueError("SKU is not in the supported format.")
        if value_name == "sku" and value is not None:
            result[value_name] = value.upper()

    if result["time_window"] not in ALLOWED_TIME_WINDOWS:
        raise ValueError("Time window is not supported.")
    if not isinstance(result["priorities"], list) or any(
        priority not in PRIORITIES for priority in result["priorities"]
    ):
        raise ValueError("Priority filters must use the supplied P1-P4 labels.")
    if result["status"] is not None and (
        not isinstance(result["status"], str)
        or result["status"].casefold() not in ALLOWED_STATUSES
    ):
        raise ValueError("Status filter is not supported.")
    if not isinstance(result["group_by"], list) or any(
        field not in ALLOWED_GROUP_FIELDS for field in result["group_by"]
    ):
        raise ValueError("Grouping field is not supported.")

    if result["status"] is not None:
        result["status"] = result["status"].casefold()
    return result


def _read_csv(filename: str, data_used: list[str]) -> pd.DataFrame:
    """Read a known project CSV, preserving IDs and dates as text at first."""
    path = DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(filename)
    if filename not in data_used:
        data_used.append(filename)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _resolve_products(products: pd.DataFrame, sku: str | None, name: str | None) -> pd.DataFrame:
    """Find product rows using an exact SKU/name first, then a safe name match."""
    if sku:
        return products[products["sku"].str.casefold() == sku.casefold()]
    if not name:
        return products.iloc[0:0]

    clean_name = name.strip().casefold()
    exact_match = products[products["name"].str.casefold() == clean_name]
    if not exact_match.empty:
        return exact_match
    return products[products["name"].str.casefold().str.contains(re.escape(clean_name), na=False)]


def _time_window_bounds(time_window: str) -> tuple[pd.Timestamp | None, pd.Timestamp]:
    """Return start and fixed end timestamps for the evaluation window."""
    if time_window == "last_24_hours":
        return EVALUATION_TIME - pd.Timedelta(hours=24), EVALUATION_TIME
    if time_window == "last_30_days":
        return EVALUATION_TIME - pd.Timedelta(days=30), EVALUATION_TIME
    return None, EVALUATION_TIME


def _format_percent(value: str) -> str:
    """Format a numeric ratio stored in a CSV as a readable percentage."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").mul(100).iloc[0]
    return f"{number:.0f}%" if pd.notna(number) else "not available"


def _inventory_answer(filters: dict) -> dict:
    """Compare inventory with reorder levels and inspect related receipts."""
    data_used = []
    products = _read_csv("product_master.csv", data_used)
    matched_products = _resolve_products(products, filters["sku"], filters["product_name"])
    if matched_products.empty:
        return {
            "answer": "I could not match that product to the supplied product list.",
            "facts": [],
            "data_used": data_used,
            "notes": "No matching SKU or product name was found in product_master.csv.",
        }
    if len(matched_products) > 1:
        options = ", ".join(matched_products["name"].head(5).tolist())
        return {
            "answer": "I found more than one possible product. Please specify the exact product or SKU.",
            "facts": [],
            "data_used": data_used,
            "notes": f"Possible matches include: {options}.",
        }

    product = matched_products.iloc[0]
    sku = product["sku"]
    inventory = _read_csv("inventory.csv", data_used)
    inventory = inventory[inventory["sku"].eq(sku)].copy()
    if filters["store_id"]:
        relevant_locations = [filters["store_id"]]
        if filters["intent"] == "stockout_root_cause":
            # The public academy-order scenario also depends on warehouse stock.
            relevant_locations.append("WH001")
        inventory = inventory[inventory["store_id"].isin(relevant_locations)]

    for column in ("on_hand", "system_qty", "reorder_point"):
        inventory[column] = pd.to_numeric(inventory[column], errors="coerce")

    if inventory.empty:
        return {
            "answer": f"No inventory row was found for {product['name']} ({sku}) at the requested location.",
            "facts": [f"sku={sku}"],
            "data_used": data_used,
            "notes": "The product exists in product_master.csv, but the requested location has no inventory record.",
        }

    receipts = _read_csv("receipts.csv", data_used)
    receipts = receipts[receipts["sku"].eq(sku)].copy()
    if filters["store_id"]:
        # Include the requested location and warehouse receipts that could
        # explain a warehouse-depleted stock-out.
        receipts = receipts[receipts["store_id"].isin([filters["store_id"], "WH001"])]
    receipts["receipt_date_parsed"] = pd.to_datetime(receipts["receipt_date"], errors="coerce")
    ordered_quantities = pd.to_numeric(receipts["po_qty"], errors="coerce")
    receipts["completion_rate"] = (
        pd.to_numeric(receipts["received_qty"], errors="coerce")
        / ordered_quantities.where(ordered_quantities.ne(0))
    )
    receipts = receipts.sort_values("receipt_date_parsed", ascending=False)

    vendor_id = product["vendor_id"]
    vendors = _read_csv("vendor_master.csv", data_used)
    vendor = vendors[vendors["vendor_id"].eq(vendor_id)]

    # Count matching cases only; never return employee IDs or raw case text.
    cases = _read_csv("support_cases.csv", data_used)
    matching_cases = cases[cases["sku"].eq(sku)]
    if filters["store_id"]:
        matching_cases = matching_cases[matching_cases["store_id"].eq(filters["store_id"])]

    facts = [f"product={product['name']}", f"sku={sku}", f"high_demand={product['high_demand']}"]
    inventory_locations = []
    for _, row in inventory.iterrows():
        location_facts = (
            f"{row['store_id']}: on_hand={row['on_hand']}, system_qty={row['system_qty']}, "
            f"reorder_point={row['reorder_point']}"
        )
        inventory_locations.append(location_facts)
        facts.append(location_facts)

    receipt_lines = []
    for _, receipt in receipts.head(5).iterrows():
        completion_rate = receipt["completion_rate"]
        completion = f"{completion_rate:.0%}" if pd.notna(completion_rate) else "not available"
        receipt_lines.append(
            f"{receipt['store_id']} receipt on {receipt['receipt_date']}: "
            f"{receipt['received_qty']} received of {receipt['po_qty']} ordered ({completion})"
        )
        if receipt["damage_notes"]:
            receipt_lines[-1] += f"; note: {receipt['damage_notes']}"
        facts.append(receipt_lines[-1])

    vendor_summary = ""
    if not vendor.empty:
        vendor_row = vendor.iloc[0]
        vendor_summary = (
            f" Vendor {vendor_row['name']} has a { _format_percent(vendor_row['fill_rate']) } fill rate "
            f"and { _format_percent(vendor_row['on_time_rate']) } on-time rate."
        )
        facts.append(f"vendor={vendor_row['name']} ({vendor_id})")
        facts.append(f"vendor_fill_rate={_format_percent(vendor_row['fill_rate'])}")
        facts.append(f"vendor_on_time_rate={_format_percent(vendor_row['on_time_rate'])}")

    if not matching_cases.empty:
        facts.append(f"matching_support_cases={matching_cases.shape[0]}")

    location_label = f" at {filters['store_id']}" if filters["store_id"] else " across listed locations"
    shown_inventory = inventory_locations[:5]
    answer = f"Inventory for {product['name']}{location_label}: " + "; ".join(shown_inventory)
    if len(inventory_locations) > len(shown_inventory):
        answer += f"; plus {len(inventory_locations) - len(shown_inventory)} other listed locations"
    answer += "."
    if receipt_lines:
        answer += " Recent receipt evidence: " + "; ".join(receipt_lines[:2]) + "."
    answer += vendor_summary

    notes = ""
    zero_locations = inventory[inventory["on_hand"].eq(0)]
    if len(zero_locations) > 1 and "WH001" in zero_locations["store_id"].values:
        notes = "Both the requested location and WH001 have zero on hand; the warehouse receipt notes should be reviewed."
    elif (
        pd.to_numeric(inventory["on_hand"], errors="coerce")
        .le(pd.to_numeric(inventory["reorder_point"], errors="coerce"))
        .any()
    ):
        notes = "At least one listed location is at or below its reorder point."

    return {"answer": answer, "facts": facts, "data_used": data_used, "notes": notes}


def _support_cases_answer(filters: dict) -> dict:
    """Filter support incidents locally and return grouped counts only."""
    data_used = []
    cases = _read_csv("support_cases.csv", data_used)
    cases["created_ts_parsed"] = pd.to_datetime(cases["created_ts"], errors="coerce")

    start, end = _time_window_bounds(filters["time_window"])
    if start is not None:
        cases = cases[cases["created_ts_parsed"].between(start, end, inclusive="both")]
    else:
        cases = cases[cases["created_ts_parsed"].le(end)]
    if filters["priorities"]:
        cases = cases[cases["priority_label"].isin(filters["priorities"])]
    if filters["store_id"]:
        cases = cases[cases["store_id"].eq(filters["store_id"])]

    status = filters["status"] or "unresolved"
    if status == "unresolved":
        cases = cases[cases["status"].str.casefold().isin(["open", "in progress", "unresolved"])]
    elif status != "all":
        cases = cases[cases["status"].str.casefold().eq(status)]

    if cases.empty:
        return {
            "answer": "No support cases matched the requested filters in the supplied snapshot.",
            "facts": ["matching_case_count=0"],
            "data_used": data_used,
            "notes": "The snapshot may not include the requested period or matching incidents.",
        }

    # Aggregate by safe business fields. Do not expose employee IDs or issue text.
    grouped = (
        cases.groupby(["store_id", "category_label", "status"], dropna=False)
        .size()
        .reset_index(name="case_count")
        .sort_values(["store_id", "category_label", "status"])
    )
    facts = [
        f"store={row.store_id}, category={row.category_label}, status={row.status}, count={row.case_count}"
        for row in grouped.itertuples(index=False)
    ]
    start_label = start.strftime("%Y-%m-%d %H:%M") if start is not None else "all available dates"
    answer = (
        f"Found {cases.shape[0]} matching support cases from {start_label} through "
        f"{end.strftime('%Y-%m-%d %H:%M')} (evaluation time). Counts are grouped by store, category, and status."
    )
    notes = "The time range uses the fixed evaluation timestamp 2026-10-10 12:00 IST."
    return {"answer": answer, "facts": facts, "data_used": data_used, "notes": notes}


def _quality_complaints_answer(filters: dict, by_vendor: bool) -> dict:
    """Join quality records to products and vendors, then calculate counts."""
    data_used = []
    cases = _read_csv("support_cases.csv", data_used)
    returns = _read_csv("returns_quality.csv", data_used)
    products = _read_csv("product_master.csv", data_used)
    vendors = _read_csv("vendor_master.csv", data_used)

    selected_window = filters["time_window"]
    if by_vendor and selected_window == "none":
        selected_window = "last_30_days"
    start, end = _time_window_bounds(selected_window)
    if start is not None:
        cases["created_ts_parsed"] = pd.to_datetime(cases["created_ts"], errors="coerce")
        cases = cases[cases["created_ts_parsed"].between(start, end, inclusive="both")]
        returns["created_ts_parsed"] = pd.to_datetime(returns["created_ts"], errors="coerce")
        returns = returns[returns["created_ts_parsed"].between(start, end, inclusive="both")]
    else:
        cases["created_ts_parsed"] = pd.to_datetime(cases["created_ts"], errors="coerce")
        cases = cases[cases["created_ts_parsed"].le(end)]
        returns["created_ts_parsed"] = pd.to_datetime(returns["created_ts"], errors="coerce")
        returns = returns[returns["created_ts_parsed"].le(end)]

    quality_cases = cases[cases["category_label"].eq("Product Quality and Safety")].copy()
    quality_returns = returns[
        returns["reason"].str.contains(
            r"quality|seam|split|crack|slipped|malfunction|unsafe|damage|defect|broken|warped|torn|lost air|faulty",
            case=False,
            regex=True,
            na=False,
        )
    ].copy()

    if filters["sku"] or filters["product_name"]:
        matched_products = _resolve_products(products, filters["sku"], filters["product_name"])
        if matched_products.empty:
            return {
                "answer": "I could not match that product to the supplied product list.",
                "facts": [],
                "data_used": data_used,
                "notes": "No matching SKU or product name was found in product_master.csv.",
            }
        target_skus = set(matched_products["sku"])
        quality_cases = quality_cases[quality_cases["sku"].isin(target_skus)]
        quality_returns = quality_returns[quality_returns["sku"].isin(target_skus)]

    if filters["store_id"]:
        quality_cases = quality_cases[quality_cases["store_id"].eq(filters["store_id"])]
        quality_returns = quality_returns[quality_returns["store_id"].eq(filters["store_id"])]

    case_products = quality_cases.merge(products[["sku", "name", "vendor_id"]], on="sku", how="left")
    return_products = quality_returns.merge(products[["sku", "name", "vendor_id"]], on="sku", how="left")

    if by_vendor:
        combined = pd.concat(
            [
                case_products[["vendor_id"]].assign(source="support_cases.csv"),
                return_products[["vendor_id"]].assign(source="returns_quality.csv"),
            ],
            ignore_index=True,
        )
        combined = combined[combined["vendor_id"].ne("")]
        if combined.empty:
            return {
                "answer": "No matching quality records could be linked to a vendor in the supplied snapshot.",
                "facts": [],
                "data_used": data_used,
                "notes": "A product-to-vendor match was not available for the selected period.",
            }

        counts = combined.groupby("vendor_id").size().rename("record_count").reset_index()
        counts = counts.merge(vendors[["vendor_id", "name", "fill_rate", "on_time_rate"]], on="vendor_id", how="left")
        counts = counts.sort_values(["record_count", "vendor_id"], ascending=[False, True])
        facts = []
        for row in counts.itertuples(index=False):
            support_count = int(combined[(combined["vendor_id"] == row.vendor_id) & (combined["source"] == "support_cases.csv")].shape[0])
            return_count = int(combined[(combined["vendor_id"] == row.vendor_id) & (combined["source"] == "returns_quality.csv")].shape[0])
            facts.append(
                f"vendor={row.name or row.vendor_id} ({row.vendor_id}), total_records={row.record_count}, "
                f"support_cases={support_count}, returns_quality={return_count}, "
                f"fill_rate={_format_percent(row.fill_rate)}, on_time_rate={_format_percent(row.on_time_rate)}"
            )

        top_vendor = counts.iloc[0]
        answer = (
            f"The highest linked count in this snapshot is {top_vendor['name']} ({top_vendor['vendor_id']}) "
            f"with {int(top_vendor['record_count'])} quality records in the selected window. "
            "This is a signal for review, not proof that the vendor caused the complaints."
        )
        notes = (
            "Counts combine support-case and returns/quality rows; the files do not share a common incident ID, "
            "so potentially related rows cannot be deduplicated."
        )
    else:
        facts = [f"quality_support_cases={case_products.shape[0]}", f"quality_returns={return_products.shape[0]}"]
        answer = (
            f"The selected window contains {case_products.shape[0]} product-quality support cases and "
            f"{return_products.shape[0]} matching returns/quality records."
        )
        notes = "Only records with a product-quality category or quality-related return reason were counted."

    return {"answer": answer, "facts": facts, "data_used": data_used, "notes": notes}


class RetailDataAgent:
    """Retail data agent performing pandas analysis over local CSVs.

    Adheres to Single Responsibility Principle: executes data analytics only.
    Safety screening is handled exclusively by the SafetyGate.
    """

    def query(self, question: str) -> dict:
        """Run allowlisted intent extraction and pandas computation on CSVs."""
        safe_text = question.strip() if isinstance(question, str) else ""
        if not safe_text:
            return _empty_result("No question provided.", "Empty input to data agent.")

        try:
            filters = _extract_query(safe_text)
        except Exception as error:
            log_failure("data_agent.intent_generation", error)
            return _empty_result(
                "I could not safely identify a supported data query.",
                "The intent response was unavailable or invalid, so no generated query was run and no CSV was analyzed.",
            )

        intent = filters["intent"]
        if intent == "unsupported":
            return _empty_result(
                "The supplied CSV files do not support this question as a data lookup.",
                "No approved query intent matched the request.",
            )

        try:
            if intent in {"inventory_lookup", "stockout_root_cause"}:
                return _inventory_answer(filters)
            if intent == "recent_support_cases":
                return _support_cases_answer(filters)
            if intent == "quality_complaints_by_vendor":
                return _quality_complaints_answer(filters, by_vendor=True)
            if intent == "product_quality_by_sku":
                return _quality_complaints_answer(filters, by_vendor=False)
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            log_failure("data_agent.query", error)
            return _empty_result(
                "I could not complete the data lookup because a required CSV file is unavailable or unreadable.",
                "Check the supplied files under data/ and try again.",
            )
        except (KeyError, ValueError, TypeError) as error:
            log_failure("data_agent.query", error)
            return _empty_result(
                "I could not safely calculate an answer from the supplied data.",
                "A required field or value was missing or could not be interpreted.",
            )
        except Exception as error:
            log_failure("data_agent.query", error)
            raise

        return _empty_result("No data result was produced.", "The selected intent has no implementation.")


def answer_data_question(question: str) -> dict:
    """Top-level entry point delegating directly to RetailDataAgent."""
    return RetailDataAgent().query(question)


def _answer_screened_data_question(screened_input: object) -> dict:
    """Compatibility entry point accepting either a text string or dictionary."""
    if isinstance(screened_input, dict):
        text = screened_input.get("cleaned_text") or screened_input.get("question") or ""
    elif isinstance(screened_input, str):
        text = screened_input
    else:
        text = ""
    return answer_data_question(text)

