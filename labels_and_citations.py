"""Shared category, priority, and policy citation conventions."""

import re


# Canonical labels used by triage, orchestration, and response validation.
CATEGORIES = {
    "POS and Payments",
    "Inventory and Replenishment",
    "Pricing and Promotions",
    "Returns and Refunds",
    "Vendor and Supply Chain",
    "Product Quality and Safety",
    "Customer Service",
    "Other or Security",
}
PRIORITIES = {"P1", "P2", "P3", "P4"}
MARKDOWN_HEADING_PATTERN = re.compile(r"(?m)^#{1,6}[ \t]+(.+?)\s*#*\s*$")


def heading_to_anchor(heading: str) -> str:
    """Convert a Markdown heading to the citation anchor format."""
    anchor = heading.casefold().replace("&", " and ").replace("`", "")
    anchor = re.sub(r"[^a-z0-9\s-]", "", anchor)
    return re.sub(r"[\s-]+", "-", anchor).strip("-")


def citations_for_markdown(filename: str, markdown: str) -> set[str]:
    """Return citation IDs for the real headings in one Markdown file."""
    return {
        f"{filename}#{heading_to_anchor(match.group(1).strip())}"
        for match in MARKDOWN_HEADING_PATTERN.finditer(markdown)
    }
