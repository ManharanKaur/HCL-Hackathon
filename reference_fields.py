"""Parsing for semicolon-separated citation and data-source fields."""


def parse_reference_field(value: object) -> list[str]:
    """Normalize a delimited string or list of references into clean strings."""
    if isinstance(value, str):
        return [part.strip() for part in value.split(";") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if part is not None and str(part).strip()]
    return []
