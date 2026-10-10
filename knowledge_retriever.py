"""Local policy search backed by ChromaDB and the configured embedding API."""

import hashlib
import re
from pathlib import Path

from llm_client import get_embedding


DEFAULT_EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent / "knowledge_base"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "retail_policy_sections"

# Chroma cosine distance is lower for closer matches. This limit prevents a
# completely unrelated section from being presented as policy evidence.
MAX_COSINE_DISTANCE = 0.85

CATEGORY_BY_FILE = {
    "01_retail_operations_sop.md": [
        "POS and Payments",
        "Inventory and Replenishment",
        "Pricing and Promotions",
        "Returns and Refunds",
        "Vendor and Supply Chain",
        "Product Quality and Safety",
        "Customer Service",
        "Other or Security",
    ],
    "02_pos_pricing_returns_policy.md": [
        "POS and Payments",
        "Pricing and Promotions",
        "Returns and Refunds",
        "Customer Service",
    ],
    "03_inventory_replenishment_policy.md": ["Inventory and Replenishment"],
    "04_vendor_quality_policy.md": [
        "Vendor and Supply Chain",
        "Product Quality and Safety",
    ],
    "05_responsible_ai_policy.md": ["Other or Security"],
}

HEADING_PATTERN = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)\s*#*\s*$")


def _get_collection():
    """Open the persistent policy collection, or explain a missing dependency."""
    try:
        import chromadb
    except ImportError as error:
        raise RuntimeError(
            "Policy search requires ChromaDB. Install the 'chromadb' package before searching."
        ) from error

    # PersistentClient writes the vector index under the project-root directory.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _make_stable_id(filename: str, heading: str, occurrence: int = 1) -> str:
    """Create the same ID whenever a file section keeps the same heading."""
    identity = f"{filename}::{heading.casefold().strip()}::{occurrence}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _read_policy_sections() -> list[dict]:
    """Read Markdown files and turn each heading plus its body into a section."""
    sections = []

    for policy_path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        markdown = policy_path.read_text(encoding="utf-8")
        headings = list(HEADING_PATTERN.finditer(markdown))
        categories = CATEGORY_BY_FILE.get(policy_path.name, ["Other or Security"])
        heading_occurrences = {}

        if not headings:
            # Still index a valid policy file even if it has no Markdown headings.
            heading = policy_path.stem.replace("_", " ")
            body = markdown.strip()
            if body:
                sections.append(
                    {
                        "id": _make_stable_id(policy_path.name, heading),
                        "filename": policy_path.name,
                        "heading": heading,
                        "categories": categories,
                        "document": f"{heading}\n{body}",
                    }
                )
            continue

        for index, heading_match in enumerate(headings):
            heading = heading_match.group(2).strip()
            heading_key = heading.casefold()
            heading_occurrences[heading_key] = heading_occurrences.get(heading_key, 0) + 1
            occurrence = heading_occurrences[heading_key]
            body_start = heading_match.end()
            body_end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
            body = markdown[body_start:body_end].strip()

            # Keep any document preamble with the first section so source IDs
            # and applicability notes remain searchable with the policy text.
            if index == 0:
                preamble = markdown[: heading_match.start()].strip()
                if preamble:
                    body = f"{preamble}\n\n{body}" if body else preamble

            document = f"{heading}\n{body}" if body else heading
            sections.append(
                {
                    "id": _make_stable_id(policy_path.name, heading, occurrence),
                    "filename": policy_path.name,
                    "heading": heading,
                    "categories": categories,
                    "document": document,
                }
            )

    return sections


def _ensure_indexed(collection) -> None:
    """Upsert new or changed sections and remove sections no longer on disk.

    Stable IDs make repeated indexing replace the existing section instead of
    creating duplicates. A content hash means unchanged sections need no new
    embedding API call when a user searches again.
    """
    sections = _read_policy_sections()
    current_ids = [section["id"] for section in sections]
    existing = collection.get(ids=current_ids, include=["metadatas"]) if current_ids else {
        "ids": [],
        "metadatas": [],
    }
    existing_hashes = {
        section_id: metadata.get("content_hash", "")
        for section_id, metadata in zip(existing["ids"], existing["metadatas"] or [])
    }

    sections_to_embed = []
    for section in sections:
        content_hash = hashlib.sha256(section["document"].encode("utf-8")).hexdigest()
        section["content_hash"] = content_hash
        if existing_hashes.get(section["id"]) != content_hash:
            sections_to_embed.append(section)

    if sections_to_embed:
        embedding_result = get_embedding(
            [section["document"] for section in sections_to_embed],
            model=DEFAULT_EMBEDDING_MODEL,
        )
        embeddings = embedding_result.get("embeddings", [])
        if len(embeddings) != len(sections_to_embed):
            raise RuntimeError("The embedding service returned an unexpected number of policy vectors.")

        # Chroma metadata values are kept simple and scalar for compatibility.
        collection.upsert(
            ids=[section["id"] for section in sections_to_embed],
            documents=[section["document"] for section in sections_to_embed],
            embeddings=embeddings,
            metadatas=[
                {
                    "filename": section["filename"],
                    "heading": section["heading"],
                    "categories": "|".join(section["categories"]),
                    "content_hash": section["content_hash"],
                }
                for section in sections_to_embed
            ],
        )

    # Remove records for headings or files that have since been deleted.
    all_existing = collection.get(include=["metadatas"])
    stale_ids = [section_id for section_id in all_existing["ids"] if section_id not in set(current_ids)]
    if stale_ids:
        collection.delete(ids=stale_ids)


def _heading_anchor(heading: str) -> str:
    """Convert a Markdown heading to the usual lowercase citation anchor."""
    anchor = heading.casefold().replace("&", " and ").replace("`", "")
    anchor = re.sub(r"[^a-z0-9\s-]", "", anchor)
    return re.sub(r"[\s-]+", "-", anchor).strip("-")


def search_policy(query: str, category: str | None = None, top_k: int = 3) -> dict:
    """Return matching policy passages with citations from stored metadata.

    The query is embedded with the separate embedding model. Chroma ranks the
    stored sections by vector distance; optional category filtering uses the
    category tags saved with each section. No citation is made up when there
    are no close matches.
    """
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return {"passages": []}

    collection = _get_collection()
    _ensure_indexed(collection)
    document_count = collection.count()
    if document_count == 0:
        return {"passages": []}

    query_result = get_embedding(query.strip(), model=DEFAULT_EMBEDDING_MODEL)
    query_vectors = query_result.get("embeddings", [])
    if not query_vectors or not query_vectors[0]:
        return {"passages": []}

    # Ask for all sections so category filtering does not hide a good match.
    matches = collection.query(
        query_embeddings=[query_vectors[0]],
        n_results=document_count,
        include=["documents", "metadatas", "distances"],
    )

    passages = []
    ids = matches.get("ids", [[]])[0]
    documents = matches.get("documents", [[]])[0]
    metadatas = matches.get("metadatas", [[]])[0]
    distances = matches.get("distances", [[]])[0]

    for section_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        if document is None or metadata is None or distance is None:
            continue
        if distance > MAX_COSINE_DISTANCE:
            continue

        stored_categories = metadata.get("categories", "").split("|")
        if category and category not in stored_categories:
            continue

        filename = metadata.get("filename", "")
        heading = metadata.get("heading", "")
        if not filename or not heading:
            continue

        passages.append(
            {
                "text": document,
                "filename": filename,
                "heading": heading,
                "citation": f"{filename}#{_heading_anchor(heading)}",
                "distance": float(distance),
                "id": section_id,
            }
        )
        if len(passages) >= top_k:
            break

    return {"passages": passages}
