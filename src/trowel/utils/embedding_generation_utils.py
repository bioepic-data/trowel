"""Utilities for generating embeddings using LinkML-Store."""

import csv
import logging
import os
from typing import Any, List, Optional, Tuple

__all__ = [
    "generate_embeddings_with_linkml_store",
    "generate_embeddings_with_curategpt",
    "export_embeddings_to_csv",
]

_TROWEL_INDEX_NAME = "embeddings"


def _duckdb_uri(db_path: str) -> str:
    """Return a LinkML-Store DuckDB URI for a local database path."""
    return f"duckdb:///{os.path.abspath(db_path)}"


def _get_linkml_store_collection(
    db_path: str,
    collection_name: str,
    *,
    recreate_if_exists: bool = False,
):
    """Import LinkML-Store lazily and return a configured collection."""
    try:
        from linkml_store import Client
    except ImportError:
        raise ImportError(
            "linkml-store is required for embedding generation. "
            "Install with: pip install linkml-store"
        )

    client = Client()
    db = client.attach_database(
        _duckdb_uri(db_path),
        alias="trowel",
        recreate_if_exists=recreate_if_exists,
    )
    collection = db.get_collection(
        collection_name,
        type="EmbeddingRow",
        create_if_not_exists=True,
    )
    return collection


def _normalize_embedding_model(model: Optional[str]) -> str:
    """Normalize legacy CurateGPT OpenAI model syntax to llm model names."""
    if model and model.startswith("openai:"):
        return model.split(":", 1)[1]
    return model or "text-embedding-ada-002"


def _is_openai_embedding_model(model: Optional[str]) -> bool:
    """Return whether a model name is handled by OpenAI in the llm package."""
    model_name = _normalize_embedding_model(model)
    return model_name.startswith("text-embedding-")


def _require_openai_key_if_needed(model: Optional[str], provider_name: str) -> None:
    if _is_openai_embedding_model(model) and not os.getenv("OPENAI_API_KEY"):
        raise ImportError(
            "OPENAI_API_KEY environment variable is not set. "
            f"{provider_name} requires an OpenAI API key for OpenAI embeddings. "
            "Set it with: export OPENAI_API_KEY='your-key-here'"
        )


def _document_text(row: dict, text_fields: Optional[List[str]]) -> str:
    if text_fields:
        text_parts = [str(row.get(field, ""))
                      for field in text_fields if field in row]
    else:
        text_parts = [str(v) for v in row.values() if v]
    return " ".join(filter(None, text_parts))


def _normalize_linkml_row(row: dict, include_embeddings: bool) -> dict:
    """Normalize a LinkML-Store row into a dictionary row for CSV export."""
    doc = dict(row)
    if include_embeddings and "__index__" in doc:
        doc["embeddings"] = doc.pop("__index__")
    elif not include_embeddings:
        doc.pop("__index__", None)
    return doc


def generate_embeddings_with_linkml_store(
    csv_path: str,
    collection_name: str = "embeddings",
    db_path: str = "./backup/db.duckdb",
    text_fields: Optional[List[str]] = None,
    limit: Optional[int] = None,
    skip: int = 0,
    model: Optional[str] = None,
) -> Tuple[str, int]:
    """Generate embeddings for CSV data using LinkML-Store with DuckDB.

    Initializes a LinkML-Store DuckDB collection, loads CSV rows into it, then
    attaches a named LLM index to generate vector embeddings.

    Args:
        csv_path: Path to the CSV file containing data to embed
        collection_name: Name of the collection to store embeddings (default: "embeddings")
        db_path: Path to DuckDB file for storage (default: "./backup/db.duckdb")
        text_fields: List of CSV column names to use for generating embeddings.
                    If None, uses all columns concatenated.
        limit: Maximum number of rows to embed (for testing/sampling)
        skip: Number of rows to skip from the beginning
        model: LinkML-Store/llm embedding model string. Legacy
               "openai:<model-name>" values are normalized to "<model-name>".

    Returns:
        Tuple of (database_path, number_of_embeddings_created)

    Raises:
        FileNotFoundError: If the CSV file does not exist
        ImportError: If linkml-store or duckdb is not installed or an OpenAI
                     model is used without OPENAI_API_KEY
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    _require_openai_key_if_needed(model, "LinkML-Store")

    try:
        __import__("duckdb")
    except ImportError:
        raise ImportError(
            "duckdb is required for embedding storage. "
            "Install with: pip install duckdb"
        )

    # Ensure database directory exists
    db_dir = os.path.dirname(os.path.abspath(db_path)) or "."
    os.makedirs(db_dir, exist_ok=True)

    try:
        from linkml_store.index.implementations.llm_indexer import LLMIndexer
    except ImportError:
        raise ImportError(
            "linkml-store with LLM indexing support is required for embedding "
            "generation. Install with: pip install linkml-store"
        )

    normalized_model = _normalize_embedding_model(model)
    logging.info(f"Initializing LinkML-Store DuckDB database at {db_path}...")
    logging.info(f"Using embedding model: {normalized_model}")

    logging.info(f"Loading data from {csv_path}...")
    rows_read = 0
    rows_to_insert: List[dict[str, Any]] = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader):
                # Handle skip parameter
                if idx < skip:
                    continue

                # Handle limit parameter
                if limit is not None and len(rows_to_insert) >= limit:
                    break

                rows_read += 1

                text_to_embed = _document_text(row, text_fields)

                if not text_to_embed.strip():
                    logging.warning(
                        f"Row {idx} has no text to embed, skipping...")
                    continue

                rows_to_insert.append(row)

        if not rows_to_insert:
            logging.warning(f"No embeddable rows found in {csv_path}")
            return db_path, 0

        collection = _get_linkml_store_collection(db_path, collection_name)
        collection.insert(rows_to_insert)
        collection.commit()

        indexer = LLMIndexer(
            name=_TROWEL_INDEX_NAME,
            index_attributes=text_fields,
            embedding_model_name=normalized_model,
        )
        collection.attach_indexer(indexer, auto_index=True)

        logging.info(
            f"Successfully embedded {len(rows_to_insert)} rows from {csv_path}")
        logging.info(
            f"Embeddings stored in collection '{collection_name}' at {db_path}")

        return db_path, len(rows_to_insert)

    except Exception as e:
        logging.error(f"Error during embedding generation: {e}")
        raise


def generate_embeddings_with_curategpt(*args, **kwargs) -> Tuple[str, int]:
    """Backward-compatible wrapper for the LinkML-Store embedding backend."""
    return generate_embeddings_with_linkml_store(*args, **kwargs)


def export_embeddings_to_csv(
    db_path: str,
    collection_name: str,
    output_path: str,
    include_embeddings: bool = False,
) -> int:
    """Export embeddings from a LinkML-Store database to CSV format.

    Retrieves all documents from a LinkML-Store collection and exports them
    to a CSV file. If embeddings are requested, reads from the internal
    LinkML-Store index collection and writes the vector as an ``embeddings``
    column.

    Args:
        db_path: Path to the LinkML-Store DuckDB database
        collection_name: Name of the collection to export
        output_path: Path where the CSV will be written
        include_embeddings: If True, include the embedding vectors as columns
                           (Note: vectors are high-dimensional, CSV may be large)

    Returns:
        Number of rows exported

    Raises:
        FileNotFoundError: If the database path does not exist
        ImportError: If linkml-store is not installed
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database path not found: {db_path}")

    collection = _get_linkml_store_collection(db_path, collection_name)
    export_collection = collection

    if include_embeddings:
        index_collection_name = collection._index_collection_name(_TROWEL_INDEX_NAME)
        export_collection = collection.parent.get_collection(
            index_collection_name,
            create_if_not_exists=False,
        )

    logging.info(f"Opening LinkML-Store database at {db_path}...")
    logging.info(f"Retrieving all documents from collection '{export_collection.alias}'...")

    try:
        raw_results = export_collection.find({}, limit=-1).rows
        docs = [
            _normalize_linkml_row(result, include_embeddings=include_embeddings)
            for result in raw_results
        ]

        field_names = []
        for doc in docs:
            for key in doc.keys():
                if key not in field_names:
                    field_names.append(key)

        if not field_names:
            logging.warning(
                f"No documents found in collection '{collection_name}'")
            return 0

        # Export to CSV
        logging.info(f"Exporting to {output_path}...")
        rows_exported = 0

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path))
                    or ".", exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()

            for doc in docs:
                writer.writerow(doc)
                rows_exported += 1

                if rows_exported % 100 == 0:
                    logging.info(f"Exported {rows_exported} rows...")

        logging.info(
            f"Successfully exported {rows_exported} rows to {output_path}")
        return rows_exported

    except Exception as e:
        logging.error(f"Error during CSV export: {e}")
        raise
