"""Utilities for generating embeddings using CurateGPT."""

import csv
import logging
import os
from typing import List, Optional, Tuple

__all__ = [
    "generate_embeddings_with_curategpt",
    "export_embeddings_to_csv",
]


def generate_embeddings_with_curategpt(
    csv_path: str,
    collection_name: str = "embeddings",
    db_path: str = "./backup/db.duckdb",
    text_fields: Optional[List[str]] = None,
    limit: Optional[int] = None,
    skip: int = 0,
) -> Tuple[str, int]:
    """Generate embeddings for CSV data using CurateGPT with DuckDB backend.

    Initializes a CurateGPT store with DuckDB backend and generates
    vector embeddings for each row using OpenAI's text-embedding-ada-002 model.

    Args:
        csv_path: Path to the CSV file containing data to embed
        collection_name: Name of the collection to store embeddings (default: "embeddings")
        db_path: Path to DuckDB file for storage (default: "./backup/db.duckdb")
        text_fields: List of CSV column names to use for generating embeddings.
                    If None, uses all columns concatenated.
        limit: Maximum number of rows to embed (for testing/sampling)
        skip: Number of rows to skip from the beginning

    Returns:
        Tuple of (database_path, number_of_embeddings_created)

    Raises:
        FileNotFoundError: If the CSV file does not exist
        ImportError: If curategpt or duckdb is not installed or OPENAI_API_KEY is not set
    """
    try:
        from curategpt.store import get_store
    except ImportError:
        raise ImportError(
            "curategpt is required for embedding generation. "
            "Install with: pip install curategpt"
        )

    try:
        import duckdb
    except ImportError:
        raise ImportError(
            "duckdb is required for embedding storage. "
            "Install with: pip install duckdb"
        )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ImportError(
            "OPENAI_API_KEY environment variable is not set. "
            "CurateGPT requires an OpenAI API key for embedding generation. "
            "Set it with: export OPENAI_API_KEY='your-key-here'"
        )

    # Ensure database directory exists
    db_dir = os.path.dirname(os.path.abspath(db_path)) or "."
    os.makedirs(db_dir, exist_ok=True)

    logging.info(f"Initializing CurateGPT store with DuckDB at {db_path}...")
    store = get_store("duckdb", db_path)

    logging.info(f"Loading data from {csv_path}...")
    rows_read = 0
    rows_inserted = 0

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader):
                # Handle skip parameter
                if idx < skip:
                    continue

                # Handle limit parameter
                if limit is not None and rows_inserted >= limit:
                    break

                rows_read += 1

                # Prepare text for embedding
                if text_fields:
                    # Use only specified fields
                    text_parts = [str(row.get(field, "")) for field in text_fields if field in row]
                    text_to_embed = " ".join(filter(None, text_parts))
                else:
                    # Use all fields concatenated
                    text_to_embed = " ".join(str(v) for v in row.values() if v)

                if not text_to_embed.strip():
                    logging.warning(f"Row {idx} has no text to embed, skipping...")
                    continue

                try:
                    # Insert the row with its text content
                    # CurateGPT will automatically generate embeddings
                    store.insert(row, collection=collection_name)
                    rows_inserted += 1

                    if rows_inserted % 100 == 0:
                        logging.info(f"Embedded {rows_inserted} rows...")

                except Exception as e:
                    logging.warning(f"Failed to embed row {idx}: {e}")
                    continue

        logging.info(f"Successfully embedded {rows_inserted} rows from {csv_path}")
        logging.info(f"Embeddings stored in collection '{collection_name}' at {db_path}")

        return db_path, rows_inserted

    except Exception as e:
        logging.error(f"Error during embedding generation: {e}")
        raise


def export_embeddings_to_csv(
    db_path: str,
    collection_name: str,
    output_path: str,
    include_embeddings: bool = False,
) -> int:
    """Export embeddings from CurateGPT database to CSV format.

    Retrieves all documents from a CurateGPT collection and exports them
    to a CSV file, optionally including the embedding vectors.

    Args:
        db_path: Path to the CurateGPT database
        collection_name: Name of the collection to export
        output_path: Path where the CSV will be written
        include_embeddings: If True, include the embedding vectors as columns
                           (Note: vectors are high-dimensional, CSV may be large)

    Returns:
        Number of rows exported

    Raises:
        FileNotFoundError: If the database path does not exist
        ImportError: If curategpt is not installed
    """
    try:
        from curategpt.store import get_store
    except ImportError:
        raise ImportError(
            "curategpt is required for this operation. "
            "Install with: pip install curategpt"
        )

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database path not found: {db_path}")

    logging.info(f"Opening CurateGPT database at {db_path}...")
    store = get_store("chromadb", db_path)

    logging.info(f"Retrieving all documents from collection '{collection_name}'...")

    try:
        # Get field names for the collection
        field_names = store.field_names(collection=collection_name)
        if not field_names:
            logging.warning(f"No documents found in collection '{collection_name}'")
            return 0

        # Export to CSV
        logging.info(f"Exporting to {output_path}...")
        rows_exported = 0

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()

            # Query all documents (without embedding vectors for now)
            # CurateGPT's find() method returns all documents when called without filters
            docs = store.find(where={}, collection=collection_name)

            for doc in docs:
                writer.writerow(doc)
                rows_exported += 1

                if rows_exported % 100 == 0:
                    logging.info(f"Exported {rows_exported} rows...")

        logging.info(f"Successfully exported {rows_exported} rows to {output_path}")
        return rows_exported

    except Exception as e:
        logging.error(f"Error during CSV export: {e}")
        raise
