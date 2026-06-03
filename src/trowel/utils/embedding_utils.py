"""Utilities for working with embeddings and embedding databases."""

import csv
import logging
from typing import Dict, List, Tuple

import numpy as np

__all__ = [
    "prepare_embedding_csv",
    "load_embeddings_from_csv",
    "load_embeddings_from_duckdb",
    "get_top_level_categories",
]


def prepare_embedding_csv(
    input_file: str,
    output_file: str,
    column_indices: List[int],
    skip_rows: int = 0,
    delimiter: str = ','
) -> None:
    """Prepare a CSV file for embedding by selecting specific columns.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        column_indices: List of column indices to extract
        skip_rows: Number of rows to skip (e.g., header row)
        delimiter: CSV delimiter character
    """
    with open(input_file, newline='', encoding='utf-8') as infile:
        all_rows = list(csv.reader(infile, delimiter=delimiter))
        if not all_rows:
            raise RuntimeError(f"Input file '{input_file}' is empty")

        full_header = all_rows[0]
        # Skip specified rows and keep the rest
        data_rows = all_rows[skip_rows + 1:] if len(all_rows) > 1 else []

        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            # Write only the selected header labels
            writer.writerow([full_header[i] if i < len(full_header) else ''
                             for i in column_indices])

            # Write selected columns for each data row
            for row in data_rows:
                writer.writerow([row[i] if i < len(row) else ''
                                 for i in column_indices])

    logging.info(
        f"Prepared CSV with selected columns written to {output_file}")


def load_embeddings_from_csv(
    embedding_file: str,
) -> Tuple[List[str], List[np.ndarray]]:
    """Load embeddings from a generated CSV file.

    Preferred format: a CSV with an ``embedding`` or ``embeddings`` column
    containing a Python-style list of floats. The older metadata tuple format is
    still accepted for compatibility.

    Args:
        embedding_file: Path to embedding CSV file

    Returns:
        Tuple of (labels_list, vectors_list)
    """
    import ast

    labels = []
    vectors = []

    def parse_vector(value):
        parsed = ast.literal_eval(value)
        return list(parsed)

    with open(embedding_file, 'r', newline='', encoding='utf-8') as f:
        first_line = f.readline()
        f.seek(0)

        if first_line and not first_line.lstrip().startswith("{"):
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            embedding_field = next(
                (
                    field
                    for field in ("embeddings", "embedding")
                    if field in fieldnames
                ),
                None,
            )
            if embedding_field:
                for line_num, row in enumerate(reader, 2):
                    try:
                        label = (
                            row.get("LABEL")
                            or row.get("ID")
                            or row.get("id")
                            or row.get("label")
                            or row.get("name")
                            or "Unknown"
                        )
                        labels.append(label)
                        vectors.append(np.array(parse_vector(row[embedding_field])))
                    except Exception as e:
                        logging.warning(f"Error parsing line {line_num}: {e}")
                logging.info(
                    f"Loaded {len(labels)} terms with embeddings from {embedding_file}")
                if vectors:
                    logging.info(f"Embedding dimension: {len(vectors[0])}")
                return labels, vectors

        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # The line format is: {'ID': '...', 'LABEL': '...'},(val1, val2, ...)
                # Find where the metadata dict ends
                dict_end = line.index('}') + 1

                # Extract metadata
                metadata_str = line[:dict_end]
                metadata = ast.literal_eval(metadata_str)

                # Get label from metadata, with fallbacks
                label = metadata.get('LABEL',
                                     metadata.get('ID',
                                                  metadata.get('name', 'Unknown')))

                # Extract embedding values - everything after the dict and comma
                embedding_str = line[dict_end + 1:]  # Skip the comma after }
                embedding_values = parse_vector(embedding_str)

                labels.append(label)
                vectors.append(np.array(embedding_values))

            except Exception as e:
                logging.warning(f"Error parsing line {line_num}: {e}")
                continue

    logging.info(
        f"Loaded {len(labels)} terms with embeddings from {embedding_file}")
    if vectors:
        logging.info(f"Embedding dimension: {len(vectors[0])}")

    return labels, vectors


def load_embeddings_from_duckdb(
    db_path: str,
    collection_name: str,
    exclude_ids: List[str] = None,
) -> Tuple[List[str], List[np.ndarray]]:
    """Load embeddings from a DuckDB database created by LinkML-Store.

    Args:
        db_path: Path to DuckDB database file
        collection_name: Name of the collection to query
        exclude_ids: List of IDs to exclude (e.g., ['__venomx__'])

    Returns:
        Tuple of (labels_list, vectors_list)
    """
    try:
        import duckdb
    except ImportError:
        raise ImportError(
            "duckdb is required for this function. Install with: pip install duckdb")

    if exclude_ids is None:
        exclude_ids = []

    conn = duckdb.connect(db_path)

    def quote_identifier(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    def table_columns(table_name: str) -> List[str]:
        rows = conn.execute(
            f"PRAGMA table_info({quote_identifier(table_name)})"
        ).fetchall()
        return [row[1] for row in rows]

    def table_exists(table_name: str) -> bool:
        rows = conn.execute("SHOW TABLES").fetchall()
        return table_name in {row[0] for row in rows}

    index_collection_name = f"internal__index__{collection_name}__embeddings"
    target_table = collection_name
    vector_column = "embeddings"

    if table_exists(collection_name):
        columns = table_columns(collection_name)
        if "embeddings" not in columns:
            target_table = index_collection_name
            vector_column = "__index__"

    if not table_exists(target_table):
        raise RuntimeError(f"Embedding table not found: {target_table}")

    columns = table_columns(target_table)
    if vector_column not in columns:
        raise RuntimeError(
            f"Embedding column '{vector_column}' not found in {target_table}")

    exclude_placeholders = " AND ".join(["id != ?" for _ in exclude_ids])
    exclude_clause = f" AND {exclude_placeholders}" if exclude_placeholders else ""

    query = f"""
    SELECT id, {quote_identifier(vector_column)}
    FROM {quote_identifier(target_table)}
    WHERE id != ''
    {exclude_clause}
    ORDER BY id
    """

    try:
        results = conn.execute(query, exclude_ids).fetchall()
    except Exception as e:
        logging.error(f"Error querying DuckDB: {e}")
        raise
    finally:
        conn.close()

    labels = []
    vectors = []

    for result in results:
        term_id = result[0]
        embedding = result[1]
        labels.append(term_id)
        vectors.append(np.array(embedding))

    logging.info(
        f"Loaded {len(labels)} terms with embeddings from {collection_name}")
    if vectors:
        logging.info(f"Embedding dimension: {len(vectors[0])}")

    return labels, vectors


def get_top_level_categories(
    source_csv: str,
    id_column: int = 0,
    label_column: int = 1,
    category_column: str = 'Category',
    skip_rows: int = 0,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build mappings of term IDs to their top-level categories.

    This function processes a hierarchical ontology where:
    - Each term has an ID, label, and category
    - Categories can be either labels or IDs
    - Top-level categories start with a specific prefix (e.g., 'BERVO:9')

    Args:
        source_csv: Path to source CSV file with term definitions
        id_column: Column index for term IDs
        label_column: Column index for term labels
        category_column: Column header for category information
        skip_rows: Number of header rows to skip

    Returns:
        Tuple of (id_to_label_dict, id_to_top_level_category_dict)
    """
    id_to_label = {}
    label_to_id = {}
    id_to_category = {}
    id_to_parent_id = {}

    # Load the source file
    with open(source_csv, newline='', encoding='utf-8') as infile:
        all_rows = list(csv.reader(infile))
        header = all_rows[0]
        data_rows = all_rows[skip_rows + 1:]  # Skip header and additional rows

    # Find column index for category
    if category_column not in header:
        logging.warning(
            f"Category column '{category_column}' not found in header")
        category_col_idx = None
    else:
        category_col_idx = header.index(category_column)

    # Build initial mappings
    for row in data_rows:
        if len(row) > id_column:
            term_id = row[id_column].strip()
            label = row[label_column].strip() if len(
                row) > label_column else ''

            if term_id:
                id_to_label[term_id] = label
                if label:
                    label_to_id[label] = term_id

                # Get the category
                if category_col_idx is not None and len(row) > category_col_idx:
                    category = row[category_col_idx].strip()
                    if category:
                        id_to_category[term_id] = category

    # Build parent relationships by resolving category labels to IDs
    for term_id, category in id_to_category.items():
        if category.startswith('BERVO:'):
            id_to_parent_id[term_id] = category
        elif category in label_to_id:
            id_to_parent_id[term_id] = label_to_id[category]

    # Function to find top-level category
    def get_top_level(term_id: str, max_depth: int = 20) -> str:
        """Find the top-level category for a given term."""
        if not term_id or not term_id.startswith('BERVO:'):
            return 'Unknown'

        # If it's already a top-level category (BERVO:9*)
        if term_id.startswith('BERVO:9'):
            return term_id

        # Traverse up the hierarchy
        visited = set()
        current = term_id
        depth = 0

        while current and depth < max_depth:
            if current in visited:  # Avoid cycles
                return 'Unknown'
            visited.add(current)

            if current.startswith('BERVO:9'):
                return current

            # Move to parent
            current = id_to_parent_id.get(current)
            depth += 1

        return 'Unknown'

    # Build final mapping
    term_to_top_level = {}
    for term_id in id_to_label.keys():
        top_level = get_top_level(term_id)
        term_to_top_level[term_id] = top_level

    logging.info(f"Built category mappings for {len(term_to_top_level)} terms")
    logging.info(
        f"Found {len(set(term_to_top_level.values()))} unique top-level categories")

    return id_to_label, term_to_top_level
