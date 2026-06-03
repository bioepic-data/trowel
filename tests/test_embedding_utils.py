"""Tests for embedding loading utilities."""

import csv
import os
import tempfile

import pytest
from linkml_store import Client

from trowel.utils.embedding_utils import (
    load_embeddings_from_csv,
    load_embeddings_from_duckdb,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_load_embeddings_from_csv_with_embeddings_column(temp_dir):
    """Test loading LinkML-Store export CSVs with an embeddings column."""
    csv_path = os.path.join(temp_dir, "embeddings.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "embeddings"])
        writer.writeheader()
        writer.writerow({
            "id": "BERVO:0000001",
            "label": "Temperature",
            "embeddings": "[0.1, 0.2, 0.3]",
        })

    labels, vectors = load_embeddings_from_csv(csv_path)

    assert labels == ["BERVO:0000001"]
    assert len(vectors) == 1
    assert vectors[0].tolist() == [0.1, 0.2, 0.3]


def test_load_embeddings_from_csv_with_legacy_embedding_column(temp_dir):
    """Test loading existing CSVs with a singular embedding column."""
    csv_path = os.path.join(temp_dir, "embeddings.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "embedding"])
        writer.writeheader()
        writer.writerow({
            "id": "BERVO:0000001",
            "label": "Temperature",
            "embedding": "[0.1, 0.2]",
        })

    labels, vectors = load_embeddings_from_csv(csv_path)

    assert labels == ["BERVO:0000001"]
    assert vectors[0].tolist() == [0.1, 0.2]


def test_load_embeddings_from_duckdb_linkml_store_index_table(temp_dir):
    """Test loading vectors from LinkML-Store's internal index collection."""
    db_path = os.path.join(temp_dir, "test.duckdb")
    db = Client().attach_database(
        f"duckdb:///{db_path}",
        alias="test",
        recreate_if_exists=True,
    )
    collection = db.get_collection("test_collection", type="EmbeddingRow")
    collection.insert([{"id": "BERVO:0000001", "label": "Temperature"}])
    collection.commit()
    index_collection = db.get_collection(
        "internal__index__test_collection__embeddings",
        type="EmbeddingRowIndex",
    )
    index_collection.insert([
        {
            "id": "BERVO:0000001",
            "label": "Temperature",
            "__index__": [0.1, 0.2],
        },
    ])
    index_collection.commit()

    labels, vectors = load_embeddings_from_duckdb(db_path, "test_collection")

    assert labels == ["BERVO:0000001"]
    assert vectors[0].tolist() == pytest.approx([0.1, 0.2])
