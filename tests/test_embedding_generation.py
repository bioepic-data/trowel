"""Tests for embedding generation utilities."""

import csv
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from linkml_store import Client

from trowel.utils.embedding_generation_utils import (
    export_embeddings_to_csv,
    generate_embeddings_with_curategpt,
    generate_embeddings_with_linkml_store,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_csv(temp_dir):
    """Create a sample CSV file for testing."""
    csv_path = os.path.join(temp_dir, "test_data.csv")
    rows = [
        {
            "id": "BERVO:0000001",
            "label": "Temperature",
            "definition": "Air temperature measurement",
        },
        {
            "id": "BERVO:0000002",
            "label": "Humidity",
            "definition": "Moisture content in air",
        },
        {
            "id": "BERVO:0000003",
            "label": "Pressure",
            "definition": "Atmospheric pressure",
        },
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "definition"])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


class TestGenerateEmbeddingsWithLinkMLStore:
    """Tests for generate_embeddings_with_linkml_store function."""

    def test_missing_input_file(self, temp_dir):
        """Test that function raises error when input file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            generate_embeddings_with_linkml_store(
                os.path.join(temp_dir, "nonexistent.csv"),
                db_path=os.path.join(temp_dir, "test.duckdb")
            )

    def test_missing_openai_api_key_for_default_model(self, sample_csv, temp_dir):
        """Test that the default OpenAI embedding model requires OPENAI_API_KEY."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ImportError, match="OPENAI_API_KEY"):
                generate_embeddings_with_linkml_store(
                    sample_csv,
                    db_path=os.path.join(temp_dir, "test.duckdb"),
                )

    def test_missing_openai_api_key_for_legacy_openai_model(
        self,
        sample_csv,
        temp_dir,
    ):
        """Test that legacy openai: model syntax is normalized and checked."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ImportError, match="OPENAI_API_KEY"):
                generate_embeddings_with_linkml_store(
                    sample_csv,
                    db_path=os.path.join(temp_dir, "test.duckdb"),
                    model="openai:text-embedding-3-small",
                )

    @patch("trowel.utils.embedding_generation_utils._get_linkml_store_collection")
    @patch("linkml_store.index.implementations.llm_indexer.LLMIndexer")
    def test_successful_embedding_generation(
        self,
        mock_indexer_cls,
        mock_get_collection,
        sample_csv,
        temp_dir,
    ):
        """Test successful embedding generation with mocked LinkML-Store."""
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        mock_indexer = MagicMock()
        mock_indexer_cls.return_value = mock_indexer
        db_path = os.path.join(temp_dir, "test.duckdb")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result_path, num_embeddings = generate_embeddings_with_linkml_store(
                sample_csv,
                collection_name="test_collection",
                db_path=db_path,
                text_fields=["label", "definition"],
                model="openai:text-embedding-3-small",
            )

        assert result_path == db_path
        assert num_embeddings == 3
        inserted_rows = mock_collection.insert.call_args[0][0]
        assert len(inserted_rows) == 3
        assert inserted_rows[0]["id"] == "BERVO:0000001"
        mock_indexer_cls.assert_called_once_with(
            name="embeddings",
            index_attributes=["label", "definition"],
            embedding_model_name="text-embedding-3-small",
        )
        mock_collection.attach_indexer.assert_called_once_with(
            mock_indexer,
            auto_index=True,
        )

    @patch("trowel.utils.embedding_generation_utils._get_linkml_store_collection")
    @patch("linkml_store.index.implementations.llm_indexer.LLMIndexer")
    def test_embedding_with_limit_and_skip(
        self,
        mock_indexer_cls,
        mock_get_collection,
        sample_csv,
        temp_dir,
    ):
        """Test that limit and skip select the expected CSV rows."""
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        mock_indexer_cls.return_value = MagicMock()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            _, num_embeddings = generate_embeddings_with_linkml_store(
                sample_csv,
                db_path=os.path.join(temp_dir, "test.duckdb"),
                skip=1,
                limit=1,
            )

        assert num_embeddings == 1
        inserted_rows = mock_collection.insert.call_args[0][0]
        assert inserted_rows[0]["id"] == "BERVO:0000002"

    @patch("trowel.utils.embedding_generation_utils._get_linkml_store_collection")
    @patch("linkml_store.index.implementations.llm_indexer.LLMIndexer")
    def test_legacy_wrapper_uses_linkml_store_backend(
        self,
        mock_indexer_cls,
        mock_get_collection,
        sample_csv,
        temp_dir,
    ):
        """Test the backward-compatible function delegates to the new backend."""
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        mock_indexer_cls.return_value = MagicMock()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            _, num_embeddings = generate_embeddings_with_curategpt(
                sample_csv,
                db_path=os.path.join(temp_dir, "test.duckdb"),
            )

        assert num_embeddings == 3
        assert mock_collection.insert.called

    @patch("trowel.utils.embedding_generation_utils._get_linkml_store_collection")
    @patch("linkml_store.index.implementations.llm_indexer.LLMIndexer")
    def test_database_directory_creation(
        self,
        mock_indexer_cls,
        mock_get_collection,
        sample_csv,
        temp_dir,
    ):
        """Test that database directory is created if it doesn't exist."""
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        mock_indexer_cls.return_value = MagicMock()
        db_path = os.path.join(temp_dir, "nested", "dir", "test.duckdb")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generate_embeddings_with_linkml_store(sample_csv, db_path=db_path)

        assert os.path.exists(os.path.dirname(db_path))


class TestExportEmbeddingsToCSV:
    """Tests for export_embeddings_to_csv function."""

    def test_missing_database(self, temp_dir):
        """Test that function raises error when database doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Database path not found"):
            export_embeddings_to_csv(
                os.path.join(temp_dir, "nonexistent.duckdb"),
                "collection",
                os.path.join(temp_dir, "output.csv")
            )

    def test_successful_csv_export(self, temp_dir):
        """Test successful CSV export from the primary collection."""
        db_path = os.path.join(temp_dir, "test.duckdb")
        db = Client().attach_database(
            f"duckdb:///{db_path}",
            alias="test",
            recreate_if_exists=True,
        )
        collection = db.get_collection("test_collection", type="EmbeddingRow")
        collection.insert([
            {"id": "BERVO:0000001", "label": "Temperature"},
            {"id": "BERVO:0000002", "label": "Humidity"},
        ])
        collection.commit()

        output_path = os.path.join(temp_dir, "output.csv")
        num_exported = export_embeddings_to_csv(
            db_path,
            "test_collection",
            output_path,
        )

        assert num_exported == 2
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["id"] == "BERVO:0000001"
            assert "embeddings" not in reader.fieldnames

    def test_csv_export_includes_linkml_store_embedding_column(self, temp_dir):
        """Test that indexed rows export vectors as an embeddings column."""
        db_path = os.path.join(temp_dir, "test.duckdb")
        db = Client().attach_database(
            f"duckdb:///{db_path}",
            alias="test",
            recreate_if_exists=True,
        )
        collection = db.get_collection("test_collection", type="EmbeddingRow")
        collection.insert([
            {"id": "BERVO:0000001", "label": "Temperature"},
        ])
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

        output_path = os.path.join(temp_dir, "output.csv")
        num_exported = export_embeddings_to_csv(
            db_path,
            "test_collection",
            output_path,
            include_embeddings=True,
        )

        assert num_exported == 1
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert "embeddings" in reader.fieldnames
            assert "__index__" not in reader.fieldnames
            assert rows[0]["embeddings"].startswith("[0.1")

    def test_empty_collection_export(self, temp_dir):
        """Test exporting an empty collection."""
        db_path = os.path.join(temp_dir, "test.duckdb")
        db = Client().attach_database(
            f"duckdb:///{db_path}",
            alias="test",
            recreate_if_exists=True,
        )
        collection = db.get_collection("empty_collection", type="EmbeddingRow")
        collection.insert([{"id": "BERVO:0000001", "label": "Temperature"}])
        collection.delete_where({})
        collection.commit()

        output_path = os.path.join(temp_dir, "output.csv")
        num_exported = export_embeddings_to_csv(
            db_path,
            "empty_collection",
            output_path,
        )

        assert num_exported == 0

    def test_csv_export_creates_directory(self, temp_dir):
        """Test that export creates output directory if it doesn't exist."""
        db_path = os.path.join(temp_dir, "test.duckdb")
        db = Client().attach_database(
            f"duckdb:///{db_path}",
            alias="test",
            recreate_if_exists=True,
        )
        collection = db.get_collection("test_collection", type="EmbeddingRow")
        collection.insert([{"id": "BERVO:0000001", "label": "Temperature"}])
        collection.commit()

        output_path = os.path.join(temp_dir, "nested", "output.csv")
        export_embeddings_to_csv(db_path, "test_collection", output_path)

        assert os.path.exists(os.path.dirname(output_path))
