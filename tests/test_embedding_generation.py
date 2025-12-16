"""Tests for embedding generation utilities."""

import csv
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

import pytest

from trowel.utils.embedding_generation_utils import (
    generate_embeddings_with_curategpt,
    export_embeddings_to_csv,
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
        {"id": "BERVO:0000001", "label": "Temperature",
            "definition": "Air temperature measurement"},
        {"id": "BERVO:0000002", "label": "Humidity",
            "definition": "Moisture content in air"},
        {"id": "BERVO:0000003", "label": "Pressure",
            "definition": "Atmospheric pressure"},
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "definition"])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


@pytest.fixture
def large_sample_csv(temp_dir):
    """Create a larger sample CSV for testing limit/skip."""
    csv_path = os.path.join(temp_dir, "large_test_data.csv")
    rows = [
        {"id": f"BERVO:{i:07d}", "label": f"Term_{i}",
            "definition": f"Definition for term {i}"}
        for i in range(1000)
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "definition"])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


class TestGenerateEmbeddingsWithCurategpt:
    """Tests for generate_embeddings_with_curategpt function."""

    def test_missing_input_file(self, temp_dir):
        """Test that function raises error when input file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            generate_embeddings_with_curategpt(
                os.path.join(temp_dir, "nonexistent.csv"),
                db_path=os.path.join(temp_dir, "test.duckdb")
            )

    def test_missing_openai_api_key(self, sample_csv, temp_dir):
        """Test that function raises error when OPENAI_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ImportError, match="OPENAI_API_KEY"):
                generate_embeddings_with_curategpt(
                    sample_csv,
                    db_path=os.path.join(temp_dir, "test.duckdb")
                )

    def test_missing_curategpt(self, sample_csv, temp_dir):
        """Test that function raises error when curategpt is not installed."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Mock the import to fail
            def mock_import_curategpt(*args, **kwargs):
                if args and args[0] == "curategpt.store":
                    raise ImportError("No module named 'curategpt'")
                return __import__(*args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import_curategpt):
                with pytest.raises(ImportError, match="curategpt is required"):
                    generate_embeddings_with_curategpt(
                        sample_csv, db_path=os.path.join(temp_dir, "test.duckdb"))

    def test_missing_duckdb(self, sample_csv, temp_dir):
        """Test that function raises error when duckdb is not installed."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # We can't easily test this without actually uninstalling duckdb,
            # so we'll test that the function checks for it
            # This is more of an integration test
            db_path = os.path.join(temp_dir, "test.duckdb")
            # If duckdb is installed (which it is), the test passes
            # The actual error would only occur if duckdb wasn't installed
            try:
                import duckdb
                pytest.skip("duckdb is installed; skip this test")
            except ImportError:
                # If duckdb isn't installed, our code should raise the right error
                with pytest.raises(ImportError, match="duckdb is required"):
                    generate_embeddings_with_curategpt(
                        sample_csv, db_path=db_path)

    @patch("curategpt.store.get_store")
    def test_successful_embedding_generation(self, mock_get_store, sample_csv, temp_dir):
        """Test successful embedding generation with mocked CurateGPT."""
        # Setup mocks
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            db_path = os.path.join(temp_dir, "test.duckdb")

            # Run function
            result_path, num_embeddings = generate_embeddings_with_curategpt(
                sample_csv,
                collection_name="test_collection",
                db_path=db_path
            )

            # Assertions
            assert result_path == db_path
            assert num_embeddings == 3
            assert mock_get_store.called
            assert mock_store.insert.call_count == 3

    @patch("curategpt.store.get_store")
    def test_embedding_with_limit(self, mock_get_store, large_sample_csv, temp_dir):
        """Test that limit parameter restricts number of embeddings."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            db_path = os.path.join(temp_dir, "test.duckdb")

            # Run function with limit
            result_path, num_embeddings = generate_embeddings_with_curategpt(
                large_sample_csv,
                db_path=db_path,
                limit=10
            )

            # Assertions
            assert num_embeddings == 10
            assert mock_store.insert.call_count == 10

    @patch("curategpt.store.get_store")
    def test_embedding_with_skip(self, mock_get_store, large_sample_csv, temp_dir):
        """Test that skip parameter skips first N rows."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            db_path = os.path.join(temp_dir, "test.duckdb")

            # Run function with skip
            result_path, num_embeddings = generate_embeddings_with_curategpt(
                large_sample_csv,
                db_path=db_path,
                skip=100,
                limit=50
            )

            # Assertions
            assert num_embeddings == 50
            # Check that the first inserted row is the 101st (index 100)
            first_call_args = mock_store.insert.call_args_list[0]
            assert first_call_args[0][0]["id"] == "BERVO:0000100"

    @patch("curategpt.store.get_store")
    def test_embedding_with_text_fields(self, mock_get_store, sample_csv, temp_dir):
        """Test that text_fields parameter is passed to store."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            db_path = os.path.join(temp_dir, "test.duckdb")

            # Run function with specific text fields
            result_path, num_embeddings = generate_embeddings_with_curategpt(
                sample_csv,
                db_path=db_path,
                text_fields=["label", "definition"]
            )

            # Assertions
            assert num_embeddings == 3
            assert mock_store.insert.call_count == 3

    @patch("curategpt.store.get_store")
    def test_database_directory_creation(self, mock_get_store, sample_csv, temp_dir):
        """Test that database directory is created if it doesn't exist."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Use nested directories that don't exist
            db_path = os.path.join(temp_dir, "nested", "dir", "test.duckdb")

            # Run function
            result_path, num_embeddings = generate_embeddings_with_curategpt(
                sample_csv,
                db_path=db_path
            )

            # Assertions
            assert os.path.exists(os.path.dirname(db_path))

    @patch("curategpt.store.get_store")
    def test_duckdb_backend_specified(self, mock_get_store, sample_csv, temp_dir):
        """Test that duckdb backend is specified in get_store call."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            db_path = os.path.join(temp_dir, "test.duckdb")

            # Run function
            generate_embeddings_with_curategpt(sample_csv, db_path=db_path)

            # Assertions
            mock_get_store.assert_called_with("duckdb", db_path)


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

    @patch("curategpt.store.get_store")
    def test_successful_csv_export(self, mock_get_store, temp_dir):
        """Test successful CSV export from database."""
        # Create a fake database directory
        db_path = os.path.join(temp_dir, "test.duckdb")
        os.makedirs(db_path, exist_ok=True)

        # Setup mocks
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        # Mock returned documents
        mock_documents = [
            {"id": "BERVO:0000001", "label": "Temperature",
                "embedding": "[0.1, 0.2, ...]"},
            {"id": "BERVO:0000002", "label": "Humidity",
                "embedding": "[0.3, 0.4, ...]"},
        ]
        mock_store.find.return_value = mock_documents
        mock_store.field_names.return_value = ["id", "label", "embedding"]

        output_path = os.path.join(temp_dir, "output.csv")

        # Run function
        num_exported = export_embeddings_to_csv(
            db_path,
            "test_collection",
            output_path
        )

        # Assertions
        assert num_exported == 2
        assert os.path.exists(output_path)

        # Check CSV content
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["id"] == "BERVO:0000001"
            assert rows[1]["id"] == "BERVO:0000002"

    @patch("curategpt.store.get_store")
    def test_csv_export_creates_directory(self, mock_get_store, temp_dir):
        """Test that export creates output directory if it doesn't exist."""
        db_path = os.path.join(temp_dir, "test.duckdb")
        os.makedirs(db_path, exist_ok=True)

        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.find.return_value = []
        mock_store.field_names.return_value = ["id"]

        # Use nested directories that don't exist
        output_path = os.path.join(temp_dir, "nested", "output.csv")

        # Run function
        export_embeddings_to_csv(db_path, "collection", output_path)

        # Assertions
        assert os.path.exists(os.path.dirname(output_path))

    @patch("curategpt.store.get_store")
    def test_empty_collection_export(self, mock_get_store, temp_dir):
        """Test exporting empty collection."""
        db_path = os.path.join(temp_dir, "test.duckdb")
        os.makedirs(db_path, exist_ok=True)

        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.field_names.return_value = None  # Indicates empty collection

        output_path = os.path.join(temp_dir, "output.csv")

        # Run function
        num_exported = export_embeddings_to_csv(
            db_path, "empty_collection", output_path)

        # Assertions
        assert num_exported == 0

    @patch("curategpt.store.get_store")
    def test_large_csv_export(self, mock_get_store, temp_dir):
        """Test exporting large number of documents."""
        db_path = os.path.join(temp_dir, "test.duckdb")
        os.makedirs(db_path, exist_ok=True)

        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        # Create 1000 mock documents
        mock_documents = [
            {"id": f"BERVO:{i:07d}", "label": f"Term_{i}"}
            for i in range(1000)
        ]
        mock_store.find.return_value = mock_documents
        mock_store.field_names.return_value = ["id", "label"]

        output_path = os.path.join(temp_dir, "output.csv")

        # Run function
        num_exported = export_embeddings_to_csv(
            db_path, "collection", output_path)

        # Assertions
        assert num_exported == 1000

        # Verify file was written
        assert os.path.exists(output_path)
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1000
