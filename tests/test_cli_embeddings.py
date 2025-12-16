"""Tests for embedding CLI commands."""

import csv
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import pytest
from click.testing import CliRunner

from trowel.cli import main, generate_embeddings


@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


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
            "definition": "Air temperature"},
        {"id": "BERVO:0000002", "label": "Humidity",
            "definition": "Moisture content"},
        {"id": "BERVO:0000003", "label": "Pressure", "definition": "Air pressure"},
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "definition"])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


class TestGenerateEmbeddingsCommand:
    """Tests for the generate-embeddings CLI command."""

    def test_command_requires_input_file(self, runner):
        """Test that command requires input file."""
        result = runner.invoke(main, ["embeddings", "generate-embeddings"])
        assert result.exit_code != 0
        assert "Error" in result.output or "required" in result.output.lower()

    def test_command_with_nonexistent_file(self, runner, temp_dir):
        """Test that command fails with nonexistent input file."""
        result = runner.invoke(main, [
            "embeddings", "generate-embeddings",
            "-i", os.path.join(temp_dir, "nonexistent.csv")
        ])
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_command_requires_openai_api_key(self, runner, sample_csv, temp_dir):
        """Test that command fails without OPENAI_API_KEY."""
        # Clear environment
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(main, [
                "embeddings", "generate-embeddings",
                "-i", sample_csv,
                "-d", os.path.join(temp_dir, "test.duckdb")
            ])
            assert result.exit_code != 0
            assert "OPENAI_API_KEY" in result.output

    @patch("trowel.cli.generate_embeddings_with_curategpt")
    def test_command_successful_execution(self, mock_generate, runner, sample_csv, temp_dir):
        """Test successful command execution."""
        db_path = os.path.join(temp_dir, "test.duckdb")
        mock_generate.return_value = (db_path, 3)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(main, [
                "embeddings", "generate-embeddings",
                "-i", sample_csv,
                "-d", db_path,
                "-c", "test_collection"
            ])

            assert result.exit_code == 0
            assert "Successfully generated 3 embeddings" in result.output
            assert "Database saved at" in result.output

    @patch("trowel.cli.generate_embeddings_with_curategpt")
    def test_command_with_custom_collection_name(self, mock_generate, runner, sample_csv, temp_dir):
        """Test command with custom collection name."""
        db_path = os.path.join(temp_dir, "test.duckdb")
        mock_generate.return_value = (db_path, 3)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(main, [
                "embeddings", "generate-embeddings",
                "-i", sample_csv,
                "-d", db_path,
                "-c", "my_collection"
            ])

            # Verify the collection name was passed
            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args[1]
            assert call_kwargs["collection_name"] == "my_collection"

    @patch("trowel.cli.export_embeddings_to_csv")
    @patch("trowel.cli.generate_embeddings_with_curategpt")
    def test_command_with_export(self, mock_generate, mock_export, runner, sample_csv, temp_dir):
        """Test command with CSV export."""
        db_path = os.path.join(temp_dir, "test.duckdb")
        export_path = os.path.join(temp_dir, "embeddings.csv")
        mock_generate.return_value = (db_path, 3)
        mock_export.return_value = 3

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(main, [
                "embeddings", "generate-embeddings",
                "-i", sample_csv,
                "-d", db_path,
                "-e", export_path
            ])

            assert result.exit_code == 0
            assert "Exporting embeddings to" in result.output
            assert "Exported 3 embeddings" in result.output
            assert mock_export.called

    @patch("trowel.cli.generate_embeddings_with_curategpt")
    def test_command_with_limit(self, mock_generate, runner, sample_csv, temp_dir):
        """Test command with limit parameter."""
        db_path = os.path.join(temp_dir, "test.duckdb")
        mock_generate.return_value = (db_path, 10)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(main, [
                "embeddings", "generate-embeddings",
                "-i", sample_csv,
                "-d", db_path,
                "-l", "10"
            ])

            # Verify limit was passed
            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args[1]
            assert call_kwargs["limit"] == 10

    @patch("trowel.cli.generate_embeddings_with_curategpt")
    def test_command_with_skip(self, mock_generate, runner, sample_csv, temp_dir):
        """Test command with skip parameter."""
        db_path = os.path.join(temp_dir, "test.duckdb")
        mock_generate.return_value = (db_path, 5)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(main, [
                "embeddings", "generate-embeddings",
                "-i", sample_csv,
                "-d", db_path,
                "-s", "5"
            ])

            # Verify skip was passed
            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args[1]
            assert call_kwargs["skip"] == 5

    @patch("trowel.cli.generate_embeddings_with_curategpt")
    def test_command_with_text_fields(self, mock_generate, runner, sample_csv, temp_dir):
        """Test command with text_fields parameter."""
        db_path = os.path.join(temp_dir, "test.duckdb")
        mock_generate.return_value = (db_path, 3)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(main, [
                "embeddings", "generate-embeddings",
                "-i", sample_csv,
                "-d", db_path,
                "-f", "label,definition"
            ])

            # Verify text_fields were parsed correctly
            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args[1]
            assert call_kwargs["text_fields"] == ["label", "definition"]

    @patch("trowel.cli.export_embeddings_to_csv")
    @patch("trowel.cli.generate_embeddings_with_curategpt")
    def test_command_export_failure_handling(self, mock_generate, mock_export, runner, sample_csv, temp_dir):
        """Test that export failure is handled gracefully."""
        db_path = os.path.join(temp_dir, "test.duckdb")
        export_path = os.path.join(temp_dir, "embeddings.csv")
        mock_generate.return_value = (db_path, 3)
        mock_export.side_effect = Exception("Export failed")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(main, [
                "embeddings", "generate-embeddings",
                "-i", sample_csv,
                "-d", db_path,
                "-e", export_path
            ])

            assert result.exit_code != 0
            assert "Failed to export embeddings" in result.output
            assert "Database was created successfully" in result.output

    @patch("trowel.cli.generate_embeddings_with_curategpt")
    def test_command_import_error_handling(self, mock_generate, runner, sample_csv, temp_dir):
        """Test that ImportError from missing dependencies is handled."""
        db_path = os.path.join(temp_dir, "test.duckdb")
        mock_generate.side_effect = ImportError("curategpt is required")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(main, [
                "embeddings", "generate-embeddings",
                "-i", sample_csv,
                "-d", db_path
            ])

            assert result.exit_code != 0
            assert "curategpt is required" in result.output

    @patch("trowel.cli.generate_embeddings_with_curategpt")
    def test_command_general_error_handling(self, mock_generate, runner, sample_csv, temp_dir):
        """Test that general exceptions are handled."""
        db_path = os.path.join(temp_dir, "test.duckdb")
        mock_generate.side_effect = Exception(
            "Unexpected error during embedding")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(main, [
                "embeddings", "generate-embeddings",
                "-i", sample_csv,
                "-d", db_path
            ])

            assert result.exit_code != 0
            assert "Embedding generation failed" in result.output

    def test_command_help_text(self, runner):
        """Test that help text is available."""
        result = runner.invoke(
            main, ["embeddings", "generate-embeddings", "--help"])
        assert result.exit_code == 0
        assert "Generate embeddings for CSV data" in result.output
        assert "OPENAI_API_KEY" in result.output
        assert "DuckDB" in result.output

    @patch("trowel.cli.generate_embeddings_with_curategpt")
    def test_command_logs_next_steps(self, mock_generate, runner, sample_csv, temp_dir):
        """Test that command logs helpful next steps after export."""
        db_path = os.path.join(temp_dir, "test.duckdb")
        export_path = os.path.join(temp_dir, "embeddings.csv")
        mock_generate.return_value = (db_path, 3)

        with patch("trowel.cli.export_embeddings_to_csv", return_value=3):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = runner.invoke(main, [
                    "embeddings", "generate-embeddings",
                    "-i", sample_csv,
                    "-d", db_path,
                    "-e", export_path
                ])

                assert "find-similar" in result.output
                assert "visualize-clusters" in result.output

    @patch("trowel.cli.generate_embeddings_with_curategpt")
    def test_default_db_path_used(self, mock_generate, runner, sample_csv, temp_dir):
        """Test that default database path is used when not specified."""
        mock_generate.return_value = ("./backup/db.duckdb", 3)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(main, [
                "embeddings", "generate-embeddings",
                "-i", sample_csv
            ])

            # Verify default db_path was used
            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args[1]
            assert call_kwargs["db_path"] == "./backup/db.duckdb"

    @patch("trowel.cli.generate_embeddings_with_curategpt")
    def test_default_collection_name_used(self, mock_generate, runner, sample_csv, temp_dir):
        """Test that default collection name is used when not specified."""
        mock_generate.return_value = ("./backup/db.duckdb", 3)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(main, [
                "embeddings", "generate-embeddings",
                "-i", sample_csv
            ])

            # Verify default collection_name was used
            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args[1]
            assert call_kwargs["collection_name"] == "embeddings"
