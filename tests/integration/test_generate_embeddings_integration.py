"""Optional end-to-end integration test for embeddings generation/export."""

import csv
import os
import uuid

import pytest
from click.testing import CliRunner

from trowel.cli import main

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_OPENAI_INTEGRATION") != "1",
    reason="Set RUN_OPENAI_INTEGRATION=1 to run OpenAI-backed integration tests.",
)
def test_generate_embeddings_workflow_end_to_end(tmp_path):
    """Run the real CLI workflow: CSV -> DuckDB embeddings -> CSV export."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for this integration test")

    input_csv = tmp_path / "integration_input.csv"
    db_path = tmp_path / "integration.duckdb"
    output_csv = tmp_path / "integration_embeddings.csv"
    collection_name = f"integration_{uuid.uuid4().hex[:8]}"

    with open(input_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "definition"])
        writer.writeheader()
        writer.writerow(
            {
                "id": "BERVO:9999999",
                "label": "Integration test term",
                "definition": "A minimal row to validate end-to-end embedding flow.",
            }
        )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "embeddings",
            "generate-embeddings",
            "-i",
            str(input_csv),
            "-d",
            str(db_path),
            "-c",
            collection_name,
            "-f",
            "label,definition",
            "-l",
            "1",
            "-e",
            str(output_csv),
        ],
    )

    assert result.exit_code == 0, result.output
    assert db_path.exists()
    assert output_csv.exists()

    with open(output_csv, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0].get("id") == "BERVO:9999999"
