"""Pytest configuration and shared fixtures."""

import csv
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_embedding_file(temp_dir):
    """Create a sample embedding file similar to bervo_embeds.csv format.

    This is based on the actual BERVO embedding files in backup/ directory.
    """
    csv_path = os.path.join(temp_dir, "sample_embeddings.csv")

    # Create sample data with ID and embedding vector
    # Real embeddings are 1536 dimensions, we'll use smaller for tests
    rows = [
        {
            "id": "BERVO:0000001",
            "label": "Temperature",
            "embedding": "[0.1, 0.2, 0.3, 0.4, 0.5]"
        },
        {
            "id": "BERVO:0000002",
            "label": "Humidity",
            "embedding": "[0.2, 0.3, 0.4, 0.5, 0.6]"
        },
        {
            "id": "BERVO:0000003",
            "label": "Pressure",
            "embedding": "[0.3, 0.4, 0.5, 0.6, 0.7]"
        },
        {
            "id": "BERVO:0000004",
            "label": "Wind Speed",
            "embedding": "[0.4, 0.5, 0.6, 0.7, 0.8]"
        },
        {
            "id": "BERVO:0000005",
            "label": "Precipitation",
            "embedding": "[0.5, 0.6, 0.7, 0.8, 0.9]"
        }
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "embedding"])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


@pytest.fixture
def sample_prepared_csv(temp_dir):
    """Create a sample prepared CSV file for embedding generation.

    This would be the output of prepare-embeddings command.
    """
    csv_path = os.path.join(temp_dir, "prepared.csv")

    rows = [
        {
            "id": "BERVO:0000001",
            "label": "Temperature",
            "definition": "Temperature is a physical quantity that expresses hot and cold"
        },
        {
            "id": "BERVO:0000002",
            "label": "Humidity",
            "definition": "Humidity is the amount of water vapor present in the air"
        },
        {
            "id": "BERVO:0000003",
            "label": "Pressure",
            "definition": "Pressure is the force applied perpendicular to the surface"
        },
        {
            "id": "BERVO:0000004",
            "label": "Wind Speed",
            "definition": "Wind speed is the rate at which air moves horizontally"
        },
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "definition"])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


@pytest.fixture
def large_prepared_csv(temp_dir):
    """Create a larger prepared CSV file for testing performance.

    Useful for testing limit/skip parameters.
    """
    csv_path = os.path.join(temp_dir, "large_prepared.csv")

    rows = [
        {
            "id": f"BERVO:{i:07d}",
            "label": f"Term_{i}",
            "definition": f"This is the definition for term number {i} in the BERVO ontology"
        }
        for i in range(500)
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "definition"])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before and after each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)
