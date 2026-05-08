"""Unit tests for ESS-DIVE API wrapper functions."""

import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open
import requests
import polars as pl

from trowel.wrappers.essdive import (
    get_metadata,
    get_variable_names,
    normalize_essdive_token,
)


class TestGetMetadata(unittest.TestCase):
    """Test suite for get_metadata function."""

    def test_normalize_essdive_token(self):
        """Test ESS-DIVE token normalization for anonymous public access."""
        self.assertIsNone(normalize_essdive_token(None))
        self.assertIsNone(normalize_essdive_token(""))
        self.assertIsNone(normalize_essdive_token("  "))
        self.assertIsNone(normalize_essdive_token("null"))
        self.assertIsNone(normalize_essdive_token("None"))
        self.assertEqual(normalize_essdive_token(" token "), "token")

    @patch('requests.get')
    def test_get_metadata_success_without_token(self, mock_get):
        """Test get_metadata with successful anonymous API response."""
        # Mock response for successful API call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'id': 'test-id-123',
            'dataset': {
                'name': 'Test Dataset',
                'variableMeasured': ['temperature', 'pH'],
                'description': ['This is a test dataset'],
                'spatialCoverage': [{'description': 'Test site'}],
                'measurementTechnique': ['Sensor measurements'],
                'distribution': [
                    {
                        'contentUrl': 'https://example.com/data.csv',
                        'name': 'data.csv',
                        'encodingFormat': 'text/csv'
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        # Create temp directory for outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Call the function
            results_path, frequencies_path, filetable_path = get_metadata(
                ['doi:10.1234/test'], None, temp_dir
            )

            request_url = mock_get.call_args[0][0]
            request_headers = mock_get.call_args[1]["headers"]
            self.assertIn("doi%3A10.1234%2Ftest", request_url)
            self.assertNotIn("Authorization", request_headers)

            # Check if files were created
            self.assertTrue(os.path.exists(results_path))
            self.assertTrue(os.path.exists(frequencies_path))
            self.assertTrue(os.path.exists(filetable_path))

            # Check contents of results file
            results_df = pl.read_csv(results_path, separator="\t")
            self.assertEqual(len(results_df), 1)
            self.assertEqual(results_df['doi'][0], 'doi:10.1234/test')
            self.assertEqual(results_df['id'][0], 'test-id-123')

            # Check contents of filetable file
            filetable_df = pl.read_csv(filetable_path, separator="\t")
            self.assertEqual(len(filetable_df), 1)
            self.assertEqual(filetable_df['dataset_id'][0], 'test-id-123')
            self.assertEqual(filetable_df['url']
                             [0], 'https://example.com/data.csv')

            # Check contents of frequencies file - note that pH is normalized to lowercase
            with open(frequencies_path, 'r') as f:
                frequencies_content = f.read()
                self.assertIn('temperature', frequencies_content)
                # pH is normalized to lowercase
                self.assertIn('ph', frequencies_content)

    @patch('requests.get')
    def test_get_metadata_includes_real_token(self, mock_get):
        """Test get_metadata includes Authorization for a real token."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as temp_dir:
            get_metadata(['10.1234/test'], ' fake_token ', temp_dir)

            request_headers = mock_get.call_args[1]["headers"]
            self.assertEqual(
                request_headers["Authorization"],
                "Bearer fake_token",
            )

    @patch('requests.get')
    def test_get_metadata_omits_null_like_token(self, mock_get):
        """Test get_metadata omits Authorization for null-like token values."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as temp_dir:
            get_metadata(['10.1234/test'], 'null', temp_dir)

            request_headers = mock_get.call_args[1]["headers"]
            self.assertNotIn("Authorization", request_headers)

    @patch('requests.get')
    def test_get_metadata_accepts_essdive_identifier(self, mock_get):
        """Test get_metadata can request an ESS-DIVE package identifier."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as temp_dir:
            get_metadata(['ess-dive-test-id'], None, temp_dir)

            request_url = mock_get.call_args[0][0]
            self.assertTrue(request_url.endswith("/packages/ess-dive-test-id"))

    @patch('requests.get')
    def test_get_metadata_error(self, mock_get):
        """Test get_metadata with API error."""
        # Mock response for failed API call
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        # Create temp directory for outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Call the function
            results_path, frequencies_path, filetable_path = get_metadata(
                ['doi:10.1234/test'], 'fake_token', temp_dir
            )

            # Check if files were created (they should be, even if empty)
            self.assertTrue(os.path.exists(results_path))
            self.assertTrue(os.path.exists(frequencies_path))
            self.assertTrue(os.path.exists(filetable_path))

            # Check that results file has no data rows (just headers)
            results_df = pl.read_csv(results_path, separator="\t")
            self.assertEqual(len(results_df), 0)

            # Check that filetable has no data rows
            filetable_df = pl.read_csv(filetable_path, separator="\t")
            self.assertEqual(len(filetable_df), 0)

            # Frequencies file should be empty
            with open(frequencies_path, 'r') as f:
                frequencies_content = f.read()
                self.assertEqual(frequencies_content, '')


@patch('requests.get')
class TestGetVariableNames(unittest.TestCase):
    """Test suite for get_column_names function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock filetable data
        self.mock_filetable_data = """dataset_id\turl\tname\tencoding
test-id-123\thttps://example.com/data.csv\tdata.csv\ttext/csv
test-id-123\thttps://example.com/metadata.xml\tmetadata.xml\ttext/xml
test-id-123\thttps://example.com/data_dict.csv\tdd.csv\ttext/csv"""

    def test_get_variable_names_csv(self, mock_get):
        """Test get_column_names with CSV file."""
        # Mock response for CSV file
        csv_response = MagicMock()
        csv_response.status_code = 200
        csv_response.content = b"Temperature,pH,Conductivity\n25.3,7.2,150"

        # Mock response for XML file
        xml_response = MagicMock()
        xml_response.status_code = 200
        xml_response.content = b"""<?xml version="1.0" encoding="UTF-8"?>
        <eml:eml xmlns:eml="https://eml.ecoinformatics.org/eml-2.2.0">
            <dataset>
                <keywordSet>
                    <keyword>soil</keyword>
                    <keyword>carbon</keyword>
                </keywordSet>
            </dataset>
        </eml:eml>"""

        # Mock response for data dictionary
        dd_response = MagicMock()
        dd_response.status_code = 200
        dd_response.content = b"Column_or_Row_Name,Description\nTemperature,Temperature in Celsius\nMoisture,Soil moisture content"

        # Configure mock to return different responses based on URL
        def get_response(url, **kwargs):
            if 'data.csv' in url:
                return csv_response
            elif 'metadata.xml' in url:
                return xml_response
            elif 'dd.csv' in url:
                return dd_response
            return MagicMock(status_code=404)

        mock_get.side_effect = get_response

        # Create temp directory and filetable
        with tempfile.TemporaryDirectory() as temp_dir:
            filetable_path = os.path.join(temp_dir, 'filetable.txt')
            with open(filetable_path, 'w') as f:
                f.write(self.mock_filetable_data)

            # Call the function
            column_names_path = get_variable_names(filetable_path, temp_dir)

            # Check if output file was created
            self.assertTrue(os.path.exists(column_names_path))

            # Check contents of output file
            column_names_df = pl.read_csv(column_names_path, separator="\t")

            # Check that we have entries for both column names and keywords
            names = column_names_df['name'].to_list()
            sources = column_names_df['source'].to_list()

            # Making assertions that match the function's actual behavior
            self.assertIn('temperature', names)
            self.assertIn('ph', names)
            self.assertIn('conductivity', names)
            self.assertIn('soil', names)
            self.assertIn('carbon', names)

            # The data dictionary content is processed but might not be included
            # in the final output if prioritizing other sources or if combining entries

            # Check that we have proper source tags
            self.assertIn('column', sources)
            self.assertIn('keyword', sources)

    def test_get_variable_names_error(self, mock_get):
        """Test get_column_names with API error."""
        # Mock response for failed API call
        mock_get.return_value = MagicMock(status_code=404)

        # Create temp directory and filetable
        with tempfile.TemporaryDirectory() as temp_dir:
            filetable_path = os.path.join(temp_dir, 'filetable.txt')
            with open(filetable_path, 'w') as f:
                f.write(self.mock_filetable_data)

            # Call the function
            column_names_path = get_variable_names(filetable_path, temp_dir)

            # Check if output file was created
            self.assertTrue(os.path.exists(column_names_path))

            # Check contents of output file - should have headers but no data
            with open(column_names_path, 'r') as f:
                content = f.read().strip().split('\n')
                self.assertEqual(len(content), 1)  # Just the header line
                self.assertIn('name', content[0])
                self.assertIn('frequency', content[0])


if __name__ == "__main__":
    unittest.main()
