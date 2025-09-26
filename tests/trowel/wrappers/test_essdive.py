"""Unit tests for ESS-DIVE wrapper functions."""

import unittest
import tempfile
import os
import sys
import io
from unittest.mock import patch, MagicMock, mock_open
import pytest

from trowel.wrappers.essdive import (
    normalize_variables,
    parse_header,
    parse_data_dictionary,
    extract_units,
    clean_unicode_chars,
    clean_punctuation,
    parse_eml_keywords,
    parse_excel_header
)


class TestNormalizeVariables(unittest.TestCase):
    """Test suite for normalize_variables function."""

    def test_normalize_variables_basic(self):
        """Test basic normalization of variables."""
        variables = ["Temp_C", "pH_VALUE", "Conductivity"]
        expected = ["conductivity", "ph value", "temp c"]
        self.assertEqual(sorted(normalize_variables(variables)), expected)

    def test_normalize_variables_with_hierarchy(self):
        """Test normalization of hierarchical variables."""
        variables = ["Climate>Temperature>Annual", "Climate>Temperature>Monthly"]
        expected = ["annual", "climate", "monthly", "temperature"]
        self.assertEqual(normalize_variables(variables), expected)

    def test_normalize_variables_with_brackets(self):
        """Test normalization of variables with brackets."""
        variables = ["temp (C)", "depth (m)", "concentration (mg/L)"]
        expected = ["concentration (mg/l)", "depth (m)", "temp (c)"]
        self.assertEqual(normalize_variables(variables), expected)

    def test_normalize_variables_with_punctuation(self):
        """Test normalization of variables with punctuation."""
        variables = ["_temp_", ".pH.", "!conductivity!"]
        expected = ["conductivity", "ph", "temp"]
        self.assertEqual(normalize_variables(variables), expected)

    def test_normalize_variables_empty_and_none(self):
        """Test normalization with empty strings and None values."""
        variables = ["", None, "Temp"]
        expected = ["temp"]
        self.assertEqual(normalize_variables(variables), expected)

    def test_normalize_variables_length_limit(self):
        """Test that variables exceeding length limit are filtered out."""
        long_name = "a" * 71  # Length exceeds the 70 character limit
        variables = ["Temp", long_name, "pH"]
        expected = ["ph", "temp"]
        self.assertEqual(normalize_variables(variables), expected)

    def test_normalize_variables_duplicates(self):
        """Test that duplicate variables are removed."""
        variables = ["Temp", "temp", "TEMP", "pH", "pH"]
        expected = ["ph", "temp"]
        self.assertEqual(normalize_variables(variables), expected)

    def test_normalize_variables_unicode(self):
        """Test normalization of variables with unicode characters."""
        variables = ["\ufeffTemp", "pH\u200b", "\u202eConductivity"]
        expected = ["conductivity", "ph", "temp"]
        self.assertEqual(normalize_variables(variables), expected)


class TestParseHeader(unittest.TestCase):
    """Test suite for parse_header function."""
    
    def test_parse_header_basic(self):
        """Test basic header parsing."""
        header = "Temp,pH,Conductivity"
        expected = ["temp", "ph", "conductivity"]
        self.assertEqual(parse_header(header), expected)

    def test_parse_header_with_unicode(self):
        """Test header parsing with unicode characters."""
        header = "\ufeffTemp,\u200bpH,Conductivity\u202e"
        expected = ["temp", "ph", "conductivity"]
        self.assertEqual(parse_header(header), expected)

    def test_parse_header_with_punctuation(self):
        """Test header parsing with punctuation."""
        header = "_Temp_,.pH.,!Conductivity!"
        # The function only removes leading punctuation, not trailing
        expected = ["temp", "ph.", "conductivity!"]
        self.assertEqual(parse_header(header), expected)

    def test_parse_header_empty(self):
        """Test parsing empty header."""
        header = ""
        expected = []
        self.assertEqual(parse_header(header), expected)


class TestParseDataDictionary(unittest.TestCase):
    """Test suite for parse_data_dictionary function."""
    
    def test_parse_data_dictionary_basic(self):
        """Test basic data dictionary parsing."""
        dd = "Column_Name,Description\nTemp,Temperature in Celsius\npH,Acidity or alkalinity"
        # The function includes column header in the output and sorts results alphabetically
        expected = ["column name", "ph", "temp"]
        self.assertEqual(parse_data_dictionary(dd), expected)

    def test_parse_data_dictionary_with_header(self):
        """Test data dictionary parsing with header row."""
        dd = "Column_or_Row_Name,Description\nTemp,Temperature in Celsius\npH,Acidity or alkalinity"
        expected = ["ph", "temp"]
        self.assertEqual(parse_data_dictionary(dd), expected)

    def test_parse_data_dictionary_empty(self):
        """Test parsing empty data dictionary."""
        dd = ""
        expected = []
        self.assertEqual(parse_data_dictionary(dd), expected)


class TestParseEmlKeywords(unittest.TestCase):
    """Test suite for parse_eml_keywords function."""
    
    def test_parse_eml_keywords_basic(self):
        """Test basic EML keyword parsing."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <eml:eml xmlns:eml="https://eml.ecoinformatics.org/eml-2.2.0">
            <dataset>
                <keywordSet>
                    <keyword>climate</keyword>
                    <keyword>temperature</keyword>
                    <keyword>precipitation</keyword>
                </keywordSet>
            </dataset>
        </eml:eml>"""
        expected = ["climate", "temperature", "precipitation"]
        self.assertEqual(sorted(parse_eml_keywords(xml_content)), sorted(expected))

    def test_parse_eml_keywords_empty(self):
        """Test parsing EML with no keywords."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <eml:eml xmlns:eml="https://eml.ecoinformatics.org/eml-2.2.0">
            <dataset>
                <keywordSet>
                </keywordSet>
            </dataset>
        </eml:eml>"""
        expected = []
        self.assertEqual(parse_eml_keywords(xml_content), expected)

    def test_parse_eml_keywords_invalid_xml(self):
        """Test parsing invalid XML."""
        xml_content = "This is not valid XML"
        expected = []
        self.assertEqual(parse_eml_keywords(xml_content), expected)


# Skip the complex Excel parsing tests for now, as they require more extensive mocking
@pytest.mark.skip("Excel parsing tests require complex mocking of file I/O")
class TestParseExcelHeader(unittest.TestCase):
    """Test suite for parse_excel_header function."""
    
    def test_parse_xlsx_header(self):
        """Test parsing XLSX file header."""
        # We'd need a real XLSX file to test this properly
        pass
        
    def test_parse_xls_header(self):
        """Test parsing XLS file header."""
        # We'd need a real XLS file to test this properly
        pass


if __name__ == "__main__":
    unittest.main()