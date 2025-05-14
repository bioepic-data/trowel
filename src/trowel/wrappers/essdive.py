"""API wrapper for ESS-DIVE."""

# See https://api.ess-dive.lbl.gov/#/Dataset/getDataset

from io import StringIO, BytesIO
import sys
import os
import tempfile
from typing import Tuple, List
import csv
import logging
import polars as pl
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
import openpyxl
import xlrd

BASE_URL = "https://api.ess-dive.lbl.gov"

ENDPOINT = "packages"

USER_HEADERS = {
    "user_agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0",
    "content-type": "application/json",
    "Range": "bytes=0-1000",
}

PARSIBLE_ENCODINGS = ["text/csv",
                      "text/plain",
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                      "https://eml.ecoinformatics.org/eml-2.2.0",
                      ]

PARSIBLE_EXTENSIONS = [
    ".csv",
    ".txt",
    ".tsv",
    ".xlsx",
    ".xls",
    ".xml",
]

logger = logging.getLogger(__name__)


def get_metadata(
    identifiers: list, token: str, outpath: str = "."
) -> Tuple[str, dict, str]:
    """Get metadata from ESS-DIVE for a list of identifiers.
    The identifiers should be DOIs.
    This also requires an authentication token for ESS-DIVE.

    Results are streamed to files in the specified output directory as they are received.

    Args:
        identifiers: List of DOI identifiers
        token: ESS-DIVE authentication token
        outpath: Directory to write output files (defaults to current directory)

    Returns:
        Tuple containing paths to the results, variables frequency file, and file table file
    """
    # Create output file paths
    results_path = f"{outpath}/results.txt"
    filetable_path = f"{outpath}/filetable.txt"

    # Initialize empty files with headers
    results_schema = pl.DataFrame(
        schema={
            "doi": pl.Utf8,
            "id": pl.Utf8,
            "name": pl.Utf8,
            "variables": pl.Utf8,
            "description": pl.Utf8,
            "site_description": pl.Utf8,
            "methods": pl.Utf8,
        }
    )
    results_schema.write_csv(results_path, separator="\t")

    files_schema = pl.DataFrame(
        schema={
            "dataset_id": pl.Utf8,
            "url": pl.Utf8,
            "name": pl.Utf8,
            "encoding": pl.Utf8,
        }
    )
    files_schema.write_csv(filetable_path, separator="\t")

    all_variables = {}  # key is variable name, value is frequency
    headers = {"Authorization": f"Bearer {token}"}

    results_found = False
    files_found = False

    # Dictionary to collect errors
    errors = {
        "no_variables": [],
        "no_site_description": [],
        "no_methods": [],
        "no_files": [],
        "authorization": [],
        "not_found": [],
        "other_errors": []
    }

    # Add tqdm progress bar
    for identifier in tqdm(identifiers, desc="Processing identifiers", unit="entry"):
        # Check on identifier format first
        if identifier.startswith("https://doi.org/"):
            # This is a full DOI, so we need to strip it down
            identifier = identifier.replace("https://doi.org/", "")
        elif identifier.startswith("doi.org/"):
            identifier = identifier.replace("doi.org/", "")

        # clean it up
        identifier = identifier.strip()

        # Check if this is a DOI anyway
        if identifier.startswith("ess-dive"):
            sys.exit(
                f"The provided identifier {identifier} does not appear to be a DOI. Please check the format."
            )
        if not identifier.startswith("doi:"):
            identifier = "doi:" + identifier

        get_packages_url = "{}/{}/{}?&isPublic=true".format(
            BASE_URL, ENDPOINT, identifier
        )
        response = requests.get(
            get_packages_url, headers=headers, verify=True, stream=True)

        if response.status_code == 200:
            # Success - but will need to restructure
            these_results = response.json()
            # Add relevant parts to the dataframe
            essdive_id = these_results["id"]
            name = these_results["dataset"]["name"]
            try:
                variables = normalize_variables(
                    these_results["dataset"]["variableMeasured"]
                )
                for var in variables:
                    if var in all_variables:
                        all_variables[var] += 1
                    else:
                        all_variables[var] = 1
            except KeyError:
                errors["no_variables"].append(identifier)
                variables = []
            desc_text = these_results["dataset"]["description"]
            try:
                site_desc = these_results["dataset"]["spatialCoverage"][0][
                    "description"
                ]
            except KeyError:
                errors["no_site_description"].append(identifier)
                site_desc = ""
            try:
                methods = these_results["dataset"]["measurementTechnique"]
            except KeyError:
                errors["no_methods"].append(identifier)
                methods = []

            # Create entry for results and append to file
            entry = pl.DataFrame(
                {
                    "doi": [identifier],
                    "id": [essdive_id],
                    "name": [name],
                    "variables": ["|".join(variables)],
                    "description": [" ".join(desc_text)],
                    "site_description": [site_desc],
                    "methods": [" ".join(methods)],
                }
            )
            # Append to results file
            with open(results_path, "a") as f:
                entry.write_csv(f, separator="\t", include_header=False)
            results_found = True

            # See if we have file information
            if "distribution" in these_results["dataset"]:
                for raw_entry in these_results["dataset"]["distribution"]:
                    url = raw_entry["contentUrl"]
                    filename = raw_entry["name"]
                    encoding = raw_entry["encodingFormat"]
                    file_entry = pl.DataFrame(
                        {
                            "dataset_id": [essdive_id],
                            "url": [url],
                            "name": [filename],
                            "encoding": [encoding],
                        }
                    )
                    # Append to filetable
                    with open(filetable_path, "a") as f:
                        file_entry.write_csv(
                            f, separator="\t", include_header=False)
                    files_found = True
            else:
                errors["no_files"].append(identifier)
        else:
            # There was an error
            if response.status_code == 401:
                errors["authorization"].append(identifier)
                logger.error(
                    "You may need to refresh your authentication token for ESS-DIVE."
                )
                break
            elif response.status_code == 404:
                errors["not_found"].append(identifier)
            else:
                errors["other_errors"].append(
                    f"{identifier} (status code: {response.status_code})")
                break

    # Check if files have content and log errors if they don't
    if not results_found:
        logger.error(
            "No metadata results were found for the provided identifiers")

    if not files_found:
        logger.error("No files were found for the provided identifiers")

    # Write frequencies to file
    frequencies_path = f"{outpath}/frequencies.txt"
    with open(frequencies_path, "w") as freq_file:
        for freq in all_variables:
            freq_file.write(f"{freq}\t{all_variables[freq]}\n")

    # Log all collected errors
    if any(errors.values()):
        logger.error("The following errors occurred during processing:")

        if errors["no_variables"]:
            logger.error(
                f"No variables found for: {', '.join(errors['no_variables'])}")

        if errors["no_site_description"]:
            logger.error(
                f"No site description found for: {', '.join(errors['no_site_description'])}")

        if errors["no_methods"]:
            logger.error(
                f"No methods found for: {', '.join(errors['no_methods'])}")

        if errors["no_files"]:
            logger.error(
                f"No files found for: {', '.join(errors['no_files'])}")

        if errors["authorization"]:
            logger.error(
                f"Authorization errors for: {', '.join(errors['authorization'])}")

        if errors["not_found"]:
            logger.error(
                f"Datasets not found for: {', '.join(errors['not_found'])}")

        if errors["other_errors"]:
            logger.error(f"Other errors: {', '.join(errors['other_errors'])}")

    return results_path, frequencies_path, filetable_path


def parse_excel_header(content: bytes, filename: str) -> List[str]:
    """Parse header from an Excel file.
    Also normalizes column names.

    Args:
        content: The Excel file content as bytes
        filename: Name of the file (used to determine if it's .xls or .xlsx)

    Returns:
        List of normalized column names
    """
    header_names = []
    file_ext = os.path.splitext(filename.lower())[1]

    # Create a temporary file to save the Excel content
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        if file_ext == '.xlsx':
            # Use openpyxl for .xlsx files
            wb = openpyxl.load_workbook(
                temp_path, read_only=True, data_only=True)

            # Get the first worksheet
            ws = wb.active

            # Read the first row (header)
            for cell in next(ws.rows):
                if cell.value:
                    value = str(cell.value).strip()
                    if value:
                        header_names.append(value.lower().replace("_", " "))
        elif file_ext == '.xls':
            # Use xlrd for .xls files
            wb = xlrd.open_workbook(temp_path)

            # Get the first worksheet
            ws = wb.sheet_by_index(0)

            # Read the first row (header)
            for col in range(ws.ncols):
                value = ws.cell_value(0, col)
                if value:
                    value = str(value).strip()
                    if value:
                        header_names.append(value.lower().replace("_", " "))
    except Exception as e:
        logger.debug(f"Error parsing Excel file {filename}: {str(e)}")
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_path)
        except:
            pass

    return header_names


def get_file_extension(filename: str) -> str:
    """Get the lowercase extension of a file."""
    return os.path.splitext(filename.lower())[1]


def get_column_names(filetable_path: str, outpath: str = ".") -> str:
    """Get dataset column names from ESS-DIVE for a list of data identifiers.
    Takes the name/path of the table, as produced by get_metadata,
    as input.

    Column names and keywords are streamed to an output file as they are collected.

    Args:
        filetable_path: Path to the filetable created by get_metadata
        outpath: Directory to write output file (defaults to current directory)

    Returns:
        Path to the column names output file
    """
    # Define the output file path
    column_names_path = f"{outpath}/column_names.txt"

    # Initialize the column and keyword frequency dictionaries for tracking
    column_frequencies = {}
    keyword_frequencies = {}

    # Load the file as a polars dataframe
    filetable = pl.read_csv(filetable_path, separator="\t")

    # Create dictionaries for tracking errors
    errors = {
        "failed_urls": [],
        "response_errors": [],
        "encoding_errors": [],
        "unsupported_filetype": []
    }

    # Get the set of entries with an encoding value we can parse
    # This includes CSV files and other parsable formats
    tab_data_files = filetable.filter(
        (pl.col("encoding").is_in(PARSIBLE_ENCODINGS) |
         pl.col("name").str.to_lowercase().str.contains("|".join([ext + "$" for ext in PARSIBLE_EXTENSIONS]))) &
        ~pl.col("name").str.to_lowercase().str.starts_with("readme")
    )

    # Get the set of entries that are XML files (could be EML)
    xml_files = filetable.filter(
        (pl.col("name").str.to_lowercase().str.ends_with(".xml"))
    )

    # Get the set of entries that look like they are data dictionaries
    data_dict_files = filetable.filter(pl.col("name").str.contains("dd.csv$"))

    # Remove data dict files and XML files from the tabular files to avoid duplication
    tab_data_files = tab_data_files.join(data_dict_files, on="url", how="anti")
    tab_data_files = tab_data_files.join(xml_files, on="url", how="anti")

    # Initialize the output file
    with open(column_names_path, "w") as f:
        f.write("name\tfrequency\tsource\n")

    # Process files
    file_count = len(tab_data_files) + len(data_dict_files) + len(xml_files)
    logging.info(f"Processing {file_count} files...")

    # First process XML files for keywords
    if len(xml_files) > 0:
        for idx, row in enumerate(tqdm(xml_files.iter_rows(named=True), desc="Processing XML files", unit="file", total=len(xml_files))):
            url = row["url"]
            filename = row["name"]

            try:
                response = requests.get(url, headers=USER_HEADERS, verify=True)
                status_code = response.status_code

                if status_code == 200:
                    try:
                        # Try to decode the content as UTF-8
                        try:
                            response_text = response.content.decode('utf-8')
                        except UnicodeDecodeError:
                            # If UTF-8 fails, try another common encoding
                            try:
                                response_text = response.content.decode(
                                    'latin-1')
                            except UnicodeDecodeError:
                                # Last resort, ignore errors
                                response_text = response.content.decode(
                                    'utf-8', errors='ignore')
                                logger.warning(
                                    f"Had to ignore encoding errors for {url}")

                        # Extract keywords from the XML
                        keywords = parse_eml_keywords(response_text)

                        # Normalize keywords
                        keywords = normalize_variables(keywords)

                        # Update keyword frequencies
                        new_keywords = False
                        for keyword in keywords:
                            if keyword in keyword_frequencies:
                                keyword_frequencies[keyword] += 1
                            else:
                                keyword_frequencies[keyword] = 1
                                new_keywords = True

                        # If we found new keywords, append them to the file
                        if new_keywords:
                            with open(column_names_path, "a") as f:
                                for keyword, freq in keyword_frequencies.items():
                                    if freq == 1:  # Only write new keywords
                                        f.write(
                                            f"{keyword}\t{freq}\tkeyword\n")

                    except Exception as e:
                        errors["encoding_errors"].append(f"{url} ({str(e)})")
                        logger.debug(
                            f"Error processing XML file {filename}: {str(e)}")
                        continue
                else:
                    errors["response_errors"].append(
                        f"{url} (status code: {response.status_code})")
                    continue

            except Exception as e:
                errors["failed_urls"].append(f"{url} ({str(e)})")
                continue

    # Now retrieve the column names, then the data dictionaries
    for i, fileset_name in enumerate([("data files", tab_data_files), ("data dictionaries", data_dict_files)]):
        name, fileset = fileset_name

        for idx, row in enumerate(tqdm(fileset.iter_rows(named=True), desc=f"Processing {name}", unit="file", total=len(fileset))):
            url = row["url"]
            filename = row["name"]
            file_ext = get_file_extension(filename)

            try:
                response = requests.get(
                    url, headers=USER_HEADERS, verify=True, stream=True if file_ext not in [".xls", ".xlsx"] else False
                )
                status_code = response.status_code

                if status_code == 200:
                    try:
                        data_names = []

                        # Process the file based on its type
                        if i == 1 or file_ext == ".csv" or file_ext == ".txt" or file_ext == ".tsv":
                            # Data dictionary or CSV-like file
                            content = response.content

                            # Try to decode the content as UTF-8
                            try:
                                response_text = content.decode('utf-8')
                            except UnicodeDecodeError:
                                # If UTF-8 fails, try another common encoding
                                try:
                                    response_text = content.decode('latin-1')
                                except UnicodeDecodeError:
                                    # Last resort, ignore errors
                                    response_text = content.decode(
                                        'utf-8', errors='ignore')
                                    logger.warning(
                                        f"Had to ignore encoding errors for {url}")

                            if i == 1:  # Data dictionary
                                data_names = parse_data_dictionary(
                                    response_text)
                            else:  # CSV file
                                # For CSV, just get the header
                                for line in response_text.splitlines():
                                    if line.strip():  # Skip empty lines
                                        data_names = parse_header(line)
                                        break  # We only need the first line

                        elif file_ext == ".xlsx" or file_ext == ".xls":
                            # Excel file
                            content = response.content
                            data_names = parse_excel_header(content, filename)

                        else:
                            errors["unsupported_filetype"].append(
                                f"{url} (filetype: {file_ext})")
                            continue

                        # Update column frequencies
                        new_columns = False
                        for name in data_names:
                            if name in column_frequencies:
                                column_frequencies[name] += 1
                            else:
                                column_frequencies[name] = 1
                                new_columns = True

                        # If we found new columns, append them to the file
                        if new_columns:
                            with open(column_names_path, "a") as f:
                                for column, freq in column_frequencies.items():
                                    if freq == 1:  # Only write new columns
                                        f.write(f"{column}\t{freq}\tcolumn\n")

                    except Exception as e:
                        errors["encoding_errors"].append(f"{url} ({str(e)})")
                        logger.debug(f"Encoding error for {url}: {str(e)}")
                        continue
                else:
                    errors["response_errors"].append(
                        f"{url} (status code: {response.status_code})")
                    continue

            except Exception as e:
                errors["failed_urls"].append(f"{url} ({str(e)})")
                continue

    # After processing all files, update the file with final frequencies
    # First identify terms that appear in both columns and keywords and combine their sources
    # Create a dictionary to track combined term frequencies and sources
    all_terms = {}

    # Process column terms first
    for column, freq in column_frequencies.items():
        # If this term also appears as a keyword, combine sources
        if column in keyword_frequencies:
            source = "column|keyword"
            # Sum the frequencies from both sources
            combined_freq = freq + keyword_frequencies[column]
            all_terms[column] = (combined_freq, source)
            # Mark this keyword as processed
            keyword_frequencies[column] = -1  # Mark as already processed
        else:
            all_terms[column] = (freq, "column")

    # Process remaining keywords (those not already combined with columns)
    for keyword, freq in keyword_frequencies.items():
        if freq > 0:  # Only process keywords that weren't already combined
            all_terms[keyword] = (freq, "keyword")

    # Sort by frequency (highest first)
    sorted_terms = sorted(
        all_terms.items(), key=lambda item: item[1][0], reverse=True)

    with open(column_names_path, "w") as f:
        f.write("name\tfrequency\tsource\n")
        for term, (frequency, source) in sorted_terms:
            f.write(f"{term}\t{frequency}\t{source}\n")

    # Log any errors that occurred during processing
    if errors["response_errors"]:
        logger.error(
            f"Response errors for {len(errors['response_errors'])} URLs")
        for url in errors["response_errors"][:5]:  # Log first few errors
            logger.error(f"  {url}")
        if len(errors["response_errors"]) > 5:
            logger.error(f"  ...and {len(errors['response_errors']) - 5} more")

    if errors["failed_urls"]:
        logger.error(f"Failed to process {len(errors['failed_urls'])} URLs")
        for url in errors["failed_urls"][:5]:  # Log first few errors
            logger.error(f"  {url}")
        if len(errors["failed_urls"]) > 5:
            logger.error(f"  ...and {len(errors['failed_urls']) - 5} more")

    if errors["encoding_errors"]:
        logger.error(
            f"Encoding errors for {len(errors['encoding_errors'])} URLs")
        for url in errors["encoding_errors"][:5]:  # Log first few errors
            logger.error(f"  {url}")
        if len(errors["encoding_errors"]) > 5:
            logger.error(f"  ...and {len(errors['encoding_errors']) - 5} more")

    if errors["unsupported_filetype"]:
        logger.error(
            f"Unsupported file types for {len(errors['unsupported_filetype'])} URLs")
        for url in errors["unsupported_filetype"][:5]:  # Log first few errors
            logger.error(f"  {url}")
        if len(errors["unsupported_filetype"]) > 5:
            logger.error(
                f"  ...and {len(errors['unsupported_filetype']) - 5} more")

    return column_names_path


def normalize_variables(variables: list) -> list:
    """Normalize variables from ESS-DIVE."""
    normalized = []
    for var in variables:
        if ">" in var:
            # This is a hierarchy but we just want all terms
            vars = var.split(">")
            for v in vars:
                if v in normalized:
                    continue
                else:
                    normalized.append(v.lower().replace("_", " ").strip())
        else:
            normalized.append(var.lower().replace("_", " "))
    normalized.sort()
    return normalized


def parse_header(header: str) -> list:
    """Parse header from a data file.
    Also normalizes."""
    header_names = []
    reader = csv.reader(StringIO(header))
    for row in reader:
        for name in row:
            if name != "":
                header_names.append(name.lower().replace("_", " ").strip())
    return header_names


def parse_data_dictionary(dd: str) -> list:
    """Parse a data dictionary.
    Also normalizes."""
    data_names = []
    reader = csv.reader(StringIO(dd))
    for row in reader:
        name = row[0]
        if name not in ["", "Column_or_Row_Name"]:
            data_names.append(name.lower().replace("_", " ").strip())
    return data_names


def parse_eml_keywords(content: str) -> List[str]:
    """Parse keywords from an Ecological Metadata Language (EML) XML file.

    Args:
        content: String containing the XML content

    Returns:
        List of keywords extracted from the XML
    """
    keywords = []

    try:
        # Parse the XML content
        # The namespace is important for finding elements correctly
        root = ET.fromstring(content)

        # Find all keyword elements, regardless of their namespace path
        # This handles different EML structures and namespaces
        for keyword_elem in root.findall(".//*{https://eml.ecoinformatics.org/eml-2.2.0}keyword") or \
                root.findall(".//*keyword") or \
                root.findall(".//*{*}keyword"):
            if keyword_elem.text and keyword_elem.text.strip():
                keywords.append(keyword_elem.text.strip().lower())
    except Exception as e:
        logger.debug(f"Error parsing EML XML: {str(e)}")

    return keywords
