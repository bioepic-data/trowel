"""API wrapper for ESS-DIVE."""

# See https://api.ess-dive.lbl.gov/#/Dataset/getDataset

import logging

import requests

from typing import Iterator, Tuple

import polars as pl

BASE_URL = "https://api.ess-dive.lbl.gov"

ENDPOINT = "packages"

USER_HEADERS = {
    "user_agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0",
    "content-type": "application/json",
    "Range": "bytes=0-1000",
}

logger = logging.getLogger(__name__)


def get_metadata(
    identifiers: list, token: str
) -> Tuple[Iterator[dict], dict, Iterator[dict]]:
    """Get metadata from ESS-DIVE for a list of identifiers.
    The identifiers should be DOIs.
    This also requires an authentication token for ESS-DIVE.
    """
    # Store results in polars dataframe

    all_results = pl.DataFrame()
    all_variables = {}  # key is variable name, value is frequency
    all_files = pl.DataFrame()
    header_authorization = "bearer {}".format(token)
    headers = {"Authorization": header_authorization}

    for identifier in identifiers:

        # Check on identifier format first
        if not identifier.startswith("doi:"):
            identifier = "doi:" + identifier

        # clean it up
        identifier = identifier.strip()

        get_packages_url = "{}/{}/{}?&isPublic=true".format(
            BASE_URL, ENDPOINT, identifier
        )
        response = requests.get(get_packages_url, headers=headers)

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
                logger.error(f"No variables found for {identifier}")
                variables = []
            desc_text = these_results["dataset"]["description"]
            try:
                site_desc = these_results["dataset"]["spatialCoverage"][0][
                    "description"
                ]
            except KeyError:
                logger.error(f"No site description found for {identifier}")
                site_desc = ""
            try:
                methods = these_results["dataset"]["measurementTechnique"]
            except KeyError:
                logger.error(f"No methods found for {identifier}")
                methods = []
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
            all_results.vstack(entry, in_place=True)

            # See if we have file information
            if "distribution" in these_results["dataset"]:
                for raw_entry in these_results["dataset"]["distribution"]:
                    url = raw_entry["contentUrl"]
                    filename = raw_entry["name"]
                    encoding = raw_entry["encodingFormat"]
                    entry = pl.DataFrame(
                        {
                            "dataset_id": [essdive_id],
                            "url": [url],
                            "name": [filename],
                            "encoding": [encoding],
                        }
                    )
                    all_files.vstack(entry, in_place=True)
            else:
                logger.error(f"No files found for {identifier}")
        else:
            # There was an error
            if response.status_code == 401:
                logger.error(f"Error in response: {response.status_code}")
                logger.error(
                    "You may need to refresh your authentication token for ESS-DIVE."
                )
                break
            elif response.status_code == 404:
                logger.error(f"No dataset found for {identifier}.")
                logger.error(response.text)
            else:
                logger.error(f"Error in response from ESS-DIVE: {response.status_code}")
                logger.error(response.text)
                break

    # Transform dataframes to tsv before returning
    all_results_tsv = all_results.write_csv(separator="\t")
    all_files_tsv = all_files.write_csv(separator="\t")

    return all_results_tsv, all_variables, all_files_tsv


def get_column_names(filetable_path: str) -> dict:
    """Get dataset column from ESS-DIVE for a list of data identifiers.
    Takes the name/path of the table, as produced by get_metadata,
    as input.
    We don't need the entirety of each dataset, just the column names.
    """

    all_columns = {}  # key is column name, value is frequency

    # TODO: check to see if there is a data dictionary first
    # if so, we'll just read that and skip the other files
    # Otherwise, we'll iterate through all files

    # TODO: just CSV, TSV, etc

    # Load the file as a polars dataframe
    filetable = pl.read_csv(filetable_path, separator="\t")

    for url in filetable["url"]:
        try:
            response = requests.get(url, headers=USER_HEADERS, verify=True, stream=True)
            status_code = response.status_code
            if status_code == 200:
                print(response.text)
            else:
                logger.error(f"Error in response: {response.status_code}")
                return None
        except Exception as e:
            print(f"Encountered an error: {e}")
            return None


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
