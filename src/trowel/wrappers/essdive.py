"""API wrapper for ESS-DIVE."""

# See https://api.ess-dive.lbl.gov/#/Dataset/getDataset

import logging

import requests

from typing import Iterator, Tuple

import polars as pl

BASE_URL = "https://api.ess-dive.lbl.gov"

ENDPOINT = "packages"

logger = logging.getLogger(__name__)


def get_metadata(identifiers: list, token: str) -> Tuple[Iterator[dict], dict]:
    """Get metadata from ESS-DIVE for a list of identifiers.
    The identifiers should be DOIs.
    This also requires an authentication token for ESS-DIVE.
    """
    # Store results in polars dataframe

    all_results = pl.DataFrame()
    all_variables = {}  # key is variable name, value is frequency
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

    # Transform all_results to tsv before returning
    all_results_tsv = all_results.write_csv(separator="\t")

    return all_results_tsv, all_variables


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
