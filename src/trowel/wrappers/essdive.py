"""API wrapper for ESS-DIVE."""

# See https://api.ess-dive.lbl.gov/#/Dataset/getDataset

import logging

import requests

from typing import Iterator

import polars as pl

BASE_URL = "https://api.ess-dive.lbl.gov"

ENDPOINT = "packages"

logger = logging.getLogger(__name__)


def get_metadata(identifiers: list, token: str) -> Iterator[dict]:
    """Get metadata from ESS-DIVE for a list of identifiers.
    The identifiers should be DOIs.
    This also requires an authentication token for ESS-DIVE.
    """
    # Store results in polars dataframe

    all_results = pl.DataFrame()
    header_authorization = "bearer {}".format(token)
    headers = {"Authorization": header_authorization}

    for identifier in identifiers:

        # Check on identifier format first
        if not identifier.startswith("doi:"):
            identifier = "doi:" + identifier

        get_packages_url = "{}/{}/{}?&isPublic=true".format(
            BASE_URL, ENDPOINT, identifier
        )
        print(get_packages_url)
        response = requests.get(get_packages_url, headers=headers)

        if response.status_code == 200:
            # Success - but will need to restructure
            these_results = response.json()
            # Add relevant parts to the dataframe
            print(these_results)
            essdive_id = these_results["id"]
            name = these_results["dataset"]["name"]
            try:
                variables = these_results["dataset"]["variableMeasured"]
            except KeyError:
                logger.error(f"No variables found for {identifier}")
                variables = []
            desc_text = these_results["dataset"]["description"]
            site_desc = these_results["dataset"]["spatialCoverage"][0]["description"]
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
            logger.error(f"Error in response from ESS-DIVE: {response.status_code}")
            logger.error(response.text)
            if response.status_code == 401:
                logger.error(
                    "You may need to refresh your authentication token for ESS-DIVE."
                )
            break

    # Transform all_results to tsv before returning
    all_results_tsv = all_results.write_csv(separator="\t")

    return all_results_tsv
