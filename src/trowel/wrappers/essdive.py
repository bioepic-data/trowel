"""API wrapper for ESS-DIVE."""

# See https://api.ess-dive.lbl.gov/#/Dataset/getDataset

import logging

import requests

from typing import Iterator

BASE_URL = "https://api.ess-dive.lbl.gov"

ENDPOINT = "packages"

logger = logging.getLogger(__name__)


def get_metadata(identifiers: list) -> Iterator[dict]:
    """Get metadata from ESS-DIVE for a list of identifiers.
    The identifiers should be DOIs.
    """
    results = []

    # this does not work as expected
    # may need auth token even for search?

    for identifier in identifiers:
        get_packages_url = '{}/{}?providerName="{}"&isPublic=true'.format(
            BASE_URL, ENDPOINT, identifier
        )
        get_package_response = requests.get(get_packages_url)

        if get_package_response.status_code == 200:
            # Success
            results.append(get_package_response.json())
        else:
            # There was an error
            logger.error(get_package_response.text)

    return results
