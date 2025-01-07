"""CLI for trowel"""

import logging
import os
import sys

import click
from trowel.wrappers.essdive import get_metadata

__all__ = [
    "main",
]

path_option = click.option("-p", "--path", help="Path to a file or directory.")

# Get token from environment
ESSDIVE_TOKEN = os.getenv("ESSDIVE_TOKEN")
if not ESSDIVE_TOKEN:
    logging.error(
        "You must set the ESS-DIVE authentication token as the ESSDIVE_TOKEN environment variable."
        "\nSee https://docs.ess-dive.lbl.gov/programmatic-tools/ess-dive-dataset-api#get-access"
    )


@click.group()
def main():
    """
    CLI for trowel.

    :param verbose: Verbosity while running.
    :param quiet: Boolean to be quiet or verbose.
    """

    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)


@main.command()
@path_option
def get_essdive_metadata(path):
    """Given a file containing one DOI per line, return metadata from ESS-DIVE."""

    if not path:
        logging.error("You must provide a path to a file containing DOIs.")
        sys.exit()

    with open(path, "r") as f:
        identifiers = f.readlines()

    results = get_metadata(identifiers, ESSDIVE_TOKEN)
    print(results)


if __name__ == "__main__":
    main()
