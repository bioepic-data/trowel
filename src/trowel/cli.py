"""CLI for trowel"""

import logging

import click
from trowel.wrappers.essdive import get_metadata

__all__ = [
    "main",
]

path_option = click.option("-p", "--path", help="Path to a file or directory.")


def main():
    """
    CLI for trowel.

    :param verbose: Verbosity while running.
    :param quiet: Boolean to be quiet or verbose.
    """

    logging.basicConfig()
    logger = logging.root


@path_option
def get_essdive_metadata(path):
    """Given a file containing one DOI per line, return metadata from ESS-DIVE."""
    with open(path, "r") as f:
        identifiers = f.readlines()

    results = get_metadata(identifiers)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
