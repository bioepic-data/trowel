"""CLI for trowel"""

import logging
import os
import sys

import click
from trowel.wrappers.essdive import get_metadata, get_column_names

__all__ = [
    "main",
]

path_option = click.option(
    "-p", "--path", help="Path to a file or directory.", required=False)
outpath_option = click.option(
    "-o", "--outpath", help="Directory where output files should be written.", required=False, default=".")

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
@outpath_option
def get_essdive_metadata(path, outpath):
    """Given a file containing one DOI per line, return metadata from ESS-DIVE.
    Parameters:
    path: Path to a file containing DOIs, with or without the doi prefix, one per line.
    outpath: Directory where output files should be written.
    """

    if not path:
        logging.error(
            "You must provide a path to a file containing DOIs with the --path option.")
        sys.exit()

    if not os.path.exists(outpath):
        logging.error(
            f"The specified output directory '{outpath}' does not exist.")
        sys.exit()

    with open(path, "r") as f:
        identifiers = f.readlines()

    results_path, frequencies_path, filetable_path = get_metadata(
        identifiers, ESSDIVE_TOKEN, outpath)

    logging.info(f"Results written to {results_path}")
    logging.info(f"Frequencies written to {frequencies_path}")
    logging.info(f"File table written to {filetable_path}")


@main.command()
@path_option
@outpath_option
def get_essdive_column_names(path, outpath):
    """Get all column names from all data files.
    Files are those with identifiers retrieved by the get_essdive_metadata function.
    By default, this is filetable.txt.
    Parameters:
    path: Path to a directory containing the filetable file.
    outpath: Directory where output files should be written. Assumed to be the same as path if not provided.
    """

    if not path:
        logging.error(
            "You must provide a path to the directory containing output from get_essdive_metadata.")
        sys.exit()
    if not os.path.exists(path):
        logging.error(
            f"The specified path '{path}' does not exist.")
        sys.exit()

    filetable_path = os.path.join(outpath, "filetable.txt")

    if not os.path.exists(filetable_path):
        logging.error(
            "Could not find file table at specified path.")
        sys.exit()

    if not outpath:
        outpath = path
        logging.info(
            f"Output path not provided. Using the same path as input: {outpath}")
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        logging.info(
            f"Output path '{outpath}' does not exist. Creating it.")

    column_names_path = get_column_names(filetable_path, outpath)

    logging.info(f"Column names written to {column_names_path}")


if __name__ == "__main__":
    main()
