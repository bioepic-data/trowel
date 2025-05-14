"""CLI for trowel"""

import logging
import os
import sys

import click
from trowel.wrappers.essdive import get_metadata, get_column_names

__all__ = [
    "main",
]

# Configure logging to suppress DEBUG messages from urllib3 and other chatty libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("connectionpool").setLevel(logging.WARNING)

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

    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # Set up console handler with a cleaner format
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


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
    """Get all column names from all data files and keywords from XML files.
    Files are those with identifiers retrieved by the get_essdive_metadata function.
    By default, this is filetable.txt."""

    # If path is not provided, look for filetable.txt in the outpath directory
    if not path:
        filetable_path = os.path.join(outpath, "filetable.txt")
    else:
        filetable_path = path

    if not os.path.exists(filetable_path):
        logging.error(f"The filetable file {filetable_path} does not exist.")
        logging.error("You must run get_essdive_metadata first to get the filetable.")
        sys.exit()

    if not os.path.exists(outpath):
        logging.error(f"The specified output directory '{outpath}' does not exist.")
        sys.exit()

    column_names_path, keywords_path = get_column_names(filetable_path, outpath)
    
    logging.info(f"Column names written to {column_names_path}")
    logging.info(f"Keywords written to {keywords_path}")


if __name__ == "__main__":
    main()
