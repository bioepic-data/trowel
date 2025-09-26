"""CLI for trowel"""

import logging
import os
import sys

import click
from trowel.wrappers.essdive import get_metadata, get_variable_names
from trowel.utils.matching_utils import match_terms

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

    if not ESSDIVE_TOKEN:
        logging.error("ESSDIVE_TOKEN is not set. Cannot proceed.")
        sys.exit(1)

    results_path, frequencies_path, filetable_path = get_metadata(
        identifiers, ESSDIVE_TOKEN, outpath)

    logging.info(f"Results written to {results_path}")
    logging.info(f"Frequencies written to {frequencies_path}")
    logging.info(f"File table written to {filetable_path}")


@main.command()
@path_option
@outpath_option
def get_essdive_variables(path, outpath):
    """Get all variable names from all data files and keywords from XML files.
    Also extracts variable names from data dictionaries if present and
    compiles them into a single list (data_dictionaries.tsv).
    Files are those with identifiers retrieved by the get_essdive_metadata function.
    By default, this is filetable.txt."""

    # If path is not provided, look for filetable.txt in the outpath directory
    if not path:
        filetable_path = os.path.join(outpath, "filetable.txt")
    else:
        filetable_path = path

    if not os.path.exists(filetable_path):
        logging.error(f"The filetable file {filetable_path} does not exist.")
        logging.error(
            "You must run get_essdive_metadata first to get the filetable.")
        sys.exit()

    if not os.path.exists(outpath):
        logging.error(
            f"The specified output directory '{outpath}' does not exist.")
        sys.exit()

    variable_names_path = get_variable_names(filetable_path, outpath)

    logging.info(f"Variable names and keywords written to {variable_names_path}")

# TODO: incorporate LLM embedding similarity matching for terms


@main.command()
@click.option('-t', '--terms-file', help='Path to a TSV file with terms in the first column.', required=True)
@click.option('-l', '--list-file', help='Path to a file with terms, one per line.', required=True)
@click.option('-o', '--output', help='Path where the output file should be written.', required=False)
@click.option('-f', '--fuzzy', is_flag=True, help='Enable fuzzy matching for terms without exact matches.', default=False)
@click.option('-s', '--similarity-threshold', help='Minimum similarity score (0-100) for fuzzy matches.', type=float, default=80.0)
def match_term_lists(terms_file, list_file, output, fuzzy, similarity_threshold):
    """Match terms from a TSV file against a list of terms in another file.

    This command takes two files:
    1. A TSV file with terms in the first column
    2. A text file with terms, one per line

    It then produces a new file containing all terms from the first file with a new column
    indicating whether there was an exact match in the second file, and what the match was.

    If fuzzy matching is enabled with --fuzzy, it will also look for approximate matches
    when an exact match is not found, using the Levenshtein distance algorithm.
    """
    if not os.path.exists(terms_file):
        logging.error(f"The terms file {terms_file} does not exist.")
        sys.exit(1)

    if not os.path.exists(list_file):
        logging.error(f"The list file {list_file} does not exist.")
        sys.exit(1)

    if fuzzy:
        logging.info(
            f"Fuzzy matching enabled with similarity threshold of {similarity_threshold}%")

    result_path = match_terms(terms_file, list_file,
                              output, fuzzy, similarity_threshold)

    if result_path:
        logging.info(f"Matched terms written to {result_path}")
    else:
        logging.error("Failed to match terms between files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
