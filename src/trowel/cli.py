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

    results, frequencies, filetable = get_metadata(identifiers, ESSDIVE_TOKEN)

    with open(os.path.join(outpath, "frequencies.txt"), "w") as freq_file:
        for freq in frequencies:
            freq_file.write(f"{freq}\t{frequencies[freq]}\n")

    with open(os.path.join(outpath, "results.txt"), "w") as results_file:
        results_file.write(str(results))

    with open(os.path.join(outpath, "filetable.txt"), "w") as filetable_file:
        filetable_file.write(str(filetable))


@main.command()
@path_option
@outpath_option
def get_essdive_column_names(path, outpath):
    """Get all column names from all data files.
    Files are those with identifiers retrieved by the get_essdive_metadata function.
    By default, this is filetable.txt.
    Parameters:
    path: Path to a directory containing results from get_essdive_metadata.
    outpath: Directory where output files should be written. If not provided, defaults to the input path.
    """

    if not path:
        # Check current directory for filetable.txt if path not provided
        current_dir = os.getcwd()
        if os.path.exists(os.path.join(current_dir, "filetable.txt")):
            path = current_dir
            logging.info(f"Using current directory '{current_dir}' as input path.")
        else:
            logging.error(
                "No path provided and filetable.txt not found in current directory. Please use --path option.")
            sys.exit()

    if not outpath:
        outpath = path
        logging.info(
            f"No output path provided. Using input path '{path}' as output path.")
    if not os.path.exists(outpath):
        logging.error(
            f"The specified output directory '{outpath}' does not exist.")
        sys.exit()

    filetable_path = os.path.join(outpath, "filetable.txt")

    if not os.path.exists(filetable_path):
        logging.error("Couldn't find a filetable at the provided path.")
        sys.exit()

    column_names = get_column_names(filetable_path)

    with open(os.path.join(outpath, "column_names.txt"), "w") as filetable_file:
        for colname in column_names:
            filetable_file.write(f"{colname}\t{column_names[colname]}\n")


if __name__ == "__main__":
    main()
