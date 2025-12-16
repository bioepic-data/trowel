"""CLI for trowel"""

import logging
import os
import sys

import click
import requests
from trowel.wrappers.essdive import get_metadata, get_variable_names
from trowel.utils.matching_utils import match_terms
from trowel.utils.embedding_utils import (
    prepare_embedding_csv,
    load_embeddings_from_csv,
    load_embeddings_from_duckdb,
    get_top_level_categories,
)
from trowel.utils.similarity_utils import (
    compute_cosine_similarity,
    find_similar_terms,
    find_similar_terms_cross_collection,
)
from trowel.utils.visualization_utils import (
    plot_clusters_pca,
    plot_clusters_tsne,
    plot_similarity_heatmap,
    create_category_color_map,
)

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
@click.option('-o', '--output', help='Output file path for BERVO CSV.', required=False, default='bervo.csv')
def get_bervo(output):
    """Download the BERVO ontology from the official source.

    BERVO (Biogeochemical and Ecological Processes Ontology) is a comprehensive
    ontology for environmental science variables and measurements.

    Example:
        trowel get-bervo -o bervo.csv
        trowel get-bervo  # Downloads to bervo.csv by default
    """
    bervo_url = 'https://docs.google.com/spreadsheets/d/1mS8VVtr-m24vZ7nQUtUbQrN8r-UBy3AwRzTfQsmwVL8/export?exportFormat=csv'

    logging.info(f"Downloading BERVO ontology from {bervo_url}...")

    try:
        resp = requests.get(bervo_url, stream=True)
        resp.raise_for_status()

        with open(output, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logging.info(f"BERVO ontology downloaded successfully to {output}")
        logging.info(f"File size: {os.path.getsize(output)} bytes")

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download BERVO ontology: {e}")
        sys.exit(1)


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
@click.option('-w', '--workers', help='Number of parallel workers for file processing.', type=int, default=10)
def get_essdive_variables(path, outpath, workers):
    """Get all variable names from all data files and keywords from XML files.
    Also extracts variable names from data dictionaries if present and
    compiles them into a single list (data_dictionaries.tsv).
    Files are those with identifiers retrieved by the get_essdive_metadata function.
    By default, this is filetable.tsv."""

    # If path is not provided, look for filetable.tsv in the outpath directory
    if not path:
        filetable_path = os.path.join(outpath, "filetable.tsv")
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

    # Look for results file in the same directory as the filetable
    results_path = os.path.join(os.path.dirname(filetable_path), "results.tsv")
    if not os.path.exists(results_path):
        # Try in the output directory if not found with filetable
        results_path = os.path.join(outpath, "results.tsv")
        if not os.path.exists(results_path):
            logging.warning(
                "Results file not found. Dataset mapping will not be available.")
            results_path = None

    variable_names_path = get_variable_names(
        filetable_path, results_path, outpath, max_workers=workers)

    logging.info(
        f"Variable names and keywords written to {variable_names_path}")

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


@main.group()
def embeddings():
    """Commands for working with embeddings and term clustering."""
    pass


@embeddings.command()
@click.option('-i', '--input', 'input_file', help='Path to input CSV file.', required=True)
@click.option('-o', '--output', 'output_file', help='Path to output CSV file.', required=True)
@click.option('-c', '--columns', help='Comma-separated list of column indices to extract (0-indexed).', required=True)
@click.option('--skip-rows', type=int, default=0, help='Number of header rows to skip.')
def prepare_embeddings(input_file, output_file, columns, skip_rows):
    """Prepare a CSV file for embedding by selecting specific columns.

    Example:
        trowel embeddings prepare-embeddings -i bervo.csv -o bervo_prepared.csv -c 0,1,6,12
    """
    if not os.path.exists(input_file):
        logging.error(f"Input file {input_file} does not exist.")
        sys.exit(1)

    try:
        column_indices = [int(x.strip()) for x in columns.split(',')]
    except ValueError:
        logging.error("Column indices must be comma-separated integers.")
        sys.exit(1)

    prepare_embedding_csv(input_file, output_file,
                          column_indices, skip_rows=skip_rows)
    logging.info(f"Prepared CSV written to {output_file}")


@embeddings.command()
@click.option('-e', '--embeddings', 'embedding_file', help='Path to embedding CSV file from CurateGPT.', required=True)
@click.option('-o', '--output', 'output_dir', help='Directory for output files.', required=False, default='.')
def load_embeddings(embedding_file, output_dir):
    """Load embeddings from a CurateGPT-generated CSV file and compute similarity metrics.

    This command:
    1. Loads embeddings and labels
    2. Computes cosine similarity matrix
    3. Saves results to files for downstream analysis

    Example:
        trowel embeddings load-embeddings -e bervo_embeds.csv -o ./analysis
    """
    if not os.path.exists(embedding_file):
        logging.error(f"Embedding file {embedding_file} does not exist.")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info(f"Loading embeddings from {embedding_file}...")
    labels, vectors = load_embeddings_from_csv(embedding_file)

    logging.info("Computing cosine similarity matrix...")
    similarity_matrix = compute_cosine_similarity(vectors)

    # Save results
    import json
    results = {
        'num_terms': len(labels),
        'embedding_dimension': len(vectors[0]) if vectors else 0,
        'similarity_stats': {
            'min': float(similarity_matrix.min()),
            'max': float(similarity_matrix.max()),
            'mean': float(similarity_matrix.mean()),
        }
    }

    results_file = os.path.join(output_dir, 'embedding_stats.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(
        f"Embedding statistics: {len(labels)} terms, dimension {len(vectors[0]) if vectors else 0}")
    logging.info(f"Similarity stats - min: {results['similarity_stats']['min']:.4f}, "
                 f"max: {results['similarity_stats']['max']:.4f}, "
                 f"mean: {results['similarity_stats']['mean']:.4f}")
    logging.info(f"Results saved to {results_file}")


@embeddings.command()
@click.option('-e', '--embeddings', 'embedding_file', help='Path to embedding CSV file from CurateGPT.', required=True)
@click.option('-q', '--query', help='Query term to find similar terms for.', required=True)
@click.option('-n', '--top-n', type=int, default=10, help='Number of similar terms to return.')
@click.option('-o', '--output', 'output_file', help='Optional output file for results.', required=False)
def find_similar(embedding_file, query, top_n, output_file):
    """Find the most similar terms to a query term.

    Example:
        trowel embeddings find-similar -e bervo_embeds.csv -q BERVO:0000026 -n 10
    """
    if not os.path.exists(embedding_file):
        logging.error(f"Embedding file {embedding_file} does not exist.")
        sys.exit(1)

    logging.info(f"Loading embeddings from {embedding_file}...")
    labels, vectors = load_embeddings_from_csv(embedding_file)

    logging.info("Computing cosine similarity matrix...")
    similarity_matrix = compute_cosine_similarity(vectors)

    logging.info(f"Finding {top_n} terms most similar to '{query}'...")
    similar = find_similar_terms(query, labels, similarity_matrix, top_n=top_n)

    if not similar:
        logging.warning(f"No similar terms found for '{query}'.")
        sys.exit(1)

    # Print results
    print(f"\nTop {top_n} terms most similar to '{query}':")
    print("-" * 70)
    for i, (term, score) in enumerate(similar, 1):
        print(f"{i:2d}. {term:40s} (similarity: {score:.4f})")

    # Optionally write to file
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"Top {top_n} terms most similar to '{query}'\n")
            f.write("-" * 70 + "\n")
            for i, (term, score) in enumerate(similar, 1):
                f.write(f"{i:2d}. {term:40s} (similarity: {score:.4f})\n")
        logging.info(f"Results written to {output_file}")


@embeddings.command()
@click.option('-e', '--embeddings', 'embedding_file', help='Path to embedding CSV file from CurateGPT.', required=True)
@click.option('-m', '--method', type=click.Choice(['pca', 'tsne']), default='pca',
              help='Dimensionality reduction method.')
@click.option('-o', '--output', 'output_file', help='Output file path for the plot.', required=False)
@click.option('--label-interval', type=int, default=100, help='Interval for labeling points.')
def visualize_clusters(embedding_file, method, output_file, label_interval):
    """Visualize term clusters using dimensionality reduction.

    Example:
        trowel embeddings visualize-clusters -e bervo_embeds.csv -m pca -o clusters_pca.png
        trowel embeddings visualize-clusters -e bervo_embeds.csv -m tsne -o clusters_tsne.png
    """
    if not os.path.exists(embedding_file):
        logging.error(f"Embedding file {embedding_file} does not exist.")
        sys.exit(1)

    logging.info(f"Loading embeddings from {embedding_file}...")
    labels, vectors = load_embeddings_from_csv(embedding_file)

    title = f"Term Clustering using {method.upper()}"

    if method == 'pca':
        plot_clusters_pca(labels, vectors, output_file=output_file, title=title,
                          label_interval=label_interval)
    else:  # tsne
        plot_clusters_tsne(labels, vectors, output_file=output_file, title=title,
                           label_interval=label_interval)

    if output_file:
        logging.info(f"Plot saved to {output_file}")
    else:
        logging.info("Plot displayed.")


@embeddings.command()
@click.option('-s', '--source-csv', help='Path to source CSV with term definitions.', required=True)
@click.option('-e', '--embeddings', 'embedding_file', help='Path to embedding CSV file from CurateGPT.', required=True)
@click.option('-o', '--output', 'output_file', help='Output file path for the plot.', required=False)
@click.option('--label-interval', type=int, default=100, help='Interval for labeling points.')
def visualize_by_category(source_csv, embedding_file, output_file, label_interval):
    """Visualize term clusters colored by ontology category.

    Example:
        trowel embeddings visualize-by-category -s bervo.csv -e bervo_embeds.csv -o clusters_categorical.png
    """
    if not os.path.exists(source_csv):
        logging.error(f"Source CSV {source_csv} does not exist.")
        sys.exit(1)

    if not os.path.exists(embedding_file):
        logging.error(f"Embedding file {embedding_file} does not exist.")
        sys.exit(1)

    logging.info(f"Loading category mappings from {source_csv}...")
    id_to_label, id_to_top_level = get_top_level_categories(
        source_csv, skip_rows=1)

    logging.info(f"Loading embeddings from {embedding_file}...")
    labels, vectors = load_embeddings_from_csv(embedding_file)

    logging.info("Creating category-based color map...")
    colors, category_to_color = create_category_color_map(
        labels, id_to_top_level)

    title = "Term Clustering by Category"

    plot_clusters_pca(labels, vectors, colors=colors, output_file=output_file,
                      title=title, label_interval=label_interval)

    if output_file:
        logging.info(f"Plot saved to {output_file}")
    else:
        logging.info("Plot displayed.")


@embeddings.command()
@click.option('-e', '--embeddings', 'embedding_file', help='Path to embedding CSV file from CurateGPT.', required=True)
@click.option('-n', '--num-terms', type=int, default=50, help='Number of terms to include in heatmap.')
@click.option('-o', '--output', 'output_file', help='Output file path for the heatmap.', required=False)
def visualize_heatmap(embedding_file, num_terms, output_file):
    """Visualize similarity as a heatmap for a subset of terms.

    Example:
        trowel embeddings visualize-heatmap -e bervo_embeds.csv -n 50 -o similarity_heatmap.png
    """
    if not os.path.exists(embedding_file):
        logging.error(f"Embedding file {embedding_file} does not exist.")
        sys.exit(1)

    logging.info(f"Loading embeddings from {embedding_file}...")
    labels, vectors = load_embeddings_from_csv(embedding_file)

    logging.info("Computing cosine similarity matrix...")
    similarity_matrix = compute_cosine_similarity(vectors)

    plot_similarity_heatmap(labels, similarity_matrix, num_terms=num_terms,
                            output_file=output_file)

    if output_file:
        logging.info(f"Heatmap saved to {output_file}")
    else:
        logging.info("Heatmap displayed.")


@embeddings.command()
@click.option('-b', '--bervo-embeddings', help='Path to BERVO embedding CSV file.', required=True)
@click.option('-n', '--new-embeddings', help='Path to new terms embedding CSV file.', required=True)
@click.option('-t', '--top-n', type=int, default=25, help='Number of top pairs to return.')
@click.option('-o', '--output', 'output_file', help='Optional output file for results.', required=False)
def cross_collection_similarity(bervo_embeddings, new_embeddings, top_n, output_file):
    """Find the most similar term pairs between two collections.

    Example:
        trowel embeddings cross-collection-similarity -b bervo_embeds.csv -n new_vars_embeds.csv -t 25
    """
    if not os.path.exists(bervo_embeddings):
        logging.error(
            f"BERVO embedding file {bervo_embeddings} does not exist.")
        sys.exit(1)

    if not os.path.exists(new_embeddings):
        logging.error(f"New embeddings file {new_embeddings} does not exist.")
        sys.exit(1)

    logging.info("Loading embeddings...")
    bervo_labels, bervo_vectors = load_embeddings_from_csv(bervo_embeddings)
    new_labels, new_vectors = load_embeddings_from_csv(new_embeddings)

    logging.info("Computing cross-collection similarity matrix...")
    similarity_matrix = compute_cosine_similarity(bervo_vectors, new_vectors)

    logging.info(f"Finding top {top_n} cross-collection similar pairs...")
    top_pairs = find_similar_terms_cross_collection(
        bervo_labels, new_labels, similarity_matrix, top_n=top_n
    )

    # Print results
    print(f"\nTop {top_n} BERVO-to-new-variable pairs by similarity:")
    print("-" * 90)
    for rank, (term1, term2, score) in enumerate(top_pairs, 1):
        print(f"{rank:2d}. {term1:40s} <-> {term2:40s} (similarity: {score:.4f})")

    # Optionally write to file
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"Top {top_n} BERVO-to-new-variable pairs by similarity\n")
            f.write("-" * 90 + "\n")
            for rank, (term1, term2, score) in enumerate(top_pairs, 1):
                f.write(
                    f"{rank:2d}. {term1:40s} <-> {term2:40s} (similarity: {score:.4f})\n")
        logging.info(f"Results written to {output_file}")


if __name__ == "__main__":
    main()
