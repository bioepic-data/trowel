"""Utility functions for matching terms between files."""

import os
import logging
import polars as pl
from typing import Dict, Set, Tuple, List, Optional
try:
    from rapidfuzz import process, fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logging.warning(
        "rapidfuzz package not available; fuzzy matching will be disabled.")
    logging.warning(
        "To enable fuzzy matching, install rapidfuzz: pip install rapidfuzz")


def match_terms(tsv_file_path: str, term_list_path: str, output_path: str = None,
                fuzzy_match: bool = False, similarity_threshold: float = 80.0) -> Optional[str]:
    """Match terms from a TSV file against a list of terms in another file.

    This function takes two files:
    1. A TSV file with terms in the first column
    2. A text file with terms, one per line

    It then produces a new file containing all terms from the first file with a new column
    indicating whether there was an exact match in the second file, and what the match was.

    If fuzzy matching is enabled, it will also look for approximate matches when an exact match
    is not found, using the Levenshtein distance algorithm.

    Args:
        tsv_file_path: Path to a TSV file with terms in the first column
        term_list_path: Path to a file with terms, one per line
        output_path: Path where the output file should be written
                     (defaults to tsv_file_path with "_matched" appended)
        fuzzy_match: Whether to use fuzzy matching for terms that don't have exact matches
        similarity_threshold: Minimum similarity score (0-100) for fuzzy matches

    Returns:
        Path to the output file or None if an error occurred
    """
    # Validate input files
    if not os.path.exists(tsv_file_path):
        logging.error(f"The TSV file {tsv_file_path} does not exist.")
        return None

    if not os.path.exists(term_list_path):
        logging.error(f"The term list file {term_list_path} does not exist.")
        return None

    # Set default output path if not provided
    if not output_path:
        base_name, ext = os.path.splitext(tsv_file_path)
        output_path = f"{base_name}_matched{ext}"

    # Read the term list file into a set for O(1) lookups
    with open(term_list_path, 'r') as f:
        term_list = {line.strip() for line in f if line.strip()}

    # Convert to list for fuzzy matching if needed
    term_list_for_fuzzy = list(
        term_list) if fuzzy_match and FUZZY_AVAILABLE else []

    # Process the TSV file using polars for efficiency
    try:
        # Read the TSV file
        df = pl.read_csv(tsv_file_path, separator='\t')

        # Ensure there's at least one column
        if df.width == 0:
            logging.error(f"The TSV file {tsv_file_path} has no columns.")
            return None

        # Get the terms from the first column
        first_col_name = df.columns[0]

        # Create new columns for match status and matched term
        def match_term(term):
            term_str = str(term).strip()

            # Try exact match first
            if term_str in term_list:
                return (True, term_str, 100.0, "exact")

            # If fuzzy matching is enabled and the library is available, try that
            if fuzzy_match and FUZZY_AVAILABLE and term_list_for_fuzzy:
                # Find the best match using rapidfuzz
                best_match, score, _ = process.extractOne(
                    term_str,
                    term_list_for_fuzzy,
                    scorer=fuzz.ratio
                )

                if score >= similarity_threshold:
                    return (True, best_match, score, "fuzzy")

            # No match found
            return (False, "", 0.0, "none")

        # Apply the matching function to each term in the first column
        match_results = [match_term(term) for term in df[first_col_name]]
        match_status = [result[0] for result in match_results]
        matched_terms = [result[1] for result in match_results]
        match_scores = [result[2] for result in match_results]
        match_types = [result[3] for result in match_results]

        # Add new columns to the dataframe
        df = df.with_columns([
            pl.Series(name="match_found", values=match_status),
            pl.Series(name="matched_term", values=matched_terms),
            pl.Series(name="match_score", values=match_scores),
            pl.Series(name="match_type", values=match_types)
        ])

        # Write the result to the output file
        df.write_csv(output_path, separator='\t')

        logging.info(
            f"Processed {len(df)} terms with {sum(match_status)} matches")
        exact_matches = sum(1 for t in match_types if t == "exact")
        fuzzy_matches = sum(1 for t in match_types if t == "fuzzy")
        if fuzzy_match and FUZZY_AVAILABLE:
            logging.info(
                f"Found {exact_matches} exact matches and {fuzzy_matches} fuzzy matches")

        return output_path

    except Exception as e:
        logging.error(f"Error matching terms: {str(e)}")
        return None
