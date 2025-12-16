"""Utilities for similarity analysis of embeddings."""

import logging
from typing import List, Tuple

import numpy as np

__all__ = [
    "compute_cosine_similarity",
    "find_similar_terms",
    "find_similar_terms_cross_collection",
]


def compute_cosine_similarity(
    vectors1: List[np.ndarray],
    vectors2: List[np.ndarray] = None,
) -> np.ndarray:
    """Compute cosine similarity between two sets of vectors.

    Result is a matrix sim[ROW][COL] where ROW is from vectors1 and COL is from vectors2.
    If vectors2 is None, compute similarity within vectors1.

    Args:
        vectors1: List of embedding vectors
        vectors2: List of embedding vectors to compare against (optional)

    Returns:
        Cosine similarity matrix
    """
    if vectors2 is None:
        vectors2 = vectors1

    # Convert lists to numpy arrays
    matrix1 = np.array(vectors1)
    matrix2 = np.array(vectors2)

    # Normalize the vectors in both matrices
    matrix1_norm = matrix1 / np.linalg.norm(matrix1, axis=1)[:, np.newaxis]
    matrix2_norm = matrix2 / np.linalg.norm(matrix2, axis=1)[:, np.newaxis]

    # Compute dot products (resulting in cosine similarity values)
    cosine_similarity_matrix = np.dot(matrix1_norm, matrix2_norm.T)

    return cosine_similarity_matrix


def find_similar_terms(
    query_term: str,
    labels: List[str],
    similarity_matrix: np.ndarray,
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    """Find the top N most similar terms to a query term.

    Args:
        query_term: The term to find similar terms for
        labels: List of all term labels
        similarity_matrix: Precomputed cosine similarity matrix
        top_n: Number of top similar terms to return

    Returns:
        List of tuples (term_label, similarity_score)
    """
    if query_term not in labels:
        logging.warning(f"Term '{query_term}' not found in the collection.")
        return []

    # Find the index of the query term
    query_idx = labels.index(query_term)

    # Get similarities for this term
    similarities = similarity_matrix[query_idx]

    # Get indices of top N similar terms (excluding the query term itself)
    top_indices = np.argsort(-similarities)[1:top_n + 1]  # Skip first (itself)

    # Create results
    results = [(labels[idx], float(similarities[idx])) for idx in top_indices]

    return results


def find_similar_terms_cross_collection(
    labels1: List[str],
    labels2: List[str],
    similarity_matrix: np.ndarray,
    top_n: int = 25,
    exclude_self_matches: bool = True,
) -> List[Tuple[str, str, float]]:
    """Find the top N most similar term pairs across two collections.

    Args:
        labels1: List of labels from first collection
        labels2: List of labels from second collection
        similarity_matrix: Cross-collection similarity matrix (labels1 x labels2)
        top_n: Number of top pairs to return
        exclude_self_matches: If True, exclude pairs where both terms have same prefix

    Returns:
        List of tuples (term1_label, term2_label, similarity_score)
    """
    pairs = []

    # Iterate through all pairs
    for i, label_i in enumerate(labels1):
        for j, label_j in enumerate(labels2):
            similarity = similarity_matrix[i, j]

            # Check if we should exclude this pair
            if exclude_self_matches:
                # Exclude pairs where both are from same collection
                # (assuming labels from same collection share a prefix)
                if label_i.startswith('BERVO') == label_j.startswith('BERVO'):
                    continue

            pairs.append((label_i, label_j, float(similarity)))

    # Sort by similarity descending and get top N
    pairs.sort(key=lambda x: -x[2])
    top_pairs = pairs[:top_n]

    return top_pairs


def compute_distance_matrix(
    vectors: List[np.ndarray],
) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix.

    Args:
        vectors: List of embedding vectors

    Returns:
        Distance matrix
    """
    try:
        from scipy.spatial import distance_matrix
    except ImportError:
        raise ImportError(
            "scipy is required for this function. Install with: pip install scipy")

    vectors_array = np.array(vectors)
    distances = distance_matrix(vectors_array, vectors_array)

    return distances
