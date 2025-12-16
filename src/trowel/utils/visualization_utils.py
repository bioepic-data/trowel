"""Utilities for visualizing embeddings and clusters."""

import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "reduce_dimensionality_pca",
    "reduce_dimensionality_tsne",
    "plot_clusters_pca",
    "plot_clusters_tsne",
]


def reduce_dimensionality_pca(vectors: List[np.ndarray]) -> np.ndarray:
    """Reduce vectors to 2D using PCA.

    Args:
        vectors: List of embedding vectors

    Returns:
        2D reduced data array
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError("scikit-learn is required for this function. Install with: pip install scikit-learn")

    vectors_array = np.array(vectors)
    reducer = PCA(n_components=2)
    reduced_data = reducer.fit_transform(vectors_array)

    return reduced_data


def reduce_dimensionality_tsne(
    vectors: List[np.ndarray],
    perplexity: Optional[int] = None,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce vectors to 2D using t-SNE.

    Args:
        vectors: List of embedding vectors
        perplexity: Perplexity parameter (default: min(len(vectors)-1, 30))
        random_state: Random state for reproducibility

    Returns:
        2D reduced data array
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError("scikit-learn is required for this function. Install with: pip install scikit-learn")

    vectors_array = np.array(vectors)
    n_samples = len(vectors_array)

    if perplexity is None:
        perplexity = min(n_samples - 1, 30)

    reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    reduced_data = reducer.fit_transform(vectors_array)

    return reduced_data


def plot_clusters_pca(
    labels: List[str],
    vectors: List[np.ndarray],
    colors: Optional[List] = None,
    output_file: Optional[str] = None,
    title: str = "Term Clustering using PCA",
    label_interval: int = 100,
) -> None:
    """Plot 2D PCA reduction of embeddings with optional coloring.

    Args:
        labels: List of term labels
        vectors: List of embedding vectors
        colors: Optional list of colors for points
        output_file: Optional file path to save the plot
        title: Title for the plot
        label_interval: Interval for labeling points (every Nth point)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for this function. Install with: pip install matplotlib")

    logging.info("Reducing dimensionality using PCA...")
    reduced_data = reduce_dimensionality_pca(vectors)

    logging.info("Creating plot...")
    fig, ax = plt.subplots(figsize=(14, 10))

    if colors is None:
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], s=50, alpha=0.6)
    else:
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=50, alpha=0.6)

    # Add labels to a subset of points
    for i in range(0, len(labels), label_interval):
        ax.annotate(labels[i], (reduced_data[i, 0], reduced_data[i, 1]),
                   fontsize=9, ha='right', alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

    plt.tight_layout()

    if output_file:
        logging.info(f"Saving plot to {output_file}")
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_clusters_tsne(
    labels: List[str],
    vectors: List[np.ndarray],
    colors: Optional[List] = None,
    output_file: Optional[str] = None,
    title: str = "Term Clustering using t-SNE",
    perplexity: Optional[int] = None,
    label_interval: int = 100,
) -> None:
    """Plot 2D t-SNE reduction of embeddings with optional coloring.

    Args:
        labels: List of term labels
        vectors: List of embedding vectors
        colors: Optional list of colors for points
        output_file: Optional file path to save the plot
        title: Title for the plot
        perplexity: Perplexity parameter for t-SNE
        label_interval: Interval for labeling points (every Nth point)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for this function. Install with: pip install matplotlib")

    logging.info("Reducing dimensionality using t-SNE (this may take a while)...")
    reduced_data = reduce_dimensionality_tsne(vectors, perplexity=perplexity)

    logging.info("Creating plot...")
    fig, ax = plt.subplots(figsize=(14, 10))

    if colors is None:
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], s=50, alpha=0.6)
    else:
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=50, alpha=0.6)

    # Add labels to a subset of points
    for i in range(0, len(labels), label_interval):
        ax.annotate(labels[i], (reduced_data[i, 0], reduced_data[i, 1]),
                   fontsize=9, ha='right', alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

    plt.tight_layout()

    if output_file:
        logging.info(f"Saving plot to {output_file}")
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_similarity_heatmap(
    labels: List[str],
    similarity_matrix: np.ndarray,
    num_terms: int = 50,
    output_file: Optional[str] = None,
) -> None:
    """Plot a heatmap of the similarity matrix for a subset of terms.

    Args:
        labels: List of term labels
        similarity_matrix: Precomputed cosine similarity matrix
        num_terms: Number of terms to include in heatmap
        output_file: Optional file path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("matplotlib and seaborn are required. Install with: pip install matplotlib seaborn")

    # Select a subset of terms
    subset_labels = labels[:num_terms]
    subset_matrix = similarity_matrix[:num_terms, :num_terms]

    logging.info(f"Creating heatmap for {num_terms} terms...")
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(subset_matrix,
                xticklabels=subset_labels,
                yticklabels=subset_labels,
                cmap='coolwarm',
                center=0.5,
                vmin=0,
                vmax=1,
                ax=ax,
                cbar_kws={'label': 'Cosine Similarity'})

    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.title(f'Term Similarity Heatmap (first {num_terms} terms)')
    plt.tight_layout()

    if output_file:
        logging.info(f"Saving heatmap to {output_file}")
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def create_category_color_map(
    labels: List[str],
    category_mapping: Dict[str, str],
) -> Tuple[List, Dict]:
    """Create colors for points based on category mapping.

    Args:
        labels: List of term labels
        category_mapping: Mapping from term ID/label to category

    Returns:
        Tuple of (colors_list, category_to_color_dict)
    """
    try:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
    except ImportError:
        raise ImportError("matplotlib is required. Install with: pip install matplotlib")

    # Get unique categories
    unique_categories = list(set(category_mapping.values()))
    n_categories = len(unique_categories)

    logging.info(f"Creating color map for {n_categories} unique categories...")

    # Create diverse color palette
    colors1 = cm.tab20(range(min(20, n_categories)))
    colors2 = cm.Set3(range(min(12, n_categories)))
    colors3 = cm.Paired(range(min(12, n_categories)))
    all_colors = list(colors1) + list(colors2) + list(colors3)

    # Assign green shades to Plant-related categories
    green_shades = [
        (0.0, 0.5, 0.0, 1.0),      # Dark green
        (0.0, 0.7, 0.0, 1.0),      # Medium-dark green
        (0.133, 0.545, 0.133, 1.0), # Forest green
        (0.0, 0.8, 0.0, 1.0),      # Bright green
        (0.196, 0.804, 0.196, 1.0), # Lime green
        (0.486, 0.988, 0.0, 1.0),  # Lawn green
        (0.0, 1.0, 0.0, 1.0),      # Pure green
        (0.564, 0.933, 0.564, 1.0), # Light green
        (0.596, 0.984, 0.596, 1.0), # Pale green
    ]

    # Create color mapping
    category_colors = []
    green_idx = 0
    for cat in unique_categories:
        if cat.endswith('Plant'):  # Assuming Plant categories end with 'Plant'
            category_colors.append(green_shades[green_idx % len(green_shades)])
            green_idx += 1
        else:
            idx = len(category_colors) - green_idx
            category_colors.append(all_colors[idx % len(all_colors)])

    colormap = mcolors.ListedColormap(category_colors[:n_categories])
    category_to_color = {cat: colormap(i) for i, cat in enumerate(unique_categories)}

    # Map each term to its color
    colors = [category_to_color.get(category_mapping.get(label, 'Unknown'), 'gray')
              for label in labels]

    return colors, category_to_color
