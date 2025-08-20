# -*- coding: utf-8 -*-
"""
PCA Dimensionality Evaluation Script.

This script evaluates and compares two strategies for selecting the number of
principal components (n_components) for dimensionality reduction on high-dimensional
text embeddings, especially in a "small n, large p" scenario (few samples, many features).

Strategies Compared:
1.  **Fixed Dimension Strategy**: Based on the minimum number of samples in any class,
    ensuring numerical stability for subsequent algorithms like Mahalanobis distance.
    (e.g., n_components = min_samples_per_intent - 1)
2.  **Variance-Driven Strategy**: Based on the cumulative explained variance ratio,
    aiming to preserve a certain percentage of the total information (e.g., 95%).

Evaluation Metric:
-   **Intent Separability Score**: A custom metric designed to measure how well
    different intents are separated in the reduced-dimensional space. It is calculated as:
    `Mean Inter-Intent Distance / Mean Intra-Intent Spread`
    A higher score indicates better separation.
"""
import sys
from pathlib import Path

# Add the project root to the Python path to allow importing from 'src'
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

from src.dataset import CLUDataset
from src.processor import CLUProcessor

# --- Configuration ---
DATA_FILE_PATH = "data/IT_01_1.json"
EXPLAINED_VARIANCE_THRESHOLD = 0.95
MIN_SAMPLES_FOR_ANALYSIS = 15  # Same as in src.processor


def get_optimal_dimension_by_variance(
    embeddings: np.ndarray, threshold: float = EXPLAINED_VARIANCE_THRESHOLD
) -> int:
    """
    Determines the optimal number of PCA components to retain a given percentage
    of the total variance.

    This addresses the question: "How many dimensions are needed to keep X% of the data's information?"

    Args:
        embeddings: A 2D numpy array of shape (n_samples, n_features).
        threshold: The desired cumulative explained variance ratio (e.g., 0.95 for 95%).

    Returns:
        The number of components required to meet the threshold.
    """
    logger.info(f"Calculating PCA and finding dimension for {threshold:.2%} cumulative variance...")
    
    # In a "p > n" scenario, the number of significant components cannot exceed n-1.
    # We fit PCA with all possible components to analyze the full variance spectrum.
    n_max_components = min(embeddings.shape) - 1
    pca = PCA(n_components=n_max_components, random_state=42)
    pca.fit(embeddings)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find the first dimension where the cumulative variance exceeds the threshold
    optimal_dim = np.argmax(cumulative_variance >= threshold) + 1
    
    logger.success(
        f"Found optimal dimension: {optimal_dim} (explains {cumulative_variance[optimal_dim-1]:.2%} of variance)."
    )
    return optimal_dim


def calculate_intent_separability_score(
    reduced_embeddings: np.ndarray, labels: list[str]
) -> float:
    """
    Calculates a score to evaluate the quality of dimensionality reduction
    in terms of class separability.

    The score is the ratio of the average distance between intent centroids to the
    average spread (standard deviation) within each intent cluster.

    Args:
        reduced_embeddings: The embeddings after PCA reduction.
        labels: A list of intent labels corresponding to each embedding.

    Returns:
        A float representing the separability score. Higher is better.
    """
    df = pd.DataFrame(reduced_embeddings)
    df["intent"] = labels
    
    intent_centroids = df.groupby("intent").mean().to_numpy()
    
    # 1. Calculate Mean Inter-Intent Distance (between centroids)
    if len(intent_centroids) < 2:
        return 0.0
    inter_intent_distances = euclidean_distances(intent_centroids)
    # Get the upper triangle of the distance matrix (excluding the diagonal)
    mean_inter_distance = inter_intent_distances[np.triu_indices(len(inter_intent_distances), k=1)].mean()
    
    # 2. Calculate Mean Intra-Intent Spread (within clusters)
    intra_intent_spreads = []
    for intent_name, group in df.groupby("intent"):
        centroid = intent_centroids[list(df["intent"].unique()).index(intent_name)]
        distances_to_centroid = euclidean_distances(group.drop("intent", axis=1), centroid.reshape(1, -1))
        # Use average distance to centroid as a measure of spread
        intra_intent_spreads.append(np.mean(distances_to_centroid))
        
    mean_intra_spread = np.mean(intra_intent_spreads)
    
    # 3. Calculate the final score
    if mean_intra_spread == 0:
        return np.inf  # Avoid division by zero
        
    separability_score = mean_inter_distance / mean_intra_spread
    
    return separability_score


def main():
    """Main execution function to run the evaluation."""
    logger.info("Starting PCA dimensionality evaluation script.")

    # 1. Load data and get full embeddings
    if not Path(DATA_FILE_PATH).exists():
        logger.error(f"Data file not found at '{DATA_FILE_PATH}'. Please check the path.")
        return

    dataset = CLUDataset.from_json(Path(DATA_FILE_PATH))
    processor = CLUProcessor(dataset)
    embeddings_map = processor.get_all_embeddings()
    
    all_embeddings = np.array(list(embeddings_map.values()))
    all_utterances = dataset.get_utterances()
    all_labels = [utt.intent for utt in all_utterances]

    if all_embeddings.shape[0] < MIN_SAMPLES_FOR_ANALYSIS:
        logger.error("Not enough samples in the dataset to perform a meaningful analysis.")
        return
        
    logger.info(f"Loaded {all_embeddings.shape[0]} samples with {all_embeddings.shape[1]} dimensions each.")

    # 2. Determine dimensions for both strategies
    # Strategy B: Variance-Driven Dimension
    dim_variance_driven = get_optimal_dimension_by_variance(all_embeddings, EXPLAINED_VARIANCE_THRESHOLD)

    # Strategy A: Fixed Dimension (from processor logic)
    intents_for_analysis = [
        intent for intent in dataset.get_intents()
        if len(dataset.get_utterances(intent=intent.category)) >= MIN_SAMPLES_FOR_ANALYSIS
    ]
    if len(intents_for_analysis) < 2:
        logger.error("Fewer than 2 intents meet the minimum sample requirement. Cannot determine fixed dimension.")
        dim_fixed = 14 # Fallback
    else:
        n_min = min(len(dataset.get_utterances(intent=i.category)) for i in intents_for_analysis)
        dim_fixed = min(n_min - 1, all_embeddings.shape[1])
    
    # 3. Perform PCA and evaluate for both dimensions
    strategies = {
        "Fixed (Stability-Driven)": dim_fixed,
        f"Variance-Driven ({EXPLAINED_VARIANCE_THRESHOLD:.0%})": dim_variance_driven,
    }

    results = []
    for name, n_components in strategies.items():
        logger.info(f"--- Evaluating Strategy: '{name}' with n_components = {n_components} ---")
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        reduced_embeddings = pca.fit_transform(all_embeddings)
        
        # Calculate explained variance
        explained_variance = np.sum(pca.explained_variance_ratio_)
        
        # Calculate separability score
        separability_score = calculate_intent_separability_score(reduced_embeddings, all_labels)
        
        results.append({
            "Strategy": name,
            "PCA Dimensions": n_components,
            "Explained Variance": f"{explained_variance:.2%}",
            "Intent Separability Score": f"{separability_score:.4f}",
        })

    # 4. Print results and conclusion
    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("                 PCA Dimensionality Reduction Strategy Comparison")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80 + "\n")

    logger.info("Analysis:")
    fixed_score = float(results_df.loc[0, "Intent Separability Score"])
    variance_score = float(results_df.loc[1, "Intent Separability Score"])

    if fixed_score > variance_score:
        logger.info(
            "The Fixed (Stability-Driven) strategy yields a higher separability score. "
            "This suggests that for subsequent tasks like Mahalanobis distance, which are sensitive to "
            "the number of samples, sacrificing some variance to ensure numerical stability (p > n) is a "
            "reasonable trade-off. The reduced feature space is more robust for classification."
        )
    elif variance_score > fixed_score:
        logger.info(
            f"The Variance-Driven strategy (to {EXPLAINED_VARIANCE_THRESHOLD:.0%}) yields a higher separability score. "
            "This indicates that retaining more of the original data's variance leads to better-defined intent "
            "clusters in the reduced space. This approach is better if the primary goal is preserving "
            "data structure, for instance, for visualization or clustering."
        )
    else:
        logger.info(
            "Both strategies yield a similar separability score. The choice can be made based on "
            "the specific downstream task's requirements (stability vs. information preservation)."
        )

    logger.info(
        "Key takeaway: In a 'small n, large p' world, there's a tension between "
        "preserving information (variance) and ensuring the stability of downstream statistical models. "
        "This evaluation provides a quantitative basis for making that choice."
    )


if __name__ == "__main__":
    main()
