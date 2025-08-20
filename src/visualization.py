# -*- coding: utf-8 -*-
"""Functions for creating and saving visualizations."""

from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import umap

from src.dataset import CLUDataset
from src.schemas import BoundaryViolationRecord


class Visualizer:
    """Handles the creation of various plots for dataset analysis."""

    def __init__(self, dataset: CLUDataset, output_dir: Path = Path("outputs/figures")):
        """
        Initializes the visualizer.

        Args:
            dataset: The CLUDataset to visualize.
            output_dir: Directory to save the plots.
        """
        self.dataset = dataset
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")
        
        # Configure matplotlib to use a font that supports Chinese characters
        try:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
            plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of the minus sign displaying as a square
            logger.info("Set matplotlib font to 'WenQuanYi Micro Hei' for Chinese support.")
        except Exception:
            logger.warning(
                "Failed to set Chinese font 'WenQuanYi Micro Hei'. "
                "Plots may not render CJK characters correctly. "
                "Please ensure the font is installed on the system."
            )

        logger.info(f"Visualizer initialized. Figures will be saved to '{self.output_dir.resolve()}'")

    def plot_intent_similarity_heatmap(
        self, embeddings_map: Dict[str, np.ndarray], save: bool = True
    ) -> Optional[plt.Figure]:
        """
        Generates and saves a heatmap of cosine similarity between intent centroids.

        Args:
            embeddings_map: A map from utterance text to its embedding.
            save: Whether to save the plot to a file.

        Returns:
            The matplotlib Figure object if not saved, otherwise None.
        """
        logger.info("Generating intent similarity heatmap...")
        intents = self.dataset.get_intents()
        intent_names = [intent.category for intent in intents]
        intent_centroids = []

        for intent_name in intent_names:
            utterances = self.dataset.get_utterances(intent=intent_name)
            if not utterances:
                intent_centroids.append(
                    np.zeros(list(embeddings_map.values())[0].shape)
                )
                continue

            intent_embeddings = np.array(
                [
                    embeddings_map[utt.text]
                    for utt in utterances
                    if utt.text in embeddings_map
                ]
            )
            if intent_embeddings.size == 0:
                intent_centroids.append(
                    np.zeros(list(embeddings_map.values())[0].shape)
                )
                continue

            centroid = np.mean(intent_embeddings, axis=0)
            intent_centroids.append(centroid)

        similarity_matrix = cosine_similarity(np.array(intent_centroids))

        fig, ax = plt.subplots(
            figsize=(max(10, len(intent_names) * 0.5), max(8, len(intent_names) * 0.4))
        )
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            xticklabels=intent_names,
            yticklabels=intent_names,
            ax=ax,
        )
        ax.set_title("意图相似度热力图 (质心余弦相似度)")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / "intent_similarity_heatmap.png"
            fig.savefig(save_path, dpi=300)
            logger.info(f"Heatmap saved to {save_path}")
            plt.close(fig)
            return None

        return fig

    def _reduce_dimensions_umap(
        self,
        embeddings: np.ndarray,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Reduces the dimensionality of embeddings using UMAP.

        Args:
            embeddings: A numpy array of high-dimensional embeddings.
            n_neighbors: UMAP's n_neighbors parameter.
            min_dist: UMAP's min_dist parameter.
            random_state: The random seed for reproducibility.

        Returns:
            A numpy array of 2D embeddings.
        """
        logger.info(
            f"Performing UMAP dimensionality reduction (n_neighbors={n_neighbors}, min_dist={min_dist})..."
        )
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=random_state,
            n_components=2,
        )
        embeddings_2d = reducer.fit_transform(embeddings)
        logger.success("UMAP reduction complete.")
        return embeddings_2d

    def plot_global_scatterplot(
        self,
        embeddings_map: Dict[str, np.ndarray],
        boundary_violations: Optional[List[BoundaryViolationRecord]] = None,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Generates a 2D scatter plot of all utterances after UMAP reduction.

        Args:
            embeddings_map: A map from utterance text to its embedding.
            boundary_violations: A list of identified boundary violation records to highlight.
            save: Whether to save the plot to a file.

        Returns:
            The matplotlib Figure object if not saved, otherwise None.
        """
        logger.info("Generating global utterance scatter plot...")
        utterances = self.dataset.get_utterances()
        embeddings = np.array([embeddings_map[utt.text] for utt in utterances])
        intents = [utt.intent for utt in utterances]
        texts = [utt.text for utt in utterances]

        embeddings_2d = self._reduce_dimensions_umap(embeddings)

        df = pd.DataFrame(embeddings_2d, columns=["x", "y"])
        df["intent"] = intents
        df["text"] = texts
        
        # Add a column for violation status
        violation_texts = {v.text for v in boundary_violations} if boundary_violations else set()
        df["is_violation"] = df["text"].apply(lambda t: t in violation_texts)

        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Plot normal points first
        normal_df = df[~df["is_violation"]]
        sns.scatterplot(
            data=normal_df,
            x="x",
            y="y",
            hue="intent",
            palette="viridis",
            s=50,
            alpha=0.6,
            ax=ax,
            legend="full"
        )
        
        # Plot violation points on top with a distinct marker
        violation_df = df[df["is_violation"]]
        if not violation_df.empty:
            sns.scatterplot(
                data=violation_df,
                x="x",
                y="y",
                hue="intent",
                palette="viridis",
                marker="^",  # Use a triangle for violations
                s=150,       # Make them larger
                edgecolor="yellow",
                linewidth=2,
                ax=ax,
                legend=False # Do not add a second legend for violations
            )

        ax.set_title("全局语料分布 (UMAP) - 黄边三角形表示边界混淆点")
        ax.legend(title='意图', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # --- Add text box for top violations ---
        if boundary_violations:
            top_n = 15
            # Violations are pre-sorted by p-value descending from the processor
            top_violations = boundary_violations[:top_n]
            
            report_lines = [f"边界混淆语料 Top {len(top_violations)} (按p-value排序):"]
            for i, v in enumerate(top_violations):
                # Truncate long utterance text for display
                text = v.text if len(v.text) < 50 else v.text[:47] + "..."
                line = (
                    f"{i+1:2d}. [{v.original_intent}] '{text}' -> "
                    f"[{v.confused_with.intent}] (p={v.confused_with.p_value:.3f})"
                )
                report_lines.append(line)
            
            report_text = "\n".join(report_lines)

            # Adjust figure layout to make space for the text at the bottom
            fig.subplots_adjust(bottom=0.3)
            
            # Add the text box to the figure
            fig.text(0.01, 0.25, report_text, 
                     ha='left', va='top', 
                     fontsize=9, wrap=False, 
                     bbox=dict(boxstyle='round,pad=0.5', fc='#EFEFEF', alpha=0.8))

        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust tight_layout to leave space for legend

        if save:
            save_path = self.output_dir / "global_scatterplot.png"
            fig.savefig(save_path, dpi=300)
            logger.info(f"Global scatter plot saved to {save_path}")
            plt.close(fig)
            return None

        return fig

    def plot_per_intent_scatterplots(
        self,
        embeddings_map: Dict[str, np.ndarray],
        outliers_map: Dict[str, List[Dict]],
        save: bool = True
    ) -> None:
        """
        Generates a 2D scatter plot for each intent individually,
        highlighting and annotating outliers.

        Args:
            embeddings_map: A map from utterance text to its embedding.
            outliers_map: A map from intent name to a list of its outliers.
            save: Whether to save the plots to files.
        """
        logger.info("Generating per-intent scatter plots with outlier annotations...")
        intents = self.dataset.get_intents()
        
        outlier_texts_by_intent = {
            intent: {o['text'] for o in outliers} 
            for intent, outliers in outliers_map.items()
        }

        for intent in intents:
            intent_name = intent.category
            utterances = self.dataset.get_utterances(intent=intent_name)
            
            # UMAP's spectral initialization can fail with very few samples (e.g., <= 3).
            # Skip plotting for intents with insufficient data points.
            if len(utterances) < 4:
                logger.warning(
                    f"Skipping plot for intent '{intent_name}': "
                    f"not enough samples ({len(utterances)} < 4)."
                )
                continue

            embeddings = np.array([embeddings_map[utt.text] for utt in utterances])
            # Adjust n_neighbors to be less than the number of samples
            n_neighbors = min(15, len(utterances) - 1)
            embeddings_2d = self._reduce_dimensions_umap(embeddings, n_neighbors=n_neighbors)

            df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
            df['text'] = [utt.text for utt in utterances]
            
            intent_outliers = outlier_texts_by_intent.get(intent_name, set())
            df['status'] = df['text'].apply(lambda t: 'Outlier' if t in intent_outliers else 'Normal')
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.scatterplot(
                data=df,
                x='x',
                y='y',
                hue='status',
                style='status',
                palette={'Normal': 'C0', 'Outlier': 'C3'}, # Blue for Normal, Red for Outlier
                markers={'Normal': 'o', 'Outlier': 'X'},
                s=80,
                ax=ax
            )
            
            # Annotate outliers
            outlier_df = df[df['status'] == 'Outlier']
            for i, row in outlier_df.iterrows():
                ax.annotate(
                    f"{row['text'][:30]}...", # Truncate text
                    (row['x'], row['y']),
                    textcoords="offset points",
                    xytext=(5,5),
                    ha='left',
                    fontsize=9,
                    color='darkred'
                )

            ax.set_title(f"意图内部语料分布: '{intent_name}' (UMAP)")
            ax.legend(title='类型')
            plt.tight_layout()

            if save:
                # Sanitize filename
                safe_intent_name = "".join(c for c in intent_name if c.isalnum() or c in (' ', '_')).rstrip()
                save_path = self.output_dir / f"intent_{safe_intent_name}.png"
                fig.savefig(save_path, dpi=300)
                logger.info(f"Scatter plot for intent '{intent_name}' saved to {save_path}")
                plt.close(fig)
        logger.success("Per-intent scatter plots generated.")
