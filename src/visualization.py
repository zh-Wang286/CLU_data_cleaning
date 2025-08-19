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
        self, embeddings_map: Dict[str, np.ndarray], save: bool = True
    ) -> Optional[plt.Figure]:
        """
        Generates a 2D scatter plot of all utterances after UMAP reduction.

        Args:
            embeddings_map: A map from utterance text to its embedding.
            save: Whether to save the plot to a file.

        Returns:
            The matplotlib Figure object if not saved, otherwise None.
        """
        logger.info("Generating global utterance scatter plot...")
        utterances = self.dataset.get_utterances()
        embeddings = np.array([embeddings_map[utt.text] for utt in utterances])
        intents = [utt.intent for utt in utterances]

        embeddings_2d = self._reduce_dimensions_umap(embeddings)

        df = pd.DataFrame(embeddings_2d, columns=["x", "y"])
        df["intent"] = intents

        fig, ax = plt.subplots(figsize=(16, 12))
        sns.scatterplot(
            data=df,
            x="x",
            y="y",
            hue="intent",
            palette="viridis",
            s=50,
            alpha=0.7,
            ax=ax,
        )
        ax.set_title("全局话语分布 (UMAP)")
        ax.legend(title='意图', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

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
            
            if len(utterances) < 2:
                logger.warning(f"Skipping plot for intent '{intent_name}': not enough samples ({len(utterances)}).")
                continue

            embeddings = np.array([embeddings_map[utt.text] for utt in utterances])
            embeddings_2d = self._reduce_dimensions_umap(embeddings, n_neighbors=min(15, len(utterances)-1))

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

            ax.set_title(f"意图内部话语分布: '{intent_name}' (UMAP)")
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
