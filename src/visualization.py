# -*- coding: utf-8 -*-
"""重构后的可视化模块，消除UMAP重复计算，统一绘图配置。"""

from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

from src.core.dimensionality_reducer import DimensionalityReducer
from src.dataset import CLUDataset
from src.schemas import BoundaryViolationRecord


class Visualizer:
    """
    重构后的可视化器，消除UMAP重复计算。
    
    统一使用DimensionalityReducer进行降维，避免重复计算。
    """

    def __init__(self, dataset: CLUDataset, output_dir: Path = Path("outputs/figures")):
        """
        初始化可视化器和降维模块。

        Args:
            dataset: CLU数据集实例
            output_dir: 保存图片的目录
        """
        self.dataset = dataset
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化降维器
        self.dimensionality_reducer = DimensionalityReducer(dataset)
        
        # 配置绘图主题
        self._setup_plotting_environment()
        
        logger.info(f"Visualizer initialized. Figures will be saved to '{self.output_dir.resolve()}'")
    
    def _setup_plotting_environment(self):
        """统一配置绘图环境，避免重复设置。"""
        sns.set_theme(style="whitegrid")
        
        # 配置中文字体支持
        try:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
            plt.rcParams['axes.unicode_minus'] = False
            logger.info("Set matplotlib font to 'WenQuanYi Micro Hei' for Chinese support.")
        except Exception:
            logger.warning(
                "Failed to set Chinese font 'WenQuanYi Micro Hei'. "
                "Plots may not render CJK characters correctly. "
                "Please ensure the font is installed on the system."
            )

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
        intents = [utt.intent for utt in utterances]
        texts = [utt.text for utt in utterances]

        # 使用统一的UMAP降维器，避免重复计算
        umap_embeddings_map = self.dimensionality_reducer.get_umap_embeddings(embeddings_map)
        embeddings_2d = np.array([umap_embeddings_map[text] for text in texts])

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

    def plot_targeted_scatterplot(
        self,
        embeddings_map: Dict[str, np.ndarray],
        target_intents: List[str],
        boundary_violations: Optional[List[BoundaryViolationRecord]] = None,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Generates a 2D scatter plot for a targeted subset of intents after UMAP reduction.
        It uses the same global UMAP transformation for consistency.

        Args:
            embeddings_map: A map from utterance text to its embedding.
            target_intents: A list of intent names to include in the plot.
            boundary_violations: A list of identified boundary violation records to highlight.
            save: Whether to save the plot to a file.

        Returns:
            The matplotlib Figure object if not saved, otherwise None.
        """
        logger.info(f"Generating targeted scatter plot for intents: {target_intents}...")
        
        # --- 1. 使用统一UMAP降维器获得一致的2D投影 ---
        all_utterances = self.dataset.get_utterances()
        all_intents = [utt.intent for utt in all_utterances]
        all_texts = [utt.text for utt in all_utterances]
        
        # 使用统一的UMAP降维器，避免重复计算
        umap_embeddings_map = self.dimensionality_reducer.get_umap_embeddings(embeddings_map)
        embeddings_2d = np.array([umap_embeddings_map[text] for text in all_texts])
        
        df = pd.DataFrame(embeddings_2d, columns=["x", "y"])
        df["intent"] = all_intents
        df["text"] = all_texts

        # --- 2. Filter the DataFrame to include only the target intents ---
        targeted_df = df[df["intent"].isin(target_intents)].copy()
        if targeted_df.empty:
            logger.warning("No utterances found for the targeted intents. Cannot generate plot.")
            return None

        # --- 3. Highlight boundary violations within the targeted group ---
        violation_texts = {v.text for v in boundary_violations} if boundary_violations else set()
        targeted_df["is_violation"] = targeted_df["text"].apply(lambda t: t in violation_texts)
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # # Plot normal points
        # normal_df = targeted_df[~targeted_df["is_violation"]]
        sns.scatterplot(
            data=targeted_df,
            x="x",
            y="y",
            hue="intent",
            palette="bright",
            s=80,
            alpha=0.7,
            ax=ax,
            legend="full"
        )
        
        # # Plot violation points
        # violation_df = targeted_df[targeted_df["is_violation"]]
        # if not violation_df.empty:
        #     sns.scatterplot(
        #         data=violation_df,
        #         x="x",
        #         y="y",
        #         hue="intent",
        #         palette="viridis",
        #         marker="^",
        #         s=200,
        #         edgecolor="red",
        #         linewidth=2,
        #         ax=ax,
        #         legend=False
        #     )
            
        # ax.set_title(f"局部意图语料分布 (UMAP) - 红边三角形表示边界混淆点\n意图: {', '.join(target_intents)}")
        ax.legend(title='意图', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])

        if save:
            safe_filename = "".join(c for c in "_vs_".join(target_intents) if c.isalnum() or c in (' ', '_', '-')).rstrip()
            save_path = self.output_dir / f"targeted_scatterplot_{safe_filename}.png"
            fig.savefig(save_path, dpi=300)
            logger.info(f"Targeted scatter plot saved to {save_path}")
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

            # 对于单个意图的散点图，需要独立的UMAP以适应样本数
            intent_embeddings_map = {utt.text: embeddings_map[utt.text] for utt in utterances}
            n_neighbors = min(15, len(utterances) - 1)
            
            # 单独计算该意图的UMAP（因为需要调整n_neighbors）
            umap_embeddings_map = self.dimensionality_reducer.get_umap_embeddings(
                intent_embeddings_map, 
                n_neighbors=n_neighbors,
                force_recompute=True  # 强制重新计算以使用不同的n_neighbors
            )
            embeddings_2d = np.array([umap_embeddings_map[utt.text] for utt in utterances])

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
