# -*- coding: utf-8 -*-
"""统一的降维算法模块，消除PCA和UMAP重复代码。"""

from typing import Dict, Optional

from loguru import logger
import numpy as np
from sklearn.decomposition import PCA
import umap

from src.dataset import CLUDataset


class DimensionalityReducer:
    """
    统一的降维器，负责PCA和UMAP降维操作。
    消除原代码中的重复降维逻辑。
    """
    
    def __init__(self, dataset: CLUDataset):
        """
        初始化降维器。
        
        Args:
            dataset: CLU数据集实例
        """
        self.dataset = dataset
        self._pca_cache: Dict[int, Dict] = {}  # 按min_samples缓存PCA结果
        self._umap_cache: Optional[np.ndarray] = None  # 缓存UMAP结果
        logger.info("DimensionalityReducer initialized.")
    
    def get_pca_embeddings(
        self, 
        embeddings_map: Dict[str, np.ndarray], 
        min_samples_for_analysis: int
    ) -> Dict:
        """
        获取PCA降维后的嵌入向量，支持缓存。
        
        Args:
            embeddings_map: 语料文本到嵌入向量的映射
            min_samples_for_analysis: 意图被纳入分析的最小样本数
            
        Returns:
            包含降维结果的字典: {
                "reduced_map": Dict[str, np.ndarray],  # 降维后的嵌入映射
                "target_dim": int,                     # 目标维度
                "intents_for_analysis": List          # 符合条件的意图列表
            }
        """
        # 检查缓存
        if min_samples_for_analysis in self._pca_cache:
            logger.info(f"Using cached PCA results for min_samples={min_samples_for_analysis}.")
            return self._pca_cache[min_samples_for_analysis]
        
        logger.info(f"Computing PCA with min_samples_for_analysis={min_samples_for_analysis}...")
        
        all_embeddings = np.array(list(embeddings_map.values()))
        
        # 筛选符合条件的意图
        intents_for_analysis = [
            intent for intent in self.dataset.get_intents()
            if len(self.dataset.get_utterances(intent=intent.category)) >= min_samples_for_analysis
        ]
        
        if len(intents_for_analysis) < 2:
            logger.warning("Fewer than 2 intents meet min sample requirement for PCA.")
            return {"reduced_map": {}, "target_dim": 0, "intents_for_analysis": []}
        
        # 计算目标维度
        n_min = min(
            len(self.dataset.get_utterances(intent=intent.category)) 
            for intent in intents_for_analysis
        )
        target_dim = min(n_min - 1, all_embeddings.shape[1])
        
        logger.info(f"Applying PCA to reduce dimension to {target_dim} (min samples: {n_min}).")
        
        # 执行PCA
        pca = PCA(n_components=target_dim, random_state=42)
        reduced_embeddings_matrix = pca.fit_transform(all_embeddings)
        
        # 构建降维后的映射
        utterance_texts = list(embeddings_map.keys())
        reduced_embeddings_map = {
            text: reduced_vec 
            for text, reduced_vec in zip(utterance_texts, reduced_embeddings_matrix)
        }
        
        # 缓存结果
        result = {
            "reduced_map": reduced_embeddings_map, 
            "target_dim": target_dim, 
            "intents_for_analysis": intents_for_analysis
        }
        self._pca_cache[min_samples_for_analysis] = result
        
        logger.success(f"PCA completed. Reduced to {target_dim}D space.")
        return result
    
    def get_umap_embeddings(
        self,
        embeddings_map: Dict[str, np.ndarray],
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
        force_recompute: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        获取UMAP 2D降维后的嵌入向量，支持缓存。
        
        Args:
            embeddings_map: 语料文本到嵌入向量的映射
            n_neighbors: UMAP邻居数参数
            min_dist: UMAP最小距离参数
            random_state: 随机种子
            force_recompute: 是否强制重新计算
            
        Returns:
            语料文本到2D坐标的映射
        """
        if self._umap_cache is not None and not force_recompute:
            logger.info("Using cached UMAP results.")
            utterance_texts = list(embeddings_map.keys())
            return {
                text: coord for text, coord in zip(utterance_texts, self._umap_cache)
            }
        
        logger.info(
            f"Computing UMAP 2D reduction (n_neighbors={n_neighbors}, min_dist={min_dist})..."
        )
        
        utterance_texts = list(embeddings_map.keys())
        embeddings = np.array(list(embeddings_map.values()))
        
        # 执行UMAP降维
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=random_state,
            n_components=2,
        )
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # 缓存结果
        self._umap_cache = embeddings_2d
        
        # 构建映射
        umap_map = {
            text: coord for text, coord in zip(utterance_texts, embeddings_2d)
        }
        
        logger.success("UMAP 2D reduction completed.")
        return umap_map
    
    def clear_cache(self):
        """清空所有缓存。"""
        self._pca_cache.clear()
        self._umap_cache = None
        logger.info("DimensionalityReducer cache cleared.")


