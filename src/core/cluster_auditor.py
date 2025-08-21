# -*- coding: utf-8 -*-
"""统一的聚类审计模块。"""

from typing import Dict, Optional

import hdbscan
from loguru import logger
import numpy as np
import pandas as pd

from src.dataset import CLUDataset


class ClusterAuditor:
    """
    统一的聚类审计器，负责全局聚类分析。
    重构原有的audit_global_clusters逻辑。
    """
    
    def __init__(self, dataset: CLUDataset):
        """
        初始化聚类审计器。
        
        Args:
            dataset: CLU数据集实例
        """
        self.dataset = dataset
        logger.info("ClusterAuditor initialized.")
    
    def audit_clusters_globally(
        self,
        reduced_embeddings_map: Dict[str, np.ndarray],
        min_cluster_size: int = 15,
        min_samples: Optional[int] = None,
    ) -> Dict:
        """
        执行全局聚类审计，使用HDBSCAN在降维空间中发现潜在重叠。
        
        Args:
            reduced_embeddings_map: 降维后的嵌入向量映射
            min_cluster_size: HDBSCAN的最小簇大小
            min_samples: HDBSCAN的最小样本数参数
            
        Returns:
            包含聚类结果和审计信息的字典:
            - summary: 聚类摘要统计
            - clusters: 簇纯度分析列表
            - raw_labels: 原始聚类标签数据
        """
        logger.info(
            f"Starting global clustering audit: min_cluster_size={min_cluster_size}, "
            f"min_samples={min_samples}..."
        )
        
        if not reduced_embeddings_map:
            logger.warning("Empty reduced embeddings map. Cannot perform clustering.")
            return {"summary": {}, "clusters": [], "raw_labels": []}
        
        # 准备数据
        utterances = self.dataset.get_utterances()
        texts = [utt.text for utt in utterances]
        intents = [utt.intent for utt in utterances]
        
        # 构建降维嵌入矩阵
        reduced_embeddings = np.array([
            reduced_embeddings_map[text] for text in texts
            if text in reduced_embeddings_map
        ])
        
        if len(reduced_embeddings) == 0:
            logger.warning("No reduced embeddings available for clustering.")
            return {"summary": {}, "clusters": [], "raw_labels": []}
        
        # 执行HDBSCAN聚类
        cluster_labels = self._perform_hdbscan_clustering(
            reduced_embeddings, min_cluster_size, min_samples
        )
        
        # 分析聚类结果
        return self._analyze_clustering_results(texts, intents, cluster_labels)
    
    def _perform_hdbscan_clustering(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int,
        min_samples: Optional[int]
    ) -> np.ndarray:
        """
        执行HDBSCAN聚类。
        
        Args:
            embeddings: 降维后的嵌入向量矩阵
            min_cluster_size: 最小簇大小
            min_samples: 最小样本数
            
        Returns:
            聚类标签数组
        """
        logger.info("Performing HDBSCAN clustering...")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
        )
        
        cluster_labels = clusterer.fit_predict(embeddings)
        
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_ratio = np.sum(cluster_labels == -1) / len(cluster_labels)
        
        logger.info(f"HDBSCAN completed: {num_clusters} clusters, {noise_ratio:.2%} noise.")
        return cluster_labels
    
    def _analyze_clustering_results(
        self,
        texts: list,
        intents: list,
        cluster_labels: np.ndarray
    ) -> Dict:
        """
        分析聚类结果，计算簇纯度等指标。
        
        Args:
            texts: 语料文本列表
            intents: 对应的意图标签列表
            cluster_labels: HDBSCAN聚类标签
            
        Returns:
            聚类分析结果字典
        """
        logger.info("Analyzing clustering results...")
        
        # 创建分析DataFrame
        df = pd.DataFrame({
            "text": texts,
            "original_intent": intents,
            "cluster": cluster_labels
        })
        
        # 计算基础统计
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_ratio = np.sum(cluster_labels == -1) / len(cluster_labels)
        
        # 分析每个簇
        cluster_audit = []
        for cluster_id in sorted(set(cluster_labels)):
            if cluster_id == -1:  # 跳过噪声点
                continue
            
            cluster_df = df[df["cluster"] == cluster_id]
            audit_info = self._analyze_single_cluster(cluster_df, cluster_id)
            
            if audit_info:  # 只保留有效的簇（大小>=3）
                cluster_audit.append(audit_info)
        
        # 按纯度和大小排序，突出显示问题簇
        cluster_audit.sort(key=lambda x: (x["purity"], -x["size"]))
        
        logger.success("Clustering analysis completed.")
        
        return {
            "summary": {
                "num_clusters": num_clusters,
                "noise_ratio": noise_ratio,
                "total_utterances": len(df),
            },
            "clusters": cluster_audit,
            "raw_labels": df.to_dict("records"),
        }
    
    def _analyze_single_cluster(self, cluster_df: pd.DataFrame, cluster_id: int) -> Optional[Dict]:
        """
        分析单个簇的组成和纯度。
        
        Args:
            cluster_df: 簇的DataFrame
            cluster_id: 簇ID
            
        Returns:
            簇分析信息字典，如果簇太小则返回None
        """
        intent_counts = cluster_df["original_intent"].value_counts()
        
        majority_intent = intent_counts.idxmax()
        majority_count = intent_counts.max()
        total_samples = len(cluster_df)
        
        # 过滤过小的簇
        if total_samples < 3:
            logger.debug(f"Filtering out small cluster {cluster_id} with {total_samples} samples.")
            return None
        
        purity = majority_count / total_samples
        
        return {
            "cluster_id": cluster_id,
            "size": total_samples,
            "majority_intent": majority_intent,
            "purity": purity,
            "intent_distribution": intent_counts.to_dict(),
        }


