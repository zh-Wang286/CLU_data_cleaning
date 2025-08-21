# -*- coding: utf-8 -*-
"""统一的异常点检测模块。"""

from typing import Dict, List, Literal

from loguru import logger
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from src.dataset import CLUDataset


class OutlierDetector:
    """
    统一的异常点检测器，负责意图内异常点检测。
    重构原有的detect_intra_intent_outliers逻辑。
    """
    
    def __init__(self, dataset: CLUDataset):
        """
        初始化异常点检测器。
        
        Args:
            dataset: CLU数据集实例
        """
        self.dataset = dataset
        logger.info("OutlierDetector initialized.")
    
    def find_outliers_within_intents(
        self,
        embeddings_map: Dict[str, np.ndarray],
        method: Literal["knn", "lof"] = "knn",
        k: int = 1,
        threshold_policy: Literal["90pct", "95pct", "iqr"] = "95pct",
    ) -> Dict[str, List[Dict]]:
        """
        检测每个意图内部的异常点。
        
        Args:
            embeddings_map: 语料文本到嵌入向量的映射
            method: 异常点检测方法 ('knn' 或 'lof')，当前只实现了 'knn'
            k: k-NN距离中的k值
            threshold_policy: 阈值策略 ('90pct', '95pct', 或 'iqr')
            
        Returns:
            意图名到异常点记录列表的映射，每个异常点记录包含:
            - original_idx: 在意图语料列表中的索引
            - text: 语料文本
            - score: 异常分数
            - threshold: 判断阈值
            - rank: 排名
        """
        logger.info(
            f"Starting intra-intent outlier detection: method='{method}', k={k}, "
            f"threshold='{threshold_policy}'..."
        )
        
        if method != "knn":
            raise NotImplementedError(f"Method '{method}' is not implemented yet. Only 'knn' is supported.")
        
        all_outliers = {}
        
        for intent in self.dataset.get_intents():
            intent_name = intent.category
            utterances = self.dataset.get_utterances(intent=intent_name)
            
            # 检查样本数是否足够
            if len(utterances) <= k + 1:
                logger.warning(
                    f"Skipping intent '{intent_name}': insufficient samples "
                    f"({len(utterances)} <= {k + 1})."
                )
                continue
            
            # 获取意图的嵌入向量
            intent_embeddings = np.array([
                embeddings_map[utt.text] for utt in utterances
                if utt.text in embeddings_map
            ])
            
            if len(intent_embeddings) == 0:
                logger.warning(f"No embeddings found for intent '{intent_name}'.")
                continue
            
            # 计算k-NN距离
            outlier_records = self._detect_knn_outliers(
                intent_embeddings, utterances, k, threshold_policy
            )
            
            if outlier_records:
                all_outliers[intent_name] = outlier_records
                logger.info(f"Found {len(outlier_records)} outliers in intent '{intent_name}'.")
        
        logger.success(f"Outlier detection completed. Found outliers in {len(all_outliers)} intents.")
        return all_outliers
    
    def _detect_knn_outliers(
        self,
        embeddings: np.ndarray,
        utterances: List,
        k: int,
        threshold_policy: str
    ) -> List[Dict]:
        """
        使用k-NN距离检测异常点。
        
        Args:
            embeddings: 意图的嵌入向量矩阵
            utterances: 对应的语料对象列表
            k: k-NN中的k值
            threshold_policy: 阈值策略
            
        Returns:
            异常点记录列表
        """
        # 计算余弦距离矩阵
        distances = cosine_distances(embeddings)
        
        # 排除自距离
        distances_no_self = distances.copy()
        np.fill_diagonal(distances_no_self, np.inf)
        
        # 计算k-NN距离
        sorted_distances = np.sort(distances_no_self, axis=1)
        k_nearest_distances = sorted_distances[:, k - 1]
        
        # 数值稳定性检查
        if not np.isfinite(k_nearest_distances).all():
            logger.error(f"Invalid k-NN distances found. k={k} may be too large.")
            return []
        
        # 计算阈值
        threshold = self._calculate_threshold(k_nearest_distances, threshold_policy)
        if threshold is None:
            return []
        
        # 识别异常点
        outlier_indices = np.where(k_nearest_distances > threshold)[0]
        
        if len(outlier_indices) == 0:
            return []
        
        # 构建异常点记录
        outlier_records = []
        sorted_indices = outlier_indices[np.argsort(-k_nearest_distances[outlier_indices])]
        
        for rank, idx in enumerate(sorted_indices):
            outlier_records.append({
                "original_idx": idx,
                "text": utterances[idx].text,
                "score": k_nearest_distances[idx],
                "threshold": threshold,
                "rank": rank + 1,
            })
        
        return outlier_records
    
    def _calculate_threshold(
        self,
        distances: np.ndarray,
        threshold_policy: str
    ) -> float:
        """
        根据策略计算异常点判断阈值。
        
        Args:
            distances: k-NN距离数组
            threshold_policy: 阈值策略
            
        Returns:
            计算得到的阈值，如果计算失败返回None
        """
        finite_distances = distances[np.isfinite(distances)]
        
        if len(finite_distances) == 0:
            logger.warning("No finite distances available for threshold calculation.")
            return None
        
        if threshold_policy == "90pct":
            threshold = np.percentile(finite_distances, 90)
        elif threshold_policy == "95pct":
            threshold = np.percentile(finite_distances, 95)
        elif threshold_policy == "iqr":
            q1, q3 = np.percentile(finite_distances, [25, 75])
            iqr = q3 - q1
            threshold = q3 + 1.5 * iqr
        else:
            raise ValueError(f"Unknown threshold policy: {threshold_policy}")
        
        logger.debug(f"Calculated threshold: {threshold:.4f} using policy '{threshold_policy}'.")
        return threshold


