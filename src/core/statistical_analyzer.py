# -*- coding: utf-8 -*-
"""统一的统计分析模块，处理马氏距离和边界检测。"""

from typing import Dict, List, Optional, Tuple

from loguru import logger
import numpy as np
from scipy.stats import chi2

from src.dataset import CLUDataset
from src.schemas import BoundaryViolationRecord


class StatisticalAnalyzer:
    """
    统一的统计分析器，负责马氏距离计算和边界违规检测。
    消除原代码中重复的统计模型构建逻辑。
    """
    
    def __init__(self, dataset: CLUDataset):
        """
        初始化统计分析器。
        
        Args:
            dataset: CLU数据集实例
        """
        self.dataset = dataset
        logger.info("StatisticalAnalyzer initialized.")
    
    def build_intent_statistical_models(
        self,
        reduced_embeddings_map: Dict[str, np.ndarray],
        target_intents: List[str],
        min_samples_for_analysis: int,
        regularization_factor: float = 1e-6
    ) -> Dict[str, Dict]:
        """
        为指定意图构建统计模型（均值和逆协方差矩阵）。
        
        Args:
            reduced_embeddings_map: 降维后的嵌入向量映射
            target_intents: 目标意图列表
            min_samples_for_analysis: 最小样本数要求
            regularization_factor: 正则化因子，确保协方差矩阵可逆
            
        Returns:
            意图名到统计模型的映射，每个模型包含 {"mean": 均值向量, "inv_cov": 逆协方差矩阵}
        """
        logger.info(f"Building statistical models for {len(target_intents)} intents...")
        
        intent_stats = {}
        
        # 筛选符合条件的意图
        valid_intents = [
            intent for intent in self.dataset.get_intents() 
            if (intent.category in target_intents and 
                self.dataset.count_utterances(intent=intent.category) >= min_samples_for_analysis)
        ]
        
        for intent in valid_intents:
            intent_name = intent.category
            utterances = self.dataset.get_utterances(intent=intent_name)
            
            # 获取该意图的降维嵌入向量
            intent_embeddings_reduced = np.array([
                reduced_embeddings_map[utt.text] 
                for utt in utterances 
                if utt.text in reduced_embeddings_map
            ])
            
            if len(intent_embeddings_reduced) < min_samples_for_analysis:
                logger.warning(f"Intent '{intent_name}' has insufficient samples after filtering.")
                continue
            
            # 计算均值和协方差矩阵
            mean_vector = np.mean(intent_embeddings_reduced, axis=0)
            cov_matrix = np.cov(intent_embeddings_reduced, rowvar=False)
            
            # 正则化协方差矩阵
            reg_identity = np.identity(cov_matrix.shape[0]) * regularization_factor
            
            try:
                # 使用伪逆确保数值稳定性
                inv_cov_matrix = np.linalg.pinv(cov_matrix + reg_identity)
                intent_stats[intent_name] = {
                    "mean": mean_vector,
                    "inv_cov": inv_cov_matrix,
                }
                logger.debug(f"Statistical model built for intent '{intent_name}'.")
            except np.linalg.LinAlgError:
                logger.error(f"Failed to compute inverse covariance for intent '{intent_name}'.")
                continue
        
        logger.info(f"Successfully built statistical models for {len(intent_stats)} intents.")
        return intent_stats
    
    def calculate_mahalanobis_distance(
        self,
        point: np.ndarray,
        mean: np.ndarray,
        inv_cov: np.ndarray
    ) -> Tuple[float, float]:
        """
        计算单个点到分布的马氏距离和对应的p值。
        
        Args:
            point: 数据点
            mean: 分布均值
            inv_cov: 逆协方差矩阵
            
        Returns:
            (马氏距离, p值)元组
        """
        delta = point - mean
        mahalanobis_sq = delta.T @ inv_cov @ delta
        
        # 数值稳定性检查
        if mahalanobis_sq < 0:
            logger.warning("Negative Mahalanobis squared distance detected.")
            return 0.0, 1.0
        
        mahalanobis_dist = np.sqrt(mahalanobis_sq)
        p_value = 1 - chi2.cdf(mahalanobis_sq, df=len(point))
        
        return mahalanobis_dist, p_value
    
    def detect_boundary_violations(
        self,
        reduced_embeddings_map: Dict[str, np.ndarray],
        intent_stats: Dict[str, Dict],
        target_intents: Optional[List[str]] = None,
        p_value_threshold: float = 0.05
    ) -> List[BoundaryViolationRecord]:
        """
        检测边界违规，统一处理全局和局部边界分析。
        
        Args:
            reduced_embeddings_map: 降维后的嵌入向量映射
            intent_stats: 意图统计模型映射
            target_intents: 目标意图列表，None表示全局分析
            p_value_threshold: p值阈值
            
        Returns:
            边界违规记录列表
        """
        logger.info(f"Detecting boundary violations (p_value > {p_value_threshold})...")
        
        violations = []
        embedding_dim = len(next(iter(reduced_embeddings_map.values())))
        
        # 确定要检查的语料范围
        if target_intents is None:
            # 全局分析：检查所有语料
            utterances_to_check = self.dataset.get_utterances()
            logger.info("Running global boundary violation analysis.")
        else:
            # 局部分析：只检查目标意图的语料
            utterances_to_check = []
            for intent_name in target_intents:
                utterances_to_check.extend(self.dataset.get_utterances(intent=intent_name))
            logger.info(f"Running targeted boundary violation analysis for: {target_intents}")
        
        # 逐个检查语料
        for utterance in utterances_to_check:
            original_intent = utterance.intent
            
            # 获取降维后的嵌入向量
            reduced_embedding = reduced_embeddings_map.get(utterance.text)
            if reduced_embedding is None:
                continue
            
            best_violation = None
            
            # 与其他意图的分布进行比较
            for target_intent, stats in intent_stats.items():
                if target_intent == original_intent:
                    continue
                
                # 计算马氏距离和p值
                mahalanobis_dist, p_value = self.calculate_mahalanobis_distance(
                    reduced_embedding, stats["mean"], stats["inv_cov"]
                )
                
                # 检查是否超过阈值
                if p_value > p_value_threshold:
                    if best_violation is None or p_value > best_violation.confused_with.p_value:
                        best_violation = BoundaryViolationRecord(
                            text=utterance.text,
                            original_intent=original_intent,
                            confused_with={
                                "intent": target_intent,
                                "p_value": p_value,
                                "mahalanobis_distance": mahalanobis_dist,
                            },
                        )
            
            if best_violation:
                violations.append(best_violation)
        
        # 按p值降序排序
        violations.sort(key=lambda v: v.confused_with.p_value, reverse=True)
        
        logger.success(f"Boundary violation detection completed. Found {len(violations)} violations.")
        return violations


