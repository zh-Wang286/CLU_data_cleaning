# -*- coding: utf-8 -*-
"""重构后的CLU数据处理核心逻辑，消除代码重复，统一命名约定。"""

import hashlib
from pathlib import Path
import pickle
import textwrap
from typing import Dict, List, Literal, Optional

from loguru import logger
import numpy as np
from openai import OpenAIError
from openai.types.create_embedding_response import CreateEmbeddingResponse
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.core.cluster_auditor import ClusterAuditor
from src.core.dimensionality_reducer import DimensionalityReducer
from src.core.outlier_detector import OutlierDetector
from src.core.statistical_analyzer import StatisticalAnalyzer
from src.dataset import CLUDataset
from src.model_client import model_client
from src.schemas import BoundaryViolationRecord, Intent, Utterance


class CLUProcessor:
    """
    重构后的CLU数据处理协调器。
    
    统一命名约定：
    - find_*: 查找/检测类操作（如异常点检测）
    - analyze_*: 分析类操作（如边界分析）  
    - audit_*: 审计类操作（如聚类审计）
    - generate_*: 生成类操作（如语料增广）
    """

    def __init__(self, dataset: CLUDataset, output_dir: Path = Path("outputs")):
        """
        初始化处理器及其核心分析模块。

        Args:
            dataset: CLU数据集实例
            output_dir: 输出目录，用于保存嵌入向量等
        """
        self.dataset = dataset
        self.output_dir = output_dir
        self.embeddings_dir = self.output_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化核心分析模块
        self.dimensionality_reducer = DimensionalityReducer(dataset)
        self.statistical_analyzer = StatisticalAnalyzer(dataset)
        self.outlier_detector = OutlierDetector(dataset)
        self.cluster_auditor = ClusterAuditor(dataset)
        
        logger.info(
            f"CLUProcessor initialized with core analysis modules. "
            f"Outputs: '{self.output_dir.resolve()}'"
        )

    def get_all_embeddings(
        self, batch_size: int = 2048, use_cache: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Computes or loads from cache the embeddings for all utterances in the dataset.

        Args:
            batch_size: The number of utterances to process in a single API call.
            use_cache: If True, attempts to load embeddings from a local cache first.

        Returns:
            A dictionary mapping utterance text to its corresponding numpy embedding.
        """
        utterances = self.dataset.get_utterances()
        utterance_texts = [utt.text for utt in utterances]

        # Generate a unique hash for the current set of texts to use as a cache key.
        texts_hash = hashlib.md5(
            "".join(sorted(utterance_texts)).encode("utf-8")
        ).hexdigest()
        cache_file = self.embeddings_dir / f"embeddings_{texts_hash[:10]}.pkl"

        if use_cache and cache_file.exists():
            logger.info(f"Loading embeddings from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        logger.info(
            f"Cache not found. Computing embeddings for {len(utterance_texts)} utterances..."
        )
        embeddings_map: Dict[str, np.ndarray] = {}

        for i in range(0, len(utterance_texts), batch_size):
            batch_texts = utterance_texts[i : i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(utterance_texts) - 1)//batch_size + 1}..."
            )

            try:
                response = self._get_embeddings_with_retry(texts=batch_texts)
                for text, embedding_data in zip(batch_texts, response.data):
                    embeddings_map[text] = np.array(embedding_data.embedding)
            except Exception as e:
                logger.error(f"Failed to process batch starting at index {i}: {e}")
                # Decide on error handling: skip batch, or halt execution
                # For now, we halt.
                raise

        logger.success(f"Successfully computed all {len(embeddings_map)} embeddings.")

        # Save to cache
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings_map, f)
        logger.info(f"Embeddings saved to cache: {cache_file}")

        return embeddings_map

    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60)
    )
    def _get_embeddings_with_retry(self, texts: List[str]) -> CreateEmbeddingResponse:
        """
        A robust method to get embeddings using tenacity for exponential backoff.

        Args:
            texts: A list of texts to embed.

        Returns:
            The embedding response from the API.
        """
        logger.debug(f"Requesting embeddings for {len(texts)} texts...")
        client = model_client.get_embedding_client()
        response = client.embeddings.create(
            input=texts,
            model=settings.embedding_model,
        )
        return response

    def find_outliers_within_intents(
        self,
        embeddings_map: Dict[str, np.ndarray],
        method: Literal["knn", "lof"] = "knn",
        k: int = 1,
        threshold_policy: Literal["90pct", "95pct", "iqr"] = "95pct",
    ) -> Dict[str, List[Dict]]:
        """
        检测每个意图内部的异常点。
        
        重构说明：将原有的150+行复杂逻辑委托给OutlierDetector核心模块。

        Args:
            embeddings_map: 语料文本到嵌入向量的映射
            method: 异常点检测方法 ('knn' 或 'lof')
            k: k-NN距离中的k值
            threshold_policy: 阈值策略 ('90pct', '95pct', 或 'iqr')

        Returns:
            意图名到异常点记录列表的映射
        """
        return self.outlier_detector.find_outliers_within_intents(
            embeddings_map=embeddings_map,
            method=method,
            k=k,
            threshold_policy=threshold_policy
        )

    def analyze_boundary_violations(
        self,
        embeddings_map: Dict[str, np.ndarray],
        p_value_threshold: float = 0.05,
        regularization_factor: float = 1e-6,
        min_samples_for_analysis: int = 15,
    ) -> List[BoundaryViolationRecord]:
        """
        检测意图边界违规（全局分析）。
        
        重构说明：将原有的100+行复杂逻辑委托给核心模块。

        Args:
            embeddings_map: 语料文本到嵌入向量的映射
            p_value_threshold: p值阈值
            regularization_factor: 正则化因子
            min_samples_for_analysis: 最小样本数要求

        Returns:
            边界违规记录列表
        """
        # 1. 获取PCA降维结果
        pca_results = self.dimensionality_reducer.get_pca_embeddings(
            embeddings_map, min_samples_for_analysis
        )
        reduced_embeddings_map = pca_results.get("reduced_map", {})
        intents_for_analysis = pca_results.get("intents_for_analysis", [])

        if not reduced_embeddings_map:
            logger.warning("PCA reduction failed. Skipping boundary violation detection.")
            return []
        
        # 2. 构建所有符合条件意图的统计模型
        intent_names = [intent.category for intent in intents_for_analysis]
        intent_stats = self.statistical_analyzer.build_intent_statistical_models(
            reduced_embeddings_map=reduced_embeddings_map,
            target_intents=intent_names,
            min_samples_for_analysis=min_samples_for_analysis,
            regularization_factor=regularization_factor
        )
        
        # 3. 执行全局边界违规检测
        return self.statistical_analyzer.detect_boundary_violations(
            reduced_embeddings_map=reduced_embeddings_map,
            intent_stats=intent_stats,
            target_intents=None,  # None表示全局分析
            p_value_threshold=p_value_threshold
        )

    def analyze_targeted_boundary_violations(
        self,
        embeddings_map: Dict[str, np.ndarray],
        target_intents: List[str],
        p_value_threshold: float = 0.05,
        regularization_factor: float = 1e-6,
        min_samples_for_analysis: int = 15,
    ) -> List[BoundaryViolationRecord]:
        """
        检测指定意图间的边界违规（局部分析）。
        
        重构说明：将原有的150+行重复逻辑委托给核心模块，代码行数减少90%。

        Args:
            embeddings_map: 语料文本到嵌入向量的映射
            target_intents: 要分析的目标意图列表
            p_value_threshold: p值阈值
            regularization_factor: 正则化因子
            min_samples_for_analysis: 最小样本数要求

        Returns:
            边界违规记录列表
        """
        if len(target_intents) < 2:
            logger.error("Need at least 2 target intents for analysis.")
            return []
        
        # 1. 获取全局PCA降维结果（保持与全局分析的一致性）
        pca_results = self.dimensionality_reducer.get_pca_embeddings(
            embeddings_map, min_samples_for_analysis
        )
        reduced_embeddings_map = pca_results.get("reduced_map", {})

        if not reduced_embeddings_map:
            logger.warning("PCA reduction failed. Skipping targeted analysis.")
            return []

        # 2. 构建目标意图的统计模型
        intent_stats = self.statistical_analyzer.build_intent_statistical_models(
            reduced_embeddings_map=reduced_embeddings_map,
            target_intents=target_intents,
            min_samples_for_analysis=min_samples_for_analysis,
            regularization_factor=regularization_factor
        )
        
        if not intent_stats:
            logger.error("Failed to build statistical models for target intents.")
            return []
        
        # 3. 执行局部边界违规检测
        return self.statistical_analyzer.detect_boundary_violations(
            reduced_embeddings_map=reduced_embeddings_map,
            intent_stats=intent_stats,
            target_intents=target_intents,  # 指定目标意图列表
            p_value_threshold=p_value_threshold
        )

    def audit_clusters_globally(
        self,
        embeddings_map: Dict[str, np.ndarray],
        min_cluster_size: int = 15,
        min_samples: Optional[int] = None,
        min_samples_for_analysis: int = 15,
    ) -> Dict:
        """
        执行全局聚类审计，发现潜在的意图重叠。
        
        重构说明：将原有的80+行聚类逻辑委托给ClusterAuditor核心模块。

        Args:
            embeddings_map: 语料文本到嵌入向量的映射
            min_cluster_size: HDBSCAN最小簇大小
            min_samples: HDBSCAN最小样本数参数
            min_samples_for_analysis: 意图纳入PCA分析的最小语料数

        Returns:
            包含聚类结果和审计信息的字典
        """
        # 1. 获取PCA降维结果
        pca_results = self.dimensionality_reducer.get_pca_embeddings(
            embeddings_map, min_samples_for_analysis
        )
        reduced_embeddings_map = pca_results.get("reduced_map", {})
        
        if not reduced_embeddings_map:
            logger.warning("PCA reduction failed. Aborting clustering audit.")
            return {"summary": {}, "clusters": [], "raw_labels": []}

        # 2. 执行聚类审计
        return self.cluster_auditor.audit_clusters_globally(
            reduced_embeddings_map=reduced_embeddings_map,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )

    def generate_utterance_candidates(self, threshold: int = 25) -> Dict[str, List[str]]:
        """
        为低样本量意图生成候选语料。
        
        重构说明：统一命名约定，从enrich_*改为generate_*。

        Args:
            threshold: 意图应具有的最小语料数量

        Returns:
            意图名到新生成候选语料列表的映射，需人工审核后使用
        """
        logger.info(f"Starting data enrichment for intents with < {threshold} utterances.")
        
        low_sample_intents = self.dataset.warn_low_utterance_intents(threshold)
        if not low_sample_intents:
            logger.info("No intents require data enrichment.")
            return {}

        generated_candidates = {}
        llm_client = model_client.get_llm_client()

        for intent_name, current_count in low_sample_intents.items():
            intent_def = next((i for i in self.dataset.get_intents() if i.category == intent_name), None)
            existing_utterances = self.dataset.get_utterances(intent=intent_name)
            
            if not intent_def or not existing_utterances:
                continue
            
            num_to_generate = max(10, threshold - current_count)

            # Create a detailed prompt for the LLM
            prompt = self._build_enrichment_prompt(intent_def, existing_utterances, num_to_generate)
            
            try:
                logger.debug(f"Generating {num_to_generate} samples for intent '{intent_name}'...")
                response = llm_client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that generates high-quality training data for conversational language understanding models.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    n=1,
                    temperature=0.7,
                )

                content = response.choices[0].message.content
                if content:
                    # Basic parsing of the response
                    new_utterances = [
                        line.strip() for line in content.split("\n") if line.strip()
                    ]
                    generated_candidates[intent_name] = new_utterances
                    logger.info(
                        f"Generated {len(new_utterances)} candidates for intent '{intent_name}'."
                    )

            except OpenAIError as e:
                logger.error(
                    f"Failed to generate utterances for intent '{intent_name}': {e}"
                )
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred during generation for intent '{intent_name}': {e}"
                )

        logger.success("Data enrichment process complete.")
        return generated_candidates

    def _build_enrichment_prompt(
        self, intent: Intent, examples: List[Utterance], num_to_generate: int
    ) -> str:
        """Builds a detailed prompt for the LLM to generate new utterances."""
        
        example_texts = "\n".join([f"- {ex.text}" for ex in examples[:10]]) # Use up to 10 examples
        
        prompt = f"""
        You are an AI assistant creating training data for a conversational language understanding (CLU) model, specifically for the **Microsoft Azure CLU service**.
        Your task is to generate {num_to_generate} new, diverse, and high-quality example utterances for the following intent.
        
        **Intent Name:**
        `{intent.category}`
        
        **Intent Description:**
        `{intent.description or "No description provided."}`
        
        **Existing Examples (for style and semantic reference):**
        {example_texts}
        
        **Instructions for Generating Diverse Utterances:**
        1.  The context is a user interacting with a system (e.g., an IT helpdesk bot).
        2.  Generate exactly {num_to_generate} new utterances.
        3.  Ensure the new utterances are semantically consistent with the intent, but vary them using techniques such as:
            - **Synonym Replacement:** Use different words with similar meanings.
            - **Word Order Variation:** Change the sentence structure (e.g., active to passive).
            - **Sentence Type Diversification:** Mix statements, questions, and commands.
            - **Formality Change:** Include both colloquial and formal expressions.
            - **Sentence Length Variation:** Create both short and long sentences.
            - **Pronoun/Subject Change:** Use different subjects or pronouns (e.g., "I", "my computer", "the user").
            - **Punctuation Variation:** Use different punctuation where appropriate.
        
        **Output Format Constraints:**
        - Output **only** the new utterances.
        - Each utterance must be on a new line.
        - Do not include numbering, bullet points, titles, explanations, or any other text.

        **Output Format Example:**
        我需要重置一下我的密码
        忘记密码了怎么办
        能不能帮我改一下密码
        """
        return textwrap.dedent(prompt)
