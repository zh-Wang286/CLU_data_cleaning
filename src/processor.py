# -*- coding: utf-8 -*-
"""Core data processing logic for CLU data analysis."""

import hashlib
from pathlib import Path
import pickle
import textwrap
import time
from typing import Dict, List, Literal, Optional

import hdbscan
from loguru import logger
import numpy as np
from openai import OpenAIError
from openai.types.create_embedding_response import CreateEmbeddingResponse
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.dataset import CLUDataset
from src.model_client import model_client
from src.schemas import Utterance


class CLUProcessor:
    """
    Orchestrates the analysis of a CLU dataset, including embedding,
    outlier detection, and clustering audits.
    """

    def __init__(self, dataset: CLUDataset, output_dir: Path = Path("outputs")):
        """
        Initializes the processor with a dataset.

        Args:
            dataset: The CLUDataset instance to be processed.
            output_dir: The root directory for saving outputs like embeddings.
        """
        self.dataset = dataset
        self.output_dir = output_dir
        self.embeddings_dir = self.output_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"CLUProcessor initialized. Outputs will be saved to '{self.output_dir.resolve()}'"
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

    def detect_intra_intent_outliers(
        self,
        embeddings_map: Dict[str, np.ndarray],
        method: Literal["knn", "lof"] = "knn",
        k: int = 1,
        threshold_policy: Literal["95pct", "iqr"] = "95pct",
    ) -> Dict[str, List[Dict]]:
        """
        Detects outliers within each intent based on embedding distances.

        Args:
            embeddings_map: A map from utterance text to its embedding.
            method: The outlier detection method to use ('knn' or 'lof'). Currently, only 'knn' is implemented.
            k: The number of nearest neighbors to consider for the k-NN distance.
            threshold_policy: The policy to determine the outlier threshold ('95pct' or 'iqr').

        Returns:
            A dictionary where keys are intent names and values are lists of
            outlier records, each containing the utterance index, text, score, etc.
        """
        logger.info(
            f"Starting intra-intent outlier detection using method='{method}' with k={k} and threshold='{threshold_policy}'..."
        )
        all_outliers = {}

        for intent in self.dataset.get_intents():
            intent_name = intent.category
            utterances = self.dataset.get_utterances(intent=intent_name)

            if len(utterances) <= k + 1:
                logger.warning(
                    f"Skipping intent '{intent_name}' for outlier detection: not enough samples ({len(utterances)})."
                )
                continue

            # This assumes that the order is preserved and embeddings_map contains all utterances
            intent_embeddings = np.array(
                [embeddings_map[utt.text] for utt in utterances]
            )

            # Calculate pairwise cosine distances
            distances = cosine_distances(intent_embeddings)

            # For each point, find the distance to its k-th nearest neighbor
            np.partition(distances, k, axis=1)
            k_nearest_distances = distances[:, k]

            # Determine threshold
            if threshold_policy == "95pct":
                threshold = np.percentile(k_nearest_distances, 95)
            elif threshold_policy == "iqr":
                q1, q3 = np.percentile(k_nearest_distances, [25, 75])
                iqr = q3 - q1
                threshold = q3 + 1.5 * iqr
            else:
                raise ValueError(f"Unknown threshold policy: {threshold_policy}")

            # Identify outliers
            outlier_indices = np.where(k_nearest_distances > threshold)[0]
            if len(outlier_indices) > 0:
                outlier_records = []
                # Sort outliers by their distance score
                sorted_indices = outlier_indices[
                    np.argsort(-k_nearest_distances[outlier_indices])
                ]

                for rank, idx in enumerate(sorted_indices):
                    outlier_records.append(
                        {
                            "original_idx": idx,  # Use the index within the intent's utterance list
                            "text": utterances[idx].text,
                            "score": k_nearest_distances[idx],
                            "threshold": threshold,
                            "rank": rank + 1,
                        }
                    )
                all_outliers[intent_name] = outlier_records
                logger.info(
                    f"Detected {len(outlier_records)} outliers in intent '{intent_name}' (threshold={threshold:.4f})."
                )

        logger.success(
            f"Outlier detection complete. Found outliers in {len(all_outliers)} intents."
        )
        return all_outliers

    def audit_global_clusters(
        self,
        embeddings_map: Dict[str, np.ndarray],
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
    ) -> Dict:
        """
        Performs a global clustering audit using HDBSCAN to find potential overlaps.

        Args:
            embeddings_map: A map from utterance text to its embedding.
            min_cluster_size: The minimum size of clusters to be considered by HDBSCAN.
            min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

        Returns:
            A dictionary containing the clustering results and an audit of each cluster.
        """
        logger.info(
            f"Starting global clustering audit with min_cluster_size={min_cluster_size}..."
        )

        utterances = self.dataset.get_utterances()
        texts = [utt.text for utt in utterances]
        intents = [utt.intent for utt in utterances]
        embeddings = np.array([embeddings_map[text] for text in texts])

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",  # HDBSCAN works well with euclidean on normalized vectors
        )
        # Normalize embeddings for distance calculation. This is mathematically equivalent
        # to using cosine similarity for clustering purposes.
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        cluster_labels = clusterer.fit_predict(embeddings)

        # Create a DataFrame for easy analysis
        df = pd.DataFrame({"text": texts, "original_intent": intents, "cluster": cluster_labels})

        # --- Analysis ---
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_ratio = np.sum(cluster_labels == -1) / len(cluster_labels)
        logger.info(
            f"HDBSCAN found {num_clusters} clusters and {noise_ratio:.2%} noise."
        )

        cluster_audit = []
        for cluster_id in sorted(set(cluster_labels)):
            if cluster_id == -1:
                continue

            cluster_df = df[df["cluster"] == cluster_id]
            intent_counts = cluster_df["original_intent"].value_counts()

            majority_intent = intent_counts.idxmax()
            majority_count = intent_counts.max()
            total_samples = len(cluster_df)
            purity = majority_count / total_samples

            if total_samples < 3:  # As per fds.md, filter small clusters
                continue

            cluster_audit.append(
                {
                    "cluster_id": cluster_id,
                    "size": total_samples,
                    "majority_intent": majority_intent,
                    "purity": purity,
                    "intent_distribution": intent_counts.to_dict(),
                }
            )

        # Sort clusters by purity and size to highlight the most problematic ones
        cluster_audit.sort(key=lambda x: (x["purity"], -x["size"]))

        logger.success("Global clustering audit complete.")
        return {
            "summary": {
                "num_clusters": num_clusters,
                "noise_ratio": noise_ratio,
                "total_utterances": len(df),
            },
            "clusters": cluster_audit,
            "raw_labels": df.to_dict("records"),
        }

    def enrich_low_utterance_intents(self, threshold: int = 25) -> Dict[str, List[str]]:
        """
        Generates new utterance candidates for intents with low sample counts.
        The number of generated utterances for each intent will be the difference
        between the threshold and its current utterance count.

        Args:
            threshold: The minimum number of utterances an intent should have.

        Returns:
            A dictionary where keys are intent names and values are lists of
            newly generated utterance candidates. This is for review and does not
            modify the original dataset.
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
        self, intent: "Intent", examples: List[Utterance], num_to_generate: int
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
