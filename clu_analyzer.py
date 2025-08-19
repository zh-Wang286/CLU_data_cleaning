#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Azure CLU数据质量分析器

本模块实现对Azure CLU训练数据的语义分析，包括：
1. Intent间低耦合性分析（语义区分度）
2. Intent内高内聚性分析（样本一致性）
3. 数据可视化与质量报告

作者: Data Science Team
创建时间: 2025-01-27
"""

import json
import logging
import os
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from openai import AzureOpenAI
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from mpl_toolkits.mplot3d import Axes3D

# 忽略特定的FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn.utils.deprecation')
warnings.filterwarnings("ignore", category=UserWarning, module='umap.umap_')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLUDataAnalyzer:
    """Azure CLU数据质量分析器主类"""
    
    def __init__(self, 
                 api_key: str,
                 azure_endpoint: str,
                 deployment_name: str = "NNIT-Ada-3-large",
                 api_version: str = "2024-12-01-preview"):
        """
        初始化分析器
        
        Args:
            api_key: Azure OpenAI API密钥
            azure_endpoint: Azure OpenAI服务端点
            deployment_name: 部署模型名称
            api_version: API版本
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version
        )
        self.deployment_name = deployment_name
        
        # 数据存储
        self.intents: List[str] = []
        self.utterances: List[Dict[str, Any]] = []
        self.intent_utterances: Dict[str, List[str]] = defaultdict(list)
        self.text_to_embedding: Dict[str, np.ndarray] = {}
        
        logger.info(f"CLU分析器初始化完成，使用模型: {deployment_name}")
    
    def load_data(self, json_file_path: str) -> None:
        """
        加载CLU训练数据
        
        Args:
            json_file_path: JSON数据文件路径
        """
        logger.info(f"正在加载数据文件: {json_file_path}")
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取Intent列表
            self.intents = [intent['category'] for intent in data['assets']['intents']]
            
            # 提取utterances
            self.utterances = data['assets']['utterances']
            
            # 按Intent分组utterances
            for utterance in self.utterances:
                intent = utterance['intent']
                text = utterance['text']
                self.intent_utterances[intent].append(text)
            
            logger.info(f"数据加载完成: {len(self.intents)}个Intent, {len(self.utterances)}条utterance")
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise
    
    def _generate_all_embeddings(self):
        """一次性为所有唯一的utterance生成嵌入向量，以提高效率。"""
        logger.info("开始为所有utterances生成嵌入向量...")
        all_texts = [u['text'] for u in self.utterances]
        
        unique_texts = list(set(all_texts))
        logger.info(f"总共 {len(all_texts)} utterances, 其中 {len(unique_texts)} 是唯一的。")
        
        if not unique_texts:
            logger.warning("数据中没有找到任何utterance文本。")
            return

        unique_embeddings_array = self.get_embeddings(unique_texts)
        
        self.text_to_embedding = {text: emb for text, emb in zip(unique_texts, unique_embeddings_array)}
        
        logger.info("所有唯一utterances的嵌入向量已生成。")

    def get_embeddings(self, texts: List[str], batch_size: int = 100, max_workers: int = 20) -> np.ndarray:
        """
        批量获取文本嵌入向量（多线程）
        
        Args:
            texts: 待嵌入的文本列表
            batch_size: 批处理大小
            max_workers: 并发线程数
            
        Returns:
            嵌入向量矩阵
        """
        logger.info(f"正在获取{len(texts)}个文本的嵌入向量 (使用{max_workers}个线程)")
        
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        # 使用字典确保嵌入向量按原始顺序排列
        embeddings_dict: Dict[int, List[List[float]]] = {}

        def get_batch_embedding(batch_texts: List[str], batch_index: int):
            """获取单个批次的嵌入向量"""
            try:
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.deployment_name
                )
                embeddings_dict[batch_index] = [data.embedding for data in response.data]
                logger.info(f"批次 {batch_index + 1}/{len(batches)} 处理完成")
            except Exception as e:
                logger.error(f"批次 {batch_index} 嵌入向量获取失败: {str(e)}")
                # 标记失败，但不存储任何内容
                embeddings_dict[batch_index] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有批次任务
            futures = [executor.submit(get_batch_embedding, batch, i) for i, batch in enumerate(batches)]
            # 等待所有任务完成
            for future in futures:
                future.result() # 等待任务完成，并捕获异常
        
        # 按批次索引排序并合并结果
        full_embeddings = []
        for i in range(len(batches)):
            batch_result = embeddings_dict.get(i)
            if batch_result is None or not batch_result:
                # 如果某个批次失败，抛出异常
                raise RuntimeError(f"批次 {i} 的嵌入向量获取失败，分析中止")
            full_embeddings.extend(batch_result)
        
        return np.array(full_embeddings)
    
    def analyze_inter_intent_coupling(self) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
        """
        分析Intent间的低耦合性（语义区分度）
        
        Returns:
            相似度矩阵, 分析结果DataFrame, 和用于生成矩阵的intent名称列表
        """
        logger.info("开始Intent间低耦合性分析")
        
        # 计算每个Intent的代表性嵌入（平均值）
        intent_embeddings = []
        ordered_intent_names = []
        
        for intent in self.intents:
            if intent in self.intent_utterances and self.intent_utterances[intent]:
                utterance_texts = self.intent_utterances[intent]
                embeddings = np.array([self.text_to_embedding[text] for text in utterance_texts])
                
                # 使用平均嵌入作为Intent的代表
                intent_embedding = np.mean(embeddings, axis=0)
                intent_embeddings.append(intent_embedding)
                ordered_intent_names.append(intent)
        
        intent_embeddings = np.array(intent_embeddings)
        
        # 计算Intent间相似度矩阵
        similarity_matrix = cosine_similarity(intent_embeddings)
        
        # 分析高相似度的Intent对
        high_similarity_pairs = []
        threshold = 0.8  # 高相似度阈值
        
        for i in range(len(ordered_intent_names)):
            for j in range(i + 1, len(ordered_intent_names)):
                similarity = similarity_matrix[i][j]
                if similarity > threshold:
                    high_similarity_pairs.append({
                        'Intent1': ordered_intent_names[i],
                        'Intent2': ordered_intent_names[j],
                        'Similarity': similarity
                    })
        
        results_df = pd.DataFrame(high_similarity_pairs)
        results_df = results_df.sort_values('Similarity', ascending=False)
        
        logger.info(f"发现{len(high_similarity_pairs)}对高相似度Intent")
        
        return similarity_matrix, results_df, ordered_intent_names
    
    def analyze_intra_intent_cohesion(self, min_cluster_size: int = 3) -> Dict[str, Dict]:
        """
        分析Intent内的高内聚性（样本一致性）
        
        Args:
            min_cluster_size: HDBSCAN最小聚类大小
            
        Returns:
            每个Intent的聚类分析结果
        """
        logger.info("开始Intent内高内聚性分析")
        
        cohesion_results = {}
        
        for intent in self.intents:
            if intent not in self.intent_utterances or len(self.intent_utterances[intent]) < min_cluster_size:
                continue
            
            logger.info(f"分析Intent: {intent}")
            
            utterance_texts = self.intent_utterances[intent]
            embeddings = np.array([self.text_to_embedding[text] for text in utterance_texts])
            
            # 使用HDBSCAN进行聚类
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='euclidean' # OpenAI嵌入向量已归一化, 使用欧氏距离等价于余弦相似度
            )
            
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # 计算聚类质量指标
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            # 计算轮廓系数（如果有多个聚类）
            silhouette_avg = 0.0
            if n_clusters > 1:
                # 过滤噪声点
                valid_indices = cluster_labels != -1
                if np.sum(valid_indices) > 1:
                    silhouette_avg = silhouette_score(
                        embeddings[valid_indices], 
                        cluster_labels[valid_indices],
                        metric='cosine'
                    )
            
            cohesion_results[intent] = {
                'n_samples': len(utterance_texts),
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': n_noise / len(utterance_texts),
                'silhouette_score': silhouette_avg,
                'cluster_labels': cluster_labels,
                'utterances': utterance_texts,
                'embeddings': embeddings
            }
        
        logger.info(f"完成{len(cohesion_results)}个Intent的内聚性分析")
        
        return cohesion_results
    
    def create_visualizations(self, 
                            similarity_matrix: np.ndarray,
                            intent_names: List[str],
                            cohesion_results: Dict[str, Dict],
                            output_dir: str = "output") -> None:
        """
        创建数据可视化图表
        
        Args:
            similarity_matrix: Intent间相似度矩阵
            intent_names: Intent名称列表
            cohesion_results: 内聚性分析结果
            output_dir: 输出目录
        """
        logger.info("开始创建可视化图表")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 强制重建matplotlib字体缓存以识别新安装的字体
        try:
            import matplotlib
            cache_dir = matplotlib.get_cachedir()
            for file in os.listdir(cache_dir):
                if file.startswith('fontlist-') and file.endswith('.json'):
                    os.remove(os.path.join(cache_dir, file))
            logger.info("Matplotlib字体缓存已清除，将进行重建。")
        except Exception as e:
            logger.warning(f"无法清除Matplotlib字体缓存: {e}")

        # 动态设置支持中文的字体
        try:
            from matplotlib.font_manager import FontProperties
            
            # 常见的支持中文的字体列表
            font_list = ['WenQuanYi Micro Hei', 'SimHei', 'WenQuanYi Zen Hei', 'Microsoft YaHei', 'Heiti TC', 'sans-serif']
            
            # 查找并设置可用字体
            font_path = None
            for font in font_list:
                try:
                    # 尝试查找字体
                    font_path = FontProperties(font).get_name()
                    plt.rcParams['font.sans-serif'] = [font]
                    logger.info(f"成功设置中文字体: {font}")
                    break
                except:
                    continue
            
            if not font_path:
                 logger.warning("未找到可用的中文字体，图表中的中文可能无法正常显示。请安装'SimHei'或'WenQuanYi Zen Hei'等字体。")

        except ImportError:
            logger.warning("matplotlib.font_manager未找到，无法动态设置中文字体。")

        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
        
        # 1. Intent相似度热力图
        plt.figure(figsize=(20, 16))
        
        # 只显示Intent名称的后半部分（去掉分类前缀）
        display_names = [name.split('_')[-1] if '_' in name else name for name in intent_names]
        
        sns.heatmap(
            similarity_matrix,
            xticklabels=display_names,
            yticklabels=display_names,
            annot=False,
            cmap='RdYlBu_r',
            center=0.5,
            square=True,
            linewidths=0.1
        )
        
        plt.title('Intent Semantic Similarity Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Intent Categories', fontsize=12)
        plt.ylabel('Intent Categories', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/intent_similarity_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Intent内聚性分析图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 聚类数量分布
        cluster_counts = [result['n_clusters'] for result in cohesion_results.values()]
        ax1.hist(cluster_counts, bins=range(0, max(cluster_counts) + 2), alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Distribution of Clusters per Intent', fontweight='bold')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Number of Intents')
        ax1.grid(True, alpha=0.3)
        
        # 噪声比例分布
        noise_ratios = [result['noise_ratio'] for result in cohesion_results.values()]
        ax2.hist(noise_ratios, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_title('Distribution of Noise Ratios', fontweight='bold')
        ax2.set_xlabel('Noise Ratio')
        ax2.set_ylabel('Number of Intents')
        ax2.grid(True, alpha=0.3)
        
        # 轮廓系数分布
        silhouette_scores = [result['silhouette_score'] for result in cohesion_results.values() 
                           if result['silhouette_score'] > 0]
        if silhouette_scores:
            ax3.hist(silhouette_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('Distribution of Silhouette Scores', fontweight='bold')
        ax3.set_xlabel('Silhouette Score')
        ax3.set_ylabel('Number of Intents')
        ax3.grid(True, alpha=0.3)
        
        # 样本数量分布
        sample_counts = [result['n_samples'] for result in cohesion_results.values()]
        ax4.hist(sample_counts, bins=20, alpha=0.7, color='gold', edgecolor='black')
        ax4.set_title('Distribution of Sample Counts per Intent', fontweight='bold')
        ax4.set_xlabel('Number of Samples')
        ax4.set_ylabel('Number of Intents')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/intent_cohesion_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 2D可视化散点图（使用UMAP降维）
        logger.info("创建2D可视化散点图")
        
        # 收集所有嵌入向量和标签
        all_embeddings = []
        all_labels = []
        intent_colors = {}
        
        # 为每个Intent分配颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(cohesion_results)))
        
        for idx, (intent, result) in enumerate(cohesion_results.items()):
            embeddings = result['embeddings']
            all_embeddings.append(embeddings)
            all_labels.extend([intent] * len(embeddings))
            intent_colors[intent] = colors[idx]
        
        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)
            
            # 使用UMAP进行降维
            reducer = UMAP(n_components=2, random_state=42, metric='cosine')
            embedding_2d = reducer.fit_transform(all_embeddings)
            
            # 创建散点图
            plt.figure(figsize=(14, 10))
            
            for intent in intent_colors:
                mask = np.array(all_labels) == intent
                if np.any(mask):
                    plt.scatter(
                        embedding_2d[mask, 0],
                        embedding_2d[mask, 1],
                        c=[intent_colors[intent]],
                        label=intent.split('_')[-1] if '_' in intent else intent,
                        alpha=0.6,
                        s=30
                    )
            
            plt.title('2D Visualization of Intent Utterances (UMAP)', fontsize=16, fontweight='bold')
            plt.xlabel('UMAP Dimension 1', fontsize=12)
            plt.ylabel('UMAP Dimension 2', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/intent_2d_visualization.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. 3D可视化散点图（使用UMAP降维）
            logger.info("创建3D可视化散点图")
            
            # 使用UMAP进行3D降维
            reducer_3d = UMAP(n_components=3, random_state=42, metric='cosine')
            embedding_3d = reducer_3d.fit_transform(all_embeddings)

            # 创建3D散点图
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')

            for intent in intent_colors:
                mask = np.array(all_labels) == intent
                if np.any(mask):
                    ax.scatter(
                        embedding_3d[mask, 0],
                        embedding_3d[mask, 1],
                        embedding_3d[mask, 2],
                        c=[intent_colors[intent]],
                        label=intent.split('_')[-1] if '_' in intent else intent,
                        alpha=0.6,
                        s=30
                    )
            
            ax.set_title('3D Visualization of Intent Utterances (UMAP)', fontsize=16, fontweight='bold')
            ax.set_xlabel('UMAP Dimension 1', fontsize=12)
            ax.set_ylabel('UMAP Dimension 2', fontsize=12)
            ax.set_zlabel('UMAP Dimension 3', fontsize=12)
            ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left', fontsize=8)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/intent_3d_visualization.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"可视化图表已保存到 {output_dir} 目录")
    
    def generate_report(self, 
                       similarity_results: pd.DataFrame,
                       cohesion_results: Dict[str, Dict],
                       output_dir: str = "output") -> None:
        """
        生成分析报告
        
        Args:
            similarity_results: 相似度分析结果
            cohesion_results: 内聚性分析结果
            output_dir: 输出目录
        """
        logger.info("生成分析报告")
        
        os.makedirs(output_dir, exist_ok=True)
        
        report_lines = [
            "# Azure CLU数据质量分析报告",
            "",
            f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 1. 数据概览",
            f"- Intent总数: {len(self.intents)}",
            f"- Utterance总数: {len(self.utterances)}",
            f"- 平均每个Intent的Utterance数量: {len(self.utterances) / len(self.intents):.1f}",
            "",
            "## 2. Intent间低耦合性分析",
            ""
        ]
        
        if not similarity_results.empty:
            report_lines.extend([
                f"发现 {len(similarity_results)} 对高相似度Intent（相似度 > 0.8）:",
                ""
            ])
            
            for _, row in similarity_results.head(10).iterrows():
                report_lines.append(
                    f"- {row['Intent1']} ↔ {row['Intent2']} (相似度: {row['Similarity']:.3f})"
                )
            
            if len(similarity_results) > 10:
                report_lines.append(f"... 还有 {len(similarity_results) - 10} 对")
        else:
            report_lines.append("✅ 未发现高相似度Intent对，语义区分度良好")
        
        report_lines.extend([
            "",
            "## 3. Intent内高内聚性分析",
            ""
        ])
        
        # 统计内聚性问题
        multi_cluster_intents = []
        high_noise_intents = []
        low_silhouette_intents = []
        
        for intent, result in cohesion_results.items():
            if result['n_clusters'] > 1:
                multi_cluster_intents.append((intent, result['n_clusters']))
            
            if result['noise_ratio'] > 0.2:  # 噪声比例超过20%
                high_noise_intents.append((intent, result['noise_ratio']))
            
            if result['silhouette_score'] > 0 and result['silhouette_score'] < 0.5:
                low_silhouette_intents.append((intent, result['silhouette_score']))
        
        if multi_cluster_intents:
            report_lines.extend([
                f"### 3.1 多聚类Intent ({len(multi_cluster_intents)}个)",
                "以下Intent内部可能语义过宽，建议拆分:",
                ""
            ])
            
            for intent, n_clusters in sorted(multi_cluster_intents, key=lambda x: x[1], reverse=True)[:10]:
                report_lines.append(f"- {intent} ({n_clusters}个聚类)")
        
        if high_noise_intents:
            report_lines.extend([
                "",
                f"### 3.2 高噪声Intent ({len(high_noise_intents)}个)",
                "以下Intent存在较多异常样本，建议检查数据质量:",
                ""
            ])
            
            for intent, noise_ratio in sorted(high_noise_intents, key=lambda x: x[1], reverse=True)[:10]:
                report_lines.append(f"- {intent} (噪声比例: {noise_ratio:.1%})")
        
        # 保存报告
        with open(f"{output_dir}/analysis_report.md", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # 保存详细结果
        similarity_results.to_csv(f"{output_dir}/high_similarity_intents.csv", index=False)
        
        cohesion_df = pd.DataFrame([
            {
                'Intent': intent,
                'Samples': result['n_samples'],
                'Clusters': result['n_clusters'],
                'Noise': result['n_noise'],
                'NoiseRatio': result['noise_ratio'],
                'SilhouetteScore': result['silhouette_score']
            }
            for intent, result in cohesion_results.items()
        ])
        
        cohesion_df.to_csv(f"{output_dir}/intent_cohesion_metrics.csv", index=False)
        
        logger.info(f"分析报告已保存到 {output_dir} 目录")
    
    def run_full_analysis(self, json_file_path: str, output_dir: str = "output") -> None:
        """
        运行完整的数据质量分析流程
        
        Args:
            json_file_path: 输入JSON文件路径
            output_dir: 输出目录
        """
        logger.info("开始完整的CLU数据质量分析")
        
        # 1. 加载数据
        self.load_data(json_file_path)
        
        # 2. 一次性生成所有嵌入向量
        self._generate_all_embeddings()
        
        # 3. Intent间低耦合性分析
        similarity_matrix, similarity_results, heatmap_intent_names = self.analyze_inter_intent_coupling()
        
        # 4. Intent内高内聚性分析
        cohesion_results = self.analyze_intra_intent_cohesion()
        
        # 5. 创建可视化
        self.create_visualizations(
            similarity_matrix, 
            heatmap_intent_names, 
            cohesion_results, 
            output_dir
        )
        
        # 6. 生成报告
        self.generate_report(similarity_results, cohesion_results, output_dir)
        
        logger.info("CLU数据质量分析完成")


def main():
    """主函数"""
    # Azure OpenAI配置
    API_KEY = "7218515241f04d98b3b5d9869a25b91f"  # 请替换为实际API密钥
    AZURE_ENDPOINT = "https://nnitasia-openai-01-ins.openai.azure.com/"
    DEPLOYMENT_NAME = "NNIT-Ada-3-large"
    
    # 数据文件路径
    JSON_FILE_PATH = "data/IT_01_1.json"
    OUTPUT_DIR = "output"
    
    try:
        # 创建分析器实例
        analyzer = CLUDataAnalyzer(
            api_key=API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            deployment_name=DEPLOYMENT_NAME
        )
        
        # 运行完整分析
        analyzer.run_full_analysis(JSON_FILE_PATH, OUTPUT_DIR)
        
        print("✅ CLU数据质量分析完成！")
        print(f"📊 结果已保存到 {OUTPUT_DIR} 目录")
        print("📈 请查看生成的可视化图表和分析报告")
        
    except Exception as e:
        logger.error(f"分析过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
