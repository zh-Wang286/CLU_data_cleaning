# -*- coding: utf-8 -*-
"""Generates a structured Markdown report from analysis results."""

from datetime import datetime
from pathlib import Path
import textwrap
from typing import Dict, List

from loguru import logger

from src.config import settings
from src.dataset import CLUDataset
from src.schemas import BoundaryViolationRecord


class ReportGenerator:
    """
    Aggregates analysis results into a comprehensive Markdown report.
    """

    def __init__(self, dataset: CLUDataset, output_dir: Path = Path("outputs/reports")):
        """
        Initializes the report generator.

        Args:
            dataset: The CLUDataset that was analyzed.
            output_dir: The directory to save the report.
        """
        self.dataset = dataset
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_parts: List[str] = []
        logger.info(
            f"ReportGenerator initialized. Reports will be saved to '{self.output_dir.resolve()}'"
        )

    def add_run_parameters(self, params: Dict[str, any]):
        """Adds a summary of the run parameters to the report."""
        run_params_header = """
        ## 0. 运行参数摘要

        本节记录了用于生成此报告的分析任务所使用的全部参数。
        """
        self.report_parts.append(textwrap.dedent(run_params_header))

        table = "| 参数 | 描述 | 值 |\n|---|---|---|\n"
        # A dictionary to hold human-readable descriptions for each parameter
        descriptions = {
            "input_file": "用于分析的 CLU 数据源文件路径",
            "output_dir": "所有分析产物（报告、图表）的根目录",
            "min_samples": "意图被纳入边界分析和聚类分析所需的最小语料数",
            "sort_by": "边界混淆报告的排序依据",
            "outlier_threshold": "意图内异常点检测的判断阈值策略",
        }

        for key, value in params.items():
            description = descriptions.get(key, "N/A")
            table += f"| `{key}` | {description} | `{value}` |\n"

        self.report_parts.append(table)

    def add_header(self):
        """Adds the main title and summary section to the report."""
        project_name = self.dataset.project.metadata.projectName
        run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header = f"""
        # CLU 数据质量审计报告
        
        - **项目:** `{project_name}`
        - **运行ID:** `{settings.run_id}`
        - **时间戳:** `{run_time}`
        
        ---
        """
        self.report_parts.append(textwrap.dedent(header))

    def add_dataset_summary(self, low_utterance_intents: Dict[str, int], threshold: int):
        """Adds the dataset summary statistics to the report."""
        summary = f"""
        ## 1. 数据集概览
        
        - **总意图数:** {self.dataset.count_intents()}
        - **总语料数:** {self.dataset.count_utterances()}
        - **语料数少于 {threshold} 的意图数量:** {len(low_utterance_intents)}
        
        """
        self.report_parts.append(textwrap.dedent(summary))
        if low_utterance_intents:
            table = "| 意图 | 语料数量 |\n|---|---|\n"
            for intent, count in sorted(low_utterance_intents.items(), key=lambda item: item[1]):
                table += f"| `{intent}` | {count} |\n"
            self.report_parts.append(table)

    def add_outlier_report(self, outliers: Dict[str, List[Dict]]):
        """Adds the intra-intent outlier detection results to the report."""
        report = f"""
        ## 2. 意图内部异常点检测
        
        本节列出了在语义上与其所属意图的中心相距甚远的语料。这些可能是标注错误或意图定义不清的信号。
        
        """
        self.report_parts.append(textwrap.dedent(report))

        if not outliers:
            self.report_parts.append("未检测到明显的异常点。\n")
            return

        for intent, records in outliers.items():
            intent_header = f"### 意图: `{intent}`\n\n"
            table = "| 排名 | 异常分数 | 阈值 | 语料文本 |\n|---|---|---|---|\n"
            for record in records:
                table += f"| {record['rank']} | {record['score']:.4f} | {record['threshold']:.4f} | `{record['text']}` |\n"
            self.report_parts.append(intent_header + table + "\n")

    def add_cluster_audit_report(self, cluster_audit: Dict):
        """Adds the global clustering audit results to the report."""
        summary = cluster_audit.get('summary', {})
        report = f"""
        ## 3. 全局聚类审计 (HDBSCAN)
        
        此审计无视原始意图标签，将所有语料按语义相似度进行分组，以识别潜在的意图重叠或定义不一致的问题。
        
        - **发现的簇数:** {summary.get('num_clusters', 'N/A')}
        - **噪声比例:** {summary.get('noise_ratio', 0):.2%} (未被分配到任何簇的语料)
        
        ### 簇纯度分析
        
        下表展示了纯度较低的簇，这些簇中混合了多个意图。这可能表明意图定义过于相似，或部分语料被错误标注。
        
        | 簇ID | 大小 | 主要意图 | 纯度 | 意图分布 |
        |---|---|---|---|---|
        """
        self.report_parts.append(textwrap.dedent(report))

        clusters = cluster_audit.get('clusters', [])
        for cluster in clusters:
            # Format the intent distribution for better readability in the table
            top_n = 3
            intent_dist = cluster['intent_distribution']
            sorted_intents = sorted(intent_dist.items(), key=lambda item: item[1], reverse=True)
            
            dist_parts = [f"`{intent}` ({count})" for intent, count in sorted_intents[:top_n]]
            if len(sorted_intents) > top_n:
                dist_parts.append("...")
            
            dist_str = ", ".join(dist_parts)

            self.report_parts.append(
                f"| {cluster['cluster_id']} | {cluster['size']} | `{cluster['majority_intent']}` | {cluster['purity']:.2%} | {dist_str} |\n"
            )
    
    def add_boundary_violation_report(self, violations: List[BoundaryViolationRecord], sort_by: str = "p_value"):
        """Adds the intent boundary violation analysis to the report."""
        report = f"""
        ## 4. 意图边界混淆分析

        本节使用马氏距离来识别那些在统计上可能属于另一个意图分布的语料。
        高 p-value ( > 0.05) 意味着一个语料可以被合理地视为另一个意图的成员，这表明意图边界存在模糊。

        """
        self.report_parts.append(textwrap.dedent(report))

        if not violations:
            self.report_parts.append("未检测到明显的意图边界混淆。\n")
            return

        # Sort based on the user's choice
        if sort_by == 'intent':
            sorted_violations = sorted(violations, key=lambda v: v.original_intent)
            report_header = report + " (按原始意图排序)\n"
        else: # Default to p_value
            sorted_violations = sorted(violations, key=lambda v: v.confused_with.p_value, reverse=True)
            report_header = report + " (按 p-value 降序排序)\n"

        self.report_parts.append(textwrap.dedent(report_header))

        table = "| 原始意图 | 语料文本 | 最可能混淆的意图 | P-value | 马氏距离 |\n|---|---|---|---|---|\n"
        for record in sorted_violations:
            table += (
                f"| `{record.original_intent}` "
                f"| `{record.text}` "
                f"| `{record.confused_with.intent}` "
                f"| {record.confused_with.p_value:.4f} "
                f"| {record.confused_with.mahalanobis_distance:.2f} |\n"
            )
        self.report_parts.append(table)

    def add_enrichment_report(self, generated_candidates: Dict[str, List[str]]):
        """Adds the data enrichment suggestions to the report."""
        report = f"""
        ## 5. 低样本意图增广建议
        
        以下是为样本量不足的意图生成的候选语料，建议由人工审核后加入数据集。
        
        """
        self.report_parts.append(textwrap.dedent(report))

        if not generated_candidates:
            self.report_parts.append("未生成任何候选语料。\n")
            return

        for intent, utterances in generated_candidates.items():
            intent_header = f"### 意图: `{intent}`\n\n"
            utterance_list = ""
            for utt in utterances:
                utterance_list += f"- `{utt}`\n"
            self.report_parts.append(intent_header + utterance_list + "\n")

    def generate(self, filename: str = "clu_audit_report_zh.md") -> Path:
        """
        Combines all parts and saves the final report to a file.

        Args:
            filename: The name of the output Markdown file.

        Returns:
            The path to the generated report.
        """
        final_report = "\n".join(self.report_parts)
        save_path = self.output_dir / filename

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(final_report)
            logger.success(f"Successfully generated report: {save_path}")
        except IOError as e:
            logger.error(f"Failed to write report to {save_path}: {e}")
            raise

        return save_path

    def generate_targeted_report(self, violations: List[BoundaryViolationRecord], target_intents: List[str], sort_by: str = "p_value") -> Path:
        """
        Generates and saves a specific report for targeted boundary violation analysis.

        Args:
            violations: A list of detected boundary violation records.
            target_intents: The list of intents that were analyzed.
            sort_by: The criteria to sort the violation records.

        Returns:
            The path to the generated report file.
        """
        run_time = datetime.now()
        project_name = self.dataset.project.metadata.projectName

        # --- Report Header ---
        header = f"""
        # CLU 局部边界混淆分析报告
        
        - **项目:** `{project_name}`
        - **运行ID:** `{settings.run_id}`
        - **时间戳:** `{run_time.strftime("%Y-%m-%d %H:%M:%S")}`
        
        ---
        """
        self.report_parts.append(textwrap.dedent(header))

        # --- Analysis Parameters ---
        params_header = """
        ## 1. 分析参数
        
        本节记录了本次局部对比分析所使用的参数。
        """
        self.report_parts.append(textwrap.dedent(params_header))
        
        table = "| 参数 | 值 |\n|---|---|\n"
        table += f"| `target_intents` | `[{', '.join(target_intents)}]` |\n"
        table += f"| `sort_by` | `{sort_by}` |\n"
        self.report_parts.append(table)

        # --- Re-use the boundary violation section ---
        # Temporarily clear other report parts to only include the violation section
        original_parts = self.report_parts
        self.report_parts = []
        self.add_boundary_violation_report(violations, sort_by)
        violation_report_part = "\n".join(self.report_parts)

        # Restore original parts and add the new one
        self.report_parts = original_parts
        self.report_parts.append(violation_report_part)
        
        # --- Finalize and Save ---
        # Create a clean filename from the target intents
        safe_filename_intents = "_vs_".join(
            "".join(c for c in intent if c.isalnum()).rstrip() for intent in target_intents
        )
        report_filename = f"targeted_analysis_report_{safe_filename_intents}.md"

        final_report_content = "\n".join(self.report_parts)
        save_path = self.output_dir / report_filename

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(final_report_content)
            logger.success(f"Successfully generated targeted analysis report: {save_path}")
        except IOError as e:
            logger.error(f"Failed to write targeted report to {save_path}: {e}")
            raise
            
        return save_path
