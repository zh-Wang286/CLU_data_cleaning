# -*- coding: utf-8 -*-
"""重构后的CLU数据清洗CLI入口，消除参数重复定义。"""

from pathlib import Path
from datetime import datetime

import click
from loguru import logger

from src.dataset import CLUDataset
from src.processor import CLUProcessor
from src.reporting import ReportGenerator
from src.utils.logging import setup_logging
from src.visualization import Visualizer


# 通用参数装饰器，消除CLI参数重复定义
def common_input_file_option(f):
    """通用输入文件参数装饰器。"""
    return click.option(
        "--input-file", "-i",
        type=click.Path(exists=True, dir_okay=False, resolve_path=True),
        required=True,
        help="CLU JSON文件路径"
    )(f)


def common_output_dir_option(f):
    """通用输出目录参数装饰器。"""
    return click.option(
        "--output-dir", "-o",
        type=click.Path(file_okay=False, resolve_path=True),
        default="outputs",
        help="输出目录"
    )(f)


def common_min_samples_option(f):
    """通用最小样本数参数装饰器。"""
    return click.option(
        "--min-samples",
        type=int,
        default=15,
        help="意图纳入边界违规和聚类分析的最小样本数"
    )(f)


def common_sort_by_option(f):
    """通用排序参数装饰器。"""
    return click.option(
        "--sort-by",
        type=click.Choice(['p_value', 'intent'], case_sensitive=False),
        default='p_value',
        help="边界违规报告排序方式 ('p_value' 或 'intent')"
    )(f)


@click.group()
def cli():
    """A CLI tool for analyzing and cleaning Azure CLU datasets."""
    setup_logging()


@cli.command()
@common_input_file_option
@common_output_dir_option
@common_min_samples_option
@common_sort_by_option
@click.option(
    "--outlier-threshold",
    type=click.Choice(['90pct', '95pct', 'iqr'], case_sensitive=False),
    default='95pct',
    help="异常点检测阈值策略 ('90pct', '95pct', 或 'iqr')",
)
def run_all(input_file: str, output_dir: str, min_samples: int, sort_by: str, outlier_threshold: str):
    """
    Run the entire analysis pipeline: validation, outlier detection,
    clustering, visualization, and reporting.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    input_path = Path(input_file)
    output_path = Path(output_dir)
    
    # Create a dictionary of the run parameters for reporting
    run_params = {
        "input_file": input_file,
        "output_dir": output_dir,
        "min_samples": min_samples,
        "sort_by": sort_by,
        "outlier_threshold": outlier_threshold,
    }
    
    # Create timestamped directories for figures
    figure_path = output_path / "figures" / run_timestamp
    figure_path.mkdir(parents=True, exist_ok=True)
    
    report_path_dir = output_path / "reports"
    report_path_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting full analysis pipeline...")

    # 1. Data Layer
    try:
        dataset = CLUDataset.from_json(input_path)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    low_utterance_intents = dataset.warn_low_utterance_intents()

    # 2. Processing Layer
    processor = CLUProcessor(dataset, output_dir=output_path)
    embeddings_map = processor.get_all_embeddings()

    outliers = processor.find_outliers_within_intents(embeddings_map, threshold_policy=outlier_threshold)
    cluster_audit = processor.audit_clusters_globally(embeddings_map, min_samples_for_analysis=min_samples)
    boundary_violations = processor.analyze_boundary_violations(embeddings_map, min_samples_for_analysis=min_samples)

    # 3. Visualization Layer
    visualizer = Visualizer(dataset, output_dir=figure_path)
    visualizer.plot_intent_similarity_heatmap(embeddings_map)
    visualizer.plot_global_scatterplot(embeddings_map, boundary_violations=boundary_violations)
    visualizer.plot_per_intent_scatterplots(embeddings_map, outliers)
    
    # 4. Reporting Layer
    report_filename = f"clu_audit_report_{run_timestamp}.md"
    reporter = ReportGenerator(dataset, output_dir=report_path_dir)
    reporter.add_header()
    reporter.add_run_parameters(run_params)
    reporter.add_dataset_summary(low_utterance_intents, threshold=25)
    reporter.add_outlier_report(outliers)
    reporter.add_cluster_audit_report(cluster_audit)
    reporter.add_boundary_violation_report(boundary_violations, sort_by=sort_by)
    report_path = reporter.generate(filename=report_filename)

    logger.success(f"Full analysis pipeline completed. Report saved to: {report_path}")


@cli.command()
@common_input_file_option
@common_output_dir_option
@click.option(
    "--threshold",
    type=int,
    default=25,
    help="意图需要增广的语料数量阈值",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="仅在控制台打印生成的候选语料，不保存报告文件",
)
def enrich(input_file: str, output_dir: str, threshold: int, dry_run: bool):
    """
    Generate new utterance candidates for low-sample intents.
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    
    logger.info(f"Starting data enrichment with threshold={threshold}...")
    
    try:
        dataset = CLUDataset.from_json(input_path)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Failed to load dataset: {e}")
        return
        
    processor = CLUProcessor(dataset, output_dir=output_path)
    generated_candidates = processor.generate_utterance_candidates(threshold=threshold)
    
    if not generated_candidates:
        logger.info("未生成任何候选语料。")
        return

    if dry_run:
        click.echo("\n--- 生成的语料候选集 (试运行) ---")
        for intent, utterances in generated_candidates.items():
            click.echo(f"\n意图: {intent}")
            for utt in utterances:
                click.echo(f"  - {utt}")
        click.echo("\n--- 试运行结束 ---")
    else:
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path_dir = output_path / "reports"
        report_path_dir.mkdir(parents=True, exist_ok=True)
        
        reporter = ReportGenerator(dataset, output_dir=report_path_dir)
        reporter.add_header()
        reporter.add_enrichment_report(generated_candidates)
        
        report_filename = f"enrichment_report_{run_timestamp}.md"
        report_path = reporter.generate(filename=report_filename)
        logger.success(f"Enrichment report saved successfully to: {report_path}")


@cli.command()
@common_input_file_option
@click.option(
    '--target-intents', '-t',
    multiple=True,
    required=True,
    help="指定要分析的意图，使用多次此选项（至少需要2个意图）",
)
@common_output_dir_option
@common_min_samples_option
@common_sort_by_option
def compare_intents(
    input_file: str,
    target_intents: tuple[str, ...],
    output_dir: str,
    min_samples: int,
    sort_by: str,
):
    """
    Run a targeted boundary violation analysis on a specific subset of intents.
    """
    if len(target_intents) < 2:
        logger.error("At least two intents must be specified using the '-t' option.")
        return

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    input_path = Path(input_file)
    
    root_output_path = Path(output_dir)
    
    # Create a descriptive, specific subdirectory for this analysis run's outputs
    safe_intents_str = "_vs_".join("".join(c for c in intent if c.isalnum()) for intent in target_intents)
    targeted_output_path = root_output_path / f"targeted_analysis_{safe_intents_str}_{run_timestamp}"

    figure_path = targeted_output_path / "figures"
    report_path_dir = targeted_output_path / "reports"
    figure_path.mkdir(parents=True, exist_ok=True)
    report_path_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting targeted analysis for intents: {list(target_intents)}")
    logger.info(f"Outputs will be saved to: {targeted_output_path.resolve()}")

    # 1. Initialization
    dataset = CLUDataset.from_json(input_path)
    # Initialize processor with the ROOT output dir to use the shared cache
    processor = CLUProcessor(dataset, output_dir=root_output_path)
    # Initialize visualizer and reporter with the specific run directory
    visualizer = Visualizer(dataset, output_dir=figure_path)
    reporter = ReportGenerator(dataset, output_dir=report_path_dir)
    
    # 2. Processing
    logger.info("Step 1/3: Computing embeddings...")
    embeddings_map = processor.get_all_embeddings()
    
    logger.info("Step 2/3: Running targeted boundary violation analysis...")
    violations = processor.analyze_targeted_boundary_violations(
        embeddings_map=embeddings_map,
        target_intents=list(target_intents),
        min_samples_for_analysis=min_samples,
    )
    
    # 3. Reporting and Visualization
    logger.info("Step 3/3: Generating outputs...")
    report_path = reporter.generate_targeted_report(
        violations=violations, target_intents=list(target_intents), sort_by=sort_by
    )
    
    visualizer.plot_targeted_scatterplot(
        embeddings_map=embeddings_map,
        target_intents=list(target_intents),
        boundary_violations=violations,
    )
    
    logger.success(f"Targeted analysis complete. Report saved to: {report_path}")


if __name__ == "__main__":
    cli()
