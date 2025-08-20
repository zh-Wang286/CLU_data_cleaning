# 命令行接口 (CLI) 使用说明

本文档详细介绍了 `main.py` 提供的命令行接口，用于运行 CLU 数据清洗与分析系统的各项功能。

## 1. 概览

本工具通过 `click` 库构建，提供了一系列子命令来执行不同的任务。所有命令均以 `python main.py` 作为前缀。

---

## 2. `run-all` 命令

此命令是工具的核心，用于执行完整的端到端分析流程。它会依次进行数据加载、异常点检测、全局聚类审计、意图边界混淆分析，并最终生成所有相关的可视化图表和一份综合审计报告。

### 2.1. 基本用法

```bash
python main.py run-all --input-file <你的数据文件路径>
```

### 2.2. 参数详解

| 参数 | 类型 | 默认值 | 必需 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| `--input-file` | 文件路径 | N/A | 是 | 指向你的 CLU 项目 JSON 文件的路径 (例如: `data/IT_01_1.json`)。 |
| `--output-dir` | 目录路径 | `outputs` | 否 | 用于存放所有输出结果（报告、图表、缓存）的根目录。 |
| `--min-samples` | 整数 | `15` | 否 | 设定一个意图必须包含的最小语料数。样本量低于此值的意图，将不会被纳入“意图边界混淆分析”和“全局聚类审计”这两项计算中。 |
| `--outlier-threshold` | `90pct` `95pct` 或 `iqr` | `95pct` | 否 | 意图内异常点检测的阈值策略。<br>- `90pct`: 使用90百分位。<br>- `95pct`: 使用95百分位。<br>- `iqr`: 使用标准IQR（四分位距）方法。 |
| `--sort-by` | `p_value` 或 `intent` | `p_value` | 否 | 指定“意图边界混淆分析”报告表格的排序方式。<br>- `p_value`: 按 p-value 从高到低排序，便于快速定位最严重的混淆问题。<br>- `intent`: 按原始意图的字母顺序排序，便于对特定意图进行系统性审查。 |

### 2.3. 使用示例

-   **以最小样本数为 10，并让报告按意图名称排序:**
    ```bash
    python main.py run-all --input-file data/IT_01_1.json --min-samples 10 --sort-by intent
    ```
-   **使用90百分位作为异常点检测阈值:**
    ```bash
    python main.py run-all --input-file data/IT_01_1.json --outlier-threshold 90pct
    ```

---

## 3. `enrich` 命令

此命令用于为样本量不足的意图调用大语言模型 (LLM) 来生成新的候选语料，以扩充数据集。

### 3.1. 基本用法

```bash
python main.py enrich --input-file <你的数据文件路径>
```

### 3.2. 参数详解

| 参数 | 类型 | 默认值 | 必需 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| `--input-file` | 文件路径 | N/A | 是 | 指向你的 CLU 项目 JSON 文件的路径。 |
| `--output-dir` | 目录路径 | `outputs` | 否 | 用于存放生成的增广报告的根目录。 |
| `--threshold` | 整数 | `25` | 否 | 意图语料数的阈值。样本量低于此值的意图，将会被纳入增广流程。 |
| `--dry-run` | 布尔标志 | `False` | 否 | 如果设置此标志，生成的候选语料将直接打印在控制台，而不会生成 Markdown 报告文件。 |

### 3.3. 使用示例

-   **直接在控制台预览为样本数少于 30 的意图生成的候选语料:**
    ```bash
    python main.py enrich --input-file data/IT_01_1.json --threshold 30 --dry-run
    ```
