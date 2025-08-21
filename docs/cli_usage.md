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

| 参数 | 短选项 | 类型 | 默认值 | 必需 | 描述 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--input-file` | `-i` | 文件路径 | N/A | 是 | CLU项目JSON文件路径 (例如: `data/IT_05.json`) |
| `--output-dir` | `-o` | 目录路径 | `outputs` | 否 | 输出目录，用于存放报告、图表、缓存等所有结果 |
| `--min-samples` | 无 | 整数 | `15` | 否 | 意图纳入边界违规和聚类分析的最小样本数 |
| `--outlier-threshold` | 无 | 选择项 | `95pct` | 否 | 异常点检测阈值策略<br>• `90pct`: 90百分位阈值<br>• `95pct`: 95百分位阈值<br>• `iqr`: IQR（四分位距）方法 |
| `--sort-by` | 无 | 选择项 | `p_value` | 否 | 边界违规报告排序方式<br>• `p_value`: 按p值降序排序（推荐）<br>• `intent`: 按意图名称排序 |

**参数使用提示:**
- 使用 `-i` 简化输入文件指定
- 使用 `-o` 快速指定输出目录  
- `--min-samples` 较小值会包含更多意图进行分析，但可能引入噪声
- `p_value` 排序优先显示最严重的边界混淆问题

### 2.3. 使用示例

**基础用法:**
```bash
# 使用短选项的简化命令
python main.py run-all -i data/IT_05.json

# 完整参数示例
python main.py run-all \
  --input-file data/IT_05.json \
  --output-dir outputs \
  --min-samples 15 \
  --sort-by p_value \
  --outlier-threshold 95pct
```

**高级用法:**
```bash
# 降低最小样本数，按意图名排序
python main.py run-all -i data/IT_05.json --min-samples 10 --sort-by intent

# 使用更严格的异常点检测
python main.py run-all -i data/IT_05.json --outlier-threshold 90pct

# 指定自定义输出目录
python main.py run-all -i data/IT_05.json -o /path/to/custom/outputs
```

---

## 3. `enrich` 命令

此命令用于为样本量不足的意图调用大语言模型 (LLM) 来生成新的候选语料，以扩充数据集。

### 3.1. 基本用法

```bash
python main.py enrich --input-file <你的数据文件路径>
```

### 3.2. 参数详解

| 参数 | 短选项 | 类型 | 默认值 | 必需 | 描述 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--input-file` | `-i` | 文件路径 | N/A | 是 | CLU项目JSON文件路径 |
| `--output-dir` | `-o` | 目录路径 | `outputs` | 否 | 输出目录，用于存放增广报告 |
| `--threshold` | 无 | 整数 | `25` | 否 | 意图需要增广的语料数量阈值 |
| `--dry-run` | 无 | 标志 | `False` | 否 | 仅在控制台打印生成的候选语料，不保存报告文件 |


### 3.3. 使用示例

**基础用法:**
```bash
# 简化命令
python main.py enrich -i data/IT_05.json

# 试运行模式（仅显示，不保存）
python main.py enrich -i data/IT_05.json --dry-run
```

**高级用法:**
```bash
# 自定义阈值和输出目录
python main.py enrich -i data/IT_05.json --threshold 30 -o custom_outputs

# 预览生成结果（用于测试）
python main.py enrich -i data/IT_05.json --threshold 20 --dry-run
```

## 4. 局部意图对比分析 (`compare-intents`)

此命令针对您指定的2个或多个意图，运行一次局部的、高精度的边界混淆分析。

它专门用于深入研究那些在全局分析中发现的、可能存在定义重叠的意图对。与全局分析（`run-all`）不同，此命令的输出（报告和散点图）将只包含您指定的意图，从而提供一个更清晰、更聚焦的视图。

**核心优势**:
-   **聚焦分析**: 排除无关意图的干扰，让您可以专注于辨析特定意图间的边界问题。
-   **全局一致性**: 分析是在与 `run-all` 命令相同的全局向量空间（经过全局PCA降维）中进行的，确保了结果的可比性和一致性。

### 4.1. 参数详解

| 参数 | 短选项 | 类型 | 默认值 | 必需 | 描述 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--input-file` | `-i` | 文件路径 | N/A | 是 | CLU项目JSON文件路径 |
| `--target-intents` | `-t` | 多选项 | N/A | 是 | 指定要分析的意图，使用多次此选项（至少需要2个意图） |
| `--output-dir` | `-o` | 目录路径 | `outputs` | 否 | 输出目录，自动创建带时间戳的子目录 |
| `--min-samples` | 无 | 整数 | `15` | 否 | 意图纳入边界违规和聚类分析的最小样本数 |
| `--sort-by` | 无 | 选择项 | `p_value` | 否 | 边界违规报告排序方式 |


### 4.2. 使用示例

**基础用法:**
```bash
# 分析两个意图的边界混淆
python main.py compare-intents \
  -i data/IT_05.json \
  -t "密码重置" \
  -t "账户解锁"
```

**高级用法:**
```bash
# 分析多个意图，指定输出目录
python main.py compare-intents \
  -i data/IT_05.json \
  -t "密码重置" \
  -t "账户解锁" \
  -t "权限申请" \
  -o custom_outputs

# 降低最小样本数要求，按意图名排序
python main.py compare-intents \
  -i data/IT_05.json \
  -t "意图A" \
  -t "意图B" \
  --min-samples 10 \
  --sort-by intent
```

**输出结构:**
分析完成后，会在指定输出目录下生成：
- `reports/targeted_analysis_report_*.md` - 局部分析报告
- `figures/targeted_scatterplot_*.png` - 目标意图散点图
