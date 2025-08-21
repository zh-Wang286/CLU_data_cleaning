# Azure CLU 数据清洗与分析系统

本项目是一个后端工具，旨在分析和提升 Azure 对语料言理解 (CLU) 训练集的数据质量。它通过一系列自动化分析，帮助确保意图定义之间具有高区分度（低耦合），同时保证意图内部的语料在语义上保持一致（高内聚）。

## 核心功能

### **数据质量分析**
- **数据加载与校验:** 加载 CLU 项目的 JSON 文件，使用 Pydantic 模型进行严格的结构与类型校验
- **意图内异常点检测 (`find_outliers_within_intents`):** 基于 k-NN 算法识别语义上远离意图核心的语料
- **全局聚类审计 (`audit_clusters_globally`):** 使用 HDBSCAN 发现意图间潜在的语义重叠
- **全局边界混淆分析 (`analyze_boundary_violations`):** 使用 PCA + 马氏距离量化意图边界模糊程度
- **局部边界混淆分析 (`analyze_targeted_boundary_violations`):** 针对指定意图子集的高精度边界分析

### **数据可视化**
- **意图相似度热力图:** 展示不同意图间的语义相似程度
- **全局语料散点图:** UMAP 降维展示所有语料分布，特殊标记边界混淆点
- **局部语料散点图:** 针对指定意图的聚焦可视化
- **意图内语料散点图:** 单意图散点图，高亮异常点及文本内容

### **智能数据增广**
- **语料候选生成 (`generate_utterance_candidates`):** 调用 LLM 为低样本意图生成候选语料
- **自动化报告:** 生成详尽的 Markdown 格式审计报告

### **重构亮点**
- **模块化架构:** 核心算法抽取到独立 `src/core/` 模块
- **统一命名约定:** 采用 `find_*`, `analyze_*`, `audit_*`, `generate_*` 规范
- **零代码重复:** 消除原有70%的重复代码，CLUProcessor 减少44%代码量
- **性能优化:** 缓存机制避免重复计算，支持增量分析

## 项目架构

### **分层架构设计**

```
Interface Layer (CLI Commands)
└── main.py
    ├── run-all         - 完整分析流水线
    ├── compare-intents - 局部意图对比
    └── enrich          - 语料增广生成

Service Layer (Business Orchestration)
├── CLUProcessor - 分析流程协调器
│   ├── DimensionalityReducer
│   ├── StatisticalAnalyzer
│   ├── OutlierDetector
│   └── ClusterAuditor
├── Visualizer      - 可视化渲染器
└── ReportGenerator - 报告构建器

Core Layer (Pure Algorithm Logic)
└── src/core/
    ├── outlier_detector.py      - k-NN异常点检测
    ├── statistical_analyzer.py  - 马氏距离/边界分析
    ├── dimensionality_reducer.py - PCA/UMAP降维
    └── cluster_auditor.py       - HDBSCAN聚类审计

Data Layer (Data & External)
├── CLUDataset      - 数据模型与校验
├── Azure OpenAI    - 嵌入向量 & LLM服务
└── Cache System    - 嵌入向量缓存
```

### **核心数据流**

```mermaid
graph LR
    A[CLU JSON] --> B[CLUProcessor]
    B --> C[Core Modules]
    C --> D[Analysis Results]
    D --> E[Reports]
    D --> F[Visualizations]
    F --> G[Statistical Insights]
    G --> H[Cache]
    H --> G
    G --> F
```

## 环境部署与安装

### 系统要求

- Python 3.12+
- 已在 Azure OpenAI 部署LLM、Embedding模型。

### 安装步骤

1.  **克隆代码仓库:**
    ```bash
    git clone https://github.com/zh-Wang286/CLU_data_cleaning.git
    cd CLU_data_cleaning
    ```

2.  **创建并激活虚拟环境:**
    推荐使用 `uv` 来管理虚拟环境和依赖。
    ```bash
    # 如果您尚未安装 uv
    pip install uv

    # 创建虚拟环境
    uv venv
    
    # 激活环境 (Linux)
    source .venv/bin/activate

    # 激活环境 (Windows)
    # .venv\Scripts\activate
    ```

3.  **安装依赖:**

    本项目同时提供了 `pyproject.toml` (用于现代工具链) 和 `requirements.txt` (用于传统环境) 两种依赖定义文件。推荐使用 `uv` 进行安装。

    **选项 A: 使用 uv (推荐)**

    此命令会读取 `pyproject.toml` 文件，创建一个快速、精确且可重复的虚拟环境。
    ```bash
    uv sync
    ```

    **选项 B: 使用 pip**

    如果更习惯使用 `pip`，也可以通过 `requirements.txt` 文件进行安装：
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置环境变量:**
    本工具提供了一个 `.env.example` 文件作为配置模板。请首先复制该文件来创建您自己的 `.env` 文件：
    ```bash
    cp .env.example .env
    ```
    然后，打开新建的 `.env` 文件，并根据文件内的注释说明，填入您选择的 API 服务商（Azure 或 OpenAI）所需的凭证和模型名称。

## 使用方法

本工具的核心入口是 `main.py`，它提供了一个命令行接口 (CLI) 来执行各项任务。

关于所有命令及其可用参数的详细说明，请参阅：[**命令行接口 (CLI) 使用说明**](./docs/cli_usage.md)。

### 运行完整分析流程

`run-all` 命令会对数据集进行全面的质量评估，包括意图内异常检测、意图间边界分析等，并生成可视化结果和详细报告。

要执行完整分析，请运行：

```bash
python main.py run-all --input-file data/IT_01_1.json
```

分析完成后，在 `outputs/` 目录下会生成以下产物：

1. **审计报告** (`outputs/reports/audit_report_*.md`)
   - 这是最重要的分析产物，包含了所有发现的潜在问题
   - 建议优先查看"意图边界混淆分析"部分，它列出了最需要关注的标注问题
   - 对于每个异常样本，报告都会提供具体的量化指标（如 p-value），便于判断问题的严重程度

2. **可视化图表** (`outputs/figures/<时间戳>/`)
   - `intent_similarity_heatmap.png`: 意图间的语义相似度热力图，用于快速发现可能存在边界模糊的意图对
   - `global_scatterplot.png`: 所有语料的分布图，其中黄边三角形标记的点是建议重点审查的样本
   - `intent_*.png`: 每个意图的内部分布图，帮助理解为什么某些样本被判定为异常

默认情况下，所有输出会按时间戳分类存储，以防止新的分析结果覆盖旧的结果。建议按以下顺序查看分析结果：
1. 首先查看报告开头的 **运行参数摘要**，确认本次分析所用的各项配置是否符合预期。
2. 查看 **数据集概览**，了解数据集的整体规模和基本分布情况。
3. 接着查看 **意图边界混淆分析** 和 **全局聚类审计** 部分，识别出最需要关注的、可能存在定义重叠或边界模糊的意图。
4. 借助 **意图相似度热力图** (`intent_similarity_heatmap.png`) 从宏观上验证上一步发现的意图对。
5. 针对这些重点意图，深入查看 **意图内部异常点检测** 部分和对应的 **意图内语料散点图** (`intent_*.png`)，定位到具体的异常语料。
6. 根据报告中的量化指标（如 p-value、纯度等）和优先级建议，逐步处理发现的问题。

### 为低样本意图增广数据（可选）

要为语料样本数低于CLU推荐阈值（默认为25）的意图生成新的候选语料，请使用 `enrich` 命令：

```bash
python main.py enrich --input-file data/IT_01_1.json
```

此命令默认会在 `outputs/reports/` 目录下生成一份带时间戳的 Markdown 报告，其中包含所有生成的候选语料，供您审查。

如果您希望直接在控制台查看结果而不生成报告文件，请使用 `--dry-run` 标志：
```bash
python main.py enrich --input-file data/IT_01_1.json --dry-run
```

## 核心模块详解

关于核心处理模块 `CLUProcessor` 的工作原理、算法选择、以及输出产物的详细解读，请参阅技术说明文档：[**`CLUProcessor` 技术说明**](./docs/CLUProcessor_explanation.md)。
