# Azure CLU 数据清洗与分析系统

本项目是一个后端工具，旨在分析和提升 Azure 对话语言理解 (CLU) 训练集的数据质量。它通过一系列自动化分析，帮助确保意图定义之间具有高区分度（低耦合），同时保证意图内部的话语在语义上保持一致（高内聚）。

## 功能特性

- **数据加载与校验:** 加载 CLU 项目的 JSON 文件，并使用 Pydantic 模型进行严格的结构与类型校验。
- **意图内异常点检测:** 基于文本向量的余弦距离，使用 k-NN 算法识别在语义上远离其所属意图核心的话语。
- **全局聚类审计:** 使用 HDBSCAN 对所有话语进行全局聚类，以发现不同意图之间潜在的语义重叠或边界模糊问题。
- **数据可视化:** 生成多种图表以直观展示数据集的内部结构，包括：
  - **意图相似度热力图:** 展示不同意图之间的语义相似程度。
  - **全局话语散点图:** 通过 UMAP 降维，展示所有话语在二维空间中的分布。
  - **意图内话语散点图:** 针对每个意图单独生成散点图，并高亮标注异常点及其文本内容。
- **数据增广:** 对样本量不足的意图，调用大语言模型 (LLM) 生成新的候选话语，供人工审核。
- **自动化报告:** 将所有分析结果汇总成一份详尽的 Markdown 格式审计报告。

## 环境部署与安装

### 系统要求

- Python 3.12+
- 已在 Azure OpenAI 部署LLM、Embedding模型。

### 安装步骤

1.  **克隆代码仓库:**
    ```bash
    git clone <repository-url>
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

### 运行完整分析流程

要对指定数据集执行所有分析步骤（包括校验、异常检测、聚类、可视化和报告生成），请使用 `run-all` 命令：

```bash
python main.py run-all --input-file data/IT_01_1.json
```

默认情况下，所有输出（报告、图表、缓存）将保存在 `outputs/` 目录下。每次运行生成的图表会存放在以当前时间命名的独立文件夹中，报告文件名也包含时间戳，以防覆盖。

### 为低样本意图增广数据（可选）

要为语料样本数低于CLU推荐阈值（默认为25）的意图生成新的候选话语，请使用 `enrich` 命令：

```bash
python main.py enrich --input-file data/IT_01_1.json
```

此命令默认会在 `outputs/reports/` 目录下生成一份带时间戳的 Markdown 报告，其中包含所有生成的候选话语，供您审查。

如果您希望直接在控制台查看结果而不生成报告文件，请使用 `--dry-run` 标志：
```bash
python main.py enrich --input-file data/IT_01_1.json --dry-run
```

## 核心模块详解

关于核心处理模块 `CLUProcessor` 的工作原理、算法选择、以及输出产物的详细解读，请参阅技术说明文档：[**`CLUProcessor` 技术说明**](./docs/CLUProcessor_explanation.md)。
