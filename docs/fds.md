## FDS｜Azure CLU 数据清洗与分析系统（后端主导）

### 1. 目标与范围
1. **确保不同Intent之间的语义区分度高**（低耦合：用语义相似度与分类边界清晰度衡量）  
2. **确保同一Intent内的utterances语义一致**（高内聚：Intent内部样本分布紧密）  
3. 对潜在的标签噪声和意图定义重叠进行检测与提示
4. 对训练数据中样本数量不足的 Intent 提供警告与补充手段
- **目标**: 面向 Azure CLU 训练集的数据质量治理，确保意图间低耦合、意图内高内聚，发现与消减噪声，补齐数据不足意图并生成可审计结果。
- **范围**: 对 `data/IT_01_1.json` 进行加载、验证、分析、可视化与报告生成；意图定义不变更，仅对样本进行标注/提示/建议；可选 LLM 增广。

### 2. 技术路径（端到端）
- **数据与校验**: `pydantic v2` 定义严格模型；加载 JSON 并做字段/值域校验；统计与一致性检查（空文本、无意图、非法语言码）。
- **向量化**: Azure OpenAI `text-embedding-3-small`；分批嵌入、缓存与重用；失败重试与速率控制。
- **类内异常检测（主输出）**:
  - kNN 异常（k=1），或 LOF；距离分位阈值（95%/IQR 可配）；输出每意图的异常样本索引与分数。
- **全局聚类审计（次输出）**:
  - HDBSCAN 聚类，统计簇与原意图的重叠度；小簇过滤；提示潜在边界重叠与意图过宽。
- **降维与可视化**: UMAP 到 2D；绘制意图相似度热力图、全局散点与类内散点。
- **数据增广（可选/手动）**: 对 <25 条的意图，用 LLM 在现有样本与意图描述下补齐；使用 `dataset` 字段标为 `Augment`；输出清单供人工复核。
- **报告与导出**: 结构化 JSON + 图表（PNG/SVG）；文本/表格化“异常审计报告”。
- **日志（新增要求）**: 关键步骤必须记录到控制台与文件，文件名带日期时间，支持轮转与保留策略（详见第 10 节）。

### 3. 总体架构（后端主导）
- **层次**:
  - 数据层：`CLUDataset`（加载/验证/检索/统计）
  - 服务层：`ModelClient`（OpenAI/Embedding/LLM 适配器）
  - 处理层：`CLUProcessor`（异常检测、聚类审计、增广）
  - 展示层：可视化与报告（热力图、散点、审计表）
  - 接口层：CLI 命令（无前端）
- **关键约束**: 不改动意图定义；显式暴露 `endpoint` 与 `api_key`；禁止过度封装；贯穿全链路的可观测日志。

### 4. 目录结构
- **代码**:
  - `main.py`：程序入口
  - `src/dataset.py`：`CLUDataset`
  - `src/processor.py`：`CLUProcessor`
  - `src/model_client.py`：`ModelClient`（OpenAI 适配）
  - `src/schemas.py`：`pydantic` 数据模型映射 `data/json_schema`
  - `src/visualization.py`：UMAP/热力图/散点图
  - `src/reporting.py`：审计聚合与导出（Markdown）
  - `src/config.py`：配置与环境变量解析
  - `src/utils/logging.py`：`loguru` 日志初始化（控制台 + 文件）
- **数据与输出**:
  - `data/IT_01_1.json`、`data/json_schema`
  - `outputs/embeddings/`、`outputs/figures/`、`outputs/reports/`
  - `logs/`（新增：日志持久化目录）

### 5. 数据模型与校验（对齐 json_schema）
- **映射**:
  - `ProjectMetadata`、`Assets`、`Intent`、`Utterance` 等 `pydantic` 模型
  - `utterances[*].text`、`utterances[*].intent` 为必填；`language` 若缺省按全局 `metadata.language`
- **扩展**:
  - `utterances[*].dataset` 使用值域：`Train|Validate|Augment`（兼容 schema 的自由字符串）
- **校验规则**:
  - 文本去重与空白剔除；非法语言码告警；不存在的 `intent` 引用告警
  - 每意图计数与 `<threshold=25` 告警列表

### 6. 关键算法设计
- **Embedding**:
  - 批量大小自适应（默认 128）；指数退避重试
- **类内异常**:
  - 距离度量：余弦距离
  - 阈值：`95pct` 或 IQR：`Q3 + 1.5*IQR`
  - 输出：`{intent: [{idx, text, score, threshold, rank}], ...}`
- **全局审计**:
  - HDBSCAN：`min_cluster_size=5`（可配），过滤簇大小 <3
  - 对齐：簇→多数表决原意图；报告跨意图混合率、未分配（-1）比例
- **相似度热力图**:
  - 每意图取均值向量，计算意图间余弦相似度；对角置 1
- **降维**:
  - UMAP：`n_neighbors=15`，`min_dist=0.1`，`metric='cosine'`，`random_state=42`

### 7. 类与接口（最小封装）
- **`ModelClient`**
  - 配置：`endpoint`, `api_key`, `embedding_model`, `llm_model`
  - 方法最少化：`get_embedding_model()`，`get_llm_model()`，`health_check()`
  - 直接暴露底层客户端实例（保持透明）
- **`CLUDataset`**
  - `get_intents()`，`get_utterances(intent)`，`count_intents()`，`count_utterances(intent|None)`，`warn_low_utterance_intents(th=25)`
- **`CLUProcessor`**
  - `detect_intra_intent_outliers(method='knn'|'lof', k=1, threshold='95pct'|'iqr')`
  - `audit_global_clusters(min_cluster_size=5)`
  - `enrich_low_utterance_intents(threshold=25)`：仅生成候选不落盘，输出待审列表
- 所有公开方法均输出结构化字典，便于报告与二次处理。
<!-- 
### 8. CLI 设计（非交互、可脚本化）
- **命令**:
  - `clu validate --input data/IT_01_1.json`
  - `clu stats --input ...`
  - `clu outliers --input ... --method knn --threshold 95pct`
  - `clu audit --input ... --min-cluster-size 5`
  - `clu enrich --input ... --threshold 25 --dry-run`
  - `clu visualize --input ... --plots heatmap,global,per-intent`
  - `clu report --input ... --out outputs/reports/it_01_1.md` -->
- **日志相关参数（新增）**:
  - `--log-level {DEBUG,INFO,WARNING,ERROR}`（默认 `INFO`）
  - `--log-dir PATH`（默认 `./logs`）
  - `--log-rotate {size:50 MB|time:00:00|off}`（默认 `time:00:00` 每日轮转）
  - `--log-retention "14 days"`（默认 `14 days`）
  - `--log-compress {zip|off}`（默认 `zip`）

### 9. 环境与依赖
- **Python**: `>=3.12`
- **项目依赖（取自 pyproject）**:
  - `openai>=1.99.9`、`pydantic>=2.11.7`
  - `hdbscan>=0.8.40`、`umap-learn>=0.5.6`、`numpy<2.0`、`numba>=0.58`、`llvmlite>=0.41`
  - `matplotlib>=3.10.5`、`seaborn>=0.13.2`、`pandas>=2.3.1`、`plotly>=6.3.0`、`loguru>=0.7.3`
- **系统依赖（CentOS8）**:
- **配置**:
  - 环境变量：`AZURE_OPENAI_ENDPOINT`、`AZURE_OPENAI_API_KEY`、`OPENAI_API_VERSION`
  - 日志环境变量（新增）：`LOG_LEVEL`、`LOG_DIR`、`LOG_ROTATE`、`LOG_RETENTION`、`LOG_COMPRESS`
- **安装**:
  - 推荐使用 `uv`（仓库已有 `uv.lock`）：`uv sync`
  - 或 `pip`: `pip install -e .`

### 10. 日志与审计（新增要求，强制）
- **目标**: 关键步骤均有结构化日志；日志输出到控制台与文件；文件名包含日期时间，可轮转与保留，支持压缩。
- **实现**:
  - 提供 `src/utils/logging.py` 统一初始化：
    - 控制台 sink：彩色、简要、适合阅读的格式。
    - 文件 sink：路径 `logs/`，文件名模式 `clu_{time}.log`（如 `clu_2025-08-19_14-30-05.log`）。
    - 轮转策略：
      - 时间轮转：默认每天 `00:00` 新文件（`rotation="00:00"`）。
      - 或文件大小：如 `50 MB`（通过 CLI/env 可选）。
    - 并发/多进程：开启队列 `enqueue=True`，避免竞争。
    - 日志格式：`"{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {process} | {name}:{function}:{line} - {message}"`。
    - 追踪上下文：在 CLI 入口生成 `run_id`（UUID），注入到日志上下文，串联一次运行内所有记录。
  - **关键步骤必须打印（最小清单）**：
    - 配置载入与参数快照（INFO）
    - JSON 加载与 schema 校验结果（INFO/ERROR）
    - 意图/样本计数、低样本阈值告警（WARN）
    - Embedding 批处理起止、吞吐、失败重试（INFO/WARN）
    - 类内异常检测参数与结果摘要（INFO）
    - 聚类参数与簇统计、混合率（INFO）
    - 可视化文件生成路径（INFO）
    - 报告导出路径与摘要指标（INFO）
    - 增广意图与新增样本数（INFO），含审计标识（`Augment`）
    - 运行耗时与资源占用摘要（INFO）

### 11. 任务拆分与里程碑（UTC+8）
- **M1 数据层（1d）**: `pydantic` 模型/加载/校验/统计/低样本告警
- **M2 模型接入（1d）**: `ModelClient`、健康检查、嵌入缓存
- **M3 异常检测（2d）**: kNN/LOF、阈值策略、结果结构化
- **M4 聚类与可视化（2d）**: HDBSCAN、UMAP、热力图与散点
- **M5 报告与CLI（1d）**: 审计报告、非交互命令
- **M6 增广与复核（1d）**: LLM 生成策略、干运行/标注
- **M7 稳定化（1d）**: 性能与边界测试、日志与监控完善（新增）
- 预计：9 工日

### 12. 验收标准
- **功能**:
  - 正确解析与校验 `data/IT_01_1.json`；输出低样本意图清单（阈值 25）
  - 产出类内异常样本列表（含分数/阈值/排名）
  - 产出聚类审计（混合率、未分配比例、疑似重叠对）
  - 生成三类图：意图相似度热力图、全局散点、任意意图内散点
  - 增广命令在 `--dry-run` 下生成候选而不写入原始数据
- **非功能**:
  - CLI 无交互；所有路径/阈值可配置
  - 完整控制台日志；生成 `logs/clu_YYYY-MM-DD_HH-MM-SS.log` 文件，按配置轮转与保留（新增）
  - 记录完整参数快照与运行 `run_id`，支持追踪（新增）
  - 在本数据集上单次全流程≤15 分钟（CPU 环境）

### 13. 风险与缓解
- **HDBSCAN/Numba 构建失败**: 明确系统依赖；提供 `--skip-cluster` 降级路径
- **Azure OpenAI 配额/速率**: 批量与退避；嵌入缓存；失败重试与断点续跑
- **语义漂移/意图过宽**: 审计报告标注高风险意图，保留阈值可调
- **中文文本噪声**: 正则清洗（空白/特殊符号），统一全角半角
- **可重复性**: 固定随机种子（UMAP/HDBSCAN）；记录参数快照
- **日志膨胀**: 通过轮转/保留/压缩控制；支持按级别降噪（新增）

<!-- ### 14. 测试策略
- **单元测试**: 加载/校验/计数；阈值计算；嵌入失败与重试；日志初始化与文件生成（新增）
- **集成测试**: 小样本子集跑全流程；快照比对图与报告结构；校验日志文件存在且包含关键步骤标记（新增）
- **回归测试**: 固定数据与随机种子；比较指标（异常数、混合率） -->

### 15. 运行与运维
- **环境**:
  - 导入环境变量：`export AZURE_OPENAI_ENDPOINT=...`，`export AZURE_OPENAI_API_KEY=...`
  - 日志相关（可选）：`export LOG_LEVEL=INFO`，`export LOG_DIR=./logs`，`export LOG_RETENTION="14 days"`
<!-- - **典型运行**:
  - 验证：`clu validate --input data/IT_01_1.json`
  - 统计：`clu stats --input data/IT_01_1.json`
  - 异常：`clu outliers --input data/IT_01_1.json --method knn --threshold 95pct`
  - 审计：`clu audit --input data/IT_01_1.json --min-cluster-size 5`
  - 可视化：`clu visualize --input data/IT_01_1.json --plots heatmap,global`
  - 报告：`clu report --input data/IT_01_1.json --out outputs/reports/it_01_1.md` -->

### 16. 交付物
- **代码**: `CLUDataset`、`CLUProcessor`、`ModelClient` 
- **文档**: `README.md`（背景/依赖/运行说明）
- **图与报告**: 热力图/散点图、异常样本列表、聚类审计表
- **增广结果**: 低样本意图的候选补充（标注来源 `Augment`）
- **日志设施（新增）**: `src/utils/logging.py` 与默认 `logs/` 目录

### 17. 设计原则与编码规范
- **风格**: Google Python 风格；控制台日志必备；不超过 3 层嵌套；关键逻辑注释充分
- **格式化**: 遵循项目 `pyproject.toml` 中 `yapf/flake8/isort` 规则
- **结构**: 短小函数、早返回、显式错误处理，不做无意义抽象
- **日志要求**: 各模块需在关键节点调用 `loguru` 输出；禁止吞噬异常；错误需包含上下文信息与 `run_id`。


