# Azure CLU 数据分析与优化需求文档
## 1. 项目背景
本项目基于 **Azure CLU** 训练数据：
目标是：
1. **确保不同Intent之间的语义区分度高**（低耦合：用语义相似度与分类边界清晰度衡量）  
2. **确保同一Intent内的utterances语义一致**（高内聚：Intent内部样本分布紧密）  
3. 对潜在的标签噪声和意图定义重叠进行检测与提示
4. 对训练数据中样本数量不足的 Intent 提供警告与补充手段
---
## 2. 数据结构与文件
- 数据文件：@data/IT_01_1.json
- 数据模式：@data/json_schema
---
## 3. 面向对象设计
### 3.1 数据类：`CLUDataset`
负责数据集的加载与基础信息获取。
**职责**：
- 读取 CLU JSON 训练数据
- 获取所有 Intent 列表
- 获取指定 Intent 的所有 utterances
- Intent 计数与 utterance 计数
- 全局 utterance 计数（复用指定Intent的utterance计数方法）
- **检测并提示 utterance 数量不足的 Intent**
**主要方法**：
```python
class CLUDataset:
    def __init__(self, json_path: str):
        pass
    def get_intents(self) -> list[str]:
        """返回所有 Intent 列表"""
    def get_utterances(self, intent_name: str) -> list[str]:
        """返回指定 Intent 的 utterances 列表"""
    def count_intents(self) -> int:
        """ 返回 Intent 总数 """
    def count_utterances(self, intent_name: str = None) -> int:
        """ 若指定 intent_name，则返回该意图下 utterances 数量，否则返回全局总数 """
    def warn_low_utterance_intents(self, threshold: int = 25) -> dict:
        """
        检测 utterance 数量不足的 Intent，并返回告警字典：
        { intent_name: utterance_count, ... }
        """
```
---
### 3.2 数据优化类：`CLUProcessor`
负责数据分析与质量检测（业务规则：Intent 不能乱动）。
**功能目标**：
1. **类内异常检测**（主要输出）  
   - 方法：  
     - kNN 异常检测（k=1）  
     - 或 LOF（Local Outlier Factor）  
   - 阈值：类内距离分布的分位数（95% 或 IQR）
   - 输出：标记/删除与类内样本差异过大的 utterance
2. **全局聚类审计**（次要输出）  
   - 方法：HDBSCAN + 次数过滤  
   - 输出：将聚类结果与原 Intent 对齐，并提示可能存在的定义重叠
3. **数据增强（手动触发）**  
   - 对于 utterance 数量不足 25 条的 Intent，可调用 LLM 结合当前 Intent 和现有 utterance 自动补充，直至样本数量满 25 条。
   - 新增数据会标记来源（LLM-generated）以便人工审核。
**主要方法**：
```python
class CLUProcessor:
    def __init__(self, dataset: CLUDataset, model_client):
        pass
    def detect_intra_intent_outliers(self, method="knn", k=1, threshold="95pct") -> dict:
        """类内异常检测，返回每个 Intent 内的异常utterance列表"""
    def audit_global_clusters(self, min_cluster_size=5) -> dict:
        """全局聚类审计，返回聚类与Intent的对照分析"""
    def enrich_low_utterance_intents(self, threshold: int = 25) -> dict:
        """
        手动触发：使用 LLM 自动补充 utterance 数量不足 threshold 条的 Intent。
        返回新增的 utterance 列表，结构：
        { intent_name: [new_utterance1, new_utterance2, ...] }
        """
```
---
### 3.3 模型调用类：`ModelClient`
使用适配器模式 Adapter Pattern，只做实例封装并返回实例，其余方法直接通过实例调用，禁止过度封装方法。
统一管理 LLM 与 Embedding API 调用（禁止过度封装，对外显式暴露 endpoint 与 apikey 配置）。
**主要方法**：
```python
class ModelClient:
    def __init__(self, endpoint: str, api_key: str, 
                 embedding_model: str = "text-embedding-3-small", 
                 llm_model: str = "gpt-4"):
        """
        初始化 ModelClient
        :param endpoint: Azure OpenAI endpoint (形如 https://xxx.openai.azure.com/)
        :param api_key: Azure OpenAI API Key
        :param embedding_model: 默认embedding模型
        :param llm_model: 默认LLM模型
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # 初始化 OpenAI 客户端
        self._client = OpenAI(api_key=self.api_key, base_url=self.endpoint)
        # 启动时进行API健康检测
        self._health_check()
    def _health_check(self):
        """
        测试 API 连通性（轻量请求）
        """
            # 用 embedding 测试最安全，成本低

    def get_embedding_model(self):
        """
        获取 openai embedding 模型实例
        """

    def get_llm_model(self):
        """
        获取 openai llm 模型实例
        """
```
---
## 4. 分析任务细化
### 4.1 Intent 间低耦合性分析
- 向量化：Azure OpenAI `text-embedding-3`
- 计算不同 Intent 平均语义相似度
- 评估分类边界清晰度（Nearest Neighbor 边界距离）
- 可视化：
  - 语义相似度热力图（英文标签）
  - 最近邻边界可视化
### 4.2 Intent 内高内聚性分析
- 在每个 Intent 内运行 HDBSCAN
- 检测是否存在多个子簇（提示该 Intent 语义过宽）
- 计算类内方差与轮廓系数
- 可视化：
  - 2D 降维（UMAP）+ 聚类结果散点图
### 4.3 异常检测方法探索
- kNN 异常值检测：标记类内距离过大的样本
- LOF 异常分数计算
- 阈值调节（可在95%分位数或IQR）
---
## 5. 技术要求
- 使用 `pydantic` 控制数据类型 ( pydanic > 2.0, field_validator )
- **Embedding 与 LLM**：
  - 使用 `openai` Python SDK
  - 显式传入 `endpoint` 与 `api_key`
- **聚类与降维**：
  - 使用 `knn`、`hdbscan`、`umap-learn`
- **可视化**：
  - `matplotlib` + `seaborn`
  - 全英文标签，简洁高对比
---
## 6. 交付内容
1. **注意**禁止使用任何 emoji 表情
2. **Python程序**（实现 CLUDataset, CLUProcessor, ModelClient 及分析任务）
3. **README.md**（包含项目背景、依赖、运行说明）
4. **分析结果可视化**（语义相似度热图、聚类散点图、异常样本列表）
5. **异常审计报告**（文本与表格结合显示高风险 Intent 与样本）
6. **utterance 数量不足告警与 LLM 扩充结果**