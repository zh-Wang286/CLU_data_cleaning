你是顶尖的数据科学家，精通自然语言处理与聚类分析。  
现在有一份 **Azure CLU** 的训练数据，包含 **60个Intent类别** 和 **1594条utterance样本**。  
本项目目标是：  
1. **确保不同Intent之间的语义区分度高**（低耦合：用语义相似度与分类边界清晰度衡量）  
2. **确保同一Intent内的utterances语义一致**（高内聚：Intent内部样本分布紧密）  

### 1. 数据与结构理解
- 数据文件：`IT_01_1.json`（节选）  
- 数据模式：参考 `@data/json_schema`，包含 Intent 列表及各自的 utterances。  

### 2. 分析任务
- **2.1 Intent 间低耦合性分析**
  - 基于 Azure OpenAI `text-embedding-3` 将utterances向量化。  
  - 计算 **Intent 与 Intent 之间的平均语义相似度**。  
  - 评估 Intent 分类边界的清晰度（可用最近邻 Intent 边界距离、可视化分布等方法）。  
  - 输出哪些 Intent 之间语义重叠较高、可能需合并或优化。

- **2.2 Intent 内部高内聚性分析**
  - 针对**同一个 Intent 的 utterances**，使用 HDBSCAN 聚类，分析内部语义分布是否紧密。  
  - 如果一个 Intent 内聚类分裂为多个子群，提示该 Intent 可能语义过宽，需拆分。  

- **2.3 方法探索**
  - 提供可行的方法来量化：
    - 散点图是否能表示
    - Intent 间低耦合（边界清晰度指标、语义相似度热力图）  
    - Intent 内高内聚（类内方差、轮廓系数、聚类结果解释）  

### 3. 技术要求
- 向量化：使用 Azure OpenAI `text-embedding-3`  
  - API 调用需显式传入 `endpoint` 与 `api key` 鉴权（使用 `openai` 包）
- 聚类分析：使用 **HDBSCAN**  
- 可视化：
  - 用人类友好的方式呈现分析结果（风格参考“乔布斯美学”——简洁、对比鲜明、重点突出）  
  - 使用 `matplotlib` + `seaborn`  
  - 英文图注，避免字符编码问题  

### 4. 交付内容
1. **Python程序** 实现上述分析任务  
2. **README文档**，包含：
   - 项目说明  
   - 依赖环境（假设已安装 `hdbscan`、`matplotlib`、`seaborn`、`openai`）  
   - 使用步骤  
