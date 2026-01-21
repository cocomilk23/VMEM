# ValueMEM

ValueMEM 是面向智能体的价值驱动记忆系统。系统使用单一物理图谱承载全部记忆，在关系属性中存储价值分数，以逻辑分层方式实现高价值优先检索，兼顾速度与上下文完整性。

## 亮点
- 单 LLM 端到端价值评分，轻量低延迟
- 统一图谱 + 逻辑分层，高价值记忆优先召回
- 记忆文本与价值分数联合存储，可解释性强
- 辐射检索：高价值命中后扩展邻域上下文
- 评分缓存：相同或相似记忆复用历史评分

## 架构
1. 输入层：接收对话/文档文本
2. 处理层：事实提取 -> 价值评分 + 三元组 -> 向量嵌入
3. 存储层：Neo4j 统一图谱，关系属性存储记忆文本与价值分数
4. 检索层：高价值优先 -> 语义排序 -> 辐射扩展

## 数据模型
关系类型：`(Entity)-[:MEMORY]->(Entity)`

关系属性：
- `predicate`：关系语义
- `memory_text`：原始事实记忆
- `value_score`：0-1 价值分数
- `created_at`：创建时间
- `access_count`：访问次数
- `embedding`：语义向量

## 快速开始
1) 安装依赖
```bash
pip install -r requirements.txt
```

2) 创建并激活虚拟环境（推荐）
```bash
python -m venv .venv
.venv\\Scripts\\activate
```

3) 配置环境变量
```bash
copy .env.example .env
```
按需修改 `.env` 中的 OpenAI 与 Neo4j 配置。

4) 初始化图谱结构
```bash
python -m vmem.cli init-schema
```

5) 写入记忆
```bash
python -m vmem.cli ingest "Xiaoming met the CEO during the conference."
```

6) 检索记忆
```bash
python -m vmem.cli query "who met the CEO"
```

更多快速步骤见 `quickstart.md`，可运行示例见 `examples/`。

## YAML 配置（可选）
你也可以使用 `config.yaml`：
```bash
python -m vmem.cli --config config.yaml ingest "Alice met the CEO."
```
如果使用 `config.yaml`，可以保留 `${OPENAI_API_KEY}`，系统会自动读取环境变量。

## API 服务（可选）
```bash
uvicorn vmem.server.app:app --host 0.0.0.0 --port 8000
```
接口：
- `POST /ingest` `{ "text": "...", "source": "user" }`
- `POST /query` `{ "text": "..." }`

## 基准与示例脚本
- 写入示例：`python scripts/seed_demo.py`
- 简单基准：`python scripts/benchmark_retrieval.py`
- 向量索引：`scripts/vector_index.cypher`（如需更大规模检索）
- 可运行示例：`python examples/cli_ingest.py --text "..."` / `python examples/cli_query.py --text "..."`

## 关键配置
环境变量示例见 `.env.example`。常用参数：
- `VMEM_VALUE_THRESHOLD`：高价值阈值（默认 0.8）
- `VMEM_TOP_K`：返回结果数量
- `VMEM_CANDIDATE_K`：候选集规模
- `VMEM_WEIGHT_VALUE` / `VMEM_WEIGHT_SIM`：价值分数与语义相似度权重
- `VMEM_CACHE_PATH`：评分缓存位置
- `VMEM_CONFIG_PATH`：YAML 配置文件路径
- `VMEM_LOG_LEVEL`：日志级别
- `VMEM_VECTOR_INDEX_NAME`：向量索引名称
- `VMEM_ANSWER_SIM_THRESHOLD`：高价值命中可直接回答的相似度阈值

## 检索策略说明
1. 向量相似度检索高价值关系（`value_score >= 阈值`）
2. 命中后辐射扩展关联记忆加入候选集
3. 若相似度和命中数不足以回答，回退至全图谱向量检索
4. 价值分数与语义相似度混合排序

提示：如需启用 Neo4j 向量索引，请执行 `scripts/vector_index.cypher`。

## 目录结构
```
Vmem/
  vmem/
    llm/              LLM 事实提取与价值评分
    embeddings/       向量嵌入
    graph/            Neo4j 图谱存储
    memory/           写入流水线
    retrieval/        价值优先检索
    server/           API 服务
  docs/
    design.md
  scripts/
    bootstrap_neo4j.cypher
    seed_demo.py
    benchmark_retrieval.py
```

## 设计取舍
- 逻辑双层通过关系属性过滤实现，避免双图谱同步成本
- 评分缓存使用 sqlite，本地轻量可复用
- 三元组由 LLM 输出，支持跨领域语义表达

## 运行测试
```bash
pip install -r requirements-dev.txt
pytest -q
```

## 下一步建议
- 用业务数据调优 `VMEM_VALUE_THRESHOLD` 与权重比例
- 在 Neo4j 中启用向量索引以支持更大规模召回
