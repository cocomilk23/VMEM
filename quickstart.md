# ValueMEM 快速启动

## 1. 安装依赖
```bash
pip install -r requirements.txt
```

## 2. 配置环境变量
```bash
copy .env.example .env
```
编辑 `.env` 填写 OpenAI 和 Neo4j 配置。

## 3. 初始化图谱
```bash
python -m vmem.cli init-schema
```

## 4. 写入记忆
```bash
python -m vmem.cli ingest "Alice met the CEO during the conference."
```

## 5. 查询记忆
```bash
python -m vmem.cli query "who met the CEO"
```

## 6. 运行 API 服务（可选）
```bash
uvicorn vmem.server.app:app --host 0.0.0.0 --port 8000
```

## 7. 运行示例（可选）
```bash
python examples/cli_ingest.py --text "Alice met the CEO during the conference."
python examples/cli_query.py --text "who met the CEO"
```
