# Examples

这个目录包含 ValueMEM 的可运行示例与示例输入。

- `cli_ingest.py`: CLI 写入示例（可运行）
- `cli_query.py`: CLI 查询示例（可运行）
- `full_flow_demo.py`: 完整流程示例（写入 + 检索 + 回答）
- `ingest.json`: API 写入示例
- `query.json`: API 查询示例
- `demo.txt`: 示例文本

运行示例：
```bash
python examples/cli_ingest.py --text "Alice met the CEO during the conference."
python examples/cli_query.py --text "who met the CEO"
python examples/full_flow_demo.py
```
