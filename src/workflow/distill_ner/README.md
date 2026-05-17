# Distill NER

`distill_ner` 是一个独立的 LangGraph 工作流，用于把 OCR 后的舰艇类 Markdown 按 chunk 分块，分别调用两个 OpenAI-compatible Chat Completions 模型抽取装备实体，并评测两个模型的实体、属性、关系差异。

## 依赖

需要安装：

```bash
pip install langgraph pydantic python-dotenv openai tqdm
```

## 配置 `.env`

在项目根目录 `.env` 中配置：

```env
MODEL_A_BASE_URL=https://api.openai.com/v1
MODEL_A_API_KEY=your_model_a_api_key
MODEL_A_NAME=gpt-4o-mini

MODEL_B_BASE_URL=http://localhost:8000/v1
MODEL_B_API_KEY=EMPTY
MODEL_B_NAME=your-vllm-model-name

DISTILL_NER_INPUT_MD=data/soviet_ships_ocr.md
DISTILL_NER_OUTPUT_DIR=outputs/distill_ner
```

不要把真实 API Key 提交到代码仓库。`MODEL_A_*` 和 `MODEL_B_*` 都使用 OpenAI-compatible Chat Completions 接口，可兼容 OpenAI、vLLM、Qwen、DeepSeek 等服务。

可选人工 gold 文件：

```env
DISTILL_NER_GOLD_JSONL=outputs/distill_ner/gold_entities.jsonl
```

`gold_entities.jsonl` 每行建议格式：

```json
{"chunk_id":"chunk_0001","entities":[{"name":"P-13","type":"missile","aliases":[],"attributes":{},"relations":[],"evidence":[{"page":"1","text":"..."}]}]}
```

## 运行

使用 `.env` 中的输入输出路径：

```bash
python -m src.workflow.distill_ner.run_eval
```

覆盖输入和输出路径：

```bash
python -m src.workflow.distill_ner.run_eval --input path/to/input.md --output outputs/distill_ner
```

如需启用 LLM 辅助复杂错误判断：

```bash
python -m src.workflow.distill_ner.run_eval --llm-eval
```

LLM 辅助判断只写入 `pair_eval.jsonl` 的 `llm_judgement` 字段，最终 Precision / Recall / F1 仍由代码规则计算。

## 输入 Markdown

输入应为 UTF-8 Markdown。工作流会按以下信息综合切块：

- Markdown 标题层级。
- 页码标记，如 `第 12 页`、`page: 12`、`<!-- page: 12 -->`。
- 文本长度，每块大约 1500 到 3000 个中文字符。

每个 chunk 会保留：

- `chunk_id`
- `page_hint`
- `title_path`
- `text`

## 输出文件

输出目录由 `DISTILL_NER_OUTPUT_DIR` 或 `--output` 指定，包含：

- `chunks.jsonl`：切块后的文本。
- `model_a_entities.jsonl`：模型 A 抽取并规范化后的实体。
- `model_b_entities.jsonl`：模型 B 抽取并规范化后的实体。
- `pair_eval.jsonl`：逐 chunk 对比评测结果与错误列表。
- `summary_report.json`：总体指标汇总。
- `summary_report.md`：便于阅读的 Markdown 总结。

## 评测口径

如果提供了人工 `gold_entities.jsonl`，则以 gold 为准计算 model_b 的实体、属性、关系 Precision / Recall / F1。

如果没有人工 gold，则默认 `model_a` 是 teacher/pseudo-gold，评测 `model_b` 相对 `model_a` 的抽取一致性。这不代表绝对正确率，只能代表两个模型在当前抽取任务上的相对一致性。

如果后续要获得真实正确率，需要人工标注 `gold_entities.jsonl`。

## 异常处理

模型返回非法 JSON 时，代码会依次尝试：

- 直接解析完整响应。
- 从 ```json fenced block``` 中提取 JSON。
- 截取第一个 `{` 到最后一个 `}`。

如果仍失败，会把错误写入对应 chunk 的结果记录，不会让整个流程因为单个 chunk 失败而中断。
