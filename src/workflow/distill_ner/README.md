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

## v2 改进说明

本版本主要面向 Qwen3-32B-AWQ 的抽取稳定性做了增强：

1. 默认通过 `chat_template_kwargs.enable_thinking=false` 关闭 Qwen3 thinking，减少 JSON 外文本。
2. 支持 `response_format={"type":"json_object"}`，若服务不支持会自动降级。
3. 抽取 prompt 增加了舰级/型号/单艇/导弹的类型判定优先级。
4. 在用户 prompt 中加入基于规则的候选实体提示，用于提升实体召回，但仍要求模型以原文 evidence 为准。
5. Markdown 切块删除图片 URL，保留图注；一级/二级标题强制断开，避免跨章节混块。
6. 后处理阶段增加型号、导弹、K/TK 艇号、声呐、雷达、导航、指挥、动力、舱室等轻量类型纠正。
7. 增加 hull_structure、communication_system、electronic_system、weapon_system 等类型，减少全部落入 equipment。

如果 A/B 都使用同一个 Qwen3 模型，建议把 A 作为 baseline，B 使用本版本 prompt/后处理进行对比；若要得到真实正确率，仍需要 `DISTILL_NER_GOLD_JSONL` 人工标注。


## v2-gold 改动说明

本版本在 v2 基础上加入了人工/半人工 Gold 评测闭环：

```text
Markdown -> chunk 切分 -> model_a/model_b 抽取 -> Gold 对比 -> 生成真实 Precision/Recall/F1 -> 画图
```

新增文件：

- `gold/gold_entities_soviet_10pages_v1.jsonl`：根据本次 10 页 Markdown 标注的 Gold 实体、属性、关系。
- `gold_eval_model_a.jsonl`：运行后输出，model_a 与 Gold 的逐块评测。
- `gold_eval_model_b.jsonl`：运行后输出，model_b 与 Gold 的逐块评测。
- `summary_report_model_a.json/.md`、`summary_report_model_b.json/.md`：两个模型相对 Gold 的总指标。
- `gold_f1_comparison.png`、`gold_error_comparison.png`、`gold_counts_model_a.png`、`gold_counts_model_b.png`：可视化评测图。

`.env` 中配置：

```env
DISTILL_NER_GOLD_JSONL=src/workflow/distill_ner/gold/gold_entities_soviet_10pages_v1.jsonl
```

如果不配置 `DISTILL_NER_GOLD_JSONL`，代码仍退回到 v2 的 `model_a_pseudo_gold` 相对一致性评测。

注意：Gold v1 已覆盖舰级、型号、艇名、导弹、鱼雷、声呐、雷达、导航、指挥、动力、舱室、通信/电子/艇体结构等实体，并带有关键属性和关系。后续若要作为论文级评测，建议再由人工逐条复核一次，重点核对 OCR 异体字和俄文字母/希腊字母混用问题。
