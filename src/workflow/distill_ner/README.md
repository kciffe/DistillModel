# distill_ner: Entity-only Equipment NER Evaluation

当前版本用于比较线上满血模型与 30B 左右小模型在舰艇装备实体识别任务上的能力差异。

流程：

```text
Markdown -> split chunks -> Model A entity extraction -> Model B entity extraction -> Gold entity comparison -> report/charts
```

当前版本只评测实体识别，不评测：

- attributes 属性正确率
- relations 关系正确率
- 属性归属错误
- 多级嵌套关系正确率
- LLM 辅助评判

Gold 文件中如果有 `attributes` 和 `relations`，评测时会忽略，只使用 `name/type/aliases/evidence`。

## 配置

复制 `.env.example` 到项目根目录 `.env`，并修改模型和路径。

DeepSeek 默认关闭 thinking：

```env
MODEL_A_PROVIDER=deepseek
MODEL_A_ENABLE_THINKING=false
```

Qwen vLLM 默认关闭 thinking：

```env
MODEL_B_PROVIDER=qwen_vllm
MODEL_B_ENABLE_THINKING=false
```

## 运行

```bash
python -m src.workflow.distill_ner.run_eval --env .env
```

如果你的代码目录不是 `src/workflow/distill_ner/`，按照项目实际包路径运行。

## 输出文件

```text
chunks.jsonl
model_a_entities.jsonl
model_b_entities.jsonl
entity_eval_model_a.jsonl
entity_eval_model_b.jsonl
entity_summary_model_a.json
entity_summary_model_b.json
entity_summary_compare.json
entity_summary_compare.md
parse_errors_model_a.jsonl
parse_errors_model_b.jsonl
entity_f1_comparison.png
entity_precision_recall_comparison.png
entity_tp_fp_fn_comparison.png
```

## 指标

- `entity_precision`：预测实体中有多少是 Gold 中的实体。
- `entity_recall`：Gold 实体中有多少被模型抽到。
- `entity_f1`：实体 Precision 和 Recall 的调和平均。
- `type_accuracy`：名称匹配成功的实体中，类型判断正确的比例。
- `wrong_type`：名称命中但类型不一致的数量。

## Prompt 说明

Prompt 采用“开放类型定义”方式：保留输出格式、实体类型体系和基本消歧规则，但明确说明示例不是完整实体列表，避免模型只围绕示例实体抽取。
