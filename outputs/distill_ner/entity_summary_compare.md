# Entity-only Equipment NER Evaluation

本轮只评测装备实体识别能力，不评测 attributes、relations 或多级嵌套关系正确率。

- input_md: 课件/MinerU_markdown_二战后苏俄舰艇全纪录.md
- gold_jsonl: src/workflow/distill_ner/gold/gold_entities_soviet_H_article_v1_min1200_max2600.jsonl
- chunk_count: 4

## Metrics

| Model | Precision | Recall | F1 | Type Accuracy | TP | FP | FN | Wrong Type | Parse Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model A: deepseek-v4-pro | 0.9452 | 0.7113 | 0.8118 | 0.9275 | 69 | 4 | 28 | 5 | 0 |
| Model B: Qwen/Qwen3-32B-AWQ | 0.8594 | 0.5670 | 0.6832 | 0.9273 | 55 | 9 | 42 | 4 | 0 |


