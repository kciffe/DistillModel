# Distill NER Summary

- input_md: 课件/MinerU_markdown_二战后苏俄舰艇全纪录_2055582301166436352.md
- model_a: Qwen/Qwen3-32B-AWQ
- model_b: Qwen/Qwen3-32B-AWQ
- reference_source: model_a_pseudo_gold
- chunk_count: 9

## Metrics

- entity_precision: 0.963855
- entity_recall: 0.975610
- entity_f1: 0.969697
- attribute_precision: 0.925926
- attribute_recall: 0.904977
- attribute_f1: 0.915332
- relation_precision: 0.979866
- relation_recall: 0.966887
- relation_f1: 0.973333

## Errors

- missing_entity: 2
- extra_entity: 3
- wrong_type: 0
- missing_attribute: 19
- wrong_attribute_value: 2
- wrong_relation: 5
- wrong_attachment: 0
