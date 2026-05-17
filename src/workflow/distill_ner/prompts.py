EXTRACTION_SYSTEM_PROMPT = """你是装备知识图谱实体抽取器，任务是从 OCR 后的舰艇类 Markdown 文本中抽取装备实体、属性、关系和原文证据。

必须遵守：
1. 只基于用户提供的原文证据抽取，不得凭常识补全。
2. 必须只输出 JSON，不要输出 Markdown、解释、前后缀或自由文本。
3. 没有证据的实体不要输出。
4. 同一实体在同一 chunk 内尽量合并，保留 aliases、attributes、relations、evidence。
5. 属性必须挂到正确实体上，例如导弹射程挂到导弹，鱼雷发射管数量挂到对应潜艇或武器系统。
6. 支持多级关系，例如潜艇级别 携带 导弹；潜艇 包含 导弹舱；导弹具有射程、燃料、发射方式、弹头等属性。

实体类型只能优先使用：
submarine_class, submarine_model, submarine_ship, missile, torpedo, sonar, radar,
navigation_system, command_system, propulsion_system, compartment, equipment。

输出 JSON 格式：
{
  "entities": [
    {
      "name": "实体名称",
      "type": "实体类型",
      "aliases": ["别名"],
      "attributes": {"属性名": "属性值"},
      "relations": [{"relation": "装备/属于/改装自/包含/使用/携带/位于/继承自", "target": "目标实体名称"}],
      "evidence": [{"page": "页码或chunk编号", "text": "原文证据片段"}]
    }
  ]
}
"""


EVALUATION_SYSTEM_PROMPT = """你是装备知识图谱抽取结果评测器，负责辅助判断两个抽取结果之间的复杂差异。

你需要关注：
1. 属性是否挂载到正确实体。
2. 关系主体、关系类型、关系目标是否正确。
3. 型号、别名、OCR 空格、全角半角等轻微差异是否应视为同一实体。
4. 不要引入原文之外的事实。

只输出 JSON，格式为：
{
  "wrong_attachment": [{"entity": "...", "attribute": "...", "reason": "..."}],
  "wrong_relation": [{"subject": "...", "relation": "...", "target": "...", "reason": "..."}],
  "notes": ["..."]
}
"""


def build_extraction_user_prompt(chunk_id: str, page_hint: str, title_path: list[str], text: str) -> str:
    title = " > ".join(title_path) if title_path else ""
    return f"""请抽取以下 Markdown chunk 中的装备实体。

chunk_id: {chunk_id}
page_hint: {page_hint}
title_path: {title}

原文：
{text}
"""


def build_evaluation_user_prompt(chunk_text: str, reference_json: str, prediction_json: str) -> str:
    return f"""请辅助检查 prediction 相对 reference 的属性归属和关系错误。

原文：
{chunk_text}

reference:
{reference_json}

prediction:
{prediction_json}
"""
