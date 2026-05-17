from __future__ import annotations

import re
import unicodedata


EXTRACTION_SYSTEM_PROMPT = """你是装备知识图谱抽取器。你的任务是从 OCR 后的舰艇类 Markdown 文本中抽取“装备实体—属性—关系—证据”。

最高优先级规则：
1. 只能基于原文，不得用常识补全。原文没有明确证据的实体、属性、关系不要输出。
2. 只输出一个合法 JSON 对象，不要输出 Markdown、解释、前后缀、思考过程。
3. evidence.text 必须是原文中的短证据片段，建议 15～80 字。没有 evidence 的实体不要输出。
4. 不要把章节标题、国家、年代、任务评价、泛化概念当实体，除非它明确是装备/型号/系统/部件。
5. 属性必须挂到正确实体：导弹射程/燃料/弹头/发射方式挂到 missile；潜艇排水量/航速/潜深/艇员/携弹量挂到潜艇级别或型号；舱室长度/用途挂到 compartment。

实体类型体系。示例只用于说明类型含义，不是完整名单，必须根据原文自动发现实体：
- submarine_class：潜艇级别/级别名称，通常是“X级”。例如 G级、H级、Y级、D级、台风级、北风之神级。
- submarine_model：设计型号、工程编号、改装型号，通常是“数字+型/字母变体+型”。例如 629型、629A型、658M型、667A型、941型、601型、605型、619型。
- submarine_ship：具体单艇、艇名、舷号或编号。例如 K-145、K-140、K-88、K-153、TK-208、“尤里·多尔戈鲁基”号。
- missile：导弹/弹道导弹/巡航导弹/潜射导弹。例如 P-11ΦM、P-13、P-21、P-27、P-29、P-31、P-39、PCM-25、4K-75、“圆锤”/“布拉瓦”。
- torpedo：鱼雷、火箭鱼雷、鱼雷发射管、鱼雷武器系统。
- sonar：声呐、声呐站、声呐系统、噪声测向仪等水声设备。
- radar：雷达、雷达站、侦察雷达。
- navigation_system：导航系统、惯性导航、天文导航、卫星导航。
- command_system：作战指挥系统、导弹发射控制系统、解算系统、射击指挥系统。
- propulsion_system：动力系统、核动力装置、柴油机、电机、推进电机、螺旋桨。
- compartment：舱室、功能舱段、导弹舱、鱼雷舱、蓄电池舱、柴油机舱、电机舱、指挥舱。
- weapon_system：武器系统、导弹武器系统、发射系统。
- communication_system：通信系统、卫星通信、拖曳天线、漂浮天线、超长波接收设备。
- electronic_system：电子侦察设备、无线电设备、控制电子设备。
- hull_structure：艇体结构、耐压艇体结构、龟背、8字形结构、品字形结构等。
- equipment：其他未能归入以上类型但具有装备意义的系统、装置、部件。

类型判定优先级：
1. “K-数字 / TK-数字 / C-数字 / 中文引号中的某某号”优先 submarine_ship。
2. “X级”优先 submarine_class；但“G-2型、H-2型”这种带“型”的北约变体可作为 submarine_model，并把 G-2/H-2 放 aliases 或 relation。
3. “数字+型、数字+字母+型、设计代号”优先 submarine_model。
4. “P-/PCM-/4K- 开头，或原文称为导弹/弹道导弹/飞航导弹”优先 missile。
5. 若一个实体既是系统也是设备，优先选择更具体类型，例如“白云石解算系统”用 command_system，不用 equipment。

关系建议使用这些规范名称：
- equipped_with：装备/配备/携带某武器或系统
- belongs_to：型号属于某级，单艇属于某级/某型号
- alias_of：北约称为/亦称/又称
- modified_from：改装自/基于
- contains：包含/设有某舱室或部件
- used_for：用于/用作
- tested_on：在某艇上试验
- replaced_by：由……取代/换装为

输出 JSON 格式必须严格如下：
{
  "entities": [
    {
      "name": "实体名称",
      "type": "实体类型",
      "aliases": ["别名"],
      "attributes": {"属性名": "属性值"},
      "relations": [{"relation": "规范关系名", "target": "目标实体名称"}],
      "evidence": [{"page": "页码或chunk编号", "text": "原文证据片段"}],
      "type_reason": "简短说明为什么是该类型",
      "confidence": 0.0
    }
  ]
}

抽取策略：先识别潜艇级别/型号/单艇，再识别导弹、鱼雷、声呐、雷达、导航、指挥、动力、舱室等组件，最后补充属性和关系。不要因为候选实体提示中出现某词就强行输出，仍以原文证据为准。
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


_CANDIDATE_PATTERNS = [
    r"[A-Z]{1,3}-\s*\d{1,4}[A-Z]?(?:\s*号艇|\s*艇)?",
    r"TK-\s*\d{1,4}",
    r"\d{3,4}\s*[A-Za-zА-Яа-яⅡI]*\s*型",
    r"[A-Z]-\s*\d\s*型",
    r"(?:G|H|Y|D)\s*级",
    r"台风级|北风之神级",
    r"P\s*-\s*\d+[A-Za-zΑ-ωА-Яа-яΦФMⅡI]*\s*型?",
    r"PCM\s*-\s*\d+\s*型?",
    r"4K\s*-\s*\d+\s*型?",
    r"“[^”]{2,20}”(?:号|型|系统|导弹|声呐|雷达)?",
    r"[\u4e00-\u9fa5A-Za-z0-9ⅡIΓФΦ\-]+(?:声呐|雷达站|导航系统|解算系统|作战指挥系统|导弹系统|柴油机|推进电机|蓄电池|发射筒|发射装置|鱼雷发射管|拖曳天线|漂浮天线|耐压艇体)",
]


def _norm_candidate(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = re.sub(r"\s+", "", text)
    text = text.replace("$", "")
    text = text.replace("\\Phi", "Φ").replace("Ф", "Φ")
    return text.strip("，。；:：、（）()[]【】 ")


def extract_candidate_hints(text: str, limit: int = 80) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for pattern in _CANDIDATE_PATTERNS:
        for match in re.finditer(pattern, text):
            item = _norm_candidate(match.group(0))
            if len(item) < 2 or item in seen:
                continue
            seen.add(item)
            candidates.append(item)
            if len(candidates) >= limit:
                return candidates
    return candidates


def build_extraction_user_prompt(chunk_id: str, page_hint: str, title_path: list[str], text: str) -> str:
    title = " > ".join(title_path) if title_path else ""
    hints = extract_candidate_hints(text)
    hints_text = "、".join(hints) if hints else "无"
    return f"""请抽取以下 Markdown chunk 中的装备实体。

chunk_id: {chunk_id}
page_hint: {page_hint}
title_path: {title}

候选实体提示（仅用于提高召回，不能替代原文证据，不能无证据输出）：
{hints_text}

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
