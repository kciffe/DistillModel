from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import DistillNERConfig
from .md_loader import load_markdown as read_markdown
from .md_loader import split_markdown_chunks
from .model_client import OpenAICompatibleClient, dump_model
from .schema import Entity, EntityExtraction, MetricCounts, PairEvaluation, SummaryReport, TextChunk

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **_: Any):  # type: ignore
        return iterable


class DistillNERState(TypedDict, total=False):
    input_md: str
    output_dir: str
    markdown_text: str
    chunks: list[TextChunk]
    model_a_results: list[EntityExtraction]
    model_b_results: list[EntityExtraction]
    normalized_a: list[EntityExtraction]
    normalized_b: list[EntityExtraction]
    pair_evals: list[PairEvaluation]
    gold_evals_a: list[PairEvaluation]
    gold_evals_b: list[PairEvaluation]
    summary: SummaryReport
    summary_a: SummaryReport
    summary_b: SummaryReport
    errors: list[str]


RELATION_MAP = {
    "装备": "equipped_with",
    "携带": "equipped_with",
    "配备": "equipped_with",
    "使用": "equipped_with",
    "属于": "belongs_to",
    "为": "belongs_to",
    "为...型": "belongs_to",
    "改装自": "modified_from",
    "基于": "modified_from",
    "包含": "contains",
    "设有": "contains",
    "位于": "located_in",
    "继承自": "inherited_from",
}


def _json_dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(_json_dump(dump_model(row) if not isinstance(row, dict) else row) + "\n")


def _validate(cls: Any, data: Any) -> Any:
    if hasattr(cls, "model_validate"):
        return cls.model_validate(data)
    return cls.parse_obj(data)


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = text.replace("\\Phi", "Φ").replace("\\mathrm{M}", "M")
    text = text.replace("Ф", "Φ")  # OCR 中西里尔 Ф 与希腊 Φ 常混用；用于匹配归一。
    text = text.replace("－", "-").replace("—", "-").replace("–", "-")
    text = re.sub(r"\$|\\\(|\\\)|\{|\}", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"([Pp]|PCM|4K)-?(\d)", lambda m: f"{m.group(1).upper()}-{m.group(2)}", text)
    text = re.sub(r"(\d+)\s*([A-Za-z]+)?\s*型", lambda m: f"{m.group(1)}{m.group(2) or ''}型", text)
    return text.casefold()


def _canonical_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = text.replace("\\Phi", "Φ").replace("\\mathrm{M}", "M").replace("Ф", "Φ")
    text = re.sub(r"\$|\{|\}", "", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", "", text) if re.fullmatch(r"[A-Za-z0-9ΦΓⅡI\-\s]+(?:型|号艇|艇)?", text) else text
    text = re.sub(r"([Pp]|PCM|4K)-?(\d)", lambda m: f"{m.group(1).upper()}-{m.group(2)}", text)
    return text.strip()


def _infer_entity_type(entity: Entity) -> str:
    name = _canonical_name(entity.name)
    ev_text = " ".join(e.text for e in entity.evidence)
    current = str(entity.type or "equipment")
    if re.search(r"^(?:K|TK|C)-?\d+", name, re.I) or name.endswith("号") or "号艇" in ev_text:
        return "submarine_ship"
    if re.search(r"^(?:P|PCM|4K)-?\d+", name, re.I) or "导弹" in name:
        return "missile"
    if "鱼雷" in name:
        return "torpedo"
    if "声呐" in name or "噪声测向仪" in name:
        return "sonar"
    if "雷达" in name:
        return "radar"
    if "导航" in name or "天琴座" in name:
        return "navigation_system"
    if "解算" in name or "作战指挥" in name or "射击指挥" in name:
        return "command_system"
    if any(k in name for k in ["柴油机", "电机", "螺旋桨", "核动力", "蓄电池"]):
        return "propulsion_system"
    if "导弹系统" in name or "发射装置" in name or "发射筒" in name:
        return "weapon_system"
    if any(k in name for k in ["通信", "拖曳天线", "漂浮天线", "超长波"]):
        return "communication_system"
    if any(k in name for k in ["电子侦察", "无线电"]):
        return "electronic_system"
    if any(k in name for k in ["耐压艇体", "龟背", "8字", "品字"]):
        return "hull_structure"
    if "舱" in name:
        return "compartment"
    if name.endswith("级"):
        return "submarine_class"
    if re.search(r"\d{3,4}[A-Za-zА-Яа-яⅡI]*型$", name) or re.search(r"^[A-Z]-\d型$", name, re.I):
        return "submarine_model"
    return current

def normalize_relation(value: str) -> str:
    raw = unicodedata.normalize("NFKC", str(value or "")).strip()
    for key, normalized in RELATION_MAP.items():
        if key in raw:
            return normalized
    return normalize_text(raw)


def _entity_keys(entity: Entity) -> set[str]:
    values = {entity.name, *entity.aliases}
    return {normalize_text(v) for v in values if normalize_text(v)}


def _entities_match(left: Entity, right: Entity) -> bool:
    return bool(_entity_keys(left) & _entity_keys(right))


def _find_match(entity: Entity, candidates: list[Entity], used: set[int] | None = None) -> int | None:
    for idx, candidate in enumerate(candidates):
        if used is not None and idx in used:
            continue
        if _entities_match(entity, candidate):
            return idx
    return None


def _dedupe_relations(relations: list[Any]) -> list[Any]:
    seen: set[tuple[str, str]] = set()
    out = []
    for rel in relations:
        key = (normalize_relation(rel.relation), normalize_text(rel.target))
        if key in seen or not key[1]:
            continue
        seen.add(key)
        out.append(rel)
    return out


def _merge_entities(entities: list[Entity]) -> list[Entity]:
    merged: list[Entity] = []
    for entity in entities:
        if not entity.name.strip() or not entity.evidence:
            continue
        entity.name = _canonical_name(entity.name)
        entity.aliases = sorted({_canonical_name(a) for a in entity.aliases if a.strip() and _canonical_name(a) != entity.name})
        entity.type = _infer_entity_type(entity)
        entity.attributes = {
            unicodedata.normalize("NFKC", str(k)).strip(): unicodedata.normalize("NFKC", str(v)).strip()
            for k, v in entity.attributes.items()
            if str(k).strip() and str(v).strip()
        }
        match_idx = _find_match(entity, merged)
        if match_idx is None:
            entity.relations = _dedupe_relations(entity.relations)
            merged.append(entity)
            continue
        target = merged[match_idx]
        # 更具体类型覆盖 equipment；规则推断出的 ship/model/class 也覆盖明显错误类型。
        if target.type == "equipment" or entity.type != "equipment":
            target.type = _infer_entity_type(entity)
        target.aliases = sorted(set(target.aliases + entity.aliases + ([entity.name] if entity.name != target.name else [])))
        target.attributes.update(entity.attributes)
        target.relations = _dedupe_relations(target.relations + entity.relations)
        target.evidence.extend(entity.evidence)
        if not target.type_reason and entity.type_reason:
            target.type_reason = entity.type_reason
        if target.confidence is None:
            target.confidence = entity.confidence
    return merged

def _normalize_extractions(results: list[EntityExtraction]) -> list[EntityExtraction]:
    normalized = []
    for result in results:
        result.entities = _merge_entities(result.entities)
        normalized.append(result)
    return normalized


def _load_gold(path: str | None) -> dict[str, list[Entity]]:
    if not path:
        return {}
    gold_path = Path(path)
    if not gold_path.exists():
        return {}
    records: dict[str, list[Entity]] = {}
    with gold_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            records[str(item["chunk_id"])] = [_validate(Entity, e) for e in item.get("entities", [])]
    return records


def _attribute_items(entity: Entity) -> list[tuple[str, str, str]]:
    return [(entity.name, normalize_text(k), normalize_text(v)) for k, v in entity.attributes.items()]


def _relation_items(entity: Entity) -> list[tuple[str, str, str]]:
    return [(entity.name, normalize_relation(r.relation), normalize_text(r.target)) for r in entity.relations if r.target]


def _target_keys(target: str, entities: list[Entity]) -> set[str]:
    keys = {normalize_text(target)}
    for entity in entities:
        if normalize_text(target) in _entity_keys(entity):
            keys.update(_entity_keys(entity))
    return {key for key in keys if key}


def _score_pair(
    chunk_id: str,
    reference: list[Entity],
    prediction: list[Entity],
    reference_source: str,
    prediction_source: str = "model_b",
) -> PairEvaluation:
    errors: dict[str, list[dict[str, Any]]] = {
        "missing_entity": [],
        "extra_entity": [],
        "wrong_type": [],
        "missing_attribute": [],
        "wrong_attribute_value": [],
        "wrong_relation": [],
        "wrong_attachment": [],
    }

    entity_counts = MetricCounts()
    used_pred: set[int] = set()
    matches: dict[int, int] = {}
    for ref_idx, ref in enumerate(reference):
        pred_idx = _find_match(ref, prediction, used_pred)
        if pred_idx is None:
            entity_counts.fn += 1
            errors["missing_entity"].append({"name": ref.name, "type": ref.type})
            continue
        used_pred.add(pred_idx)
        matches[ref_idx] = pred_idx
        entity_counts.tp += 1
        if normalize_text(ref.type) != normalize_text(prediction[pred_idx].type):
            errors["wrong_type"].append(
                {"name": ref.name, "reference_type": ref.type, "prediction_type": prediction[pred_idx].type}
            )
    for idx, pred in enumerate(prediction):
        if idx not in used_pred:
            entity_counts.fp += 1
            errors["extra_entity"].append({"name": pred.name, "type": pred.type})

    attr_counts = MetricCounts()
    for ref_idx, pred_idx in matches.items():
        ref = reference[ref_idx]
        pred = prediction[pred_idx]
        pred_attrs = {normalize_text(k): normalize_text(v) for k, v in pred.attributes.items()}
        for key, value in ref.attributes.items():
            n_key = normalize_text(key)
            n_value = normalize_text(value)
            if n_key not in pred_attrs:
                attr_counts.fn += 1
                errors["missing_attribute"].append({"entity": ref.name, "attribute": key, "value": value})
            elif pred_attrs[n_key] == n_value:
                attr_counts.tp += 1
            else:
                attr_counts.fp += 1
                attr_counts.fn += 1
                errors["wrong_attribute_value"].append(
                    {
                        "entity": ref.name,
                        "attribute": key,
                        "reference_value": value,
                        "prediction_value": pred.attributes.get(key, ""),
                    }
                )
        ref_keys = {normalize_text(k) for k in ref.attributes}
        for key, value in pred.attributes.items():
            if normalize_text(key) not in ref_keys:
                attr_counts.fp += 1
                for other in reference:
                    if other is ref:
                        continue
                    other_values = {normalize_text(v) for v in other.attributes.values()}
                    if normalize_text(value) in other_values:
                        errors["wrong_attachment"].append(
                            {"predicted_entity": pred.name, "attribute": key, "value": value, "likely_entity": other.name}
                        )
                        break

    rel_counts = MetricCounts()
    pred_relations = []
    for pred in prediction:
        for rel in _relation_items(pred):
            pred_relations.append((pred, rel))
    used_rel: set[int] = set()
    for ref in reference:
        for _, ref_rel, ref_target in _relation_items(ref):
            ref_target_keys = _target_keys(ref_target, reference)
            found = None
            for idx, (pred_entity, (_, pred_rel, pred_target)) in enumerate(pred_relations):
                if idx in used_rel:
                    continue
                pred_target_keys = _target_keys(pred_target, prediction)
                if _entities_match(ref, pred_entity) and ref_rel == pred_rel and ref_target_keys & pred_target_keys:
                    found = idx
                    break
            if found is None:
                rel_counts.fn += 1
                errors["wrong_relation"].append({"subject": ref.name, "relation": ref_rel, "target": ref_target})
            else:
                used_rel.add(found)
                rel_counts.tp += 1
    rel_counts.fp += len(pred_relations) - len(used_rel)

    metrics = {}
    metrics.update(entity_counts.as_metrics("entity"))
    metrics.update(attr_counts.as_metrics("attribute"))
    metrics.update(rel_counts.as_metrics("relation"))
    return PairEvaluation(
        chunk_id=chunk_id,
        reference_source=reference_source,
        prediction_source=prediction_source,
        metrics=metrics,
        counts={
            "entity": entity_counts.model_dump() if hasattr(entity_counts, "model_dump") else entity_counts.dict(),
            "attribute": attr_counts.model_dump() if hasattr(attr_counts, "model_dump") else attr_counts.dict(),
            "relation": rel_counts.model_dump() if hasattr(rel_counts, "model_dump") else rel_counts.dict(),
        },
        errors=errors,
    )



def _aggregate_evaluations(
    evals: list[PairEvaluation],
    state: DistillNERState,
    config: DistillNERConfig,
    reference_source: str,
    prediction_source: str,
) -> SummaryReport:
    total = {"entity": MetricCounts(), "attribute": MetricCounts(), "relation": MetricCounts()}
    errors_by_type: dict[str, int] = {}
    chunk_errors: list[dict[str, Any]] = []
    for item in evals:
        for name in total:
            counts = item.counts.get(name, {})
            total[name].tp += int(counts.get("tp", 0))
            total[name].fp += int(counts.get("fp", 0))
            total[name].fn += int(counts.get("fn", 0))
        for error_type, rows in item.errors.items():
            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + len(rows)
            if rows:
                chunk_errors.append({"chunk_id": item.chunk_id, "error_type": error_type, "count": len(rows)})
    metrics: dict[str, float] = {}
    for name, counts in total.items():
        metrics.update(counts.as_metrics(name))
    return SummaryReport(
        input_md=state["input_md"],
        output_dir=state["output_dir"],
        model_a=config.model_a.name,
        model_b=config.model_b.name,
        reference_source=reference_source,
        prediction_source=prediction_source,
        chunk_count=len(state.get("chunks", [])),
        metrics=metrics,
        counts={k: (v.model_dump() if hasattr(v, "model_dump") else v.dict()) for k, v in total.items()},
        errors_by_type=errors_by_type,
        chunk_errors=chunk_errors,
    )


def _write_summary(path_json: Path, path_md: Path, summary: SummaryReport) -> None:
    path_json.write_text(json.dumps(dump_model(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Distill NER Summary",
        "",
        f"- input_md: {summary.input_md}",
        f"- model_a: {summary.model_a}",
        f"- model_b: {summary.model_b}",
        f"- reference_source: {summary.reference_source}",
        f"- prediction_source: {summary.prediction_source}",
        f"- chunk_count: {summary.chunk_count}",
        "",
        "## Metrics",
        "",
    ]
    for key, value in summary.metrics.items():
        lines.append(f"- {key}: {value:.6f}")
    lines.extend(["", "## Counts", ""])
    for name, counts in summary.counts.items():
        lines.append(f"- {name}: TP={counts.get('tp', 0)}, FP={counts.get('fp', 0)}, FN={counts.get('fn', 0)}")
    lines.extend(["", "## Errors", ""])
    for key, value in summary.errors_by_type.items():
        lines.append(f"- {key}: {value}")
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_gold_charts(out: Path, summary_a: SummaryReport | None, summary_b: SummaryReport | None) -> None:
    """Generate visual reports. If matplotlib is unavailable, skip charts without failing the workflow."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    summaries = [s for s in [summary_a, summary_b] if s is not None]
    if not summaries:
        return
    labels = [s.prediction_source for s in summaries]

    metric_names = ["entity_f1", "attribute_f1", "relation_f1"]
    x = range(len(metric_names))
    width = 0.35 if len(summaries) > 1 else 0.55
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, s in enumerate(summaries):
        offset = (i - (len(summaries) - 1) / 2) * width
        values = [s.metrics.get(m, 0.0) for m in metric_names]
        ax.bar([p + offset for p in x], values, width=width, label=labels[i])
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Entity F1", "Attribute F1", "Relation F1"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("Gold-based extraction F1")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out / "gold_f1_comparison.png", dpi=200)
    plt.close(fig)

    error_types = sorted(set().union(*(s.errors_by_type.keys() for s in summaries)))
    if error_types:
        x2 = range(len(error_types))
        fig, ax = plt.subplots(figsize=(max(9, len(error_types) * 1.0), 4.8))
        for i, s in enumerate(summaries):
            offset = (i - (len(summaries) - 1) / 2) * width
            values = [s.errors_by_type.get(e, 0) for e in error_types]
            ax.bar([p + offset for p in x2], values, width=width, label=labels[i])
        ax.set_xticks(list(x2))
        ax.set_xticklabels(error_types, rotation=35, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("Gold-based error counts")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(out / "gold_error_comparison.png", dpi=200)
        plt.close(fig)

    for summary in summaries:
        names = ["entity", "attribute", "relation"]
        tp = [summary.counts.get(n, {}).get("tp", 0) for n in names]
        fp = [summary.counts.get(n, {}).get("fp", 0) for n in names]
        fn = [summary.counts.get(n, {}).get("fn", 0) for n in names]
        fig, ax = plt.subplots(figsize=(8, 4.8))
        w = 0.25
        xs = range(len(names))
        ax.bar([p - w for p in xs], tp, width=w, label="TP")
        ax.bar(list(xs), fp, width=w, label="FP")
        ax.bar([p + w for p in xs], fn, width=w, label="FN")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(["Entity", "Attribute", "Relation"])
        ax.set_ylabel("Count")
        ax.set_title(f"TP/FP/FN: {summary.prediction_source}")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", summary.prediction_source)
        fig.savefig(out / f"gold_counts_{safe_name}.png", dpi=200)
        plt.close(fig)


def build_graph(config: DistillNERConfig, use_llm_eval: bool | None = None):
    use_llm_eval = use_llm_eval if use_llm_eval is not None else os.getenv("DISTILL_NER_USE_LLM_EVAL", "0") == "1"

    def load_markdown(state: DistillNERState) -> DistillNERState:
        try:
            return {"markdown_text": read_markdown(state["input_md"])}
        except Exception as exc:
            return {"markdown_text": "", "errors": state.get("errors", []) + [f"load_markdown: {exc}"]}

    def split_chunks(state: DistillNERState) -> DistillNERState:
        return {"chunks": split_markdown_chunks(state.get("markdown_text", ""), min_chars=config.min_chars, max_chars=config.max_chars)}

    def extract_model_a(state: DistillNERState) -> DistillNERState:
        client = OpenAICompatibleClient(config.model_a)
        results = [client.extract_entities(chunk) for chunk in tqdm(state.get("chunks", []), desc="model_a")]
        return {"model_a_results": results}

    def extract_model_b(state: DistillNERState) -> DistillNERState:
        client = OpenAICompatibleClient(config.model_b)
        results = [client.extract_entities(chunk) for chunk in tqdm(state.get("chunks", []), desc="model_b")]
        return {"model_b_results": results}

    def normalize_results(state: DistillNERState) -> DistillNERState:
        return {
            "normalized_a": _normalize_extractions(state.get("model_a_results", [])),
            "normalized_b": _normalize_extractions(state.get("model_b_results", [])),
        }

    def evaluate_pair(state: DistillNERState) -> DistillNERState:
        gold = _load_gold(config.gold_jsonl)
        by_b = {item.chunk_id: item for item in state.get("normalized_b", [])}
        by_a = {item.chunk_id: item for item in state.get("normalized_a", [])}
        chunks = {chunk.chunk_id: chunk for chunk in state.get("chunks", [])}
        llm_client = OpenAICompatibleClient(config.model_a) if use_llm_eval else None

        # Gold mode: evaluate both model_a and model_b against human/semi-human gold.
        if gold:
            evals_a: list[PairEvaluation] = []
            evals_b: list[PairEvaluation] = []
            chunk_ids = [chunk.chunk_id for chunk in state.get("chunks", [])]
            for chunk_id in chunk_ids:
                reference = gold.get(chunk_id, [])
                pred_a = by_a.get(chunk_id, EntityExtraction(chunk_id=chunk_id)).entities
                pred_b = by_b.get(chunk_id, EntityExtraction(chunk_id=chunk_id)).entities
                item_a = _score_pair(chunk_id, reference, pred_a, "gold", "model_a")
                item_b = _score_pair(chunk_id, reference, pred_b, "gold", "model_b")
                if llm_client and any(item_b.errors.values()):
                    judgement, error = llm_client.judge_evaluation(
                        chunks.get(chunk_id, TextChunk(chunk_id=chunk_id, text="")).text,
                        _json_dump([dump_model(e) for e in reference]),
                        _json_dump([dump_model(e) for e in pred_b]),
                    )
                    item_b.llm_judgement = judgement or {"error": error}
                evals_a.append(item_a)
                evals_b.append(item_b)
            return {"pair_evals": evals_b, "gold_evals_a": evals_a, "gold_evals_b": evals_b}

        # No gold: keep v2 behavior, using model_a as pseudo-gold to evaluate model_b consistency.
        evals: list[PairEvaluation] = []
        for chunk_id, pred in by_b.items():
            reference = by_a.get(chunk_id, EntityExtraction(chunk_id=chunk_id)).entities
            item = _score_pair(chunk_id, reference, pred.entities, "model_a_pseudo_gold", "model_b")
            if llm_client and any(item.errors.values()):
                judgement, error = llm_client.judge_evaluation(
                    chunks.get(chunk_id, TextChunk(chunk_id=chunk_id, text="")).text,
                    _json_dump([dump_model(e) for e in reference]),
                    _json_dump([dump_model(e) for e in pred.entities]),
                )
                item.llm_judgement = judgement or {"error": error}
            evals.append(item)
        return {"pair_evals": evals}

    def aggregate_report(state: DistillNERState) -> DistillNERState:
        gold_a = state.get("gold_evals_a", [])
        gold_b = state.get("gold_evals_b", [])
        if gold_a or gold_b:
            summary_a = _aggregate_evaluations(gold_a, state, config, "gold", "model_a") if gold_a else None
            summary_b = _aggregate_evaluations(gold_b, state, config, "gold", "model_b") if gold_b else None
            return {"summary": summary_b or summary_a, "summary_a": summary_a, "summary_b": summary_b}
        summary = _aggregate_evaluations(
            state.get("pair_evals", []),
            state,
            config,
            "model_a_pseudo_gold",
            "model_b",
        )
        return {"summary": summary}

    def save_outputs(state: DistillNERState) -> DistillNERState:
        out = Path(state["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out / "chunks.jsonl", state.get("chunks", []))
        _write_jsonl(out / "model_a_entities.jsonl", state.get("normalized_a", []))
        _write_jsonl(out / "model_b_entities.jsonl", state.get("normalized_b", []))
        _write_jsonl(out / "pair_eval.jsonl", state.get("pair_evals", []))
        if state.get("gold_evals_a"):
            _write_jsonl(out / "gold_eval_model_a.jsonl", state.get("gold_evals_a", []))
        if state.get("gold_evals_b"):
            _write_jsonl(out / "gold_eval_model_b.jsonl", state.get("gold_evals_b", []))

        summary = state.get("summary")
        summary_a = state.get("summary_a")
        summary_b = state.get("summary_b")
        if summary:
            _write_summary(out / "summary_report.json", out / "summary_report.md", summary)
        if summary_a:
            _write_summary(out / "summary_report_model_a.json", out / "summary_report_model_a.md", summary_a)
        if summary_b:
            _write_summary(out / "summary_report_model_b.json", out / "summary_report_model_b.md", summary_b)
        _make_gold_charts(out, summary_a, summary_b)
        return {}

    graph = StateGraph(DistillNERState)
    graph.add_node("load_markdown", load_markdown)
    graph.add_node("split_chunks", split_chunks)
    graph.add_node("extract_model_a", extract_model_a)
    graph.add_node("extract_model_b", extract_model_b)
    graph.add_node("normalize_results", normalize_results)
    graph.add_node("evaluate_pair", evaluate_pair)
    graph.add_node("aggregate_report", aggregate_report)
    graph.add_node("save_outputs", save_outputs)
    graph.add_edge(START, "load_markdown")
    graph.add_edge("load_markdown", "split_chunks")
    graph.add_edge("split_chunks", "extract_model_a")
    graph.add_edge("extract_model_a", "extract_model_b")
    graph.add_edge("extract_model_b", "normalize_results")
    graph.add_edge("normalize_results", "evaluate_pair")
    graph.add_edge("evaluate_pair", "aggregate_report")
    graph.add_edge("aggregate_report", "save_outputs")
    graph.add_edge("save_outputs", END)
    return graph.compile()
