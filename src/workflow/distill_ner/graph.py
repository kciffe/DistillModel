from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import DistillNERConfig
from .md_loader import load_markdown as read_markdown
from .md_loader import split_markdown_chunks
from .model_client import OpenAICompatibleClient, dump_model
from .schema import Entity, EntityEvalCounts, EntityEvaluation, EntityExtraction, EntitySummaryReport, TextChunk

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
    eval_a: list[EntityEvaluation]
    eval_b: list[EntityEvaluation]
    summary_a: EntitySummaryReport
    summary_b: EntitySummaryReport
    summary_compare: dict[str, Any]
    errors: list[str]


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


def normalize_entity_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = text.strip("《》<>「」『』“”\"'（）()[]【】")
    text = text.replace("－", "-").replace("—", "-").replace("–", "-")
    text = text.replace("\\Phi", "Φ").replace("Ф", "Φ").replace("phi", "Φ")
    text = re.sub(r"\s*[-]\s*", "-", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"P-(\d+)Φ\s*M", r"P-\1ΦM", text, flags=re.I)
    text = re.sub(r"P-(\d+)\s*Φ\s*M", r"P-\1ΦM", text, flags=re.I)
    text = re.sub(r"(\d+)\s*([A-Za-zА-Яа-я]+)\s*型", r"\1\2型", text)
    text = re.sub(r"TK\s*-\s*(\d+)", r"TK-\1", text, flags=re.I)
    text = re.sub(r"K\s*-\s*(\d+)", r"K-\1", text, flags=re.I)
    return text.casefold()


def _display_clean(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def infer_entity_type(entity: Entity) -> str:
    name = _display_clean(entity.name)
    norm = normalize_entity_name(name)
    raw = f"{name} {' '.join(entity.aliases)}" 

    if re.fullmatch(r"(?:k|tk|c)-\d+", norm) or re.search(r"号$", name):
        return "submarine_ship"
    if re.fullmatch(r"[ghyd]-\d型", norm):
        return "submarine_model"
    if re.fullmatch(r".+级", name):
        return "submarine_class"
    if re.fullmatch(r"\d+[a-zа-я]*型", norm):
        return "submarine_model"
    if re.match(r"^(p|pcm)-\d+", norm) or re.match(r"^4k-\d+", norm) or "导弹" in raw:
        return "missile"
    if "鱼雷" in raw:
        return "torpedo"
    if "声呐" in raw or "噪声测向" in raw:
        return "sonar"
    if "雷达" in raw:
        return "radar"
    if "导航" in raw:
        return "navigation_system"
    if any(k in raw for k in ["解算", "作战指挥", "控制系统", "射击指挥"]):
        return "command_system"
    if any(k in raw for k in ["柴油机", "电机", "推进", "螺旋桨", "蓄电池", "动力"]):
        return "propulsion_system"
    if "舱" in raw:
        return "compartment"
    if any(k in raw for k in ["通信", "天线", "电台"]):
        return "communication_system"
    if any(k in raw for k in ["电子侦察", "无线电"]):
        return "electronic_system"
    if any(k in raw for k in ["耐压艇体", "龟背", "8字", "品字", "双壳体"]):
        return "hull_structure"
    if any(k in raw for k in ["发射筒", "发射装置", "武器系统"]):
        return "weapon_system"
    return str(entity.type or "equipment")


def _entity_keys(entity: Entity) -> set[str]:
    values = {entity.name, *entity.aliases}
    return {normalize_entity_name(v) for v in values if normalize_entity_name(v)}


def _entities_match(left: Entity, right: Entity) -> bool:
    return bool(_entity_keys(left) & _entity_keys(right))


def _find_match(entity: Entity, candidates: list[Entity], used: set[int] | None = None) -> int | None:
    for idx, candidate in enumerate(candidates):
        if used is not None and idx in used:
            continue
        if _entities_match(entity, candidate):
            return idx
    return None


def _normalize_entity(entity: Entity) -> Entity:
    entity.name = _display_clean(entity.name)
    entity.aliases = sorted({_display_clean(a) for a in entity.aliases if _display_clean(a)})
    entity.type = infer_entity_type(entity)
    # 当前评测只看实体，保留但不使用 attributes/relations
    return entity


def _merge_entities(entities: list[Entity]) -> list[Entity]:
    merged: dict[str, Entity] = {}
    for entity in entities:
        if not entity.name.strip():
            continue
        entity = _normalize_entity(entity)
        key = normalize_entity_name(entity.name)
        if not key:
            continue
        if key not in merged:
            merged[key] = entity
            continue
        target = merged[key]
        if target.type == "equipment" and entity.type != "equipment":
            target.type = entity.type
        target.aliases = sorted(set(target.aliases + entity.aliases))
        target.evidence.extend(entity.evidence)
        if not target.parent and entity.parent:
            target.parent = entity.parent
        if target.level is None and entity.level is not None:
            target.level = entity.level
    return list(merged.values())


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
            records[str(item["chunk_id"])] = _merge_entities([_validate(Entity, e) for e in item.get("entities", [])])
    return records


def _score_entities(chunk_id: str, model_name: str, reference: list[Entity], prediction: list[Entity], parse_error: str | None = None) -> EntityEvaluation:
    errors: dict[str, list[dict[str, Any]]] = {
        "missing_entity": [],
        "extra_entity": [],
        "wrong_type": [],
        "parse_error": [],
    }
    counts = EntityEvalCounts()
    if parse_error:
        counts.parse_error_count = 1
        errors["parse_error"].append({"error": parse_error})

    used_pred: set[int] = set()
    for ref in reference:
        pred_idx = _find_match(ref, prediction, used_pred)
        if pred_idx is None:
            counts.fn += 1
            errors["missing_entity"].append({"name": ref.name, "type": ref.type})
            continue
        pred = prediction[pred_idx]
        used_pred.add(pred_idx)
        counts.tp += 1
        if normalize_entity_name(str(ref.type)) != normalize_entity_name(str(pred.type)):
            counts.wrong_type += 1
            errors["wrong_type"].append({
                "name": ref.name,
                "reference_type": ref.type,
                "prediction_type": pred.type,
            })

    for idx, pred in enumerate(prediction):
        if idx not in used_pred:
            counts.fp += 1
            errors["extra_entity"].append({"name": pred.name, "type": pred.type})

    return EntityEvaluation(
        chunk_id=chunk_id,
        model_name=model_name,
        metrics=counts.metrics(),
        counts={
            "tp": counts.tp,
            "fp": counts.fp,
            "fn": counts.fn,
            "wrong_type": counts.wrong_type,
            "parse_error_count": counts.parse_error_count,
        },
        errors=errors,
    )


def _aggregate(model_name: str, input_md: str, output_dir: str, gold_jsonl: str | None, chunk_count: int, evals: list[EntityEvaluation]) -> EntitySummaryReport:
    counts = EntityEvalCounts()
    errors_by_type: dict[str, int] = {}
    chunk_errors: list[dict[str, Any]] = []
    for item in evals:
        counts.tp += int(item.counts.get("tp", 0))
        counts.fp += int(item.counts.get("fp", 0))
        counts.fn += int(item.counts.get("fn", 0))
        counts.wrong_type += int(item.counts.get("wrong_type", 0))
        counts.parse_error_count += int(item.counts.get("parse_error_count", 0))
        for error_type, rows in item.errors.items():
            n = len(rows)
            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + n
            if n:
                chunk_errors.append({"chunk_id": item.chunk_id, "error_type": error_type, "count": n})
    return EntitySummaryReport(
        input_md=input_md,
        output_dir=output_dir,
        gold_jsonl=gold_jsonl or "",
        model_name=model_name,
        prediction_source="model_output",
        chunk_count=chunk_count,
        metrics=counts.metrics(),
        counts={
            "tp": counts.tp,
            "fp": counts.fp,
            "fn": counts.fn,
            "wrong_type": counts.wrong_type,
            "parse_error_count": counts.parse_error_count,
        },
        errors_by_type=errors_by_type,
        chunk_errors=chunk_errors,
    )


def _read_extractions(path: str | Path) -> list[EntityExtraction]:
    rows: list[EntityExtraction] = []
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"历史模型输出不存在: {cache_path}")
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(_validate(EntityExtraction, json.loads(line)))
    return rows


def _cache_path(output_dir: str, configured_path: str | None, default_name: str) -> Path:
    return Path(configured_path) if configured_path else Path(output_dir) / default_name


def _write_charts(out: Path, summary_a: EntitySummaryReport, summary_b: EntitySummaryReport) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    models = [summary_a.model_name, summary_b.model_name]
    f1 = [summary_a.metrics["entity_f1"], summary_b.metrics["entity_f1"]]
    precision = [summary_a.metrics["entity_precision"], summary_b.metrics["entity_precision"]]
    recall = [summary_a.metrics["entity_recall"], summary_b.metrics["entity_recall"]]

    plt.figure(figsize=(8, 5))
    plt.bar(models, f1)
    plt.ylabel("Entity F1")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out / "entity_f1_comparison.png", dpi=200)
    plt.close()

    x = range(len(models))
    plt.figure(figsize=(8, 5))
    plt.bar([i - 0.15 for i in x], precision, width=0.3, label="Precision")
    plt.bar([i + 0.15 for i in x], recall, width=0.3, label="Recall")
    plt.xticks(list(x), models, rotation=15, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "entity_precision_recall_comparison.png", dpi=200)
    plt.close()

    labels = ["tp", "fp", "fn"]
    a_vals = [summary_a.counts[k] for k in labels]
    b_vals = [summary_b.counts[k] for k in labels]
    x = range(len(labels))
    plt.figure(figsize=(8, 5))
    plt.bar([i - 0.15 for i in x], a_vals, width=0.3, label="Model A")
    plt.bar([i + 0.15 for i in x], b_vals, width=0.3, label="Model B")
    plt.xticks(list(x), labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "entity_tp_fp_fn_comparison.png", dpi=200)
    plt.close()


def build_graph(config: DistillNERConfig, use_llm_eval: bool = False):
    def load_markdown(state: DistillNERState) -> DistillNERState:
        return {"markdown_text": read_markdown(state["input_md"])}

    def split_chunks(state: DistillNERState) -> DistillNERState:
        chunks = split_markdown_chunks(state["markdown_text"], min_chars=config.min_chars, max_chars=config.max_chars)
        return {"chunks": chunks}

    def extract_model_a(state: DistillNERState) -> DistillNERState:
        if config.reuse_model_a_output:
            path = _cache_path(state["output_dir"], config.model_a_cache_jsonl, "model_a_entities.jsonl")
            print(f"reuse model_a entity extraction: {path}")
            return {"model_a_results": _read_extractions(path)}
        client = OpenAICompatibleClient(config.model_a)
        results = [client.extract_entities(chunk) for chunk in tqdm(state.get("chunks", []), desc="model_a entity extraction")]
        return {"model_a_results": results}

    def extract_model_b(state: DistillNERState) -> DistillNERState:
        if config.reuse_model_b_output:
            path = _cache_path(state["output_dir"], config.model_b_cache_jsonl, "model_b_entities.jsonl")
            print(f"reuse model_b entity extraction: {path}")
            return {"model_b_results": _read_extractions(path)}
        client = OpenAICompatibleClient(config.model_b)
        results = [client.extract_entities(chunk) for chunk in tqdm(state.get("chunks", []), desc="model_b entity extraction")]
        return {"model_b_results": results}

    def normalize_results(state: DistillNERState) -> DistillNERState:
        return {
            "normalized_a": _normalize_extractions(state.get("model_a_results", [])),
            "normalized_b": _normalize_extractions(state.get("model_b_results", [])),
        }

    def evaluate_entities(state: DistillNERState) -> DistillNERState:
        gold = _load_gold(config.gold_jsonl)
        if not gold:
            return {"errors": ["No gold file found. Entity evaluation requires DISTILL_NER_GOLD_JSONL."]}
        a_by_chunk = {r.chunk_id: r for r in state.get("normalized_a", [])}
        b_by_chunk = {r.chunk_id: r for r in state.get("normalized_b", [])}
        eval_a: list[EntityEvaluation] = []
        eval_b: list[EntityEvaluation] = []
        for chunk in state.get("chunks", []):
            ref = gold.get(chunk.chunk_id, [])
            a = a_by_chunk.get(chunk.chunk_id, EntityExtraction(chunk_id=chunk.chunk_id, model_name=config.model_a.name))
            b = b_by_chunk.get(chunk.chunk_id, EntityExtraction(chunk_id=chunk.chunk_id, model_name=config.model_b.name))
            eval_a.append(_score_entities(chunk.chunk_id, config.model_a.name, ref, a.entities, a.error))
            eval_b.append(_score_entities(chunk.chunk_id, config.model_b.name, ref, b.entities, b.error))
        return {"eval_a": eval_a, "eval_b": eval_b}

    def aggregate_report(state: DistillNERState) -> DistillNERState:
        summary_a = _aggregate(config.model_a.name, state["input_md"], state["output_dir"], config.gold_jsonl, len(state.get("chunks", [])), state.get("eval_a", []))
        summary_b = _aggregate(config.model_b.name, state["input_md"], state["output_dir"], config.gold_jsonl, len(state.get("chunks", [])), state.get("eval_b", []))
        compare = {
            "task": "entity_only_equipment_ner",
            "input_md": state["input_md"],
            "gold_jsonl": config.gold_jsonl or "",
            "chunk_count": len(state.get("chunks", [])),
            "model_a": {"name": config.model_a.name, "metrics": summary_a.metrics, "counts": summary_a.counts},
            "model_b": {"name": config.model_b.name, "metrics": summary_b.metrics, "counts": summary_b.counts},
            "f1_gap_model_a_minus_model_b": round(summary_a.metrics["entity_f1"] - summary_b.metrics["entity_f1"], 6),
            "recall_gap_model_a_minus_model_b": round(summary_a.metrics["entity_recall"] - summary_b.metrics["entity_recall"], 6),
        }
        return {"summary_a": summary_a, "summary_b": summary_b, "summary_compare": compare}

    def save_outputs(state: DistillNERState) -> DistillNERState:
        out = Path(state["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out / "chunks.jsonl", state.get("chunks", []))
        _write_jsonl(out / "model_a_entities.jsonl", state.get("normalized_a", []))
        _write_jsonl(out / "model_b_entities.jsonl", state.get("normalized_b", []))
        _write_jsonl(out / "entity_eval_model_a.jsonl", state.get("eval_a", []))
        _write_jsonl(out / "entity_eval_model_b.jsonl", state.get("eval_b", []))

        parse_a = [dump_model(r) for r in state.get("normalized_a", []) if r.error]
        parse_b = [dump_model(r) for r in state.get("normalized_b", []) if r.error]
        _write_jsonl(out / "parse_errors_model_a.jsonl", parse_a)
        _write_jsonl(out / "parse_errors_model_b.jsonl", parse_b)

        summary_a = state.get("summary_a")
        summary_b = state.get("summary_b")
        compare = state.get("summary_compare", {})
        if summary_a:
            (out / "entity_summary_model_a.json").write_text(json.dumps(dump_model(summary_a), ensure_ascii=False, indent=2), encoding="utf-8")
        if summary_b:
            (out / "entity_summary_model_b.json").write_text(json.dumps(dump_model(summary_b), ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "entity_summary_compare.json").write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = [
            "# Entity-only Equipment NER Evaluation",
            "",
            "本轮只评测装备实体识别能力，不评测 attributes、relations 或多级嵌套关系正确率。",
            "",
            f"- input_md: {state['input_md']}",
            f"- gold_jsonl: {config.gold_jsonl or ''}",
            f"- chunk_count: {len(state.get('chunks', []))}",
            "",
            "## Metrics",
            "",
            "| Model | Precision | Recall | F1 | Type Accuracy | TP | FP | FN | Wrong Type | Parse Errors |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for label, summary in [("Model A", summary_a), ("Model B", summary_b)]:
            if summary:
                lines.append(
                    f"| {label}: {summary.model_name} | {summary.metrics['entity_precision']:.4f} | "
                    f"{summary.metrics['entity_recall']:.4f} | {summary.metrics['entity_f1']:.4f} | "
                    f"{summary.metrics['type_accuracy']:.4f} | {summary.counts['tp']} | {summary.counts['fp']} | "
                    f"{summary.counts['fn']} | {summary.counts['wrong_type']} | {summary.counts['parse_error_count']} |"
                )
        lines.extend([
            "",
            "## Notes",
            "",
            "- Precision 低说明误抽较多。",
            "- Recall 低说明漏抽较多。",
            "- Type Accuracy 只在实体名称命中的样本上计算类型是否正确。",
            "- 该版本用于比较满血模型与 30B 模型的实体识别能力差异。",
        ])
        (out / "entity_summary_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        if summary_a and summary_b:
            _write_charts(out, summary_a, summary_b)
        return {}

    graph = StateGraph(DistillNERState)
    graph.add_node("load_markdown", load_markdown)
    graph.add_node("split_chunks", split_chunks)
    graph.add_node("extract_model_a", extract_model_a)
    graph.add_node("extract_model_b", extract_model_b)
    graph.add_node("normalize_results", normalize_results)
    graph.add_node("evaluate_entities", evaluate_entities)
    graph.add_node("aggregate_report", aggregate_report)
    graph.add_node("save_outputs", save_outputs)
    graph.add_edge(START, "load_markdown")
    graph.add_edge("load_markdown", "split_chunks")
    graph.add_edge("split_chunks", "extract_model_a")
    graph.add_edge("extract_model_a", "extract_model_b")
    graph.add_edge("extract_model_b", "normalize_results")
    graph.add_edge("normalize_results", "evaluate_entities")
    graph.add_edge("evaluate_entities", "aggregate_report")
    graph.add_edge("aggregate_report", "save_outputs")
    graph.add_edge("save_outputs", END)
    return graph.compile()
