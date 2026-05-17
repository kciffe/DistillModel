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
    summary: SummaryReport
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
    text = re.sub(r"\s+", "", text)
    text = text.replace("－", "-").replace("—", "-").replace("–", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"([A-Za-z])\s+([A-Za-z])", r"\1\2", text)
    text = re.sub(r"(\d)\s+([A-Za-z])", r"\1\2", text)
    text = re.sub(r"([A-Za-z])\s+(\d)", r"\1\2", text)
    return text.casefold()


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


def _merge_entities(entities: list[Entity]) -> list[Entity]:
    merged: dict[str, Entity] = {}
    for entity in entities:
        if not entity.name.strip() or not entity.evidence:
            continue
        entity.name = unicodedata.normalize("NFKC", entity.name).strip()
        entity.aliases = sorted({unicodedata.normalize("NFKC", a).strip() for a in entity.aliases if a.strip()})
        entity.attributes = {
            unicodedata.normalize("NFKC", str(k)).strip(): unicodedata.normalize("NFKC", str(v)).strip()
            for k, v in entity.attributes.items()
            if str(k).strip() and str(v).strip()
        }
        key = normalize_text(entity.name)
        if key not in merged:
            merged[key] = entity
            continue
        target = merged[key]
        if target.type == "equipment" and entity.type != "equipment":
            target.type = entity.type
        target.aliases = sorted(set(target.aliases + entity.aliases))
        target.attributes.update(entity.attributes)
        target.relations.extend(entity.relations)
        target.evidence.extend(entity.evidence)
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
        metrics=metrics,
        counts={
            "entity": entity_counts.model_dump() if hasattr(entity_counts, "model_dump") else entity_counts.dict(),
            "attribute": attr_counts.model_dump() if hasattr(attr_counts, "model_dump") else attr_counts.dict(),
            "relation": rel_counts.model_dump() if hasattr(rel_counts, "model_dump") else rel_counts.dict(),
        },
        errors=errors,
    )


def build_graph(config: DistillNERConfig, use_llm_eval: bool | None = None):
    use_llm_eval = use_llm_eval if use_llm_eval is not None else os.getenv("DISTILL_NER_USE_LLM_EVAL", "0") == "1"

    def load_markdown(state: DistillNERState) -> DistillNERState:
        try:
            return {"markdown_text": read_markdown(state["input_md"])}
        except Exception as exc:
            return {"markdown_text": "", "errors": state.get("errors", []) + [f"load_markdown: {exc}"]}

    def split_chunks(state: DistillNERState) -> DistillNERState:
        return {"chunks": split_markdown_chunks(state.get("markdown_text", ""))}

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
        reference_source = "gold" if gold else "model_a_pseudo_gold"
        by_b = {item.chunk_id: item for item in state.get("normalized_b", [])}
        by_a = {item.chunk_id: item for item in state.get("normalized_a", [])}
        chunks = {chunk.chunk_id: chunk for chunk in state.get("chunks", [])}
        evals: list[PairEvaluation] = []
        llm_client = OpenAICompatibleClient(config.model_a) if use_llm_eval else None
        for chunk_id, pred in by_b.items():
            reference = gold.get(chunk_id) or by_a.get(chunk_id, EntityExtraction(chunk_id=chunk_id)).entities
            item = _score_pair(chunk_id, reference, pred.entities, reference_source)
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
        total = {"entity": MetricCounts(), "attribute": MetricCounts(), "relation": MetricCounts()}
        errors_by_type: dict[str, int] = {}
        chunk_errors: list[dict[str, Any]] = []
        for item in state.get("pair_evals", []):
            for name in total:
                counts = item.counts.get(name, {})
                total[name].tp += int(counts.get("tp", 0))
                total[name].fp += int(counts.get("fp", 0))
                total[name].fn += int(counts.get("fn", 0))
            for error_type, rows in item.errors.items():
                errors_by_type[error_type] = errors_by_type.get(error_type, 0) + len(rows)
                if rows:
                    chunk_errors.append({"chunk_id": item.chunk_id, "error_type": error_type, "count": len(rows)})
        metrics = {}
        for name, counts in total.items():
            metrics.update(counts.as_metrics(name))
        reference_source = state.get("pair_evals", [PairEvaluation(chunk_id="")])[0].reference_source if state.get("pair_evals") else "model_a_pseudo_gold"
        summary = SummaryReport(
            input_md=state["input_md"],
            output_dir=state["output_dir"],
            model_a=config.model_a.name,
            model_b=config.model_b.name,
            reference_source=reference_source,
            chunk_count=len(state.get("chunks", [])),
            metrics=metrics,
            counts={k: (v.model_dump() if hasattr(v, "model_dump") else v.dict()) for k, v in total.items()},
            errors_by_type=errors_by_type,
            chunk_errors=chunk_errors,
        )
        return {"summary": summary}

    def save_outputs(state: DistillNERState) -> DistillNERState:
        out = Path(state["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out / "chunks.jsonl", state.get("chunks", []))
        _write_jsonl(out / "model_a_entities.jsonl", state.get("normalized_a", []))
        _write_jsonl(out / "model_b_entities.jsonl", state.get("normalized_b", []))
        _write_jsonl(out / "pair_eval.jsonl", state.get("pair_evals", []))
        summary = state.get("summary")
        if summary:
            (out / "summary_report.json").write_text(
                json.dumps(dump_model(summary), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            lines = [
                "# Distill NER Summary",
                "",
                f"- input_md: {summary.input_md}",
                f"- model_a: {summary.model_a}",
                f"- model_b: {summary.model_b}",
                f"- reference_source: {summary.reference_source}",
                f"- chunk_count: {summary.chunk_count}",
                "",
                "## Metrics",
                "",
            ]
            for key, value in summary.metrics.items():
                lines.append(f"- {key}: {value:.6f}")
            lines.extend(["", "## Errors", ""])
            for key, value in summary.errors_by_type.items():
                lines.append(f"- {key}: {value}")
            (out / "summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
