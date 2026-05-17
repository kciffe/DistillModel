from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


EntityType = Literal[
    "submarine_class",
    "submarine_model",
    "submarine_ship",
    "missile",
    "torpedo",
    "sonar",
    "radar",
    "navigation_system",
    "command_system",
    "propulsion_system",
    "compartment",
    "weapon_system",
    "communication_system",
    "electronic_system",
    "hull_structure",
    "equipment",
]


class Evidence(BaseModel):
    page: str = ""
    text: str = ""


class Relation(BaseModel):
    relation: str = ""
    target: str = ""


class Entity(BaseModel):
    name: str = ""
    type: EntityType | str = "equipment"
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)
    relations: list[Relation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    # 这两个字段主要用于调试和人工复核；不会参与实体匹配。
    type_reason: str = ""
    confidence: Optional[float] = None


class EntityExtraction(BaseModel):
    chunk_id: str
    page_hint: str = ""
    title_path: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    raw_response: str = ""
    error: Optional[str] = None
    model_name: str = ""


class TextChunk(BaseModel):
    chunk_id: str
    page_hint: str = ""
    title_path: list[str] = Field(default_factory=list)
    text: str


class MetricCounts(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_metrics(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_precision": round(self.precision, 6),
            f"{prefix}_recall": round(self.recall, 6),
            f"{prefix}_f1": round(self.f1, 6),
        }


class PairEvaluation(BaseModel):
    chunk_id: str
    reference_source: str = "model_a_pseudo_gold"
    prediction_source: str = "model_b"
    metrics: dict[str, float] = Field(default_factory=dict)
    counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    errors: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    llm_judgement: dict[str, Any] | None = None


class SummaryReport(BaseModel):
    input_md: str
    output_dir: str
    model_a: str
    model_b: str
    reference_source: str
    prediction_source: str = "model_b"
    chunk_count: int
    metrics: dict[str, float]
    counts: dict[str, dict[str, int]]
    errors_by_type: dict[str, int]
    chunk_errors: list[dict[str, Any]] = Field(default_factory=list)


class WorkflowState(BaseModel):
    input_md: str
    output_dir: str
    markdown_text: str = ""
    chunks: list[TextChunk] = Field(default_factory=list)
    model_a_results: list[EntityExtraction] = Field(default_factory=list)
    model_b_results: list[EntityExtraction] = Field(default_factory=list)
    normalized_a: list[EntityExtraction] = Field(default_factory=list)
    normalized_b: list[EntityExtraction] = Field(default_factory=list)
    pair_evals: list[PairEvaluation] = Field(default_factory=list)
    gold_evals_a: list[PairEvaluation] = Field(default_factory=list)
    gold_evals_b: list[PairEvaluation] = Field(default_factory=list)
    summary: SummaryReport | None = None
    summary_a: SummaryReport | None = None
    summary_b: SummaryReport | None = None
    errors: list[str] = Field(default_factory=list)
