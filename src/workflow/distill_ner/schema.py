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
    evidence: list[Evidence] = Field(default_factory=list)

    # 用于保留嵌套线索，但当前实体识别评测不强制计分
    parent: str | None = None
    level: int | None = None
    type_reason: str = ""
    confidence: float | None = None

    # 兼容旧 gold / 旧模型输出；当前 entity-only 评测直接忽略
    attributes: dict[str, str] = Field(default_factory=dict)
    relations: list[Relation] = Field(default_factory=list)


class EntityExtraction(BaseModel):
    chunk_id: str
    page_hint: str = ""
    title_path: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    raw_response: str = ""
    error: Optional[str] = None
    model_name: str = ""
    finish_reason: str = ""
    retry_count: int = 0
    json_mode_fallback: bool = False


class TextChunk(BaseModel):
    chunk_id: str
    page_hint: str = ""
    title_path: list[str] = Field(default_factory=list)
    text: str


class EntityEvalCounts(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0
    wrong_type: int = 0
    parse_error_count: int = 0

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

    @property
    def type_accuracy(self) -> float:
        return (self.tp - self.wrong_type) / self.tp if self.tp else 0.0

    def metrics(self) -> dict[str, float]:
        return {
            "entity_precision": round(self.precision, 6),
            "entity_recall": round(self.recall, 6),
            "entity_f1": round(self.f1, 6),
            "type_accuracy": round(self.type_accuracy, 6),
        }


class EntityEvaluation(BaseModel):
    chunk_id: str
    model_name: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    errors: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class EntitySummaryReport(BaseModel):
    input_md: str
    output_dir: str
    gold_jsonl: str = ""
    model_name: str
    prediction_source: str
    chunk_count: int
    metrics: dict[str, float]
    counts: dict[str, int]
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
    eval_a: list[EntityEvaluation] = Field(default_factory=list)
    eval_b: list[EntityEvaluation] = Field(default_factory=list)
    summary_a: EntitySummaryReport | None = None
    summary_b: EntitySummaryReport | None = None
    summary_compare: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
