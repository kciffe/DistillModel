from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from .config import ModelConfig
from .prompts import (
    EVALUATION_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    build_evaluation_user_prompt,
    build_extraction_user_prompt,
)
from .schema import Entity, EntityExtraction, TextChunk


def _model_validate(cls: Any, data: Any) -> Any:
    if hasattr(cls, "model_validate"):
        return cls.model_validate(data)
    return cls.parse_obj(data)


def _model_dump(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj.dict()


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    if not text:
        return None, "empty response"

    candidates = [text.strip()]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    candidates.extend(part.strip() for part in fenced)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, None
            if isinstance(parsed, list):
                return {"entities": parsed}, None
        except json.JSONDecodeError:
            continue
    return None, "failed to parse JSON object"


class OpenAICompatibleClient:
    def __init__(self, config: ModelConfig, timeout: float = 120.0):
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=timeout,
        )

    def extract_entities(self, chunk: TextChunk) -> EntityExtraction:
        raw = ""
        try:
            response = self.client.chat.completions.create(
                model=self.config.name,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_extraction_user_prompt(
                            chunk.chunk_id,
                            chunk.page_hint,
                            chunk.title_path,
                            chunk.text,
                        ),
                    },
                ],
                temperature=0,
            )
            raw = response.choices[0].message.content or ""
            parsed, error = parse_json_object(raw)
            if error or parsed is None:
                return EntityExtraction(
                    chunk_id=chunk.chunk_id,
                    page_hint=chunk.page_hint,
                    title_path=chunk.title_path,
                    raw_response=raw,
                    error=error,
                    model_name=self.config.name,
                )
            entities = []
            for item in parsed.get("entities", []):
                if not isinstance(item, dict):
                    continue
                try:
                    entity = _model_validate(Entity, item)
                    entities.append(entity)
                except Exception:
                    continue
            return EntityExtraction(
                chunk_id=chunk.chunk_id,
                page_hint=chunk.page_hint,
                title_path=chunk.title_path,
                entities=entities,
                raw_response=raw,
                model_name=self.config.name,
            )
        except Exception as exc:
            return EntityExtraction(
                chunk_id=chunk.chunk_id,
                page_hint=chunk.page_hint,
                title_path=chunk.title_path,
                raw_response=raw,
                error=f"{type(exc).__name__}: {exc}",
                model_name=self.config.name,
            )

    def judge_evaluation(self, chunk_text: str, reference_json: str, prediction_json: str) -> tuple[dict[str, Any] | None, str | None]:
        raw = ""
        try:
            response = self.client.chat.completions.create(
                model=self.config.name,
                messages=[
                    {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_evaluation_user_prompt(
                            chunk_text,
                            reference_json,
                            prediction_json,
                        ),
                    },
                ],
                temperature=0,
            )
            raw = response.choices[0].message.content or ""
            return parse_json_object(raw)
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"


def dump_model(obj: Any) -> dict[str, Any]:
    return _model_dump(obj)
