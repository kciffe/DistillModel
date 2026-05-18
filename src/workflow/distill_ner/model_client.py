from __future__ import annotations

import copy
import json
from typing import Any

from openai import OpenAI

from .config import ModelConfig
from .prompts import ENTITY_ONLY_SYSTEM_PROMPT, build_extraction_user_prompt
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
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "top-level JSON must be an object"
    if "entities" not in parsed or not isinstance(parsed["entities"], list):
        return None, "JSON object must contain an entities list"
    return parsed, None


class OpenAICompatibleClient:
    def __init__(self, config: ModelConfig, timeout: float = 180.0):
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=timeout,
        )

    def _build_kwargs(self, messages: list[dict[str, str]], use_json_mode: bool = True) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.config.name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if use_json_mode and self.config.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        extra_body = copy.deepcopy(self.config.extra_body or {})
        provider = (self.config.provider or "openai").lower()

        # openai-python 的 extra_body 会作为请求 JSON 的额外顶层字段发送。
        # DeepSeek 的 thinking 需要进入请求体顶层，不能写进 prompt。
        if provider == "deepseek":
            if not self.config.enable_thinking:
                extra_body["thinking"] = {"type": "disabled"}
        elif provider == "qwen_vllm":
            if not self.config.enable_thinking:
                chat_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
                chat_kwargs["enable_thinking"] = False
                extra_body["chat_template_kwargs"] = chat_kwargs

        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _chat(self, messages: list[dict[str, str]]) -> tuple[str, str, bool]:
        kwargs = self._build_kwargs(messages, use_json_mode=True)
        json_mode_fallback = False
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            # 本地 vLLM 可能不支持 response_format，删除后重试一次。
            if "response_format" in kwargs and "response_format" in str(exc):
                kwargs.pop("response_format", None)
                json_mode_fallback = True
                response = self.client.chat.completions.create(**kwargs)
            else:
                raise
        choice = response.choices[0]
        raw = self._message_text(choice.message)
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
        return raw, finish_reason, json_mode_fallback

    @staticmethod
    def _message_text(message: Any) -> str:
        content = getattr(message, "content", None)
        if content:
            return str(content)
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            return str(reasoning_content)
        model_extra = getattr(message, "model_extra", None)
        if isinstance(model_extra, dict) and model_extra.get("reasoning_content"):
            return str(model_extra["reasoning_content"])
        if isinstance(message, dict):
            return str(message.get("content") or message.get("reasoning_content") or "")
        return ""

    def extract_entities(self, chunk: TextChunk) -> EntityExtraction:
        raw = ""
        finish_reason = ""
        json_mode_fallback = False
        try:
            messages = [
                {"role": "system", "content": ENTITY_ONLY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_extraction_user_prompt(
                        chunk.chunk_id,
                        chunk.page_hint,
                        chunk.title_path,
                        chunk.text,
                    ),
                },
            ]
            raw, finish_reason, json_mode_fallback = self._chat(messages)
            parsed, error = parse_json_object(raw)
            if error or parsed is None:
                return EntityExtraction(
                    chunk_id=chunk.chunk_id,
                    page_hint=chunk.page_hint,
                    title_path=chunk.title_path,
                    raw_response=raw,
                    error=error,
                    model_name=self.config.name,
                    finish_reason=finish_reason,
                    json_mode_fallback=json_mode_fallback,
                )
            entities: list[Entity] = []
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
                finish_reason=finish_reason,
                json_mode_fallback=json_mode_fallback,
            )
        except Exception as exc:
            return EntityExtraction(
                chunk_id=chunk.chunk_id,
                page_hint=chunk.page_hint,
                title_path=chunk.title_path,
                raw_response=raw,
                error=f"{type(exc).__name__}: {exc}",
                model_name=self.config.name,
                finish_reason=finish_reason,
                json_mode_fallback=json_mode_fallback,
            )


def dump_model(obj: Any) -> dict[str, Any]:
    return _model_dump(obj)
