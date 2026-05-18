from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import json
import os


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        if (path / ".env").exists() or (path / "pyproject.toml").exists():
            return path
    return current


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _json_env(name: str) -> dict:
    value = os.getenv(name, "").strip()
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    api_key: str
    name: str
    provider: str = "openai"
    enable_thinking: bool = False
    extra_body: dict = field(default_factory=dict)
    temperature: float = 0.0
    max_tokens: int = 2048
    json_mode: bool = True


@dataclass(frozen=True)
class DistillNERConfig:
    model_a: ModelConfig
    model_b: ModelConfig
    input_md: str
    output_dir: str
    gold_jsonl: Optional[str] = None
    min_chars: int = 1200
    max_chars: int = 2600
    task_mode: str = "entity_only"
    reuse_model_a_output: bool = False
    reuse_model_b_output: bool = False
    model_a_cache_jsonl: str = ""
    model_b_cache_jsonl: str = ""


def load_config(
    input_md: str | None = None,
    output_dir: str | None = None,
    env_path: str | Path | None = None,
) -> DistillNERConfig:
    root = find_project_root()
    load_dotenv(dotenv_path=Path(env_path) if env_path else root / ".env")

    def required(name: str) -> str:
        value = os.getenv(name, "").strip()
        if not value:
            raise ValueError(f"Missing required environment variable: {name}")
        return value

    temperature = _float_env("DISTILL_NER_TEMPERATURE", 0.0)
    max_tokens = _int_env("DISTILL_NER_MAX_TOKENS", 2048)
    json_mode = _bool_env("DISTILL_NER_JSON_MODE", True)

    def model(prefix: str, default_provider: str) -> ModelConfig:
        return ModelConfig(
            base_url=required(f"{prefix}_BASE_URL"),
            api_key=required(f"{prefix}_API_KEY"),
            name=required(f"{prefix}_NAME"),
            provider=os.getenv(f"{prefix}_PROVIDER", default_provider).strip() or default_provider,
            enable_thinking=_bool_env(f"{prefix}_ENABLE_THINKING", False),
            extra_body=_json_env(f"{prefix}_EXTRA_BODY_JSON"),
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

    resolved_input = input_md or required("DISTILL_NER_INPUT_MD")
    resolved_output = output_dir or required("DISTILL_NER_OUTPUT_DIR")
    return DistillNERConfig(
        model_a=model("MODEL_A", "deepseek"),
        model_b=model("MODEL_B", "qwen_vllm"),
        input_md=resolved_input,
        output_dir=resolved_output,
        gold_jsonl=os.getenv("DISTILL_NER_GOLD_JSONL") or None,
        min_chars=_int_env("DISTILL_NER_MIN_CHARS", 1200),
        max_chars=_int_env("DISTILL_NER_MAX_CHARS", 2600),
        task_mode=os.getenv("DISTILL_NER_TASK_MODE", "entity_only").strip() or "entity_only",
        reuse_model_a_output=_bool_env("DISTILL_NER_REUSE_MODEL_A_OUTPUT", False),
        reuse_model_b_output=_bool_env("DISTILL_NER_REUSE_MODEL_B_OUTPUT", False),
        model_a_cache_jsonl=os.getenv("DISTILL_NER_MODEL_A_CACHE_JSONL", "").strip(),
        model_b_cache_jsonl=os.getenv("DISTILL_NER_MODEL_B_CACHE_JSONL", "").strip(),
    )
