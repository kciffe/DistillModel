from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
import json
import os


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        if (path / ".env").exists() or (path / "pyproject.toml").exists():
            return path
    return current


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    api_key: str
    name: str
    temperature: float = 0.0
    max_tokens: int = 4096
    json_mode: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DistillNERConfig:
    model_a: ModelConfig
    model_b: ModelConfig
    input_md: str
    output_dir: str
    gold_jsonl: Optional[str] = None
    min_chars: int = 1200
    max_chars: int = 2600


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_json(name: str) -> dict[str, Any]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _build_extra_body(prefix: str) -> dict[str, Any]:
    extra = _env_json(f"{prefix}_EXTRA_BODY_JSON")
    # Qwen3 在 vLLM/OpenAI-compatible 服务中常用这个开关关闭 thinking，减少无关输出并提升 JSON 稳定性。
    enable_thinking = _env_bool("DISTILL_NER_ENABLE_THINKING", False)
    if "chat_template_kwargs" not in extra:
        extra["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    elif isinstance(extra["chat_template_kwargs"], dict):
        extra["chat_template_kwargs"].setdefault("enable_thinking", enable_thinking)
    return extra


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

    temperature = _env_float("DISTILL_NER_TEMPERATURE", 0.0)
    max_tokens = _env_int("DISTILL_NER_MAX_TOKENS", 4096)
    json_mode = _env_bool("DISTILL_NER_JSON_MODE", True)

    model_a = ModelConfig(
        base_url=required("MODEL_A_BASE_URL"),
        api_key=required("MODEL_A_API_KEY"),
        name=required("MODEL_A_NAME"),
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        extra_body=_build_extra_body("MODEL_A"),
    )
    model_b = ModelConfig(
        base_url=required("MODEL_B_BASE_URL"),
        api_key=required("MODEL_B_API_KEY"),
        name=required("MODEL_B_NAME"),
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        extra_body=_build_extra_body("MODEL_B"),
    )
    resolved_input = input_md or required("DISTILL_NER_INPUT_MD")
    resolved_output = output_dir or required("DISTILL_NER_OUTPUT_DIR")
    return DistillNERConfig(
        model_a=model_a,
        model_b=model_b,
        input_md=resolved_input,
        output_dir=resolved_output,
        gold_jsonl=os.getenv("DISTILL_NER_GOLD_JSONL") or None,
        min_chars=_env_int("DISTILL_NER_MIN_CHARS", 1200),
        max_chars=_env_int("DISTILL_NER_MAX_CHARS", 2600),
    )
