from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
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


@dataclass(frozen=True)
class DistillNERConfig:
    model_a: ModelConfig
    model_b: ModelConfig
    input_md: str
    output_dir: str
    gold_jsonl: Optional[str] = None


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

    model_a = ModelConfig(
        base_url=required("MODEL_A_BASE_URL"),
        api_key=required("MODEL_A_API_KEY"),
        name=required("MODEL_A_NAME"),
    )
    model_b = ModelConfig(
        base_url=required("MODEL_B_BASE_URL"),
        api_key=required("MODEL_B_API_KEY"),
        name=required("MODEL_B_NAME"),
    )
    resolved_input = input_md or required("DISTILL_NER_INPUT_MD")
    resolved_output = output_dir or required("DISTILL_NER_OUTPUT_DIR")
    return DistillNERConfig(
        model_a=model_a,
        model_b=model_b,
        input_md=resolved_input,
        output_dir=resolved_output,
        gold_jsonl=os.getenv("DISTILL_NER_GOLD_JSONL") or None,
    )
