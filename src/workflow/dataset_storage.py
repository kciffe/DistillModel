import json
import threading
from collections import Counter
from pathlib import Path
from typing import Iterable

from .schema.schema_dataset import DatasetTask, JudgedQuestion


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS_FILE = PROJECT_ROOT / "data" / "dataset_tasks.json"
QUESTIONS_FILE = PROJECT_ROOT / "data" / "generated_questions.jsonl"
_WRITE_LOCK = threading.Lock()


def load_dataset_tasks() -> list[DatasetTask]:
    with TASKS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_generated_questions() -> list[JudgedQuestion]:
    if not QUESTIONS_FILE.exists():
        return []

    questions: list[JudgedQuestion] = []
    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def count_questions_by_category() -> Counter[str]:
    return Counter(question["category"] for question in load_generated_questions())


def load_pending_dataset_tasks() -> list[DatasetTask]:
    generated_counts = count_questions_by_category()
    pending_tasks: list[DatasetTask] = []

    for task in load_dataset_tasks():
        generated_count = generated_counts[task["category"]]
        remaining_count = max(task["count"] - generated_count, 0)
        if remaining_count == 0:
            continue

        pending_task = task.copy()
        pending_task["count"] = remaining_count
        pending_tasks.append(pending_task)

    return pending_tasks


def append_generated_questions(questions: Iterable[JudgedQuestion]) -> None:
    question_list = list(questions)
    if not question_list:
        return

    QUESTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with QUESTIONS_FILE.open("a", encoding="utf-8") as file:
            for question in question_list:
                file.write(json.dumps(question, ensure_ascii=False) + "\n")
