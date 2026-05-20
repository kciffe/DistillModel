import json
import hashlib
import threading
from collections import Counter
from pathlib import Path
from typing import Iterable

from .schema.schema_dataset import DatasetTask, JudgedQuestion
from ..utils.simhash import SimHashBucketIndex, simhash


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS_FILE = PROJECT_ROOT / "data" / "dataset_tasks.json"
QUESTIONS_FILE = PROJECT_ROOT / "data" / "generated_questions.jsonl"
DEDUPED_QUESTIONS_FILE = PROJECT_ROOT / "data" / "deduped_questions.jsonl"
JUDGED_QUESTIONS_FILE = PROJECT_ROOT / "data" / "judged_questions.jsonl"
FINGERPRINT_INDEX_FILE = PROJECT_ROOT / "data" / "question_fingerprint_index.jsonl"
PASS_SCORE = 7
_WRITE_LOCK = threading.Lock()


def question_id(question: JudgedQuestion) -> str:
    raw = f"{question['category']}\n{question['question'].strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _iter_jsonl(path: Path):
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    row_list = list(rows)
    if not row_list:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as file:
            for row in row_list:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dataset_tasks() -> list[DatasetTask]:
    with TASKS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_generated_questions() -> list[JudgedQuestion]:
    return list(iter_generated_questions())


def iter_generated_questions():
    yield from _iter_jsonl(QUESTIONS_FILE)


def count_questions_by_category() -> Counter[str]:
    return Counter(question["category"] for question in load_generated_questions())


def load_judged_questions() -> list[JudgedQuestion]:
    return list(iter_judged_questions())


def iter_judged_questions():
    yield from _iter_jsonl(JUDGED_QUESTIONS_FILE)


def iter_deduped_questions():
    yield from _iter_jsonl(DEDUPED_QUESTIONS_FILE)


def count_passed_questions_by_category(filter_score:int=PASS_SCORE) -> Counter[str]:
    return Counter(
        question["category"]
        for question in load_judged_questions()
        if question.get("score", 0) >= filter_score
    )


def load_pending_dataset_tasks() -> list[DatasetTask]:
    passed_counts = count_passed_questions_by_category()
    pending_tasks: list[DatasetTask] = []

    for task in load_dataset_tasks():
        passed_count = passed_counts[task["category"]]
        remaining_count = max(task["count"] - passed_count, 0)
        if remaining_count == 0:
            continue

        pending_task = task.copy()
        pending_task["count"] = remaining_count
        pending_tasks.append(pending_task)

    return pending_tasks


def append_generated_questions(questions: Iterable[JudgedQuestion]) -> None:
    rows = []
    for question in questions:
        row = dict(question)
        row["id"] = row.get("id") or question_id(row)
        rows.append(row)
    _append_jsonl(QUESTIONS_FILE, rows)


def append_judged_questions(questions: Iterable[JudgedQuestion]) -> None:
    rows = []
    for question in questions:
        row = dict(question)
        row["id"] = row.get("id") or question_id(row)
        rows.append(row)
    _append_jsonl(JUDGED_QUESTIONS_FILE, rows)


def has_pending_dataset_tasks() -> bool:
    return bool(load_pending_dataset_tasks())


def _load_fingerprint_index(threshold: int) -> tuple[SimHashBucketIndex, set[str]]:
    index = SimHashBucketIndex(threshold=threshold)
    indexed_ids: set[str] = set()

    for row in _iter_jsonl(FINGERPRINT_INDEX_FILE) or []:
        text = row["question"].strip()
        fingerprint = int(row["fingerprint"])
        index.add(text, fingerprint)
        indexed_ids.add(row["id"])

    return index, indexed_ids


def deduplicate_generated_questions(threshold: int) -> list[JudgedQuestion]:
    index, indexed_ids = _load_fingerprint_index(threshold)
    judged_ids = set()
    for row in iter_judged_questions():
        qid = row.get("id") or question_id(row)
        judged_ids.add(qid)
        if qid not in indexed_ids:
            index.add(row["question"].strip())
            indexed_ids.add(qid)

    deduped_questions: list[JudgedQuestion] = []
    new_index_rows = []

    for question in iter_generated_questions():
        qid = question.get("id") or question_id(question)
        if qid in indexed_ids or qid in judged_ids:
            continue

        text = question["question"].strip()
        fingerprint = simhash(text)
        duplicate_text = index.find_duplicate(text, fingerprint)
        if duplicate_text is not None:
            indexed_ids.add(qid)
            new_index_rows.append(
                {
                    "id": qid,
                    "category": question["category"],
                    "question": text,
                    "fingerprint": fingerprint,
                    "duplicate": True,
                    "duplicate_question": duplicate_text,
                }
            )
            continue

        row = dict(question)
        row["id"] = qid
        index.add(text, fingerprint)
        indexed_ids.add(qid)
        deduped_questions.append(row)
        new_index_rows.append(
            {
                "id": qid,
                "category": row["category"],
                "question": text,
                "fingerprint": fingerprint,
                "duplicate": False,
            }
        )

    _append_jsonl(DEDUPED_QUESTIONS_FILE, deduped_questions)
    _append_jsonl(FINGERPRINT_INDEX_FILE, new_index_rows)
    return deduped_questions
