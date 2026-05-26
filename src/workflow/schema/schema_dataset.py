
from typing import NotRequired, TypedDict

class DatasetTask(TypedDict):
    category: str
    count: int
    description: str
    batch_id: NotRequired[int]

class JudgedQuestion(TypedDict):
    id: NotRequired[str]
    category: str
    question: str
    score:int
    answer: str
