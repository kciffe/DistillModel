
from typing import NotRequired, TypedDict

class DatasetTask(TypedDict):
    category: str
    count: int
    description: str
    batch_id: NotRequired[int]

class JudgedQuestion(TypedDict):
    category: str
    question: str
    score:int
