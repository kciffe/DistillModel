
from typing import TypedDict

class DatasetTask(TypedDict):
    category: str
    count: int
    description: str
    batch_id: int 

class JudgedQuestion(TypedDict):
    category: str
    question: str
    score:int