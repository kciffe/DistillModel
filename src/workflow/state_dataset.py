
from typing import Annotated, List, TypedDict
from langgraph.graph.message import add_messages
from .schema.schema_dataset import DatasetTask,JudgedQuestion

class DataState(TypedDict):
    messages: Annotated[list, add_messages]

    pending_tasks: List[DatasetTask] | None        # 待处理任务队列
    current_task: DatasetTask |None         # 当前正在处理的任务
    questions: List[JudgedQuestion] | None             # 当前轮生成出来的问题
    deduplicated_questions: List[JudgedQuestion] | None             # 去重后的问题
    final_data: List[JudgedQuestion] | None            # 当前轮最终保留的数据
    filtered_questions: List[JudgedQuestion] | None  # 当前轮被过滤掉的低分问题
