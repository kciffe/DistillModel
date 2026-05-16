
import operator
from typing import Annotated, List, TypedDict
from langgraph.graph.message import add_messages
from .schema.schema_dataset import DatasetTask,JudgedQuestion

class DataState(TypedDict):
    messages: Annotated[list, add_messages]

    pending_tasks: List[DatasetTask] | None        # 待处理任务队列
    current_task: DatasetTask |None         # 当前正在处理的任务
    questions: Annotated[List[JudgedQuestion], operator.add]             # 生成出来的问题
    final_data: Annotated[List[JudgedQuestion], operator.add]            # 最终保留的数据
    failed_tasks: Annotated[List[DatasetTask], operator.add]          # 失败任务
    failed_questions: Annotated[List[JudgedQuestion], operator.add]   # 失败问题
