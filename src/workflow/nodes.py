
import threading
import time

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send
from .getmodel import get_llm_without_tools
from .prompt import GENERATE_QUESTIONS_PROMPT,JUDGE_QUESTION_PROMPT
from .state_dataset import DataState
from .dataset_storage import append_generated_questions, count_questions_by_category, load_dataset_tasks, load_pending_dataset_tasks
from ..utils.logger import log_info,log_success
# 生产者节点，生产任务
def init_pending_tasks(state:DataState):
    log_info("正在初始化任务")
    generated_counts=count_questions_by_category()
    pending_tasks=load_pending_dataset_tasks()

    for task in load_dataset_tasks():
        generated_count=generated_counts[task["category"]]
        remaining_count=max(task["count"] - generated_count, 0)
        if remaining_count == 0:
            log_success(f"任务已完成，跳过：{task['category']}，已有{generated_count}/{task['count']}条")
            continue

        log_info(f"任务待生成：{task['category']}，已有{generated_count}/{task['count']}条，剩余{remaining_count}条")

    return {
        "pending_tasks":pending_tasks
    }


# 拆分任务节点，每组任务只生成20个问题
BATCH_SIZE=20
def prepare_task_batches(state:DataState):
    log_info("正在准备任务批次")
    pending_tasks=state.get("pending_tasks")or[]
    batch_tasks=[]

    for task in pending_tasks:
        total=task["count"]
        batch_id=1

        for start in range(0,total,BATCH_SIZE):
            batch_count=min(BATCH_SIZE,total-start)
            batch_task=task.copy()
            batch_task["count"] = batch_count
            batch_task["batch_id"] = batch_id

            batch_tasks.append(batch_task)
            batch_id+=1

            log_info(f"创建任务批次：{task['category']}，批次ID：{batch_id}，批次大小：{batch_count}，批次开始索引：{start}，批次结束索引：{start+batch_count}")

    return {
        "pending_tasks":batch_tasks
    }

# 路由任务节点,并发分发任务
def route_task_batches(state:DataState):
    log_info("正在路由任务批次")
    pending_tasks=state.get("pending_tasks") or []

    return [
        Send(
            "generate_question_llm",
            {
                "current_task":task
            }
        )
        for task in pending_tasks
    ]
# 生成问题节点
def generate_question_llm(state: DataState):
    task=state.get("current_task")
    if task is None:
        return {}

    thread_id=threading.get_ident()
    started_at=time.perf_counter()
    log_info(
        f"开始生成问题：category={task['category']} batch_id={task['batch_id']} "
        f"count={task['count']} thread={thread_id}"
    )
    
    prompt=f"""
        你是一位婚姻法方面的专家，根据【特定类别】的定义，针对其中{task["category"]}方面的规定，向学生提供{task["count"]}道练习题，你能提出哪些问题？
        每个问题单独一行，只允许输出问题本身，不允许输出任何其他字符。
        禁止添加序号、编号、项目符号、Markdown列表符号或解释说明。
        正确格式示例：婚姻登记机关审查结婚登记申请时需要核验哪些条件？

        【特定类别】
        {task["category"]}：{task["description"]}
    """
    model=get_llm_without_tools()
    response=model.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )
    # 将问题每行切分
    question_list=[]
    for line in response.content.splitlines():
        line=line.strip()
        if line:
            question_list.append(line)
    # 最终问题列表
    questions=[
        {
            "question":question,
            "category":task["category"],
            "score":0,
        }
        for question in question_list
    ]
    elapsed=time.perf_counter()-started_at
    append_generated_questions(questions)
    log_success(
        f"完成生成问题：category={task['category']} batch_id={task['batch_id']} "
        f"生成数={len(questions)} 已写入文件 thread={thread_id} elapsed={elapsed:.2f}s"
    )
    return {
        "questions":questions,
        "messages":[
            AIMessage(content=response.content)
        ]
    }

# 打分节点
def judge_questino_llm(state: DataState):
    log_info("正在打分")
    model=get_llm_without_tools()
    response=model.invoke(
        [
            HumanMessage(content=JUDGE_QUESTION_PROMPT)
        ]
    )
    return {
        "messages":[
            AIMessage(content=response.content)
        ]
    }
