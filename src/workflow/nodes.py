
import re
import threading
import time

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send
from .getmodel import get_llm_without_tools
from .prompt import GENERATE_QUESTIONS_PROMPT,JUDGE_QUESTION_PROMPT
from .state_dataset import DataState
from .dataset_storage import (
    append_generated_questions,
    append_judged_questions,
    count_passed_questions_by_category,
    deduplicate_generated_questions,
    has_pending_dataset_tasks,
    load_dataset_tasks,
    load_pending_dataset_tasks,
)
from ..utils.logger import log_info,log_success
from ..utils.progress import ThreadSafeProgress, progress_bar


_GENERATION_PROGRESS=ThreadSafeProgress()


# 生产者节点，生产任务
def init_pending_tasks(state:DataState):
    log_info("正在初始化任务")
    passed_counts=count_passed_questions_by_category()
    pending_tasks=load_pending_dataset_tasks()

    for task in load_dataset_tasks():
        passed_count=passed_counts[task["category"]]
        remaining_count=max(task["count"] - passed_count, 0)
        if remaining_count == 0:
            log_success(f"任务已完成，跳过：{task['category']}，已通过{passed_count}/{task['count']}条")
            continue

        log_info(f"任务待生成：{task['category']}，已通过{passed_count}/{task['count']}条，剩余{remaining_count}条")

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

            log_info(f"创建任务批次：{task['category']}，批次ID：{batch_id}，批次大小：{batch_count}，批次开始索引：{start}，批次结束索引：{start+batch_count}")
            batch_id+=1

    _GENERATION_PROGRESS.reset(len(batch_tasks))
    log_info(f"本轮生成批次数：{len(batch_tasks)}")

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
    done,total=_GENERATION_PROGRESS.mark_done()
    log_success(
        f"完成生成问题：category={task['category']} batch_id={task['batch_id']} "
        f"生成数={len(questions)} 已写入文件 生成进度={progress_bar(done,total)} "
        f"thread={thread_id} elapsed={elapsed:.2f}s"
    )
    return {
        "messages":[
            AIMessage(content=response.content)
        ]
    }

SIMHASH_THRESHOLD=6

def deduplicate_questions(state: DataState):
    log_info("正在去重")
    deduplicated_questions=deduplicate_generated_questions(SIMHASH_THRESHOLD)
    log_success(f"去重完成：本轮新增保留{len(deduplicated_questions)}条")
    return {
        "deduplicated_questions":deduplicated_questions
    }


def route_after_init(state: DataState):
    if state.get("pending_tasks"):
        return "prepare"
    return "end"


def route_after_judge(state: DataState):
    if has_pending_dataset_tasks():
        return "continue"
    return "end"

# 打分节点
JUDGE_BATCH_SIZE=20

def _extract_scores(text: str) -> list[int]:
    scores=[]
    for line in text.splitlines():
        line=line.strip()
        if not line:
            continue
        match=re.search(r"\b[0-9]\b", line)
        if match:
            scores.append(int(match.group()))
    return scores


def judge_questino_llm(state: DataState,filter_score:int=7):
    log_info("正在打分")
    questions=state.get("deduplicated_questions") or []
    if not questions:
        log_info("没有可打分的问题")
        return {
            "final_data":[],
            "filtered_questions":[],
        }

    model=get_llm_without_tools()
    judged_questions=[]
    filtered_questions=[]

    total_batches=(len(questions)+JUDGE_BATCH_SIZE-1)//JUDGE_BATCH_SIZE
    for batch_index,start in enumerate(range(0,len(questions),JUDGE_BATCH_SIZE), start=1):
        batch_questions=questions[start:start+JUDGE_BATCH_SIZE]
        log_info(f"开始打分批次：{progress_bar(batch_index-1,total_batches)}，本批{len(batch_questions)}条")
        input_text="\n".join(
            f"{index}. 类别：{question['category']}；问题：{question['question']}"
            for index, question in enumerate(batch_questions, start=1)
        )
        score_prompt=f"""
            你是一位婚姻法方面的专家。请判断每条【问题】是否准确属于它标注的【类别】。
            给出一个0到9之间的正整数评分，其中0表示完全错误，9表示非常准确。
            每条输入都需要单独判定。

            只允许输出评分，每个评分单独一行，不允许输出任何其他字符。

            【输入】
            {input_text}

            输出举例（10个传入问题时）:
            9
            7
            8
            7
            7
            7
            7 
            0
            7
            1
            """
        response=model.invoke([HumanMessage(content=score_prompt)])
        scores=_extract_scores(response.content)

        if len(scores) != len(batch_questions):
            log_info(f"打分数量不匹配：输入{len(batch_questions)}条，输出{len(scores)}条")

        for question, score in zip(batch_questions, scores):
            judged_question=question.copy()
            judged_question["score"]=score
            if score >= filter_score:
                judged_questions.append(judged_question)
            else:
                filtered_questions.append(judged_question)

        if len(scores) < len(batch_questions):
            for question in batch_questions[len(scores):]:
                failed_question=question.copy()
                failed_question["score"]=0
                filtered_questions.append(failed_question)

        log_success(f"完成打分批次：{progress_bar(batch_index,total_batches)}")

    append_judged_questions(judged_questions + filtered_questions)
    log_success(f"打分完成：通过{len(judged_questions)}条，过滤{len(filtered_questions)}条，已写入文件")
    return {
        "final_data":judged_questions,
        "filtered_questions":filtered_questions,
        "messages":[
            AIMessage(content=f"打分完成：通过{len(judged_questions)}条，过滤{len(filtered_questions)}条")
        ]
    }
