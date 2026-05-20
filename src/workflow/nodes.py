
import hashlib
import re
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

# simhash去重节点
SIMHASH_BITS=64
SIMHASH_THRESHOLD=6

# 计算 SimHash
def _simhash(text: str) -> int:
    weights=[0]*SIMHASH_BITS
    tokens=[text[i:i+2] for i in range(max(len(text)-1,1))]

    for token in tokens:
        digest=hashlib.md5(token.encode("utf-8")).digest()
        value=int.from_bytes(digest[:8],"big")
        for bit_index in range(SIMHASH_BITS):
            if value & (1 << bit_index):
                weights[bit_index]+=1
            else:
                weights[bit_index]-=1

    fingerprint=0
    for bit_index, weight in enumerate(weights):
        if weight > 0:
            fingerprint |= 1 << bit_index
    return fingerprint

# 计算海明距离
def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def deduplicate_questions(state: DataState):
    log_info("正在去重")
    questions=state.get("questions") or []
    deduplicated_questions=[]
    # 保存已经见过的原始问题文本，用于完全相同文本去重。
    seen_questions=set()
    # 保存已经保留问题的 SimHash 指纹和文本，用于相似问题去重。
    seen_fingerprints=[]

    for question in questions:
        text=question["question"].strip()
        if text in seen_questions:
            continue

        fingerprint=_simhash(text)
        duplicate_question=None
        for seen_fingerprint, seen_text in seen_fingerprints:
            if _hamming_distance(fingerprint, seen_fingerprint) <= SIMHASH_THRESHOLD:
                duplicate_question=seen_text
                break

        if duplicate_question is not None:
            log_info(f"当前条目为：{text},库中已存在相似条目：{duplicate_question}")
            continue

        seen_questions.add(text)
        seen_fingerprints.append((fingerprint,text))
        deduplicated_questions.append(question)

    log_success(f"去重完成：原始{len(questions)}条，保留{len(deduplicated_questions)}条")
    return {
        "deduplicated_questions":deduplicated_questions
    }

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

    for start in range(0,len(questions),JUDGE_BATCH_SIZE):
        batch_questions=questions[start:start+JUDGE_BATCH_SIZE]
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

    log_success(f"打分完成：通过{len(judged_questions)}条，过滤{len(filtered_questions)}条")
    return {
        "final_data":judged_questions,
        "filtered_questions":filtered_questions,
        "messages":[
            AIMessage(content=f"打分完成：通过{len(judged_questions)}条，过滤{len(filtered_questions)}条")
        ]
    }
