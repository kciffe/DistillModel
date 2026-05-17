
import threading
import time

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send
from .getmodel import get_llm_without_tools
from .prompt import GENERATE_QUESTIONS_PROMPT,JUDGE_QUESTION_PROMPT
from .state_dataset import DataState
from ..utils.logger import log_info,log_success
# 生产者节点，生产任务
def init_pending_tasks(state:DataState):
    log_info("正在初始化任务")
    return {
        "pending_tasks":[
            {
                "category":"结婚登记与条件",
                "count":500,
                "description":"涉及结婚年龄、自愿原则、禁止近亲结婚、疾病限制、登记程序及法律效力等。",
            },
            {
                "category":"夫妻权利义务",
                "count":400,
                "description":"涵盖共同财产管理、相互扶养义务、姓名权、生育权及日常家事代理权等。",
            },
            {
                "category":"离婚程序与条件",
                "count":600,
                "description":"包括协议离婚冷静期、诉讼离婚标准、分居认定、感情破裂证明及调解程序等。",
            },
            {
                "category":"财产分割",
                "count":800,
                "description":"涉及共同财产界定、个人财产保护、债务承担、房产分割规则及隐匿财产追责等。",
            },
            {
                "category":"子女抚养与监护权",
                "count":800,
                "description":"包含抚养费计算标准、探视权实施、非婚生子女权益、抚养关系变更及教育责任等。",
            },
            {
                "category":"家庭暴力与保护令",
                "count":500,
                "description":"涵盖暴力行为认定、人身保护令申请、证据收集、紧急庇护措施及刑事责任关联等。",
            },
            {
                "category":"继承权与婚姻关系",
                "count":300,
                "description":"涉及配偶继承顺位、遗嘱效力、遗产分割冲突、再婚继承权及代位继承问题等。",
            },
            {
                "category":"无效婚姻与可撤销婚姻",
                "count":200,
                "description":"包括重婚无效、胁迫婚姻撤销、隐瞒疾病撤销的程序及法律后果等。",
            },
            {
                "category":"涉外婚姻",
                "count":300,
                "description":"涉及跨国婚姻登记、域外结婚效力认定、财产跨境分割及国际子女抚养公约适用等。",
            },
            {
                "category":"法律责任与救济措施",
                "count":200,
                "description":"包含虚假登记处罚、拒不执行判决后果、损害赔偿计算及强制执行程序等。",
            },
        ]
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

        break # 只处理一个任务
    
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
    log_success(
        f"完成生成问题：category={task['category']} batch_id={task['batch_id']} "
        f"生成数={len(questions)} thread={thread_id} elapsed={elapsed:.2f}s"
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
