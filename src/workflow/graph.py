
from langgraph.graph import StateGraph,END,START
from .nodes import (
    generate_question_llm,
    judge_questino_llm,
    init_pending_tasks,
    prepare_task_batches,
    route_task_batches,

)
from .state_dataset import DataState

def build_graph():

    G=StateGraph(DataState)
    G.add_node("init_pending_tasks",init_pending_tasks)
    G.add_node("prepare_task_batches",prepare_task_batches)
    G.add_node("generate_question_llm",generate_question_llm)
    G.add_node("judge_questino_llm",judge_questino_llm)

    G.add_edge(START,"init_pending_tasks")
    G.add_edge("init_pending_tasks","prepare_task_batches")
    G.add_conditional_edges(
        "prepare_task_batches",
        route_task_batches,
        ["generate_question_llm"]
    )
    G.add_edge("generate_question_llm",END) 
    return G.compile()

mainGraph=build_graph()
