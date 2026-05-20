
from langgraph.graph import StateGraph,END,START
from .nodes import (
    generate_question_llm,
    judge_question_batch_llm,
    judge_round_complete,
    deduplicate_questions,
    init_pending_tasks,
    prepare_task_batches,
    route_after_init,
    route_after_judge,
    route_judge_batches,
    route_task_batches,

)
from .state_dataset import DataState

def build_graph():

    G=StateGraph(DataState)
    G.add_node("init_pending_tasks",init_pending_tasks)
    G.add_node("prepare_task_batches",prepare_task_batches)
    G.add_node("generate_question_llm",generate_question_llm)
    G.add_node("deduplicate_questions",deduplicate_questions)
    G.add_node("judge_question_batch_llm",judge_question_batch_llm)
    G.add_node("judge_round_complete",judge_round_complete)

    G.add_edge(START,"init_pending_tasks")
    G.add_conditional_edges(
        "init_pending_tasks",
        route_after_init,
        {
            "prepare":"prepare_task_batches",
            "end":END,
        }
    )
    G.add_conditional_edges(
        "prepare_task_batches",
        route_task_batches,
        ["generate_question_llm"]
    )
    G.add_edge("generate_question_llm","deduplicate_questions")
    G.add_conditional_edges(
        "deduplicate_questions",
        route_judge_batches,
        {
            "judge_question_batch_llm":"judge_question_batch_llm",
            "complete":"judge_round_complete",
        }
    )
    G.add_edge("judge_question_batch_llm","judge_round_complete")
    G.add_conditional_edges(
        "judge_round_complete",
        route_after_judge,
        {
            "continue":"init_pending_tasks",
            "end":END,
        }
    )
    return G.compile()

mainGraph=build_graph()
