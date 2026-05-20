
from .state_dataset import DataState
from .graph import mainGraph
from uuid import uuid4

def build_initial_dataset_state()->DataState:

    return DataState(
        {
            "messages":[],
            "pending_tasks":[],
            "current_task":None,
            "judge_questions":[],
            "judge_batch_id":None,
            "questions":[],
            "deduplicated_questions":[],
            "final_data":[],
            "filtered_questions":[],

        }
    )

def runner():
    datasetState=build_initial_dataset_state()
    config={
        "configurable":{
            "thread_id":str(uuid4().hex),
        },
        "max_concurrency": 16,
        "recursion_limit": 200,
    }
    return mainGraph.invoke(
        datasetState,
        config=config,
    )


if __name__=="__main__":
    result=runner()
    for question in result.get("final_data", []):
        print(question)
