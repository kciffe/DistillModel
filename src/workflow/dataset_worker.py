
from typing import List
from .schema.schema_dataset import DatasetTask
from .state_dataset import DataState
from .graph import mainGraph
from uuid import uuid4

def build_initial_dataset_state()->DataState:

    return DataState(
        {
            "messages":[],
            "pending_tasks":[],
            "current_task":None,
            "questions":[],
            "final_data":[],
            "failed_tasks":[],
            "failed_questions":[],
        }
    )

def runner():
    datasetState=build_initial_dataset_state()
    config={
        "configurable":{
            "thread_id":str(uuid4().hex),
        },
        "max_concurrency": 16,
    }
    return mainGraph.invoke(
        datasetState,
        config=config,
    )


if __name__=="__main__":
    result=runner()
    for question in result.get("final_data", []):
        print(question)
