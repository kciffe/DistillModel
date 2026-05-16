import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


def _base_llm():
    return init_chat_model(
        model=os.getenv("LOCAL_MODEL"),
        model_provider=os.getenv("LOCAL_MODEL_PROVIDER", "openai"),
        base_url=os.getenv("LOCAL_MODEL_BASE_URL"),
        api_key=os.getenv("LOCAL_MODEL_API_KEY"),
    )

def get_llm(TOOLS):
    return _base_llm().bind_tools(TOOLS)

def get_llm_without_tools():
    return _base_llm()
