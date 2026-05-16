from langchain_core.messages import BaseMessage


def format_messages(messages: list[BaseMessage]) -> str:
    """Format LangChain messages for readable logging."""
    formatted_messages = []
    for message in messages:
        role = message.type.upper()
        formatted_messages.append(f"{role}: {message.content}")

    return "\n".join(formatted_messages)
