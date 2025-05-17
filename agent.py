from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool 
from typing import List

from tools import call_tool

SYSTEM_PROMPT = """
You're a teacher.
Your purpose is to help the user understand tasks. You should give hints, examples and explanations.

<instructions>
    <instruction>Always use the available tools to manage the tasks (they are stored in a database)</instruction>
    <instruction>Never use your own database when you can get the answer from tools</instruction>
    <instruction>Use `get_task_answer` tool to answer questions regarding tasks and provide examples</instruction>
    <instruction>Never duplicate tool calls</instruction>
</instructions>

Your responses should be formatted as Markdown. Prefer using tables or lists for displaying data where appropriate.
""".strip()

def create_history() -> List[BaseMessage]:
    return [SystemMessage(content=SYSTEM_PROMPT)]

async def ask(
    query: str,
    llm: BaseChatModel,
    available_tools: list[BaseTool],
    history: list[BaseMessage],
    max_iterations: int = 10,
) -> str:

    n_iterations = 0
    messages = history.copy()
    messages.append(HumanMessage(content=query))

    while n_iterations < max_iterations:
        response = await llm.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return response.content
        
        for tool_call in response.tool_calls:
            response = await call_tool(tool_call, available_tools)
            messages.append(response)
        n_iterations += 1

    raise RuntimeError(
        "Maximum number of iterations reached. Please try again with a different query."
    )