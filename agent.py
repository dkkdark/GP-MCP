from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool 
from typing import List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from tools import call_tool

class State(TypedDict):
    query: str
    llm: BaseChatModel
    tools: list[BaseTool]
    history: list[BaseMessage]
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

async def assess(
    state: State
) -> str:
    print("accessment...")
    system = SystemMessage(content="""
        You're a teacher.
        Your purpose is to assess the user's understanding of a task and help them identify any knowledge gaps. You should ask diagnostic questions, check assumptions, and suggest learning resources or mini-lessons if needed.

        <instructions>
            <instruction>Always use the available tools to manage the tasks (they are stored in a database)</instruction>
            <instruction>Never use your own database when you can get the answer from tools</instruction>
            <instruction>Use `get_task_answer` tool to retrieve or verify information from tasks</instruction>
            <instruction>When identifying a gap, suggest explanations, resources, or steps to address it</instruction>
            <instruction>Never duplicate tool calls</instruction>
        </instructions>

        Your responses should be formatted as Markdown. Use bullet points, headers, or tables where useful.
        """.strip())

    max_iterations = 10
    n_iterations = 0
    messages = state["history"].copy()
    messages.append(system)
    messages.append(HumanMessage(content=state["query"]))

    while n_iterations < max_iterations:
        response = await state["llm"].ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return {"messages": [{"role": "assistant", "content": response.content}], "history": messages}
        
        for tool_call in response.tool_calls:
            response = await call_tool(tool_call, state["tools"])
            messages.append(response)
        n_iterations += 1

    raise RuntimeError(
        "Maximum number of iterations reached. Please try again with a different query."
    )

async def clarify(
    state: State
) -> str:
    print("clarifying...")
    system = SystemMessage(content="""
        You're a teacher.
        Your purpose is to help the user understand tasks. You should give hints, examples and explanations.

        <instructions>
            <instruction>Always use the available tools to manage the tasks (they are stored in a database)</instruction>
            <instruction>Never use your own database when you can get the answer from tools</instruction>
            <instruction>Use `get_task_answer` tool to answer questions regarding tasks and provide examples</instruction>
            <instruction>Never duplicate tool calls</instruction>
        </instructions>

        Your responses should be formatted as Markdown. Prefer using tables or lists for displaying data where appropriate.
        """.strip())

    max_iterations = 10
    n_iterations = 0
    messages = state["history"].copy()
    messages.append(system)
    messages.append(HumanMessage(content=state["query"]))

    while n_iterations < max_iterations:
        response = await state["llm"].ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return {"messages": [{"role": "assistant", "content": response.content}], "history": messages}
        
        for tool_call in response.tool_calls:
            response = await call_tool(tool_call, state["tools"])
            messages.append(response)
        n_iterations += 1

    raise RuntimeError(
        "Maximum number of iterations reached. Please try again with a different query."
    )

async def classify_message(
    state: State
    ) -> Dict:
        system = SystemMessage(content="""
        You are a classifier agent. Your job is to analyze the user's query and decide which specialized agent should handle it:

        - Return 'clarification' if the user is asking to better understand the task, wants the task rephrased, or seeks examples or explanations.
        - Return 'assessment' if the user wants to check their understanding, identify knowledge gaps, or needs resources to study.

        Respond ONLY with one word: clarification or assessment.
        """.strip())

        messages = [system, HumanMessage(content=state["query"])]
        result = await state["llm"].ainvoke(messages)
        label = result.content.strip().lower()

        print(f"query label: {label}")
        if label not in ["clarification", "assessment"]:
            label = "clarification"

        return {"next": label}


graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("clarification", clarify)
graph_builder.add_node("assessment", assess)
graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("clarification", END)
graph_builder.add_edge("assessment", END)

graph_builder.add_conditional_edges(
    "classifier",
    lambda state: state.get("next"),
    {"clarification": "clarification", "assessment": "assessment"}
)

graph = graph_builder.compile()

async def setupState(
    query: str,
    llm: BaseChatModel,
    available_tools: list[BaseTool],
    history: list[BaseMessage]
):
    parameters = {
        "query": query,
        "llm": llm,
        "tools": available_tools,
        "history": history
    }
    state = await graph.ainvoke(parameters)
    return state
