from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool 
from typing import List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from tools import call_tool
import prompts
import json

class State(TypedDict):
    query: str
    llm: BaseChatModel
    tools: list[BaseTool]
    history: list[BaseMessage]
    messages: Annotated[list, add_messages]
    vector: Dict[str, float]

graph_builder = StateGraph(State)


async def assess(state: State) -> Dict:
    print("assessment...")
    query = state["query"]
    llm = state["llm"]
    tools = state["tools"]
    history = state["messages"]

    system = SystemMessage(content=prompts.get_assessment_prompt().strip())

    messages = history + [system, HumanMessage(content=query)]
    for _ in range(10):
        response = await llm.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return {"messages": messages, "vector": state["vector"]}
        for tool_call in response.tool_calls:
            tool_response = await call_tool(tool_call, tools)
            messages.append(tool_response)
    raise RuntimeError("Max iterations reached in assess")

async def clarify(state: State) -> Dict:
    print("clarify...")
    query = state["query"]
    llm = state["llm"]
    tools = state["tools"]
    history = state["messages"]

    system = SystemMessage(content=prompts.get_clarification_prompt().strip())

    messages = history + [system, HumanMessage(content=query)]
    for _ in range(10):
        response = await llm.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return {"messages": messages, "vector": state["vector"]}
        for tool_call in response.tool_calls:
            tool_response = await call_tool(tool_call, tools)
            messages.append(tool_response)
    raise RuntimeError("Max iterations reached in clarify")

async def plan(state: State) -> Dict:
    print("plan...")
    query = state["query"]
    llm = state["llm"]
    tools = state["tools"]
    history = state["messages"]

    system = SystemMessage(content=prompts.get_planning_prompt().strip())

    messages = history + [system, HumanMessage(content=query)]
    for _ in range(10):
        response = await llm.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return {"messages": messages, "vector": state["vector"]}
        for tool_call in response.tool_calls:
            tool_response = await call_tool(tool_call, tools)
            messages.append(tool_response)
    raise RuntimeError("Max iterations reached in clarify")

async def ideat(state: State) -> Dict:
    print("ideat...")
    query = state["query"]
    llm = state["llm"]
    tools = state["tools"]
    history = state["messages"]

    system = SystemMessage(content=prompts.get_ideation_prompt().strip())

    messages = history + [system, HumanMessage(content=query)]
    for _ in range(10):
        response = await llm.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return {"messages": messages, "vector": state["vector"]}
        for tool_call in response.tool_calls:
            tool_response = await call_tool(tool_call, tools)
            messages.append(tool_response)
    raise RuntimeError("Max iterations reached in clarify")

async def motivate(state: State) -> Dict:
    print("motivate...")
    query = state["query"]
    llm = state["llm"]
    tools = state["tools"]

    system = SystemMessage(content=prompts.get_motivation_prompt().strip())

    messages = [system, HumanMessage(content=query)]
    for _ in range(10):
        response = await llm.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return {"messages": messages, "vector": state["vector"]}
        for tool_call in response.tool_calls:
            tool_response = await call_tool(tool_call, tools)
            messages.append(tool_response)
    raise RuntimeError("Max iterations reached in motivate")

async def classify_message(state: State) -> Dict:
    vector = state["vector"]
    query = state["query"]
    llm = state["llm"]

    system = SystemMessage(content=prompts.get_step_prompt(vector).strip())
    messages = [system, HumanMessage(content=query)]
    result = await llm.ainvoke(messages)
    label = result.content.strip().lower()
    print(f"query label: {label}")
    return {"next": label if label in ["orientation", "conceptualization", "solution ideation", "planning", "execution support"] else "orientation", "vector": state["vector"]}

async def set_scores(state: State) -> Dict:
    query = state["query"]
    history = state["messages"]
    vector = state.get("vector")

    print(f"messages: {vector}")
    system = SystemMessage(content=prompts.get_score_prompt(history, query, vector).strip())
    messages = [system, HumanMessage(content=query)]
    result = await state["llm"].ainvoke(messages)

    try:
        print(f"json {result.content}")
        updated_vector = json.loads(result.content)
    except json.JSONDecodeError:
        raise ValueError("LLM did not return valid JSON")

    return {"vector": updated_vector}


graph_builder.add_node("scoring", set_scores)
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("orientation", clarify)
graph_builder.add_node("execution support", motivate)
graph_builder.add_node("solution ideation", ideat)
graph_builder.add_node("planning", plan)
graph_builder.add_node("conceptualization", assess)

graph_builder.add_edge(START, "scoring")
graph_builder.add_edge("scoring", "classifier")
graph_builder.add_conditional_edges("classifier", lambda state: state["next"], {
    "orientation": "orientation",
    "conceptualization": "conceptualization",
    "solution ideation": "solution ideation",
    "planning": "planning",
    "execution support": "execution support"
})
graph_builder.add_edge("orientation", END)
graph_builder.add_edge("conceptualization", END)
graph_builder.add_edge("solution ideation", END)
graph_builder.add_edge("execution support", END)
graph_builder.add_edge("planning", END)

graph = graph_builder.compile()


async def setupState(
    query: str,
    llm: BaseChatModel,
    available_tools: list[BaseTool],
    messages: list[BaseMessage],
    vector: dict | None = None
):
    parameters = {
        "query": query,
        "llm": llm,
        "tools": available_tools,
        "messages": messages,
        "vector": vector
    }
    state = await graph.ainvoke(parameters)
    return state
