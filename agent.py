from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool 
from typing import Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from tools import call_tool
from prompts import Prompts
import json

class State(TypedDict):
    query: str
    llm: BaseChatModel
    tools: list[BaseTool]
    history: list[BaseMessage]
    messages: Annotated[list, add_messages]
    vector: Dict[str, float]
    current_document: str

class Agent:
    async def prepare_rag_query(history, query, llm, current_document):
        prompts = Prompts(history, query, None, current_document)
        prompt = prompts.get_rag_query_prompt()
        system = SystemMessage(content=prompt.strip())
        messages = [system]
        result = await llm.ainvoke(messages)
        return result.content.strip()

    async def assess(self, state: State) -> Dict:
        query = state["query"]
        llm = state["llm"]
        tools = state["tools"]
        history = state["messages"]
        vector = state.get("vector")
        current_document = state.get("current_document")

        prompts = Prompts(history, query, vector, current_document)

        rag_query = await self.prepare_rag_query(history, query, llm, current_document)

        system = SystemMessage(content=prompts.get_assessment_prompt().strip())

        messages = history + [system, HumanMessage(content=rag_query)]
        for _ in range(10):
            response = await llm.ainvoke(messages)
            messages.append(response)
            if not response.tool_calls:
                return {"messages": messages, "vector": state["vector"]}
            for tool_call in response.tool_calls:
                tool_response = await call_tool(tool_call, tools)
                messages.append(tool_response)
        raise RuntimeError("Max iterations reached in assess")

    async def clarify(self, state: State) -> Dict:
        query = state["query"]
        llm = state["llm"]
        tools = state["tools"]
        history = state["messages"]
        vector = state.get("vector")
        current_document = state.get("current_document")

        prompts = Prompts(history, query, vector, current_document)

        rag_query = await self.prepare_rag_query(history, query, llm, current_document)

        system = SystemMessage(content=prompts.get_clarification_prompt().strip())

        messages = history + [system, HumanMessage(content=rag_query)]
        for _ in range(10):
            response = await llm.ainvoke(messages)
            messages.append(response)
            if not response.tool_calls:
                return {"messages": messages, "vector": state["vector"]}
            for tool_call in response.tool_calls:
                tool_response = await call_tool(tool_call, tools)
                messages.append(tool_response)
        raise RuntimeError("Max iterations reached in clarify")

    async def motivate(self, state: State) -> Dict:
        query = state["query"]
        llm = state["llm"]
        tools = state["tools"]
        history = state["messages"]
        vector = state.get("vector")
        current_document = state.get("current_document")

        prompts = Prompts(history, query, vector, current_document)

        rag_query = await self.prepare_rag_query(history, query, llm, current_document)

        system = SystemMessage(content=prompts.get_motivation_prompt().strip())

        messages = [system, HumanMessage(content=rag_query)]
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
        history = state["messages"]
        llm = state["llm"]

        prompts = Prompts(history, query, vector, None)

        system = SystemMessage(content=prompts.get_step_prompt().strip())
        messages = [system, HumanMessage(content=query)]
        result = await llm.ainvoke(messages)
        label = result.content.strip().lower()
        print(f"query: {query}")
        print(f"query label: {label}")
        return {"next": label if label in ["orientation", "conceptualisation", "executive_support"] else "orientation", "vector": state["vector"]}

    def create_graph(self):
        graph_builder = StateGraph(State)

        graph_builder.add_node("classifier", self.classify_message)
        graph_builder.add_node("orientation", self.clarify)
        graph_builder.add_node("conceptualisation", self.assess)
        graph_builder.add_node("executive_support", self.motivate)

        graph_builder.add_edge(START, "classifier")
        graph_builder.add_conditional_edges("classifier", lambda state: state["next"], {
            "orientation": "orientation",
            "conceptualisation": "conceptualisation",
            "executive_support": "executive_support"
        })
        graph_builder.add_edge("orientation", END)
        graph_builder.add_edge("conceptualisation", END)
        graph_builder.add_edge("executive_support", END)

        graph = graph_builder.compile()


    async def setupState(
        self,
        query: str,
        llm: BaseChatModel,
        available_tools: list[BaseTool],
        messages: list[BaseMessage],
        vector: dict | None = None,
        current_document: str | None = None
    ):
        parameters = {
            "query": query,
            "llm": llm,
            "tools": available_tools,
            "messages": messages,
            "vector": vector,
            "current_document": current_document
        }
        graph = self.create_graph()
        state = await graph.ainvoke(parameters)
        return state
