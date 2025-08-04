import asyncio
import random
import nest_asyncio
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import streamlit as st
from agent import Agent
from client import connect_to_server
from tools import load_tools
from langchain_openai import ChatOpenAI
import threading
from queue import Queue
import os

load_dotenv()
nest_asyncio.apply()

LOADING_MESSAGES = [
    "Processing your request...",
    "Hang tight, thinking deeply...",
    "Let me check that for you..."
]

folder_path = './/data/tasks' 

document_options = [
    os.path.splitext(f)[0]
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

st.set_page_config(page_title="Teaching", layout="centered")
st.title("Teaching Assistant")

if "current_document" not in st.session_state:
    st.session_state.current_document = document_options[0]

selected_document = st.selectbox(
    "Select the task you're working on:",
    document_options,
    index=document_options.index(st.session_state.current_document)
)
st.session_state.current_document = selected_document

if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model='gpt-4.1-mini', api_key=os.getenv("OPENAI_API_KEY"))

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector" not in st.session_state:
    st.session_state.vector = {
        "Orientation": 0.0,
        "Conceptualization": 0.0,
        "Solution Ideation": 0.0,
        "Planning": 0.0,
        "Execution Support": 0.0
    }

async def handle_query(prompt, llm, messages, vector, current_document):
    async with connect_to_server() as session:
        tools = await load_tools(session)
        llm_with_tools = llm.bind_tools(tools)

        agent = Agent()

        result_state = await agent.setupState(
            query=prompt,
            llm=llm_with_tools,
            available_tools=tools,
            messages=messages,
            vector=vector,
            current_document=current_document
        )
        return result_state

def run_async_function(coro):
    q = Queue()

    def runner():
        try:
            result = asyncio.run(coro)
            q.put(result)
        except BaseException as e:
            import traceback
            traceback.print_exc()
            q.put(e)

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()

    result = q.get()
    if isinstance(result, BaseException):
        raise result
    return result

for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue
    is_user = isinstance(message, HumanMessage)
    avatar = "👤" if is_user else "🤖"
    with st.chat_message("user" if is_user else "ai", avatar=avatar):
        st.markdown(message.content)

if prompt := st.chat_input("What can I help you with?"):
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.status(random.choice(LOADING_MESSAGES), state="running")

        try:
            result = run_async_function(handle_query(
                prompt,
                st.session_state.llm,
                st.session_state.messages,
                st.session_state.vector,
                st.session_state.current_document
            ))

            last_message = ""
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1].content

            placeholder.markdown(last_message)

            if "messages" in result and last_message:
                st.session_state.messages.append(AIMessage(content=last_message))

            if "vector" in result:
                st.session_state.vector = result["vector"]

        except Exception as e:
            placeholder.error(f"An error occurred: {e}")
