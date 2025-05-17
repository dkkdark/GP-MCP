import asyncio
import random

import nest_asyncio
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import streamlit as st
from agent import ask, create_history
from client import connect_to_server
from tools import load_tools
from langchain_openai import ChatOpenAI
import os

load_dotenv()

LOADING_MESSAGES = [
    "Processing your request..."
]

def create_llm():
    return ChatOpenAI(model='gpt-4.1-mini')

async def get_response_async(user_query: str, history: list, llm):
    async with connect_to_server() as session:
        tools = await load_tools(session)
        llm_with_tools = llm.bind_tools(tools)
        response_content = await ask(user_query, llm_with_tools, tools, history.copy())
        return response_content

nest_asyncio.apply()

st.set_page_config(
    page_title="Teaching",
    layout="centered",
)

st.title("Teaching")

if "llm" not in st.session_state:
    st.session_state.llm = create_llm()

if "messages" not in st.session_state:
    st.session_state.messages = create_history()

for message in st.session_state.messages:
    if type(message) is SystemMessage:
        continue
    is_user = type(message) is HumanMessage
    avatar = "👤" if is_user else "🤖"
    with st.chat_message("user" if is_user else "ai", avatar=avatar):
        st.markdown(message.content)

import threading
from queue import Queue

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


if prompt := st.chat_input("What can I help you with?"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        message_placeholder.status(random.choice(LOADING_MESSAGES), state="running")

        try:
            response = run_async_function(
                get_response_async(prompt, st.session_state.messages, st.session_state.llm)
            )
            message_placeholder.markdown(response)
            st.session_state.messages.append(AIMessage(content=response))
        except Exception as e:
            message_placeholder.error(f"An error occurred: {e}")
