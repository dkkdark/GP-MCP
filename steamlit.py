import asyncio
import random
import nest_asyncio
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import streamlit as st
from agent import setupState
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

st.set_page_config(page_title="Teaching", layout="centered")
st.title("Teaching Assistant")

# Document selection dropdown
if "current_document" not in st.session_state:
    st.session_state.current_document = "tasks"

document_options = ["tasks", "task2"]
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

        result_state = await setupState(
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

# Add a toggle for student simulation mode
simulate_student = st.sidebar.checkbox("Simulate student (AI)", value=False)

# Determine the initial prompt
prompt = None
if simulate_student and not st.session_state.messages:
    prompt = "Hello, I want to start. Tell me about the task"
else:
    user_prompt = st.chat_input("What can I help you with?")
    if user_prompt:
        prompt = user_prompt

if prompt:
    if not simulate_student:
        st.session_state.messages.append(HumanMessage(content=prompt))

    # Render user message (first turn)
    st.chat_message("user", avatar="👤").markdown(prompt)

    # Render assistant response (first turn)
    placeholder = None
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

        if placeholder:
            placeholder.markdown(last_message)

        if "messages" in result and last_message:
            st.session_state.messages.append(AIMessage(content=last_message))

        if "vector" in result:
            st.session_state.vector = result["vector"]

        # Student simulation loop
        if simulate_student:
            import prompts
            max_turns = 10
            turns = 0
            while turns < max_turns:
                # Prepare history and last assistant response
                history = '\n'.join([m.content for m in st.session_state.messages if isinstance(m, (HumanMessage, AIMessage))][:-1])
                last_response = last_message
                student_prompt = run_async_function(
                    st.session_state.llm.ainvoke([
                        SystemMessage(content=prompts.get_student_simulation_prompt(history, last_response))
                    ])
                ).content.strip()
                if not student_prompt or "done" in student_prompt.lower():
                    break
                st.session_state.messages.append(HumanMessage(content=student_prompt))
                st.chat_message("user", avatar="👤").markdown(student_prompt)
                # Get next assistant response
                result = run_async_function(handle_query(
                    student_prompt,
                    st.session_state.llm,
                    st.session_state.messages,
                    st.session_state.vector,
                    st.session_state.current_document
                ))
                last_message = ""
                if "messages" in result and result["messages"]:
                    last_message = result["messages"][-1].content
                st.chat_message("assistant", avatar="🤖").markdown(last_message)
                if "messages" in result and last_message:
                    st.session_state.messages.append(AIMessage(content=last_message))
                if "vector" in result:
                    st.session_state.vector = result["vector"]
                turns += 1
    except Exception as e:
        if placeholder:
            placeholder.error(f"An error occurred: {e}")
