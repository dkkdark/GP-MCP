import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import asyncio
import random
import nest_asyncio
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import streamlit as st
from chat_client.agent import Agent
from chat_client.client import connect_to_server
from chat_client.tools import load_tools
from langchain_openai import ChatOpenAI
import threading
from queue import Queue
from server.document_processor import DocumentProcessor

load_dotenv()
nest_asyncio.apply()

LOADING_MESSAGES = [
    "Processing your request...",
    "Let me check that for you..."
]

os.makedirs("./data/tasks", exist_ok=True)
os.makedirs("./data/materials", exist_ok=True)

st.set_page_config(page_title="Teaching", layout="centered")
st.title("Teaching Assistant")

@st.cache_resource
def get_document_processor():
    return DocumentProcessor(db_path="./teaching_chroma_db")

processor = get_document_processor()

def get_document_options():
    folder_path = './data/tasks'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return []
    
    return [
        os.path.splitext(f)[0]
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.docx')
    ]

def save_uploaded_file(uploaded_file, target_folder):
    try:
        target_path = os.path.join(target_folder, uploaded_file.name)
        
        if os.path.exists(target_path):
            return False, f"File {uploaded_file.name} already exists"
        
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return True, f"File {uploaded_file.name} successfully uploaded"
    except Exception as e:
        return False, f"Error uploading: {str(e)}"

def process_uploaded_file(file_path):
    try:
        docs = processor.process_single_file(file_path)
        return True, len(docs)
    except Exception as e:
        return False, str(e)

document_options = get_document_options()

with st.sidebar:
    st.header("📁 Upload files")
    
    file_type = st.radio(
        "File type:",
        ["Tasks", "Materials"],
        help="Select the folder to upload files"
    )
    
    if file_type == "Tasks":
        target_folder = "./data/tasks"
    else:
        target_folder = "./data/materials"
    
    st.info(f"📂 Folder: `{target_folder}`")
    
    uploaded_files = st.file_uploader(
        "Upload .docx files",
        type=['docx'],
        accept_multiple_files=True,
        help="Only .docx files are supported"
    )
    
    if uploaded_files:
        st.write(f"Selected files: {len(uploaded_files)}")
        
        for file in uploaded_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 {file.name}")
            with col2:
                st.write(f"{file.size / 1024:.1f} KB")
        
        if st.button("🚀 Upload files", type="primary"):
            success_count = 0
            error_count = 0
            processed_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    success, message = save_uploaded_file(uploaded_file, target_folder)
                    
                    if success:
                        success_count += 1
                        status_text.text(f"Uploaded: {uploaded_file.name}")
                        
                        file_path = os.path.join(target_folder, uploaded_file.name)
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            process_success, result = process_uploaded_file(file_path)
                            if process_success:
                                processed_count += 1
                                st.success(f"✅ {uploaded_file.name} processed ({result} chunks)")
                            else:
                                st.error(f"❌ Error processing {uploaded_file.name}: {result}")
                    else:
                        error_count += 1
                        st.warning(message)
                    
                except Exception as e:
                    error_count += 1
                    st.error(f"Error uploading {uploaded_file.name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if success_count > 0:
                st.success(f"✅ Successfully uploaded {success_count} files")
                if processed_count > 0:
                    st.success(f"🔄 Processed {processed_count} files")
                    st.balloons()
                    st.info("🎉 New files are ready to use!")
                if error_count > 0:
                    st.warning(f"⚠️ Errors uploading: {error_count}")
                
                st.rerun()
            else:
                st.error("❌ Failed to upload any files")

if "current_document" not in st.session_state:
    st.session_state.current_document = document_options[0] if document_options else None

document_options = get_document_options()

if document_options:
    selected_document = st.selectbox(
        "Select the task you're working on:",
        document_options,
        index=document_options.index(st.session_state.current_document) if st.session_state.current_document in document_options else 0
    )
    st.session_state.current_document = selected_document
    
    st.info(f"📄 Working on task: **{selected_document}**")
else:
    st.info("📝 No available tasks. Upload files in the sidebar.")
    st.session_state.current_document = None

if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model='gpt-4.1-mini', api_key=os.getenv("OPENAI_API_KEY"))

if "messages" not in st.session_state:
    st.session_state.messages = []

if "step" not in st.session_state:
    st.session_state.step = "orientation"

async def handle_query(prompt, llm, messages, step, current_document):
    async with connect_to_server() as session:
        tools = await load_tools(session)
        llm_with_tools = llm.bind_tools(tools)

        agent = Agent()

        result_state = await agent.setupState(
            query=prompt,
            llm=llm_with_tools,
            available_tools=tools,
            messages=messages,
            step=step,
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
    if not st.session_state.current_document:
        st.error("⚠️ Please upload and select a task in the sidebar.")
    else:
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
                    st.session_state.step,
                    st.session_state.current_document
                ))

                last_message = ""
                if "messages" in result and result["messages"]:
                    last_message = result["messages"][-1].content

                placeholder.markdown(last_message)

                if "messages" in result and last_message:
                    st.session_state.messages.append(AIMessage(content=last_message))

                if "step" in result:
                    st.session_state.step = result["step"]

            except Exception as e:
                placeholder.error(f"An error occurred: {e}")
