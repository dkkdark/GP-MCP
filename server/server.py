from mcp.server.fastmcp import FastMCP
from common.config import Config
from server.document_processor import DocumentProcessor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

processor = DocumentProcessor(db_path="./teaching_chroma_db")
db = processor.initialize_or_load_db(input_dir="./data") 
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 6})

model = ChatOpenAI(model='gpt-4.1-mini', api_key=os.getenv("OPENAI_API_KEY"))
template = """
Here are some relevant data related to the question (data): {data}

Here is the question to answer: {question}

Give the answer based only on the information you found in the data
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

mcp = FastMCP("Teaching AI")

@mcp.tool()
def get_task_answer(question: str, step: str, current_document: str) -> str:
    print(f"question {question}")
    if step:
        chunks = processor.get_chunks_for_step(step, retriever, question, current_document)
        data = [doc.page_content for doc in chunks]
    else:
        data = retriever.invoke(question)
    result = chain.invoke({"data": data, "question": question}).content
    return result

if __name__ == "__main__":
    mcp.run(transport=Config.Server.TRANSPORT)
