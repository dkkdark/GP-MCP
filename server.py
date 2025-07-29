from mcp.server.fastmcp import FastMCP
from config import Config
from vector2 import retriever, get_chunks_for_step
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("Teaching AI")

model = ChatOpenAI(model='gpt-4.1-mini', api_key=os.getenv("OPENAI_API_KEY")) 

template = """
Here are some relevant data related to the question (data): {data}

Here is the question to answer: {question}

Give only the information you found in data
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

@mcp.tool()
def get_task_answer(question: str, step: str, current_document: str) -> int:
    print(f"qqqq question {question}")
    print(f"qqqq step {step}")
    if step:
        # Получаем только релевантные чанки для этапа
        chunks = get_chunks_for_step(step, retriever, question, current_document)
        data = [doc.page_content for doc in chunks]
    else:
        # Старое поведение — все чанки
        data = retriever.invoke(question)
    print(data)
    result = chain.invoke({"data": data, "question": question})
    return result

if __name__ == "__main__":
    mcp.run(transport=Config.Server.TRANSPORT)