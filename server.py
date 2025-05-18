from mcp.server.fastmcp import FastMCP
from config import Config
from vector import retriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

mcp = FastMCP("Teaching AI")

model = ChatOpenAI(model='gpt-4.1-mini') 

template = """
Here are some relevant data related to the question (data): {data}

Here is the question to answer: {question}

Give only the information you found in data
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

@mcp.tool()
def get_task_answer(question: str) -> int:
    data = retriever.invoke(question)
    print(data)
    result = chain.invoke({"data": data, "question": question})
    return result

if __name__ == "__main__":
    mcp.run(transport=Config.Server.TRANSPORT)