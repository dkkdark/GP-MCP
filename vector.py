from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from langchain.embeddings import OpenAIEmbeddings
import os

embeddings = OpenAIEmbeddings()

db_location = ".//chrome_langchain_db"
add_documents = not os.path.exists(db_location)

splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)

raw_documents = SimpleDirectoryReader(
    input_files=[".//data/tasks.docx"]
).load_data()

documents = [
    LangchainDocument(page_content=doc.get_content(), metadata={"source": "tasks.docx"})
    for doc in raw_documents
]

nodes = splitter.split_documents(documents)

if add_documents:
    db = Chroma.from_documents(nodes, embedding=embeddings, persist_directory=db_location)
else:
    db = Chroma(persist_directory=db_location, embedding_function=embeddings)
    db.add_documents(nodes)

        
retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

