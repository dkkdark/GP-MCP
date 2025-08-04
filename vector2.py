
import os
import json
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from openai import OpenAI
import prompts

def load_docx_plain(filepath):
    doc = DocxDocument(filepath)
    full_text = []
    for element in doc.element.body:
        if element.tag.endswith('p'):
            para = element.xpath(".//w:t")
            if para:
                text = ''.join([t.text for t in para if t.text])
                full_text.append(text.strip())
        elif element.tag.endswith('tbl'):
            table = element
            for row in table.xpath(".//w:tr"):
                cells = row.xpath(".//w:tc")
                row_text = []
                for cell in cells:
                    cell_text = ''.join([t.text for t in cell.xpath(".//w:t") if t.text])
                    row_text.append(cell_text.strip())
                full_text.append(' | '.join(row_text))
    return '\n'.join(full_text)

def extract_semantic_chunks(doc_text):
    system_prompt = prompts.get_chunck_splitter_prompt()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": doc_text}
        ],
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)

def chunk_large_items(semantic_chunks, doc_id, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    final_chunks = []
    for i, chunk in enumerate(semantic_chunks):
        if len(chunk["text"]) > chunk_size:
            parts = splitter.split_text(chunk["text"])
            for j, part in enumerate(parts):
                final_chunks.append({
                    "id": f"{doc_id}_{chunk['type']}_{i}_{j}",
                    "text": part,
                    "type": chunk["type"]
                })
        else:
            final_chunks.append({
                "id": f"{doc_id}_{chunk['type']}_{i}_0",
                "text": chunk["text"],
                "type": chunk["type"]
            })
    return final_chunks

def to_langchain_documents(chunks):
    return [
        Document(
            page_content=chunk["text"],
            metadata={"type": chunk["type"], "doc_id": chunk["id"].split("_")[0]} # document name, topic, keywords
        ) for chunk in chunks
    ]

def store_in_chroma(docs, db_path):
    db = Chroma(
        persist_directory=db_path,
        embedding_function=OpenAIEmbeddings()
    )
    db.add_documents(docs)
    return db

def process_directory(input_dir, db_path):
    all_docs = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".docx"):
            filepath = os.path.join(input_dir, filename)
            doc_id = os.path.splitext(filename)[0]
            doc_text = load_docx_plain(filepath)
            semantic_chunks = extract_semantic_chunks(doc_text)
            final_chunks = chunk_large_items(semantic_chunks, doc_id)
            docs = to_langchain_documents(final_chunks)
            all_docs.extend(docs)

            print(final_chunks)

    db = store_in_chroma(all_docs, db_path)
    return db

def get_chunk_types_for_step(step):
    step_to_types = {
        "orientation": ["concept", "example", "instruction", "definition"],
        "conceptualization": ["concept", "definition", "example"],
        "solution ideation": ["solution", "example", "qa"],
        "planning": ["instruction", "solution", "table"],
        "execution support": ["instruction", "solution", "qa", "table"],
    }
    return step_to_types.get(step, [])


def get_chunks_for_step(step, retriever, query="*", current_document=None):
    types = get_chunk_types_for_step(step)
    results = retriever.get_relevant_documents(query)
    filtered = [doc for doc in results if doc.metadata.get("type") in types]
    
    if current_document:
        print(f"current_documents = {filtered}")
        filtered = [doc for doc in filtered if current_document in doc.metadata.get("doc_id", "")]
    
    return filtered

filepath = ".//data/tasks.docx"
db_path = "./teaching_chroma_db"

db = process_directory(".//data", db_path)
# db = Chroma(
#         persist_directory=db_path,
#         embedding_function=OpenAIEmbeddings()
#     )
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10})
print("All documents processed and stored. Retriever is ready.")
