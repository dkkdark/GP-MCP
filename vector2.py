
import os
import json
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from openai import OpenAI

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
    system_prompt = """
You are an AI assistant designed to deeply analyze and structure educational assignments for students. Your task is to extract and, if necessary, generate the following types of semantic chunks from the provided assignment text:

- \"concept\": The main idea or core definition of the assignment. If not explicitly stated, infer and formulate it yourself.
- \"solution\": Solutions, hints, or problem-solving strategies relevant to the assignment. If not present, suggest possible approaches based on the content.
- \"qa\": All questions that the student is expected to answer. Identify both explicit and implicit questions.
- \"example\": Illustrative examples that clarify the assignment. If none are provided, create a suitable example.
- \"definition\": Clear term-definition pairs. Extract or generate these as needed.
- \"instruction\": Specific tasks or steps the student must perform. Identify all actionable instructions.
- \"table\": Any tables of data present in the assignment. Structure them clearly; if none exist, do not fabricate.

Guidelines:
- Each chunk should be at least 100 words long, if possible.
- Do not omit any sentence from the assignment text; ensure all content is included in at least one chunk.
- If a required chunk type is missing from the assignment, generate it based on your analysis and understanding.
- Return ONLY a list of JSON objects, one per chunk, in the following format:
[
  {\"type\": \"concept\", \"text\": \"...\"},
  {\"type\": \"qa\", \"text\": \"...\"},
  ...
]
- Be thorough and ensure the output is as complete and helpful as possible for downstream educational applications.
"""
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
            metadata={"type": chunk["type"]} # document name, topic, keywords
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

            print(semantic_chunks)

    db = store_in_chroma(all_docs, db_path)
    return db

filepath = ".//data/tasks.docx"
db_path = "./teaching_chroma_db"

db = process_directory(".//data", db_path)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10})
print("All documents processed and stored. Retriever is ready.")
