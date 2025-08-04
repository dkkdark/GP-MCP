import os
import json
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from openai import OpenAI
from common.prompts import Prompts

class DocumentProcessor:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.embedding_function = OpenAIEmbeddings()
        self.db = None

    def load_docx_plain(self, filepath):
        doc = DocxDocument(filepath)
        full_text = []
        for element in doc.element.body:
            if element.tag.endswith('p'):
                para = element.xpath(".//w:t")
                if para:
                    text = ''.join([t.text for t in para if t.text])
                    full_text.append(text.strip())
            elif element.tag.endswith('tbl'):
                for row in element.xpath(".//w:tr"):
                    cells = row.xpath(".//w:tc")
                    row_text = [''.join([t.text for t in cell.xpath(".//w:t") if t.text]).strip() for cell in cells]
                    full_text.append(' | '.join(row_text))
        return '\n'.join(full_text)

    def extract_semantic_chunks(self, doc_text):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompts = Prompts()
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": prompts.get_chunck_splitter_prompt()},
                {"role": "user", "content": doc_text}
            ],
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)

    def chunk_large_items(self, semantic_chunks, doc_id, filepath):
        final_chunks = []
        for i, chunk in enumerate(semantic_chunks):
            final_chunks.append({
                "id": f"{filepath}_{doc_id}_{chunk['type']}_{i}",
                "text": chunk["text"],
                "type": chunk["type"]
            })
        return final_chunks

    def to_langchain_documents(self, chunks):
        return [
            Document(
                page_content=chunk["text"],
                metadata={"type": chunk["type"], "doc_id": chunk["id"]}
            ) for chunk in chunks
        ]

    def process_directory(self, input_dir):
        all_docs = []
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if filename.endswith(".docx"):
                    filepath = os.path.join(root, filename)
                    doc_id = os.path.splitext(filename)[0]
                    doc_text = self.load_docx_plain(filepath)
                    semantic_chunks = self.extract_semantic_chunks(doc_text)
                    final_chunks = self.chunk_large_items(semantic_chunks, doc_id, filepath)
                    docs = self.to_langchain_documents(final_chunks)
                    all_docs.extend(docs)
                    
        self.db = Chroma(persist_directory=self.db_path, embedding_function=self.embedding_function)
        self.db.add_documents(all_docs)

        with open('chuncks.txt', 'w') as f:
            for item in all_docs:
                f.write(f"{item}\n")

        return self.db

    def load_existing_db(self):
        self.db = Chroma(persist_directory=self.db_path, embedding_function=self.embedding_function)
        return self.db

    def get_retriever(self, search_type="mmr", k=6):
        db = self.load_existing_db()
        return db.as_retriever(search_type=search_type, search_kwargs={"k": k})

    @staticmethod
    def get_chunk_types_for_step(step):
        step_to_types = {
            "orientation": ["concept", "example", "instruction", "definition", "table"],
            "conceptualization": ["concept", "definition", "example", "table"],
            "solution ideation": ["solution", "example", "qa", "table"],
            "planning": ["instruction", "solution", "table"],
            "execution support": ["instruction", "solution", "qa", "table"],
        }
        return step_to_types.get(step, [])

    def get_chunks_for_step(self, step, retriever, query="*", current_document=None):
        types = self.get_chunk_types_for_step(step)
        results = retriever.get_relevant_documents(query)
        print(results)
        filtered = [doc for doc in results]
        if current_document:
            filtered = [
                doc for doc in filtered
                if current_document in doc.metadata.get("doc_id", "") or "materials" in doc.metadata.get("doc_id", "")
            ]
            print(f"filter {filtered}")
        return filtered
    
    def initialize_or_load_db(self, input_dir):
        if not os.path.exists(self.db_path):
            print("No existing DB found. Creating new one...")
            return self.process_directory(input_dir)
        else:
            print("Existing DB found. Loading...")
            return self.load_existing_db()
