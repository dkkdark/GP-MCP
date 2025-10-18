import os
import json
from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
from common.prompts import Prompts
from dotenv import load_dotenv
load_dotenv(override=True)

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

    def process_single_file(self, filepath):
        if not filepath.endswith(".docx"):
            raise ValueError("Only .docx files are supported")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        print(f"Processing file: {filepath}")
        
        doc_id = os.path.splitext(os.path.basename(filepath))[0]
        
        doc_text = self.load_docx_plain(filepath)
        print(f"Extracted text length: {len(doc_text)} characters")
        
        semantic_chunks = self.extract_semantic_chunks(doc_text)
        print(f"Created semantic chunks: {len(semantic_chunks)}")
        
        final_chunks = self.chunk_large_items(semantic_chunks, doc_id, filepath)
        print(f"Created final chunks: {len(final_chunks)}")
        
        docs = self.to_langchain_documents(final_chunks)
        
        if os.path.exists(self.db_path):
            print("Loading existing database...")
            self.db = Chroma(persist_directory=self.db_path, embedding_function=self.embedding_function)
        else:
            print("Creating new database...")
            self.db = Chroma(persist_directory=self.db_path, embedding_function=self.embedding_function)
        
        print("Adding documents to database...")
        self.db.add_documents(docs)
        
        self._update_chunks_file(docs)
        
        print(f"File {filepath} successfully processed and added to database")
        return docs

    def _update_chunks_file(self, new_docs):
        try:
            existing_chunks = []
            if os.path.exists('chuncks.txt'):
                with open('chuncks.txt', 'r', encoding='utf-8') as f:
                    existing_chunks = f.readlines()
            
            with open('chuncks.txt', 'w', encoding='utf-8') as f:
                for chunk in existing_chunks:
                    f.write(chunk)
                for doc in new_docs:
                    f.write(f"{doc}\n")
        except Exception as e:
            print(f"Error updating chunks file: {e}")

    def remove_file_from_db(self, filepath):
        try:
            file_name = os.path.splitext(os.path.basename(filepath))[0]
            print(f"Removing file {file_name} from database...")
            
            if not os.path.exists(self.db_path):
                print("Database does not exist")
                return False, 0
            
            self.db = Chroma(persist_directory=self.db_path, embedding_function=self.embedding_function)
            
            all_docs = self.db.get()
            
            docs_to_remove = []
            for i, doc_id in enumerate(all_docs['ids']):
                if file_name in doc_id:
                    docs_to_remove.append(doc_id)
            
            if docs_to_remove:
                self.db.delete(ids=docs_to_remove)
                print(f"Removed {len(docs_to_remove)} chunks from database")
                
                self._remove_from_chunks_file(docs_to_remove)
                
                return True, len(docs_to_remove)
            else:
                print("Documents to remove not found")
                return True, 0
                
        except Exception as e:
            print(f"Error removing from chunks file: {e}")
            return False, str(e)

    def _remove_from_chunks_file(self, removed_ids):
        try:
            if not os.path.exists('chuncks.txt'):
                return
            
            with open('chuncks.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            filtered_lines = []
            for line in lines:
                should_keep = True
                for removed_id in removed_ids:
                    if removed_id in line:
                        should_keep = False
                        break
                if should_keep:
                    filtered_lines.append(line)
            
            with open('chuncks.txt', 'w', encoding='utf-8') as f:
                f.writelines(filtered_lines)
                
        except Exception as e:
            print(f"Error removing from chunks file: {e}")

    def get_db_stats(self):
        try:
            if not os.path.exists(self.db_path):
                return {"total_documents": 0, "message": "Database does not exist"}
            
            self.db = Chroma(persist_directory=self.db_path, embedding_function=self.embedding_function)
            all_docs = self.db.get()
            
            unique_files = set()
            for doc_id in all_docs['ids']:
                parts = doc_id.split('_')
                if len(parts) >= 2:
                    unique_files.add(parts[0])
            
            return {
                "total_documents": len(all_docs['ids']),
                "unique_files": len(unique_files),
                "file_names": list(unique_files)
            }
        except Exception as e:
            return {"error": str(e)}

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
            "execution support": ["instruction", "solution", "qa", "table", "example"],
        }
        return step_to_types.get(step, [])

    def get_chunks_for_step(self, step, retriever, query="*", current_document=None):
        types = self.get_chunk_types_for_step(step)
        results = retriever.invoke(query)
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
