from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5
from functools import lru_cache
import tempfile
import faiss
import os
import pickle
import numpy as np
import time

# Configure FAISS for optimal performance
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 6)
faiss.omp_set_num_threads(os.cpu_count() or 6)

class ResearchAssistantLangChain:
    def __init__(self, persist_dir: Optional[str] = None):
        # Initialize Cohere components
        self.llm = ChatCohere(
            model="command-r",
            temperature=0.1,
            max_tokens=4096
        )

        self.embeddings = CohereEmbeddings(model="embed-english-v3.0")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            length_function=len
        )

        self.persist_directory = Path(persist_dir) if persist_dir else Path(tempfile.mkdtemp())
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.vector_db = self._load_vector_store()
        self.url_cache = self._load_url_cache()
        self.MIN_CONTENT_LENGTH = 500

        # Optimize index on startup
        self._optimize_faiss_index()

    def _load_vector_store(self) -> Optional[FAISS]:
        index_files = ["index.faiss", "index.pkl"]
        if all((self.persist_directory / f).exists() for f in index_files):
            try:
                return FAISS.load_local(
                    str(self.persist_directory),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading vector store: {e}")
        return None

    def _load_url_cache(self) -> Dict[str, List[Document]]:
        cache_file = self.persist_directory / "url_cache.pkl"
        try:
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading URL cache: {e}")
        return {}

    def _save_url_cache(self):
        with open(self.persist_directory / "url_cache.pkl", "wb") as f:
            pickle.dump(self.url_cache, f)

    def _optimize_faiss_index(self):
        """Convert to IVFFlat index for faster searches"""
        if self.vector_db and not isinstance(self.vector_db.index, faiss.IndexIVFFlat):
            print("Optimizing FAISS index structure...")
            index = self.vector_db.index
            nlist = min(1024, index.ntotal)
            quantizer = faiss.IndexFlatL2(index.d)
            new_index = faiss.IndexIVFFlat(quantizer, index.d, nlist)
            
            if index.ntotal > 0:
                vectors = index.reconstruct_n(0, index.ntotal)
                if not new_index.is_trained:
                    new_index.train(vectors)
                new_index.add(vectors)
                
            new_index.nprobe = 16  # Balance speed/accuracy
            self.vector_db.index = new_index
            self.vector_db.save_local(str(self.persist_directory))

    def load_urls(self, urls: List[str]) -> List[Document]:
        new_urls = [url for url in urls if md5(url.encode()).hexdigest() not in self.url_cache]
        
        if new_urls:
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(self._process_single_url, new_urls))
                
            for url, docs in zip(new_urls, results):
                if docs:
                    self.url_cache[md5(url.encode()).hexdigest()] = docs
                    self._save_url_cache()
                    
        return [doc for url in urls for doc in self.url_cache.get(md5(url.encode()).hexdigest(), [])]

    def _process_single_url(self, url: str) -> List[Document]:
        try:
            docs = WebBaseLoader(url).load()
            return [doc for doc in self.text_splitter.split_documents(docs) 
                   if len(doc.page_content) >= self.MIN_CONTENT_LENGTH]
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return []

    def create_vector_store(self, documents: List[Document]):
        if documents:
            start = time.time()
            if self.vector_db:
                self.vector_db.add_documents(documents)
            else:
                self.vector_db = FAISS.from_documents(documents, self.embeddings)
            
            # Save and optimize
            self.vector_db.save_local(str(self.persist_directory))
            self._optimize_faiss_index()
            print(f"Index updated in {time.time()-start:.2f}s (Total docs: {self.vector_db.index.ntotal})")

    @lru_cache(maxsize=100)
    def query_data(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Optimized retrieval with caching"""
        if not self.vector_db:
            return {"answer": "No data loaded", "sources": []}

        try:
            retriever = self.vector_db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": num_results, "score_threshold": 0.3}
            )

            prompt = ChatPromptTemplate.from_template("""
            Answer concisely using these facts: {context}
            Question: {input}
            Answer in 3-5 bullet points with sources like [1],[2]:
            """)

            chain = create_retrieval_chain(
                retriever,
                create_stuff_documents_chain(self.llm, prompt)
            )

            start = time.time()
            result = chain.invoke({"input": query})
            print(f"Retrieval completed in {time.time()-start:.2f}s")
            
            return {
                "answer": result["answer"],
                "sources": list({doc.metadata.get('source', '') for doc in result["context"]})
            }
        except Exception as e:
            print(f"Query error: {e}")
            return {"answer": "Error processing query", "sources": []}

    def summarize_document(self, text: str) -> str:
        return self.llm.invoke([HumanMessage(
            content=f"Summarize key points from this text in 5 bullets:\n{text}"
        )]).content