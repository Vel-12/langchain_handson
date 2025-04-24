from typing import List, Dict, Any
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
import tempfile


class ResearchAssistantLangChain:
    def __init__(self):
        """Initialize the Research Assistant with Ollama model"""
        # Use a valid installed model name (e.g., "llama3", "mistral")
        self.llm = ChatOllama(
            base_url="http://localhost:11434", 
            model="llama3"  # Updated model name
        )
        # Explicitly set embeddings model (same as LLM or compatible)
        self.embeddings = OllamaEmbeddings(model="llama3")  # Added model parameter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_db = None
        self.persist_directory = tempfile.mkdtemp()
        print(f"Created temporary FAISS directory: {self.persist_directory}")


    def load_urls(self, urls: List[str]) -> List[Document]:
        all_documents = []
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                data = loader.load()
                split_documents = self.text_splitter.split_documents(data)
                all_documents.extend(split_documents)
                print(f"Successfully loaded and processed {url}")
            except Exception as e:
                print(f"Error loading {url}: {e}")
        return all_documents


    def create_vector_store(self, documents: List[Document]):
        try:
            self.vector_db = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            # Optional: Save the FAISS index to the persist directory
            self.vector_db.save_local(self.persist_directory)
            print(f"Created FAISS vector store with {len(documents)} documents")
        except Exception as e:
            print(f"Error creating FAISS vector store: {e}")
    

    def query_data(self, query: str, num_results: int = 5) -> Dict[str, Any]:

        if not self.vector_db:
            raise ValueError(f"Vector store not initialized. Please load documents first")
        
        retriever = self.vector_db.as_retriever(search_kwargs = {"k" : num_results})
        
        promt_text = """
        Answer the following question based only on the provided context:

        <context>
        {context}
        </context>

        Question: {input}
        """

        prompt = ChatPromptTemplate.from_template(promt_text)

        docuemnt_chain = create_stuff_documents_chain(self.llm, prompt)

        retrieval_chain = create_retrieval_chain(retriever, docuemnt_chain)

        response = retrieval_chain.invoke({"input": query})

        return {
            "answer": response["answer"],
            "source_documents": response["context"]
        }


    def summarize_documents(self, text: str) -> str:
        message = HumanMessage(content=f"Summarize the following document in a concise but comprehensive manner:\n\n{text}")

        response = self.llm.invoke([message])

        return response.content