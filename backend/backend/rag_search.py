"""
PRODUCTION-READY HYBRID RETRIEVAL SYSTEM (Streamlit Compatible)
-----------------------------------------------------------
Features:
✔ Document ingestion (txt, pdf, docx, html)
✔ Text splitting
✔ Embeddings with MPNet (768d)
✔ Chroma Vector Store
✔ Wikipedia Retriever
✔ Hybrid Retrieval (local + Wikipedia)
✔ Answer generation using Groq LLaMA 3.3 70B
✔ Logging + Error Handling
✔ Dynamic Model Fetching
"""

import os
import logging
import requests
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader,
    Docx2txtLoader, UnstructuredHTMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import WikipediaRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure Logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load Environment
load_dotenv()

# Configuration
# Path to the 'documents' folder at project root (backend/backend/../../documents)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "backend/documents"
CHROMA_DIR ="chroma_db"

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

def get_groq_api_key():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        logger.warning("GROQ_API_KEY is missing in environment variables.")
    return key

# ============================================================
# Core Functions
# ============================================================

def fetch_groq_models():
    """Fetches available models from Groq API."""
    key = get_groq_api_key()
    if not key:
        return ["llama-3.3-70b-versatile"] # Fallback
        
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # Filter for Text Generation / Chat models only
            # Exclude TTS, Whisper, etc.
            all_models = [model['id'] for model in data.get('data', [])]
            chat_models = [
                m for m in all_models 
                if any(x in m.lower() for x in ['llama', 'mixtral', 'gemma', 'deepseek']) 
                and not any(x in m.lower() for x in ['tts', 'whisper', 'vision', 'guard'])
            ]
            return chat_models if chat_models else ["llama-3.3-70b-versatile"]
        else:
            logger.error(f"Failed to fetch models: {response.text}")
            return ["llama-3.3-70b-versatile"]
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return ["llama-3.3-70b-versatile"]

def load_documents(data_dir: Path):
    """Loads and returns documents using multiple loader types."""
    logger.info(f"Loading documents from: {data_dir}")
    loaders_map = {
        "*.txt": TextLoader,
        "*.pdf": PyPDFLoader,
        "*.docx": Docx2txtLoader,
        "*.html": UnstructuredHTMLLoader,
    }
    all_docs = []
    
    if not data_dir.exists():
        logger.error(f"Document directory not found: {data_dir}")
        return []

    for pattern, loader_cls in loaders_map.items():
        try:
            # glob needs string path if using DirectoryLoader's glob arg? 
            # DirectoryLoader takes path as first arg.
            loader = DirectoryLoader(
                str(data_dir), glob=pattern, loader_cls=loader_cls, show_progress=False
            )
            docs = loader.load()
            all_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} files ({pattern})")
        except Exception as e:
            logger.error(f"Error loading pattern {pattern}: {e}")

    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    # Check if GPU is available (mps for mac), otherwise cpu
    # For simplicity and broad compatibility, let's stick to default or cpu if mps causes issues in some envs
    # but user code had 'mps'.
    device = 'cpu' # default safer
    try:
        import torch
        if torch.backends.mps.is_available(): 
             device = 'mps'
    except:
        pass

    # Authenticate with Hugging Face if token is present
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        # caching usually works without explicit set if env is set, but explicit is safer
        os.environ["HF_TOKEN"] = hf_token 
    else:
        logger.warning("HF_TOKEN is missing. You may hit rate limits with Hugging Face models.")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={'device': device, 'token': hf_token},
            encode_kwargs={'normalize_embeddings': True}
        )
    except TypeError:
        # Fallback for older sentence-transformers versions
        logger.warning("Failed to init with 'token', trying 'use_auth_token'...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={'device': device, 'use_auth_token': hf_token},
            encode_kwargs={'normalize_embeddings': True}
        )

    vector_store = Chroma(
        collection_name="wikipedia_docs", # Keeping user's collection name
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    if chunks:
        # Check if already populated to avoid duplicates? 
        # Chroma .add_texts appends. If we want idempotency we might need to check.
        # utilizing persist_directory implies we might be loading existing.
        # For this assignment, let's assume we reload if we pass chunks.
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        
        # Batching might be needed for large sets, but for assignment likely fine.
        vector_store.add_texts(texts=texts, metadatas=metadatas)
        logger.info("Vector store updated successfully")

    return vector_store

def create_retrievers(vector_store):
    vector_ret = vector_store.as_retriever(search_kwargs={"k": 3})
    wiki_ret = WikipediaRetriever(top_k_results=3, doc_content_chars_max=2000)
    return vector_ret, wiki_ret

def create_llm(model_name=LLM_MODEL):
    key = get_groq_api_key()
    if not key:
        return None
    return ChatGroq(
        temperature=0.3,
        model_name=model_name,
        groq_api_key=key
    )

def create_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """You are a precise Question Answering AI.
Your goal is to answer the user's SPECIFIC question directly and concisely using only the context provided.
- Do NOT summarize the documents unless explicitly asked involved in the question.
- Focus on finding the exact answer.
- If the answer is not in the context, say "I cannot find the answer in the provided documents."
"""),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

def create_summary_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """You are an expert Research Assistant and Summarizer.
Your goal is to provide a comprehensive summary of the information found in the context related to the user's topic.
- Synthesize the information into a cohesive narrative or structured list.
- Do NOT just answer a single question; look for the broader themes in the context.
- Capture key details, dates, and concepts.
- Strict Adherence to Length: {length_instruction}
"""),
        ("human", "Context:\n{context}\n\nTopic/Query:\n{question}\n\nGuidance: {length_instruction}")
    ])

def answer_query(llm, prompt, question, context):
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "context": context,
        "question": question
    })

def generate_summary(llm, prompt, question, context, length="Medium"):
    length_map = {
        "Short": "Concise, around 2-3 sentences.",
        "Medium": "Balanced, around 1 paragraph.",
        "Long": "Detailed, comprehensive coverage, possibly using bullet points."
    }
    instruction = length_map.get(length, length_map["Medium"])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "context": context,
        "question": question,
        "length_instruction": instruction
    })

def initialize_system():
    """
    Initializes the RAG system components (Vector Store, Retrievers, LLM).
    Returns (vector_ret, wiki_ret, prompt, summary_prompt)
    """
    logger.info("Initializing system...")
    
    # 1. Init Vector Store (DB) without chunks first
    try:
        vector_store = create_vector_store(chunks=None)
        # 2. Check if DB is empty
        # Chroma wrapper usually exposes the underlying collection via _collection
        collection_count = vector_store._collection.count()
    except Exception as e:
        logger.error(f"Failed to load Vector Store (likely corrupt or incompatible): {e}")
        logger.info("Attempting to reset and rebuild Vector Store...")
        import shutil
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        
        # Retry creation
        vector_store = create_vector_store(chunks=None)
        collection_count = 0

    if collection_count == 0:
        logger.info("Vector store is empty. Ingesting documents...")
        docs = load_documents(DATA_DIR)
        chunks = split_documents(docs)
        if chunks:
            vector_store.add_documents(chunks)
            logger.info(f"Ingested {len(chunks)} chunks into ChromaDB.")
    else:
        logger.info(f"Vector store loaded with {collection_count} existing documents. Skipping ingestion.")
    
    # 3. Retrievers
    vector_ret, wiki_ret = create_retrievers(vector_store)
    
    # 4. Prompts (LLM is now created dynamically)
    prompt = create_prompt()
    summary_prompt = create_summary_prompt()
    
    return vector_ret, wiki_ret, prompt, summary_prompt

def hybrid_retrieve(vector_ret, wiki_ret, query):
    all_docs = []
    
    # Vector Retrieve
    vec_res = ""
    try:
        docs = vector_ret.invoke(query)
        if docs:
            vec_res = "=== COURSE MATERIAL ===\n" + "\n\n".join([d.page_content for d in docs])
            for d in docs:
                d.metadata['source_type'] = 'Local Document'
                all_docs.append(d)
        else:
            vec_res = "No vector results found."
    except Exception as e:
        vec_res = f"Vector retrieval error: {e}"

    # Wikipedia Retrieve
    wiki_res = ""
    try:
        wiki_docs = wiki_ret.invoke(query)
        if wiki_docs:
            wiki_res = "=== WIKIPEDIA ===\n" + "\n\n".join([d.page_content for d in wiki_docs])
            for d in wiki_docs:
                d.metadata['source_type'] = 'Wikipedia'
                all_docs.append(d)
        else:
            wiki_res = "No Wikipedia results found."
    except Exception as e:
        wiki_res = f"Wikipedia retrieval error: {e}"

    combined_context = vec_res + "\n\n" + wiki_res
    return combined_context, all_docs

if __name__ == "__main__":
    initialize_system()
