"""
PRODUCTION-READY HYBRID RETRIEVAL SYSTEM (Streamlit Compatible)
-----------------------------------------------------------
✔ Document ingestion (txt, pdf, docx, html)
✔ Text splitting
✔ MPNet embeddings (768d)
✔ Chroma Vector Store (tenant-safe)
✔ Wikipedia Retriever
✔ Hybrid Retrieval (Local + Wikipedia)
✔ Answer generation using Groq LLaMA 3.3 70B
✔ Logging + Error Handling
✔ Streamlit Cloud compatible
"""

import os
import logging
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

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ------------------------------------------------------------------
# Paths & Config
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "backend/documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

CHROMA_TENANT = "default_tenant"
CHROMA_DATABASE = "default_database"
COLLECTION_NAME = "wikipedia_docs"

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def get_groq_api_key():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        logger.warning("GROQ_API_KEY not found")
    return key

# ------------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------------
def create_embeddings():
    device = "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

# ------------------------------------------------------------------
# Document Loading
# ------------------------------------------------------------------
def load_documents(data_dir: Path):
    logger.info(f"Loading documents from {data_dir}")

    loaders = {
        "*.txt": TextLoader,
        "*.pdf": PyPDFLoader,
        "*.docx": Docx2txtLoader,
        "*.html": UnstructuredHTMLLoader,
    }

    documents = []
    if not data_dir.exists():
        logger.warning("Documents directory not found")
        return documents

    for pattern, loader_cls in loaders.items():
        try:
            loader = DirectoryLoader(
                str(data_dir),
                glob=pattern,
                loader_cls=loader_cls,
                show_progress=False
            )
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} files ({pattern})")
        except Exception as e:
            logger.error(f"Failed loading {pattern}: {e}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents

# ------------------------------------------------------------------
# Text Splitting
# ------------------------------------------------------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

# ------------------------------------------------------------------
# Retrievers
# ------------------------------------------------------------------
def create_retrievers(vector_store):
    vector_ret = vector_store.as_retriever(search_kwargs={"k": 3})
    wiki_ret = WikipediaRetriever(top_k_results=3, doc_content_chars_max=2000)
    return vector_ret, wiki_ret

# ------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------
def create_llm(model_name=LLM_MODEL):
    key = get_groq_api_key()
    if not key:
        return None
    return ChatGroq(
        model_name=model_name,
        temperature=0.3,
        groq_api_key=key
    )

# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------
def create_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise Question Answering AI. "
         "Answer ONLY using the provided context. "
         "If the answer is not found, say so clearly."),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

def create_summary_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional research summarizer. "
         "Follow the length instruction strictly: {length_instruction}"),
        ("human", "Context:\n{context}\n\nTopic:\n{question}")
    ])

# ------------------------------------------------------------------
# QA Helpers
# ------------------------------------------------------------------
def answer_query(llm, prompt, question, context):
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

def generate_summary(llm, prompt, question, context, length="Medium"):
    length_map = {
        "Short": "2–3 sentences",
        "Medium": "1 paragraph",
        "Long": "Detailed explanation with bullet points"
    }
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "context": context,
        "question": question,
        "length_instruction": length_map.get(length, "1 paragraph")
    })

# ------------------------------------------------------------------
# INITIALIZATION (TENANT-SAFE & CLOUD-SAFE)
# ------------------------------------------------------------------
def initialize_system():
    logger.info("Initializing Hybrid RAG system")

    embeddings = create_embeddings()

    if not CHROMA_DIR.exists():
        logger.info("Chroma DB not found. Creating new vector store...")

        docs = load_documents(DATA_DIR)
        chunks = split_documents(docs)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE
        )
        vector_store.persist()
        logger.info(f"Vector DB created with {len(chunks)} chunks")

    else:
        logger.info("Existing Chroma DB found. Loading...")

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE
        )

    vector_ret, wiki_ret = create_retrievers(vector_store)
    prompt = create_prompt()
    summary_prompt = create_summary_prompt()

    return vector_ret, wiki_ret, prompt, summary_prompt

# ------------------------------------------------------------------
# HYBRID RETRIEVAL
# ------------------------------------------------------------------
def hybrid_retrieve(vector_ret, wiki_ret, query):
    context_parts = []

    try:
        vec_docs = vector_ret.invoke(query)
        if vec_docs:
            context_parts.append(
                "=== LOCAL DOCUMENTS ===\n" +
                "\n\n".join(d.page_content for d in vec_docs)
            )
    except Exception as e:
        logger.error(f"Vector retrieval failed: {e}")

    try:
        wiki_docs = wiki_ret.invoke(query)
        if wiki_docs:
            context_parts.append(
                "=== WIKIPEDIA ===\n" +
                "\n\n".join(d.page_content for d in wiki_docs)
            )
    except Exception as e:
        logger.error(f"Wikipedia retrieval failed: {e}")

    return "\n\n".join(context_parts)

# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------
if __name__ == "__main__":
    initialize_system()
