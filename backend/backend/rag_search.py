

import os
import logging
import tempfile
import shutil
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


# ------------------------------------------------------------------
# Paths & Config
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "backend/documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
COLLECTION_NAME = "ever_quint"

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def get_groq_api_key():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        logger.warning("GROQ_API_KEY not found")
    return key

def fetch_groq_models():
    """Dynamically fetch available chat models from Groq API (text generation/summarization only)."""
    api_key = get_groq_api_key()
    
    if not api_key:
        logger.error("GROQ_API_KEY not found. Cannot fetch models.")
        return []
    
    try:
        import requests
        url = "https://api.groq.com/openai/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Filter for chat completion models (text generation/summarization)
            # Include models with these keywords, exclude safety/audio models
            models = []
            for model in data.get("data", []):
                model_id = model.get("id", "")
                model_id_lower = model_id.lower()
                
                # Include chat/text generation models
                is_chat_model = any(keyword in model_id_lower for keyword in [
                    "versatile", "instruct", "instant", "chat"
                ])
                
                # Also include specific model families known for chat
                is_llm_family = any(family in model_id_lower for family in [
                    "llama-3", "mixtral", "gemma"
                ])
                
                # Exclude non-chat models
                is_excluded = any(excluded in model_id_lower for excluded in [
                    "guard", "whisper", "distil-whisper", "embedding"
                ])
                
                if (is_chat_model or is_llm_family) and not is_excluded:
                    models.append(model_id)
            
            if models:
                logger.info(f"✓ Fetched {len(models)} chat models from Groq API")
                return sorted(models, reverse=True)  # Newest first
            else:
                logger.warning("No suitable chat models found in API response")
                return []
        else:
            logger.error(f"API request failed with status {response.status_code}")
            return []
        
    except Exception as e:
        logger.error(f"Failed to fetch models from Groq API: {e}")
        return []

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

def process_uploaded_file(uploaded_file):
    """
    Process a Streamlit uploaded_file object:
    1. Save to temp file
    2. Load with appropriate loader
    3. Split into chunks
    4. Return chunks
    """
    try:
        # Create a temp file with the same extension
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        logger.info(f"Processing uploaded file: {uploaded_file.name} (temp: {tmp_path})")
        
        # Select Loader
        if suffix == ".txt":
            loader = TextLoader(tmp_path)
        elif suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        elif suffix == ".html":
            loader = UnstructuredHTMLLoader(tmp_path)
        else:
            logger.error(f"Unsupported file type: {suffix}")
            os.remove(tmp_path)
            return []
            
        docs = loader.load()
        # Clean up temp file
        os.remove(tmp_path)
        
        # Split
        chunks = split_documents(docs)
        return chunks
        
    except Exception as e:
        logger.error(f"Failed to process uploaded file: {e}")
        return []

def create_ephemeral_retriever(chunks):
    """Create an in-memory vector store retriever for specific chunks."""
    if not chunks:
        return None
        
    embeddings = create_embeddings()
    try:
        # Chroma with no persist_directory is in-memory
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="ephemeral_upload"
        )
        return vector_store.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        logger.error(f"Failed to create ephemeral vector store: {e}")
        return None

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
# INITIALIZATION (FINAL, STABLE)
# ------------------------------------------------------------------
def initialize_system():
    """Initialize RAG system: load existing VDB or create new one with all documents."""
    logger.info("Initializing RAG system")

    embeddings = create_embeddings()

    if not CHROMA_DIR.exists():
        logger.info("Vector DB not found. Processing all documents...")
        docs = load_documents(DATA_DIR)
        chunks = split_documents(docs)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR)
        )
        logger.info(f"✓ Created vector DB with {len(chunks)} chunks")
    else:
        logger.info("Loading existing vector DB...")
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR)
        )
        logger.info("✓ Vector DB loaded successfully")

    vector_ret, wiki_ret = create_retrievers(vector_store)
    prompt = create_prompt()
    summary_prompt = create_summary_prompt()

    return vector_ret, wiki_ret, prompt, summary_prompt

# ------------------------------------------------------------------
# HYBRID RETRIEVAL
# ------------------------------------------------------------------
def hybrid_retrieve(vector_ret, wiki_ret, query):
    """Retrieve from both vector store and Wikipedia, return (context, docs)."""
    context_parts = []
    all_docs = []

    # Vector Store retrieval
    vec_docs = []
    if vector_ret:
        try:
            vec_docs = vector_ret.invoke(query)
            if vec_docs:
                context_parts.append(
                    "=== LOCAL DOCUMENTS ===\n" +
                    "\n\n".join(d.page_content for d in vec_docs)
                )
                for doc in vec_docs:
                    doc.metadata['source_type'] = 'Local Document'
                    all_docs.append(doc)
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")

    # Wikipedia retrieval
    wiki_docs = []
    if wiki_ret:
        try:
            wiki_docs = wiki_ret.invoke(query)
            if wiki_docs:
                context_parts.append(
                    "=== WIKIPEDIA ===\n" +
                    "\n\n".join(d.page_content for d in wiki_docs)
                )
                for doc in wiki_docs:
                    doc.metadata['source_type'] = 'Wikipedia'
                    all_docs.append(doc)
        except Exception as e:
            logger.error(f"Wikipedia retrieval failed: {e}")

    combined_context = "\n\n".join(context_parts)
    return combined_context, all_docs

# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------
if __name__ == "__main__":
    initialize_system()