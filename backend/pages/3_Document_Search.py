import streamlit as st
import sys
import os
import logging

# fix path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.backend.logger_setup import setup_logging
from backend.backend.rag_search import (
    initialize_system, hybrid_retrieve, answer_query, generate_summary, 
    fetch_groq_models, create_llm, 
    process_uploaded_file, create_ephemeral_retriever
)

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="RAG Search", layout="wide")
st.title("EverQuint DOC AI", help="""
**How it works**:
1. **Global Search**: By default, the system searches the EverQuint knowledge base and Wikipedia.
2. **Document Search**: Upload a file (PDF, TXT, DOCX) to restrict the search to *only* that document.
3. **Modes**:
    - **Summarization**: Generates a concise summary of the content.
    - **Q&A**: Answers specific questions based on the context.
""")
# st.markdown("Retrieves from **Local Documents** (Vector) and **Wikipedia**.") # Removed as requested
logger.info("Document Search (RAG) app loaded")

# 1. Fetch Models (Cached)
@st.cache_data
def get_models_v2():
    logger.info("Fetching available Groq models...")
    models = fetch_groq_models()
    logger.info(f"Retrieved {len(models)} models")
    return models

available_models = get_models_v2()

# 2. Initialize System (Cached) - NOW WITHOUT LLM
@st.cache_resource
def get_rag_system_v5():
    return initialize_system()

try:
    with st.spinner("Initializing RAG System (Loading Docs & Retrievers)..."):
        logger.info("Initializing RAG system components...")
        # Unpack 4 values: ret, ret, prompt, prompt
        vector_ret, wiki_ret, prompt, summary_prompt = get_rag_system_v5()
    logger.info("RAG system initialized successfully")
    # st.success("System Initialized!") # Removed persistent success message
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)
    st.error(f"Failed to initialize system: {e}")
    st.stop()

# Global History Cache (Shared across users/sessions)
@st.cache_resource
def get_global_history():
    return []
global_history = get_global_history()

# 3. Top Configuration Area
st.markdown("---")
# Layout: Left for Settings, Right for Context (Upload)
col_settings, col_upload = st.columns([1.5, 1], gap="large")

with col_settings:
    st.subheader("⚙️ Settings")
    c1, c2 = st.columns(2)
    with c1:
        # Find index of Llama 3.3 Versatile if possible
        default_ix = 0
        for idx, m in enumerate(available_models):
            if "llama-3.3-70b-versatile" in m:
                default_ix = idx
                break
        selected_model = st.selectbox("LLM Model", options=available_models, index=default_ix)
    with c2:
        mode = st.selectbox("Mode", ["Summarization", "Q&A"])
    
    if mode == "Summarization":
        summary_length = st.select_slider("Summary Length", options=["Short", "Medium", "Long"])
    else:
        summary_length = "Medium"
        
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

with col_upload:
    st.subheader("📄 Context Source")
    uploaded_file = st.file_uploader("Upload a document (PDF, TXT, DOCX) to focus search", type=["pdf", "txt", "docx", "html"])
    
    # Logic for file processing
    if uploaded_file:
        file_key = f"uploader_{uploaded_file.name}_{uploaded_file.size}"
        if "current_file_key" not in st.session_state or st.session_state.current_file_key != file_key:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                chunks = process_uploaded_file(uploaded_file)
                if chunks:
                    retriever = create_ephemeral_retriever(chunks)
                    st.session_state.ephemeral_retriever = retriever
                    st.session_state.current_file_key = file_key
                    # Clear history for new document
                    st.session_state.messages = []
                    st.success(f"Locked on: {uploaded_file.name}")
                else:
                    st.error("Failed to process file.")
        
        # Override System
        if "ephemeral_retriever" in st.session_state:
            vector_ret = st.session_state.ephemeral_retriever
            wiki_ret = None
            st.info(f"✅ **Active Context**: {uploaded_file.name}")
    else:
        # Default State
        st.caption("Using: **Global Knowledge Base + Wikipedia**")

st.markdown("---")

if "ephemeral_retriever" not in st.session_state:
    st.toast("💡 Tip: Want to Upload Document Go Above 👆", icon="📄")

# 4. Chat State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handler for suggestion clicks
def handle_suggestion(q):
    st.session_state.messages.append({"role": "user", "content": q})

# 5. UI Logic
if not st.session_state.messages:
    # --- LANDING PAGE STATE ---
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; margin-bottom: 50px;">
        <h1>What's on your mind today?</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Suggestions Grid (Centered-ish via columns)
    st.markdown("### Suggested Topics")
    
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        if st.button("📝 Comprehensive overview of EverQuint", use_container_width=True):
            handle_suggestion("Comprehensive overview of EverQuint and its AI services")
            st.rerun()
        if st.button("🤖 Summarize AI Engineering capabilities", use_container_width=True):
            handle_suggestion("Summarize the AI Engineering capabilities")
            st.rerun()
            
    with s_col2:
        if st.button("👤 Who is the CEO of EverQuint?", use_container_width=True):
            handle_suggestion("Who is the CEO of EverQuint?")
            st.rerun()
        if st.button("🛠 What services does EverQuint offer?", use_container_width=True):
            handle_suggestion("What specific services does EverQuint offer?")
            st.rerun()

    # Shared History (if any)
    if global_history:
        st.markdown("### Recent User Queries")
        recent = list(dict.fromkeys(global_history))[-4:][::-1]
        cols = st.columns(len(recent) if recent else 1)
        for i, hist_q in enumerate(recent):
            with cols[i]:
                if st.button(f"Results for: {hist_q[:20]}...", key=f"hist_btn_{i}", help=hist_q, use_container_width=True):
                    handle_suggestion(hist_q)
                    st.rerun()

else:
    # --- CHAT HISTORY STATE ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Retrieved Resources"):
                     for i, doc in enumerate(message["sources"]):
                        meta = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
                        content = doc.page_content if hasattr(doc, 'page_content') else doc.get('page_content', '')
                        
                        source_type = meta.get('source_type', 'Unknown')
                        source_name = meta.get('source', 'Unknown Source')
                        
                        st.markdown(f"**{i+1}. {source_type} - {source_name}**")
                        st.caption(content[:300] + "...")

# 6. Chat Input (Always persistent at bottom)
if query := st.chat_input("Ask about your document, EverQuint services, or AI technology..."):
    # Allow user to type as well
    st.session_state.messages.append({"role": "user", "content": query})
    st.rerun()

# 7. Response Generation (run after rerun to render user msg first? 
# Streamlit chat flow usually requires inline handling or immediate rerun loop.
# Correct pattern: If last message is user, generate answer.)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]
    logger.info(f"Processing user query: {user_query[:100]}...")  # Log first 100 chars
    
    # Add to global history
    if user_query not in global_history:
        global_history.append(user_query)

    with st.chat_message("assistant"):
        llm = create_llm(selected_model)
        if not llm:
             logger.error("LLM initialization failed - missing API key")
             st.error("LLM not initialized. Check API Key.")
             output = "Error: LLM missing."
             retrieved_docs = []
        else:
            with st.spinner(f"Searching & Generating {mode}..."):
                try:
                    # Retrieve
                    logger.info(f"Retrieving context for query with mode: {mode}")
                    context, retrieved_docs = hybrid_retrieve(vector_ret, wiki_ret, user_query)
                    logger.info(f"Retrieved {len(retrieved_docs)} documents")
                    
                    # Generate
                    if mode == "Summarization":
                        output = generate_summary(llm, summary_prompt, user_query, context, length=summary_length)
                    else:
                        output = answer_query(llm, prompt, user_query, context)
                    
                    logger.info(f"Generated response (length: {len(output)} chars)")
                    st.markdown(output)
                except Exception as e:
                    logger.error(f"Error during retrieval/generation: {e}", exc_info=True)
                    output = f"Error: {str(e)}"
                    retrieved_docs = []
                    st.error(output)
                
                # Show Sources underneath
                if retrieved_docs:
                    with st.expander("Retrieved Resources"):
                        for i, doc in enumerate(retrieved_docs):
                            meta = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
                            source_type = meta.get('source_type', 'Unknown')
                            source_name = meta.get('source', 'Unknown Source')
                            st.markdown(f"**{i+1}. {source_type} - {source_name}**")
                            st.caption(doc.page_content[:300] + "...")
        
        # Add to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": output,
            "sources": retrieved_docs
        })
