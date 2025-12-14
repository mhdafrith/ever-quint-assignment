import streamlit as st
import os
import sys

# Ensure backend acts as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.backend.logger_setup import setup_logging
setup_logging()

st.set_page_config(
    page_title="AI Engineer Assignment",
    page_icon="🤖",
    layout="wide"
)

st.title("Ever Quint AI Engineer Interview Assignment")

st.markdown("""
## Overview
This application solves the 4 tasks required for the AI Engineer Assignment.
Select a project from the sidebar to view the solution.

### Projects
1. **Max Profit Problem**: Optimization algorithm for property development.
2. **Water Tank Problem**: Visualization of trapped water (Vanilla JS).
3. **Document Search**: Hybrid RAG system with summarization.
4. **Reasoning Agent**: Multi-step AI agent with self-verification.
""")

st.sidebar.success("Select a task above.")
