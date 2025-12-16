import streamlit as st
import sys
import os
import logging

# fix path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.backend.logger_setup import setup_logging
from backend.backend.multistep_agent import solve, LLMClient

setup_logging()
logger = logging.getLogger(__name__)

st.title("Multi-step Reasoning Agent")
logger.info("Reasoning Agent app loaded")
st.markdown("""
**Architecture:** Planner → Executor → Verifier
1. **Planner**: Breaks down complex queries.
2. **Executor**: Solves step-by-step (with Groq).
3. **Verifier**: Checks the logic and arithmetic. Retries if failed.
""")

question = st.text_area("Enter a question (math/time/logic)", "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?")

if st.button("Solve with Reasoning"):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not found")
        st.error("GROQ_API_KEY not found in environment.")
    else:
        logger.info(f"Solving question: {question[:100]}...")  # Log first 100 chars
        with st.spinner("Agent is Planning -> Executing -> Verifying..."):
            try:
                llm_client = LLMClient(api_key=api_key)
                result = solve(question, llm_client=llm_client)
                
                if result["status"] == "success":
                    logger.info(f"Successfully solved question. Answer: {result.get('answer', '')[:100]}")
                    st.success("Solved!")
                    st.write(f"**Answer:** {result['answer']}")
                    st.info(f"**Reasoning:** {result['reasoning_visible_to_user']}")
                    with st.expander("Show Agent Internal Metadata (JSON)"):
                         st.json(result)
                else:
                    logger.warning(f"Failed to solve question. Status: {result.get('status')}")
                    st.error("Failed to solve.")
                    st.json(result)
            except Exception as e:
                logger.error(f"Error during reasoning: {e}", exc_info=True)
                st.error(f"Error: {str(e)}")
