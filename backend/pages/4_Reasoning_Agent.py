import streamlit as st
import sys
import os

# fix path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.backend.multistep_agent import solve, LLMClient

st.title("Multi-step Reasoning Agent")
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
        st.error("GROQ_API_KEY not found in environment.")
    else:
        with st.spinner("Agent is Planning -> Executing -> Verifying..."):
            llm_client = LLMClient(api_key=api_key)
            result = solve(question, llm_client=llm_client)
            
            if result["status"] == "success":
                st.success("Solved!")
                st.write(f"**Answer:** {result['answer']}")
                st.info(f"**Reasoning:** {result['reasoning_visible_to_user']}")
                with st.expander("Show Agent Internal Metadata (JSON)"):
                     st.json(result)
            else:
                st.error("Failed to solve.")
                st.json(result)
