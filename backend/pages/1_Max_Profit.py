import streamlit as st
import sys
import os

# fix path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.backend.max_profit import max_profit_schedule

st.title("Max Profit Scheduling (Mars Land)")

n = st.number_input("Total time units (n)", min_value=0, max_value=1000, value=13, step=1)
if st.button("Compute optimal schedule"):
    res = max_profit_schedule(n)
    st.write("**Total Earnings:**", f"${res['profit']}")
    
    solutions = res['solutions']
    st.write(f"Found **{len(solutions)}** optimal solution(s):")
    
    for i, sol in enumerate(solutions):
        with st.expander(f"Solution {i+1}", expanded=(i==0)):
            st.write("**Counts:**")
            st.write(f"- Theatre (T): {sol['counts']['T']}")
            st.write(f"- Pub (P): {sol['counts']['P']}")
            st.write(f"- Commercial Park (C): {sol['counts']['C']}")
            
            st.markdown("**Schedule (chronological)**")
            for start, finish, b in sol['schedule']:
                st.write(f"- **{b.name}** built from t={start} to t={finish} (Earnings: {b.earning_per_unit} * {n-finish} = ${b.earning_per_unit * (n-finish)})")
