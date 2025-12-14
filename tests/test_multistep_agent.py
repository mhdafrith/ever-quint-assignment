# tests/test_multistep_agent.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from backend.multistep_agent import solve, LLMClient

def test_mock_agent():
    # Use mock LLM
    llm = LLMClient(api_key=None)
    
    # Test simple time question (handled by mock execution logic)
    q1 = "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?"
    res1 = solve(q1, llm=llm)
    assert res1['status'] == 'success'
    assert "3 hours 35 minutes" in res1['answer']

    # Test fallback
    q2 = "What is 6 * 7?"
    res2 = solve(q2, llm=llm)
    assert res2['status'] == 'success'
    assert res2['answer'] == "42" # Mock returns 42 for everything else

if __name__ == "__main__":
    test_mock_agent()
    print("Agent tests passed!")
