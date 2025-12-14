import sys
import os
import json
import logging
from dotenv import load_dotenv

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from backend.multistep_agent import solve, LLMClient

# Load Environment
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Eval")

EASY_QUESTIONS = [
    "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?",
    "Alice has 3 red apples and twice as many green apples as red. How many apples does she have in total?",
    "Calculate (25 * 4) + (100 / 2).",
    "A meeting starts at 10:00 AM and lasts 90 minutes. When does it end?",
    "If I have 50 dollars and spend 15 on lunch and 5 on coffee, how much is left?"
]

TRICKY_QUESTIONS = [
    "I have 3 marbles. I lost 1, but then found 2 more. Then I doubled my count. How many do I have now?",
    "A meeting needs 60 minutes. There are free slots: 09:00–09:30, 09:45–10:30, 11:00–12:00. Which slots can fit the meeting?",
    "John is twice as old as Mary. Mary is 10 years younger than Tom. Tom is 30. How old is John?"
]

def run_tests():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found.")
        return

    client = LLMClient(api_key=api_key)
    results_log = []

    print("=== STARTING AGENT EVALUATION ===\n")

    # Easy Tests
    print("--- EASY QUESTIONS ---")
    for i, q in enumerate(EASY_QUESTIONS):
        print(f"Running Easy Q{i+1}: {q}")
        res = solve(q, client)
        print(f"Status: {res['status']} | Retries: {res['metadata']['retries']}")
        print(f"Answer: {res['answer']}\n")
        results_log.append({"type": "easy", "question": q, "result": res})

    # Tricky Tests
    print("--- TRICKY QUESTIONS ---")
    for i, q in enumerate(TRICKY_QUESTIONS):
        print(f"Running Tricky Q{i+1}: {q}")
        res = solve(q, client)
        print(f"Status: {res['status']} | Retries: {res['metadata']['retries']}")
        print(f"Answer: {res['answer']}\n")
        results_log.append({"type": "tricky", "question": q, "result": res})
        
    # Save full log to file
    output_path = os.path.join(os.path.dirname(__file__), '..', 'run_logs', 'agent_evaluation_log.json')
    with open(output_path, "w") as f:
        json.dump(results_log, f, indent=2)
    
    print(f"Evaluation Complete. Full logs saved to {output_path}")

if __name__ == "__main__":
    run_tests()
