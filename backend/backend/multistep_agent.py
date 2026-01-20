# from typing import Dict, Any, List, Optional
# import os
# import json
# import logging
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# # Configure Logging
# logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
# logger = logging.getLogger(__name__)

# # --- PROMPTS ---

# PLANNER_PROMPT = """
# You are a Planner Agent.
# Given a user question, create a concise, step-by-step plan to solve it.
# Output ONLY the plan as a numbered list.
# Do not solve the problem yourself.
# Do not output any introductory text.

# Example 1:
# Question: If a train leaves at 14:30 and arrives at 18:05, how long is the journey?
# 1. Parse the departure time (14:30) and arrival time (18:05).
# 2. Calculate the difference in hours and minutes.
# 3. Format the duration as "X hours Y minutes".

# Example 2:
# Question: Alice has 3 red apples and twice as many green apples as red. How many apples does she have in total?
# 1. Identify the number of red apples (3).
# 2. Calculate the number of green apples (2 * red).
# 3. Sum the red and green apples to get the total.
# 4. Output the final count.

# Question: {question}
# """

# EXECUTOR_PROMPT = """
# You are an Executor Agent.
# Your task is to follow the provided plan strictly to solve the user's question.
# You must show your work for each step.
# Return the final answer clearly.

# Example 1:
# Question: Alice has 3 red apples and twice as many green apples as red. Total?
# Plan:
# 1. Identify red (3).
# 2. Calc green (2*3).
# 3. Sum total.
# Output:
# {{
#     "intermediate_steps": "Red apples = 3. Green apples = 2 * 3 = 6. Total = 3 + 6 = 9.",
#     "final_result": "9 apples"
# }}

# Example 2:
# Question: Duration from 14:30 to 18:05?
# Plan:
# 1. Parse times.
# 2. Calc diff.
# Output:
# {{
#     "intermediate_steps": "Start: 14:30. End: 18:05. 14:30 to 18:05 is 3 hours (17:30) plus 35 minutes.",
#     "final_result": "3 hours 35 minutes"
# }}

# Question: {question}
# Plan:
# {plan}

# Output Format:
# Return a JSON object with:
# {{
#     "intermediate_steps": "string describing the work done",
#     "final_result": "string containing the final answer"
# }}
# """

# VERIFIER_PROMPT = """
# You are a Verifier Agent.
# Your task is to check the proposed solution for correctness, consistency, and constraints.
# Check arithmetic, logic, and if the answer actually addresses the question.
# If the solution is correct, approve it.
# If it is incorrect, explain why.

# Example 1:
# Question: 3 red, twice as many green. Total?
# Result: 9 apples
# Steps: Red=3, Green=6. Total=9.
# Output:
# {{
#     "passed": true,
#     "details": "Calculations are correct (3 + 6 = 9)."
# }}

# Example 2:
# Question: 3 red, twice as many green. Total?
# Result: 8 apples
# Steps: Red=3, Green=6. Total=8.
# Output:
# {{
#     "passed": false,
#     "details": "Arithmetic error: 3 + 6 should be 9, not 8."
# }}

# Question: {question}
# Proposed Result: {result}
# Reasoning/Steps: {intermediate_steps}

# Output Format:
# Return a JSON object with:
# {{
#     "passed": true/false,
#     "details": "Explanation of why it passed or failed"
# }}
# """

# FINAL_FORMAT_PROMPT = """
# You are a text formatter. 
# Rewrite the following explanation to be concise and user-friendly for a general audience.
# Do NOT show raw calculations or chain-of-thought.
# Target Field: "reasoning_visible_to_user"

# Input: {reasoning}
# """

# # --- AGENT CLASSES ---

# class LLMClient:
#     def __init__(self, api_key: str = None, model="openai/gpt-oss-20b"):
#         self.api_key = api_key or os.getenv("GROQ_API_KEY")
#         if not self.api_key:
#             logger.warning("GROQ_API_KEY not found. Agent will fail if called.")
        
#         self.llm = ChatGroq(
#             temperature=0.1, # Low temp for reasoning
#             model_name=model,
#             groq_api_key=self.api_key
#         ) if self.api_key else None

#     def call_str(self, prompt_text: str) -> str:
#         if not self.llm:
#             raise NotImplementedError("Groq API Key missing. Cannot call LLM.")
#         return self.llm.invoke(prompt_text).content

#     def call_json(self, prompt_text: str) -> Dict[str, Any]:
#         """Helper to force JSON parsing from LLM output"""
#         if not self.llm:
#             raise NotImplementedError("Groq API Key missing.")
        
#         # We append a JSON instruction just in case, though prompts usually handle it
#         response_text = self.llm.invoke(prompt_text + "\n\nRespond strictly in valid JSON.").content
        
#         # Clean up code blocks if generic model adds them
#         if "```json" in response_text:
#             response_text = response_text.split("```json")[1].split("```")[0]
#         elif "```" in response_text:
#             response_text = response_text.split("```")[1].split("```")[0]
            
#         try:
#             return json.loads(response_text)
#         except json.JSONDecodeError:
#             logger.error(f"Failed to parse JSON: {response_text}")
#             return {"error": "JSON parse error", "raw": response_text}


# def planner(question: str, client: LLMClient) -> List[str]:
#     prompt = PLANNER_PROMPT.format(question=question)
#     plan_text = client.call_str(prompt)
#     # Convert text list to python list
#     steps = [line.strip() for line in plan_text.split('\n') if line.strip()]
#     return steps

# def executor(question: str, plan: List[str], client: LLMClient) -> Dict[str, Any]:
#     plan_str = "\n".join(plan)
#     prompt = EXECUTOR_PROMPT.format(question=question, plan=plan_str)
#     return client.call_json(prompt)

# def verifier(question: str, execution: Dict[str, Any], client: LLMClient) -> Dict[str, Any]:
#     intermediate = execution.get("intermediate_steps", "")
#     result = execution.get("final_result", "")
#     prompt = VERIFIER_PROMPT.format(question=question, result=result, intermediate_steps=intermediate)
#     return client.call_json(prompt)

# def format_user_reasoning(raw_reasoning: str, client: LLMClient) -> str:
#     prompt = FINAL_FORMAT_PROMPT.format(reasoning=raw_reasoning)
#     return client.call_str(prompt)

# def solve(question: str, llm_client: LLMClient = None, max_retries: int = 2) -> Dict[str, Any]:
#     if llm_client is None:
#         llm_client = LLMClient()
    
#     # 1. PLAN
#     try:
#         plan = planner(question, llm_client)
#     except Exception as e:
#         return {"status": "failed", "answer": f"Error during planning: {e}", "metadata": {}}

#     retries = 0
#     checks_log = []

#     while retries <= max_retries:
#         # 2. EXECUTE
#         try:
#             exec_out = executor(question, plan, llm_client)
#             # Handle possible JSON failure in executor
#             if "error" in exec_out:
#                 raise ValueError("Executor failed to produce valid JSON.")
#         except Exception as e:
#             # If execution fails, we might want to retry or just fail
#             logger.error(f"Execution error: {e}")
#             retries += 1
#             continue

#         # 3. VERIFY
#         try:
#             check = verifier(question, exec_out, llm_client)
#             passed = check.get("passed", False)
#             details = check.get("details", "No details")
            
#             checks_log.append({
#                 "check_name": f"attempt_{retries+1}",
#                 "passed": passed,
#                 "details": details
#             })
            
#             if passed:
#                 # Success!
#                 final_answer = exec_out.get("final_result", "")
#                 intermediate = exec_out.get("intermediate_steps", "")
                
#                 # Format the reasoning for the user
#                 user_reasoning = format_user_reasoning(intermediate, llm_client)
                
#                 return {
#                     "answer": final_answer,
#                     "status": "success",
#                     "reasoning_visible_to_user": user_reasoning,
#                     "metadata": {
#                         "plan": plan,
#                         "checks": checks_log,
#                         "retries": retries
#                     }
#                 }
#             else:
#                 logger.info(f"Verification failed: {details}. Retrying...")
                
#         except Exception as e:
#             logger.error(f"Verification error: {e}")
            
#         retries += 1

#     # If we exit loop, we failed
#     return {
#         "answer": "Unable to solve with high confidence.",
#         "status": "failed",
#         "reasoning_visible_to_user": "The agent could not verify a correct solution after multiple attempts.",
#         "metadata": {
#             "plan": plan,
#             "checks": checks_log,
#             "retries": retries
#         }
#     }
from typing import Dict, Any, List
import os
import json
import logging

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# -------------------------------------------------
# ENV & LOGGING
# -------------------------------------------------

load_dotenv()  # ✅ REQUIRED to load GROQ_API_KEY

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- PROMPTS ---

PLANNER_PROMPT = """
You are a Planner Agent.
Given a user question, create a concise, step-by-step plan to solve it.
Output ONLY the plan as a numbered list.
Do not solve the problem yourself.
Do not output any introductory text.

Example 1:
Question: If a train leaves at 14:30 and arrives at 18:05, how long is the journey?
1. Parse the departure time (14:30) and arrival time (18:05).
2. Calculate the difference in hours and minutes.
3. Format the duration as "X hours Y minutes".

Example 2:
Question: Alice has 3 red apples and twice as many green apples as red. How many apples does she have in total?
1. Identify the number of red apples (3).
2. Calculate the number of green apples (2 * red).
3. Sum the red and green apples to get the total.
4. Output the final count.

Question: {question}
"""

EXECUTOR_PROMPT = """
You are an Executor Agent.
Your task is to follow the provided plan strictly to solve the user's question.
You must show your work for each step.
Return the final answer clearly.

Example 1:
Question: Alice has 3 red apples and twice as many green apples as red. Total?
Plan:
1. Identify red (3).
2. Calc green (2*3).
3. Sum total.
Output:
{{
    "intermediate_steps": "Red apples = 3. Green apples = 2 * 3 = 6. Total = 3 + 6 = 9.",
    "final_result": "9 apples"
}}

Example 2:
Question: Duration from 14:30 to 18:05?
Plan:
1. Parse times.
2. Calc diff.
Output:
{{
    "intermediate_steps": "Start: 14:30. End: 18:05. 14:30 to 18:05 is 3 hours (17:30) plus 35 minutes.",
    "final_result": "3 hours 35 minutes"
}}

Question: {question}
Plan:
{plan}

Output Format:
Return a JSON object with:
{{
    "intermediate_steps": "string describing the work done",
    "final_result": "string containing the final answer"
}}
"""

VERIFIER_PROMPT = """
You are a Verifier Agent.
Your task is to check the proposed solution for correctness, consistency, and constraints.
Check arithmetic, logic, and if the answer actually addresses the question.
If the solution is correct, approve it.
If it is incorrect, explain why.

Example 1:
Question: 3 red, twice as many green. Total?
Result: 9 apples
Steps: Red=3, Green=6. Total=9.
Output:
{{
    "passed": true,
    "details": "Calculations are correct (3 + 6 = 9)."
}}

Example 2:
Question: 3 red, twice as many green. Total?
Result: 8 apples
Steps: Red=3, Green=6. Total=8.
Output:
{{
    "passed": false,
    "details": "Arithmetic error: 3 + 6 should be 9, not 8."
}}

Question: {question}
Proposed Result: {result}
Reasoning/Steps: {intermediate_steps}

Output Format:
Return a JSON object with:
{{
    "passed": true/false,
    "details": "Explanation of why it passed or failed"
}}
"""

FINAL_FORMAT_PROMPT = """
You are a text formatter. 
Rewrite the following explanation to be concise and user-friendly for a general audience.
Do NOT show raw calculations or chain-of-thought.
Return ONLY the rewritten text. Do not use JSON, code blocks, or any labels.

Input: {reasoning}
"""

# -------------------------------------------------
# LLM CLIENT
# -------------------------------------------------

import time

class LLMClient:
    def __init__(self, api_key: str = None, model: str = "openai/gpt-oss-120b", reasoning_effort: str = "high"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model
        self.reasoning_effort = reasoning_effort

        if not self.api_key:
            raise RuntimeError("❌ GROQ_API_KEY not found in environment")

        self.llm = ChatGroq(
            temperature=1.0,  # Recommended for gpt-oss
            model_name=model,
            groq_api_key=self.api_key,
            verbose=True,
            reasoning_effort=reasoning_effort
        )
        self.usage_history = []

    def _record_usage(self, response, duration: float):
        usage = getattr(response, "response_metadata", {}).get("token_usage", {})
        
        # Reasoning tokens are often nested in completion_tokens_details
        reasoning_tokens = usage.get("reasoning_tokens")
        if reasoning_tokens is None:
            reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)

        self.usage_history.append({
            "model": self.model_name,
            "latency": duration,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": usage.get("total_tokens", 0)
        })

    def call_str(self, prompt: str) -> str:
        start_time = time.time()
        response = self.llm.invoke(prompt)
        duration = time.time() - start_time
        self._record_usage(response, duration)
        return response.content.strip()

    def call_json(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        response = self.llm.invoke(prompt + "\nRespond strictly in valid JSON.")
        duration = time.time() - start_time
        self._record_usage(response, duration)
        
        response_text = response.content

        # Remove code blocks if present
        if "```" in response_text:
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:].strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            logger.error("JSON parse failed")
            return {"error": "json_parse_error", "raw": response_text}


# -------------------------------------------------
# AGENT FUNCTIONS
# -------------------------------------------------

def planner(question: str, client: LLMClient) -> List[str]:
    prompt = PLANNER_PROMPT.format(question=question)
    text = client.call_str(prompt)

    # ✅ Robust parsing
    return [
        line.strip().lstrip("0123456789. )")
        for line in text.split("\n")
        if line.strip()
    ]


def executor(question: str, plan: List[str], client: LLMClient) -> Dict[str, Any]:
    plan_str = "\n".join(plan)
    prompt = EXECUTOR_PROMPT.format(question=question, plan=plan_str)
    return client.call_json(prompt)


def verifier(question: str, execution: Dict[str, Any], client: LLMClient) -> Dict[str, Any]:
    prompt = VERIFIER_PROMPT.format(
        question=question,
        result=execution.get("final_result", ""),
        intermediate_steps=execution.get("intermediate_steps", "")
    )
    return client.call_json(prompt)


def format_user_reasoning(raw_reasoning: str, client: LLMClient) -> str:
    prompt = FINAL_FORMAT_PROMPT.format(reasoning=raw_reasoning)
    return client.call_str(prompt)


# -------------------------------------------------
# SOLVER PIPELINE
# -------------------------------------------------

def solve(
    question: str,
    llm_client: LLMClient = None,
    max_retries: int = 2
) -> Dict[str, Any]:

    # Use provided client or create one
    client = llm_client or LLMClient()

    # 1. PLAN
    plan = planner(question, client)

    retries = 0
    checks_log = []

    while retries <= max_retries:
        # 2. EXECUTE
        exec_out = executor(question, plan, client)

        if "error" in exec_out:
            retries += 1
            continue

        # 3. VERIFY
        check = verifier(question, exec_out, client)
        passed = check.get("passed", False)

        checks_log.append({
            "attempt": retries + 1,
            "passed": passed,
            "details": check.get("details", "")
        })

        if passed:
            user_reasoning = format_user_reasoning(
                exec_out.get("intermediate_steps", ""),
                client
            )

            # Aggregate speed insights
            total_latency = sum(call["latency"] for call in client.usage_history)
            total_tokens = sum(call["total_tokens"] for call in client.usage_history)
            total_reasoning_tokens = sum(call.get("reasoning_tokens", 0) for call in client.usage_history)

            return {
                "status": "success",
                "answer": exec_out.get("final_result", ""),
                "reasoning_visible_to_user": user_reasoning,
                "metadata": {
                    "plan": plan,
                    "checks": checks_log,
                    "retries": retries,
                    "speed_insights": {
                        "total_latency_seconds": round(total_latency, 3),
                        "total_tokens": total_tokens,
                        "total_reasoning_tokens": total_reasoning_tokens,
                        "model_used": client.model_name,
                        "reasoning_effort": client.reasoning_effort,
                        "call_details": client.usage_history
                    }
                }
            }

        retries += 1

    total_latency = sum(call["latency"] for call in client.usage_history)
    total_tokens = sum(call["total_tokens"] for call in client.usage_history)
    total_reasoning_tokens = sum(call.get("reasoning_tokens", 0) for call in client.usage_history)

    return {
        "status": "failed",
        "answer": "Unable to solve with high confidence.",
        "reasoning_visible_to_user": "The system could not verify a correct solution.",
        "metadata": {
            "plan": plan,
            "checks": checks_log,
            "retries": retries,
            "speed_insights": {
                "total_latency_seconds": round(total_latency, 3),
                "total_tokens": total_tokens,
                "total_reasoning_tokens": total_reasoning_tokens,
                "model_used": client.model_name,
                "reasoning_effort": client.reasoning_effort,
                "call_details": client.usage_history
            }
        }
    }
