# Ever Quint Assignment

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://13.61.59.119:8501)

### 👉 **[Click here to experience the Live App](http://13.61.59.119:8501/)**

This repository contains the solution for the AI Engineer Interview Assignment, consolidating four distinct projects into a single Streamlit Application.

> **Deployment Note**  
> This application is **self-hosted on an AWS EC2 instance** and runs **24/7** using a `systemd` service.  
> The app is publicly accessible via a **static Elastic IP**:
>
> **http://13.61.59.119:8501**

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- `pip`
- A [Groq API Key](https://console.groq.com/) (required for RAG and Agent)

### 2. Installation
```bash
# Clone the repository (if applicable)
# Navigate to the project directory
cd ai-engineer-assignment

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Running the App
Launch the detailed Streamlit interface:
```bash
streamlit run backend/app.py
```
Visit `http://localhost:8501` in your browser.

---

## 📂 Implementation Details

The application is structured into 4 main deliverables:

### 1. Max Profit Problem
**Objective**: Optimize property development (Theatres, Pubs, Commercial Parks) to maximize earnings over `n` time units.
- **Problem**: Knapsack-like scheduling problem.
- **Implementation**: `backend/backend/max_profit.py` (Dynamic Programming / Logic).
- **UI**: `pages/2_max_profit.py` allows inputting `n` and seeing optimal solutions.

### 2. Water Tank Problem
**Objective**: Calculate trapped water between blocks (trapping rain water problem) and visualize it.
- **Implementation**: Vanilla JS/HTML/CSS in `frontend/water_tank/`.
- **Integration**: Wrapped in Streamlit via `pages/5_water_tank_frame.py` for seamless viewing.
- **Features**: Interactive input, tabular representation, and calculation of total units.

### 3. RAG Search & Summarization (Hybrid)
**Objective**: Retrieval Augmented Generation system searching Local Docs + Wikipedia.
- **Core Logic**: `backend/backend/rag_search.py`
- **Architecture**:
    - **Vector Store**: ChromaDB with `sentence-transformers/all-mpnet-base-v2`.
    - **External**: Wikipedia Retriever.
    - **LLM**: Groq (Llama-3.3-70b) for fast inference.
- **UI Features (`pages/4_rag_search.py`)**:
    - **ChatGPT-style Interface**: Conversational UI with history.
    - **Modes**: Q&A (Precise) vs Summarization (Comprehensive).
    - **Model Selection**: Dynamically fetches available models from Groq.
    - **Source Attribution**: Displays retrieved chunks/metadata for every answer.
- **Data Prep**: Documents in `backend/documents/` are automatically ingested, split (RecursiveCharacterTextSplitter), and embedded on startup (persisted to `./chroma_db`).

### 4. Multi-Step Reasoning Agent
**Objective**: An agent that Plans, Executes, and Verifies solutions for word problems.
- **Core Logic**: `backend/backend/multistep_agent.py`
- **Architecture**:
    1.  **Planner**: Decomposes query into step-by-step plan.
    2.  **Executor**: Executes steps using LLM logic.
    3.  **Verifier**: Checks result correctness; triggers retries if failed.
- **Prompts**: Tailored few-shot prompts for each role (see `backend/backend/multistep_agent.py`).
- **Evaluation**: Run the test suite:
    ```bash
    python tests/evaluate_agent.py
    ```
    This generates `agent_evaluation_log.json` containing run logs for 8 test cases (5 Easy, 3 Tricky).

### 5. RAG Evaluation (Ragas)
**Objective**: Quantitatively assess the RAG pipeline's accuracy using **Ragas** metrics.
- **Tools used**:
    - **Ragas**: For calculating Context Precision/Recall, Faithfulness, and Answer Relevancy.
    - **Groq (Llama-3)**: As the "Judge LLM" to evaluate responses.
    - **HuggingFace Embeddings**: For vector-based similarity metrics.
- **Methodology**:
    - Used a test set of 5 Q&A pairs derived from `about_everquint.txt`.
    - Configured `RunConfig(max_workers=1)` to handle Groq API rate limits.
- **Results**:
    | Metric | Score | Interpretation |
    | :--- | :--- | :--- |
    | **Context Precision** | **1.00** | Perfect retrieval ranking. |
    | **Faithfulness** | **1.00** | Answers are fully grounded in context. |
    | **Answer Relevancy** | **0.92** | Highly relevant answers. |
    | **Context Recall** | **1.00** | All relevant info retrieved. |

- **How to Run**:
    ```bash
    # Ensure GROQ_API_KEY is set in .env
    jupyter notebook tests/ragas_evaluation.ipynb
    ```

---

## 📄 Source Code Structure

```
ai-engineer-assignment/
├── backend/
│   ├── app.py                 # Main Entry Point
│   ├── backend/               # Core Logic
│   │   ├── max_profit.py      # Project 1 Logic
│   │   ├── rag_search.py      # Project 3 Logic
│   │   ├── multistep_agent.py # Project 4 Logic
│   │   └── utils.py
│   ├── pages/                 # Streamlit UI Pages
│   │   ├── 1_home.py
│   │   ├── 2_max_profit.py
│   │   ├── 3_multistep_agent.py
│   │   ├── 4_rag_search.py
│   │   └── 5_water_tank_frame.py
│   └── documents/             # RAG Source Data
├── frontend/
│   └── water_tank/            # Project 2 (Vanilla JS)
├── tests/
│   ├── evaluate_agent.py      # Agent Test Suite
│   ├── ragas_evaluation.ipynb # Ragas Evaluation Notebook
│   ├── test_max_profit.py     # Unit Tests for Max Profit
│   ├── test_multistep_agent.py# Unit Tests for Agent
│   └── test_rag_search.py     # Unit Tests for RAG
├── chroma_db/                 # Vector Store Persistence
├── .env                       # Secrets
└── requirements.txt           # Python Dependencies
```
