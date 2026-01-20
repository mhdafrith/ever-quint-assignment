# Applied AI Systems Project: Optimization, RAG, and Agentic Reasoning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://everquintassignment.streamlit.app)

### 👉 **[Click here to experience the Live App](https://everquintassignment.streamlit.app)**

---

The project consolidates **multiple independent problem statements** into one cohesive, production-ready web application, covering:
- Algorithmic optimization
- Data visualization
- Hybrid RAG (Local Documents + Wikipedia)
- Multi-step reasoning using LLM agents

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
cd ever-quint-assignment

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
    - **LLM**: Groq (multi-model support) for fast inference.
- **UI Features (`pages/4_rag_search.py`)**:
    - **ChatGPT-style Interface**: Conversational UI with history.
    - **Modes**: Q&A (Precise) vs Summarization (Comprehensive).
    - **Summarization Lengths**: Short (2-3 sentences), Medium (1 paragraph), Long (Detailed bullet points).
    - **Model Selection**: Dynamically fetches all available chat models (Llama 3, Mixtral, Gemma, etc.) from Groq API.
    - **Source Attribution**: Displays retrieved chunks/metadata for every answer.
- **Data Ingestion**: 
    - **Supported Formats**: `.txt`, `.pdf`, `.docx`, `.html`.
    - **Static Data**: Documents in `backend/documents/` are automatically ingested and persisted to `./chroma_db`.
    - **Ephemeral Uploads**: Users can upload their own documents directly in the UI. These are processed in-memory (ephemeral vector store) for immediate chat sessions without permanently storing the data.

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

## 📂 Source Code Structure

```
ever-quint-assignment/
├── assets/                    # Project assets (images, gifs)
├── backend/
│   ├── app.py                 # Main Streamlit Entry Point
│   ├── backend/               # Core Logic & Utilities
│   │   ├── max_profit.py      # Task 1: Optimization Logic
│   │   ├── rag_search.py      # Task 3: RAG Core Logic
│   │   ├── multistep_agent.py # Task 4: Reasoning Agent Logic
│   │   └── logger_setup.py    # Global Logging Configuration
│   ├── pages/                 # Streamlit UI Components
│   │   ├── 1_Max_Profit.py
│   │   ├── 2_Water_Tank.py
│   │   ├── 3_Document_Search.py
│   │   └── 4_Reasoning_Agent.py
│   └── documents/             # Source Documents for RAG
├── frontend/
│   └── water_tank/            # Task 2: Vanilla JS Implementation
├── problem_statements/        # Original Assignment PDFs
├── tests/                     # Evaluation & Metrics
│   └── ragas_evaluation.ipynb # RAG Performance Analysis
├── run_logs/                  # Runtime Logs & Agent Evals
├── .env                       # Environment Variables (Secrets)
├── pyproject.toml             # Project Metadata
└── requirements.txt           # Python Dependencies
```
