# InsureAssist — Insurance Complaint Intelligence Copilot

A production-oriented **Applied AI / Agentic AI system** that investigates public automobile insurance complaint data and answers regulatory-guidance questions using:

- LLM-directed tool calling
- LangGraph orchestration
- Semantic RAG
- Structured SQL analytics
- Evidence-grounded generation
- Human-in-the-loop review
- Persistent workflow state
- Evaluation and tracing
- FastAPI, Docker, and CI

The goal is not to build a generic chatbot.

The goal is to build an AI system where the **LLM decides what evidence it needs**, retrieves that evidence through controlled tools, and produces an answer grounded in verifiable sources.

---

# Problem

Insurance-support questions often require combining two very different types of information:

### Structured operational data

Examples:

- What was recorded for complaint `467758`?
- What are the most common complaint issue tags?
- How many complaints were confirmed?
- How has complaint volume changed over time?

### Regulatory guidance

Examples:

- What does TDI say about total-loss valuation disputes?
- What can a consumer do if they disagree with an insurer?
- What does collision coverage pay for?

A normal RAG chatbot is not sufficient for both.

InsureAssist therefore gives the agent access to a **bounded toolset** for SQL-backed complaint analysis and semantic retrieval over official regulatory guidance.

---

# Data

The project uses real public data from the **Texas Department of Insurance (TDI)**.

### Complaint dataset

Source:

> Texas Department of Insurance — Insurance complaints: one record per complaint

The source contains approximately:

- **290K total complaint records**
- **100K+ automobile complaint records**

Important fields include:

- complaint number
- received date
- closed date
- complaint type
- coverage type
- coverage level
- complainant type
- finding type
- issue keywords

The data is normalized into a read-only SQLite evidence store.

### Guidance corpus

Official TDI consumer guidance is collected from:

- TDI Auto Insurance Guide
- TDI Auto Insurance FAQ
- TDI Insurance Complaint Help

The documents are converted into section-aware chunks and indexed for retrieval.

---

# Architecture

```text
                         User
                           |
                           v
                       FastAPI
                           |
                           v
                  LangGraph Agent
                           |
                  LLM decides next action
                           |
               +-----------+-----------+
               |           |           |
               v           v           v
       Complaint Tool   Statistics   Guidance RAG
               |           Tool          |
               |             |           |
               +------ Read-only DB      |
                             |            |
                             |     Semantic Retrieval
                             |            |
                             +------------+
                                  |
                                  v
                            Evidence Ledger
                                  |
                                  v
                       Structured Synthesis
                                  |
                                  v
                      Optional Human Review
                                  |
                                  v
                             Final Report
```

The LLM never receives unrestricted database access.

It can only call approved, validated tools.

---

# Agent Tools

The agent currently has access to four controlled tools.

### `get_complaint`

Retrieve a specific complaint by complaint number.

### `search_complaints`

Search complaints using validated filters.

### `get_complaint_statistics`

Run controlled aggregations such as:

- issue keyword frequency
- finding type
- coverage level
- complainant type
- complaint type
- monthly complaint counts

### `search_insurance_guidance`

Retrieve relevant TDI regulatory guidance using semantic search.

All tool arguments are validated before execution.

---

# Agentic Workflow

The production workflow is implemented using **LangGraph**.

```text
START
  |
initialize
  |
decide
  |
  +---- tool call ----> execute tool
  |                         |
  +-------------------------+
  |
synthesize
  |
assess
  |
optional human review
  |
finalize
  |
END
```

The model can perform multiple tool calls before generating the final answer.

A maximum iteration limit prevents uncontrolled agent loops.

---

# Semantic RAG

The regulatory corpus currently contains **62 section-aware chunks**.

Semantic retrieval uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are normalized and searched using cosine-equivalent dot-product similarity with NumPy.

A dedicated vector database was intentionally avoided because the corpus is small enough for simple in-memory similarity search.

TF-IDF retrieval is retained as a baseline.

---

# Retrieval Evaluation

Semantic search was evaluated against the TF-IDF baseline using **8 curated insurance-guidance queries**, including paraphrased consumer questions.

| Metric | TF-IDF | Semantic |
|---|---:|---:|
| Hit@1 | 75% | **100%** |
| Hit@3 | 100% | **100%** |
| MRR | 0.8542 | **1.0000** |

The largest improvements occurred for paraphrased questions where the user's wording differed from the terminology used in the source document.

Example:

```text
"The payout for my written-off vehicle seems too low.
What can I do?"
```

TF-IDF returned the relevant section at **rank 3**.

Semantic retrieval returned relevant guidance at **rank 1**.

Run the benchmark:

```bash
uv run python scripts/run_retrieval_benchmark.py
```

Results are saved to:

```text
data/evals/results/retrieval_benchmark.json
```

This is a small curated benchmark and should not be interpreted as evidence of universal 100% retrieval accuracy.

---

# Grounded Generation

Every tool result is converted into an evidence object.

Example evidence IDs:

```text
SQL-COMPLAINT-467758
SQL-STATS-KEYWORD
TDI-AUTO-GUIDE-S025-C001
```

Generated insights explicitly reference these evidence IDs.

The final report validates that every cited evidence ID actually exists in the evidence ledger.

This provides **citation integrity** and prevents the model from inventing arbitrary evidence references.

---

# Example

Question:

```text
For complaint 467758, what issue tags were recorded,
how long was the complaint open, and what does TDI
guidance say someone can do if they disagree with
an insurer about a total-loss vehicle valuation?
```

The agent independently selected:

```text
get_complaint
search_insurance_guidance
```

It then combined:

- structured complaint evidence
- semantic retrieval from official TDI guidance

into a grounded report.

A key grounding constraint is that the system explicitly distinguishes:

```text
complaint received → complaint closed duration
```

from:

```text
insurance claim settlement duration
```

because the public TDI dataset does **not** contain claim settlement timestamps.

---

# Human-in-the-Loop

The workflow supports persistent LangGraph interrupts.

A request may specify:

```json
{
  "require_human_review": true
}
```

The workflow pauses before finalization and stores its state.

A reviewer can then:

- approve
- edit
- reject

The same workflow thread resumes from its checkpoint rather than starting the agent again.

This was validated by pausing a real agent execution and resuming the persisted thread without an additional LLM call.

---

# Evaluation

## Deterministic Baseline

Before introducing LLM-directed routing, a deterministic workflow was evaluated on 12 cases.

| Metric | Result |
|---|---:|
| Task completion | 100% |
| Citation validity | 100% |
| Tool recall | 100% |
| Exact tool selection | **41.67%** |
| Retrieval success | 100% |
| p50 latency | ~2.05 ms |
| Model calls | 0 |

The deterministic system had high recall but frequently retrieved guidance unnecessarily.

This gives the agent benchmark a meaningful target:

> Can LLM-directed routing improve tool precision without sacrificing grounding?

---

## Agent Evaluation

A separate **24-case agent benchmark** covers:

- complaint retrieval
- complaint search
- SQL statistics
- regulatory guidance
- multi-tool reasoning
- missing records
- unnecessary tool avoidance
- duplicate tool calls
- grounding limitations
- complaint-duration vs claim-settlement ambiguity

The evaluator records:

- task completion
- tool recall
- exact tool selection
- duplicate tool calls
- tool failures
- citation validity
- agent iterations
- latency
- input tokens
- output tokens
- total model tokens

The evaluator is resumable so completed cases are preserved when API quota or provider-capacity limits interrupt a run.

```bash
uv run python scripts/run_agent_evals.py --max-cases 2
```

The full benchmark is being accumulated under a single model configuration before aggregate results are reported.

---

# Observability

MLflow is used for workflow tracing during local development.

```text
sqlite:///data/runtime/mlflow.db
```

The agent response and evaluation pipeline additionally capture:

- tools used
- agent iterations
- tool failures
- input tokens
- output tokens
- total tokens
- end-to-end latency

---

# API

Start the service:

```bash
uv run uvicorn insurance_copilot.main:app \
  --app-dir src \
  --reload
```

Swagger:

```text
http://localhost:8000/docs
```

Important endpoints:

```text
POST /agent/analyze
POST /agent/{thread_id}/resume
GET  /agent/{thread_id}/state

POST /analyze
POST /graph/analyze

GET /health
GET /ready
```

Example:

```bash
curl -X POST \
  http://localhost:8000/agent/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Show complaint 467758 and explain relevant TDI total-loss guidance.",
    "complaint_number": "467758",
    "guidance_top_k": 3
  }'
```

---

# Running Locally

Requires:

```text
Python 3.12
uv
```

Install dependencies:

```bash
uv sync
```

Create:

```text
.env
```

from:

```text
.env.example
```

Example configuration:

```env
APP_ENV=development
LOG_LEVEL=INFO

LLM_PROVIDER=gemini
LLM_API_KEY=<your-api-key>
LLM_MODEL=<supported-gemini-model>

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

DATABASE_PATH=data/runtime/complaints.db
CHECKPOINT_DATABASE_PATH=data/runtime/langgraph_checkpoints.sqlite

MLFLOW_TRACKING_URI=sqlite:///data/runtime/mlflow.db
MLFLOW_EXPERIMENT_NAME=insureassist-local
```

The current production provider implementation uses Gemini.

The provider interface itself is separated from the workflow so additional model providers can be added later.

---

# Build the Evidence Layer

Download TDI data:

```bash
uv run python scripts/download_data.py
```

Build the complaint database:

```bash
uv run python scripts/build_database.py --full
```

Build the regulatory guidance corpus:

```bash
uv run python scripts/build_guidance_corpus.py
```

Build the semantic index:

```bash
uv run python scripts/build_semantic_index.py
```

---

# Testing

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

Tests cover areas including:

- domain-model validation
- complaint tools
- semantic retrieval
- agent tool schemas
- retrieval limits
- workflow behavior
- checkpointing
- failure handling
- API behavior

---

# Docker

Build:

```bash
docker build -t insureassist .
```

Run:

```bash
docker run \
  --env-file .env \
  -p 8000:8000 \
  insureassist
```

The Docker build reproduces:

- TDI data download
- SQLite database construction
- guidance corpus generation
- semantic embedding generation

Secrets are provided at runtime and are not baked into the image.

---

# CI

GitHub Actions validates:

```text
dependency installation
        ↓
TDI data preparation
        ↓
SQLite database build
        ↓
guidance corpus build
        ↓
semantic index build
        ↓
Ruff
        ↓
Pytest
```

Docker packaging is also validated through the repository workflow.

---

# Limitations

The system uses public complaint data and general regulatory guidance.

It does **not** have access to:

- underlying insurance claim files
- policy contracts
- adjuster notes
- insurer decision history
- payment records
- claim settlement timestamps
- fraud investigation data

Therefore the system must not infer:

- whether a claim was correctly handled
- whether an individual policy provides coverage
- whether fraud occurred
- whether an insurer committed misconduct
- how long an underlying claim took to settle

TDI guidance is general consumer guidance and does not replace the terms of an individual insurance policy.

---

# Tech Stack

```text
Python 3.12
FastAPI
Pydantic
LangGraph
Gemini
SentenceTransformers
scikit-learn
NumPy
SQLite
MLflow
Pytest
Ruff
Docker
GitHub Actions
uv
```

---

# What This Project Demonstrates

InsureAssist is designed to demonstrate production-oriented Applied AI engineering rather than a prompt-only chatbot.

The project includes:

- real-world public data ingestion
- structured SQL evidence retrieval
- semantic RAG
- measured retrieval improvement
- LLM function/tool calling
- multi-step agent orchestration
- bounded tool execution
- structured grounded generation
- evidence citation validation
- persistent LangGraph state
- human-in-the-loop workflows
- token and latency measurement
- offline evaluation
- MLflow tracing
- FastAPI serving
- Docker reproducibility
- automated CI

The central design principle is simple:

> **Let the model reason about what information it needs, but make the system responsible for where facts come from.**
