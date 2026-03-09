# QA Spark CodeAgent

**AI-powered code review and certification for the Cloudera CDP platform.**

QA Spark CodeAgent uses a multi-agent AI architecture to review Impala SQL, PySpark, SparkSQL, Scala Spark, and Python code before it runs on your Cloudera environment. Code must score **95 or higher out of 100** to receive a certification badge. Below that threshold, execution is blocked via GitLab CI.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Agent System](#3-agent-system)
4. [File Structure](#4-file-structure)
5. [Supported Languages](#5-supported-languages)
6. [Running Locally (Streamlit UI)](#6-running-locally-streamlit-ui)
7. [Deploying to Cloudera CAI](#7-deploying-to-cloudera-cai)
8. [GitLab CI Integration](#8-gitlab-ci-integration)
9. [Configuration Reference](#9-configuration-reference)
10. [Cost and ROI Tracking](#10-cost-and-roi-tracking)
11. [Security Features](#11-security-features)
12. [Token Management](#12-token-management)
13. [Known Limitations and Next Steps](#13-known-limitations-and-next-steps)

---

## 1. Overview

### What It Does

Every piece of code that runs on the Cloudera CDP platform (queries, jobs, pipelines) passes through QA Spark CodeAgent before execution. The agent:

- Reviews code for **security vulnerabilities**, **performance anti-patterns**, and **code quality issues**
- Generates a **composite score out of 100**
- Issues a **CERTIFIED badge** if the score is 95+
- Provides **corrected, production-ready code** when issues are found
- Tracks **cost per review** in USD for ROI reporting
- **Blocks GitLab merge requests** that do not meet the threshold

### Why It Exists

Data engineers write hundreds of Spark jobs and SQL queries per month. Without automated review:
- Inefficient queries waste cluster resources (Kryo not configured, missing broadcast hints, Python UDFs)
- Security risks go undetected (hardcoded credentials, SQL injection patterns)
- Code quality degrades over time (no schema enforcement, poor null handling)

QA Spark CodeAgent acts as an automated senior engineer reviewing every commit before it reaches production.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Sources                        │
│  Streamlit UI  │  GitLab CI  │  CLI (cli.py)            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator                         │
│  1. Sanitize (mask secrets)                             │
│  2. Check embedding cache (skip if exact match)         │
│  3. Chunk large files at logical boundaries             │
│  4. Run 3 specialist agents                             │
│  5. Compute weighted composite score                    │
│  6. Store result in cache                               │
│  7. Log cost to ROI log                                 │
└──────────┬───────────────┬──────────────────────────────┘
           │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────────────────┐
    │  Security   │ │ Performance │ │   Practices      │
    │   Agent     │ │   Agent     │ │    Agent         │
    │             │ │             │ │                  │
    │ Creds/Inj/  │ │ AQE/Shuffle │ │ Schema/Nulls/    │
    │ Pickle/eval │ │ Broadcast/  │ │ Naming/Errors/   │
    │ shell=True  │ │ UDFs/RDDs   │ │ Complexity       │
    └──────┬──────┘ └──────┬──────┘ └──────┬───────────┘
           └───────────────┴───────────────┘
                           │
                    Azure OpenAI
                   (BDF-GLB-GPT-5)
                  via IQVIA Proxy
```

### Score Weighting by Language

| Language | Security | Performance | Practices |
|----------|----------|-------------|-----------|
| PySpark  | 30%      | 45%         | 25%       |
| SparkSQL | 30%      | 45%         | 25%       |
| Scala    | 30%      | 45%         | 25%       |
| Impala   | 25%      | 50%         | 25%       |
| Python   | 40%      | 25%         | 35%       |

Performance is weighted highest for Spark languages because inefficient Spark jobs have the greatest resource cost on shared Cloudera clusters.

---

## 3. Agent System

### SecurityAgent (`agent/security_agent.py`)

Detects security vulnerabilities before any LLM call using fast regex pre-scan, then sends code to the LLM for deeper analysis.

**Pre-scan rules (instant, no LLM cost):**
- Hardcoded passwords, API keys, secrets
- AWS access keys (AKIA prefix)
- Azure storage connection strings
- JDBC URLs with embedded passwords
- `eval()` and `exec()` usage
- `pickle.loads()` (insecure deserialization)
- `subprocess` with `shell=True`
- SQL string concatenation (injection risk)

**Rule:** If any CRITICAL finding is detected in the pre-scan, the security score is capped at 50 regardless of what the LLM returns. This prevents the LLM from being lenient about hardcoded credentials.

### PerformanceAgent (`agent/performance_agent.py`)

Reviews for performance anti-patterns, with language-specific expertise:

**Spark (PySpark / SparkSQL / Scala):**
- AQE configuration (enabled by default in Spark 3.2+; flags configs that disable it)
- Broadcast join threshold (10 MB; flags missing hints on small tables)
- Shuffle partition tuning
- Python UDF usage (prevents Catalyst optimization; recommend Pandas UDFs)
- RDD operations on DataFrames (bypass Catalyst)
- Cartesian / cross joins without conditions
- Unbounded ORDER BY (forces single-partition sort)
- Missing cache/persist on reused DataFrames
- Kryo serialization recommendation
- Column pruning (SELECT * on wide datasets)

**Impala SQL:**
- Partition pruning in WHERE clause
- JOIN order and broadcast hints
- LIMIT on exploratory queries
- STRAIGHT_JOIN for skewed data

**Python (non-Spark):**
- Python loops over large data vs. vectorized ops
- Generator vs. list for large iterables
- File I/O buffering
- N+1 database query patterns

### PracticesAgent (`agent/practices_agent.py`)

Reviews code quality, maintainability, and correctness:

**Spark languages:**
- Schema enforcement (flag inferred schema; recommend explicit StructType)
- Null handling on join keys
- Idempotency of writes
- Error handling patterns

**SQL languages:**
- DDL statement detection (CREATE, DROP, INSERT — blocked)
- Ambiguous column references
- Missing table aliases
- CTE vs. deeply nested subqueries
- NULL comparison bugs (`col = NULL` vs. `IS NULL`)

**Python:**
- Type hints and docstrings
- Logging vs. print()
- Mutable default arguments
- Context managers (`with open()`)
- PEP 8 compliance

### Orchestrator (`agent/orchestrator.py`)

Coordinates all three agents and computes the final result:

1. Sanitizes code (masks secrets before any LLM call)
2. Checks the embedding cache — returns instantly if identical code was reviewed before
3. Injects context from similar past reviews to reduce token usage
4. Chunks files that exceed the token budget
5. Runs Security → Performance → Practices agents
6. Computes weighted composite score
7. Applies hard rule: security score < 40 caps composite at 60
8. Stores result in cache
9. Logs cost to ROI log

---

## 4. File Structure

```
QA_Spark_CodeAgent/
│
├── agent/                          # All agent code
│   ├── base_agent.py               # Abstract base (raw openai client, token tracking)
│   ├── orchestrator.py             # Coordinates all agents end-to-end
│   ├── security_agent.py           # Security specialist
│   ├── performance_agent.py        # Performance specialist
│   ├── practices_agent.py          # Code quality specialist
│   ├── reviewer.py                 # Legacy single-agent reviewer (backwards compat)
│   ├── scorer.py                   # Score extraction from LLM output
│   ├── sanitizer.py                # Secret masking before API calls
│   ├── chunker.py                  # Large file splitting at logical boundaries
│   ├── token_budget.py             # Token estimation and chunking decisions
│   ├── embedding_cache.py          # TF-IDF cosine similarity cache
│   └── prompts/                    # Language-specific prompt JSON files
│       ├── impala.json
│       ├── pyspark.json
│       ├── sparksql.json
│       ├── scala.json
│       └── python.json
│
├── app/
│   └── streamlit_app.py            # Web UI (Streamlit)
│
├── cost/                           # Cost and ROI tracking
│   ├── pricing.py                  # Price table by provider/model
│   ├── tracker.py                  # Per-agent USD accumulation
│   └── roi_logger.py               # Append-only JSONL log
│
├── data/                           # Runtime data (auto-created)
│   ├── embedding_cache.json        # Review cache (up to 200 entries)
│   └── roi_log.jsonl               # Lifetime cost/ROI log
│
├── gitlab/
│   └── qa-spark-ci-template.yml    # GitLab CI template for MR gating
│
├── config.py                       # Central configuration (reads .env)
├── cli.py                          # Command-line entry point
├── requirements.txt                # Python dependencies
├── cdsw-build.sh                   # Cloudera CAI build script
├── .project-metadata.yaml          # Cloudera AMP metadata
├── .env.example                    # Environment variable template
└── .env                            # Local secrets (DO NOT COMMIT)
```

---

## 5. Supported Languages

| Key        | Display Name  | Notes |
|------------|---------------|-------|
| `impala`   | Impala SQL    | SELECT-only; DDL statements blocked |
| `pyspark`  | PySpark       | Spark 3.4 default; Python UDF detection |
| `sparksql` | SparkSQL      | ANSI mode, AQE, hint validation |
| `scala`    | Scala Spark   | Dataset vs DataFrame, Kryo, typed lambda anti-patterns |
| `python`   | Python        | Non-Spark Python scripts and utilities |

---

## 6. Running Locally (Streamlit UI)

### Prerequisites

- Python 3.9 or higher
- Access to the IQVIA Azure OpenAI proxy (VPN required)
- `.env` file configured (see Configuration Reference below)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/robosecure/QA_Spark_CodeAgent.git
cd QA_Spark_CodeAgent

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your Azure credentials (see Configuration Reference)
```

### Launch the UI

```bash
streamlit run app/streamlit_app.py
```

Open your browser to **http://localhost:8501**

### Using the UI

**Input Tab**
- **GitLab Repo** — paste your GitLab project URL, access token, branch, and file path to fetch code directly
- **Paste Code** — paste code directly into the text area
- **Upload File** — drag and drop `.sql`, `.py`, `.scala`, or `.txt` files

Select the language/dialect from the dropdown, then click the review button.

**Results Tab**
- Color-coded score banner (green = certified, red = not certified)
- Per-agent score breakdown with weights
- Expandable full review text from each agent
- Key findings summary
- Corrected/improved code (downloadable)
- Full report download as JSON

**Cost & ROI Tab**
- Token usage and USD cost for the current review
- Per-agent cost breakdown
- Lifetime review statistics: total reviews, pass rate, average score, total cost
- Cost breakdown by language

---

## 7. Deploying to Cloudera CAI

Cloudera CAI (formerly CML) supports deploying this project as a native **Application** (always-on web UI) and as a **Job** (on-demand review).

### Method A — Deploy as a Cloudera AMP (Recommended)

1. In CAI, go to **AMPs** → **Add AMP** → **From Repository**
2. Enter the repository URL
3. CAI reads `.project-metadata.yaml` automatically and:
   - Runs `cdsw-build.sh` to install dependencies
   - Starts the Streamlit UI as an Application on subdomain `qa-spark`
4. Set the required environment variables in the AMP setup wizard (see Configuration Reference)

### Method B — Manual Project Setup

**Step 1: Create a new CAI Project**
- Go to CAI → Projects → New Project
- Source: Git → enter repo URL
- Runtime: Python 3.9, Standard edition

**Step 2: Open a Session and install dependencies**
- Open a terminal session in CAI (1 CPU, 2 GB RAM is sufficient)
- Run:
  ```bash
  bash cdsw-build.sh
  ```

**Step 3: Set Environment Variables**
- In CAI → Project Settings → Environment Variables
- Add all variables from the Configuration Reference section below
- **Never put credentials in code or the `.env` file in CAI** — always use the Environment Variables panel

**Step 4: Deploy the Streamlit Application**
- Go to CAI → Applications → New Application
- Name: `QA Spark CodeAgent`
- Subdomain: `qa-spark`
- Script:
  ```
  streamlit run app/streamlit_app.py --server.port $CDSW_APP_PORT --server.address 127.0.0.1
  ```
- Runtime: Python 3.9
- Resources: 1 CPU, 2 GB RAM

**Step 5: Access the Application**
- CAI will provide a public URL like `https://qa-spark.your-cai-domain.com`
- Share this URL with your team

### Method C — Deploy as a CAI Job (for automated review)

- Go to CAI → Jobs → New Job
- Script: `cli.py`
- Arguments: `--file <path> --language <lang> --output json`
- Set environment variables in the Job configuration
- Jobs can be triggered via the CAI REST API v2 for CI/CD integration

---

## 8. GitLab CI Integration

Add to your project's `.gitlab-ci.yml` or include the provided template:

```yaml
include:
  - project: 'your-group/QA_Spark_CodeAgent'
    file: 'gitlab/qa-spark-ci-template.yml'
```

The CI template automatically:
- Detects `.sql`, `.py`, and `.scala` files changed in the MR
- Routes SQL files to the correct dialect (set `QA_SQL_DIALECT: sparksql` or `impala`)
- Posts a review comment directly on the MR
- **Exits with code 1** if the score is below 95, blocking the merge

Required CI/CD variables (set in GitLab → Settings → CI/CD → Variables):
- `PROVIDER`, `AZURE_TENANT_ID`, `AZURE_SERVICE_PRINCIPAL`, `AZURE_SERVICE_PRINCIPAL_SECRET`
- `AZURE_ACCOUNT_NAME`, `AZURE_DEPLOYED_MODEL`, `AZURE_ENDPOINT_BASE`, `AZURE_TOKEN_AUDIENCE`
- `PASS_THRESHOLD` (default: 95)

---

## 9. Configuration Reference

Copy `.env.example` to `.env` for local development. In Cloudera CAI, set these as Project Environment Variables.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PROVIDER` | Yes | `azure` | LLM provider: `azure`, `openai`, or `bedrock` |
| `AZURE_TENANT_ID` | Yes (azure) | — | Azure AD tenant ID |
| `AZURE_SERVICE_PRINCIPAL` | Yes (azure) | — | Service Principal client ID |
| `AZURE_SERVICE_PRINCIPAL_SECRET` | Yes (azure) | — | Service Principal secret |
| `AZURE_ACCOUNT_NAME` | Yes (azure) | — | Azure OpenAI account name |
| `AZURE_DEPLOYED_MODEL` | No | `BDF-GLB-GPT-5` | Deployed model name |
| `AZURE_API_VERSION` | No | `2024-06-01` | Azure OpenAI API version |
| `AZURE_ENDPOINT_BASE` | No | `https://openai.work.iqvia.com/cse/prod/proxy/azure` | Proxy base URL |
| `AZURE_TOKEN_AUDIENCE` | No | `api://825a47b7.../.default` | Token audience for SP auth |
| `OPENAI_API_KEY` | Yes (openai) | — | OpenAI API key (non-Azure) |
| `OPENAI_API_BASE` | No | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `MODEL_NAME` | No | `gpt-4o` | Model name for OpenAI provider |
| `PASS_THRESHOLD` | No | `95` | Minimum score for certification |
| `SPARK_VERSION` | No | `3.4` | Spark version for review context |
| `AGENT_MODE` | No | `multi` | Review mode: `multi` or `single` |
| `MAX_TOKENS_PER_CHUNK` | No | `2500` | Token limit per code chunk |
| `MAX_TOKENS_PER_REVIEW` | No | `20000` | Token limit for full review |
| `COST_INPUT_PER_1K` | No | — | Override input token price (USD/1K) |
| `COST_OUTPUT_PER_1K` | No | — | Override output token price (USD/1K) |

---

## 10. Cost and ROI Tracking

Every review is logged to `data/roi_log.jsonl` with:
- Timestamp, language, file name
- Composite score and certification status
- Review mode (multi-agent or single)
- Number of chunks processed
- Whether a cache hit occurred (no LLM cost)
- Per-agent token counts and USD cost
- Provider and model used

**The Cost & ROI tab** in the Streamlit UI surfaces this data with:
- Current review cost (per-agent breakdown)
- Lifetime totals: reviews, pass rate, average score, total spend
- Cost by language

**ROI justification example:** If a single poorly-written Spark job wastes 4 cluster hours at $12/hour, one $0.03 review that catches it pays for itself 1,600 times over.

---

## 11. Security Features

### Secret Masking

Before any code is sent to the Azure OpenAI API, the sanitizer (`agent/sanitizer.py`) scans for and masks:
- Passwords in configuration strings
- API keys and secret tokens
- AWS access keys
- Azure storage connection strings
- JDBC URLs with embedded passwords

Redacted values are replaced with `[REDACTED:type]` placeholders. The LLM never sees your actual secrets.

### Security Score Capping

If the SecurityAgent's regex pre-scan detects a CRITICAL finding (hardcoded credentials, AWS keys, etc.), the security score is hard-capped at 50 regardless of the LLM response. This ensures the LLM cannot be prompted or coaxed into passing code with secrets embedded.

### Composite Score Hard Rule

If the SecurityAgent score is below 40, the composite score cannot exceed 60, even if performance and practices scores are perfect. Security is non-negotiable.

---

## 12. Token Management

### Embedding Cache

The embedding cache (`agent/embedding_cache.py`) uses TF-IDF cosine similarity to find previously reviewed code:

- **Exact hit** (SHA-256 match): Returns cached result instantly — zero LLM cost
- **Similar code** (cosine similarity ≥ 0.35): Injects findings from similar past reviews as context, reducing the LLM's workload and token count
- Cache stores up to 200 entries in `data/embedding_cache.json`

### Code Chunking

Files that exceed the token budget are split at logical boundaries:
- Python/PySpark: `def`, `class`, `async def`
- Scala: `def`, `object`, `class`, `trait`
- SQL: statement boundaries (`;`)

Each chunk is reviewed independently and scores are combined conservatively (worst-case across chunks).

### Token Budget

| Limit | Default | Override |
|-------|---------|----------|
| Per chunk | 2,500 tokens | `MAX_TOKENS_PER_CHUNK` |
| Full review | 20,000 tokens | `MAX_TOKENS_PER_REVIEW` |

Token count is estimated at 4 characters per token. Files under the limit are sent whole.

---

## 13. Known Limitations and Next Steps

### Current Limitations

| Item | Detail |
|------|--------|
| **Token count from Azure** | Azure OpenAI via Service Principal returns token usage correctly through the raw `openai` client. If you see zeros, ensure you are on AGENT_MODE=multi (uses base_agent.py, not LangChain) |
| **Bedrock provider** | Defined but not yet implemented. Will support Claude 3.5 Sonnet and Claude 3 Opus |
| **Parallel agents** | Agents currently run sequentially. Parallel execution (using `concurrent.futures`) is the next performance upgrade |
| **GitLab permissions** | GitLab push to `iqvia.gitlab-dedicated.com/rxcorp/bdf-admin` pending SSO permission fix |
| **Cache similarity** | TF-IDF is lexical only. Phase 2 will replace with Azure OpenAI `text-embedding-ada-002` for true semantic similarity |

### Recommended Next Steps

**Priority 1 — Validation (do first)**
- [ ] Run the Streamlit UI locally and submit a real PySpark job for review
- [ ] Verify the three agent scores and composite score appear correctly
- [ ] Confirm the Cost & ROI tab logs the review to `data/roi_log.jsonl`
- [ ] Test with a file containing a hardcoded password — confirm security score caps at 50

**Priority 2 — Tuning**
- [ ] Adjust scoring rubrics in each agent's system prompt based on real review outputs
- [ ] Tune `SIMILARITY_THRESHOLD` (default 0.35) in `embedding_cache.py` if similar-code hints are too aggressive or too sparse
- [ ] Tune `PASS_THRESHOLD` if 95 is too strict or too lenient for your team
- [ ] Review and adjust weight tables in `orchestrator.py` per language

**Priority 3 — Scale**
- [ ] Deploy to Cloudera CAI dev cluster (1.5.5 SP2) as an Application
- [ ] Connect GitLab CI template to a test repository
- [ ] Push to `rxcorp/bdf-admin` once SSO permissions are resolved
- [ ] Enable parallel agent execution in `orchestrator.py`
- [ ] Upgrade embedding cache to use Azure OpenAI embeddings (Phase 2)

**Priority 4 — Expansion**
- [ ] Add Bedrock provider support (Claude 3.5 Sonnet)
- [ ] Add review history export from the Streamlit UI
- [ ] Add Confluence/wiki auto-publish for review reports
- [ ] Add Slack/Teams notification on certification failure

---

## Dependencies

```
openai>=1.30.0              # Azure OpenAI and OpenAI client
langchain-openai>=0.1.0     # Legacy single-agent mode only
azure-identity>=1.15.0      # Service Principal authentication
streamlit>=1.35.0           # Web UI
python-dotenv>=1.0.0        # .env file loading
requests>=2.31.0            # GitLab API calls
```

---

## License

Internal use only — IQVIA BDF platform tooling.
Not for external distribution.

---

*QA Spark CodeAgent — Built for the Cloudera CDP platform by the BDF team.*
*Model: BDF-GLB-GPT-5 via IQVIA Azure OpenAI proxy.*
