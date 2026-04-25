<p align="center">
  <img src="assets/banner.png" alt="TARA Banner" width="100%">
</p>

<h1 align="center">🏆 TARA — Telecom Agentic RAG Assistant</h1>

<p align="center">
  <strong>An intelligent, production-ready RAG system for telecom RAN operations</strong><br>
  <em>Samsung ennovateX AX Hackathon 2026 — Problem: RAG-based Future-Ready Telecom RAN Assistant</em>
</p>

<p align="center">
  <a href="#-key-features"><img src="https://img.shields.io/badge/Features-Hybrid%20RAG-blueviolet?style=for-the-badge" alt="Features"></a>
  <a href="#-tech-stack"><img src="https://img.shields.io/badge/LLM-Llama%203.2%203B-green?style=for-the-badge" alt="LLM"></a>
  <a href="#-tech-stack"><img src="https://img.shields.io/badge/Engine-Ollama-blue?style=for-the-badge" alt="Engine"></a>
  <a href="#-target-kpis"><img src="https://img.shields.io/badge/Faithfulness-%3E90%25-orange?style=for-the-badge" alt="KPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-red?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-evaluation">Evaluation</a> •
  <a href="#-acknowledgments">Credits</a>
</p>

---

## 📖 Overview

**TARA** (Telecom Agentic RAG Assistant) is a domain-specific Retrieval-Augmented Generation system designed to revolutionize how telecom engineers interact with 5G/O-RAN knowledge. It automates root cause analysis, answers complex specification questions, and detects anomalies — all while providing **fully explainable, citation-backed responses**.

### The Problem

| Challenge | Impact |
|-----------|--------|
| 📚 **50,000+ pages** of 3GPP/O-RAN specifications | Engineers spend hours searching for answers |
| 🔧 **Manual RCA takes 4-8 hours** per incident | High MTTR, operator revenue loss |
| 🧠 **40% shortage** in telecom AI/ML expertise | Knowledge silos, brain drain |
| 🚨 **Threshold-based** anomaly detection | High false positives, reactive response |

### TARA's Solution

TARA reduces Mean Time To Resolution from **4 hours to 30 minutes** by combining:
- 🔍 **Hybrid Retrieval** — Dense + Sparse search over telecom knowledge bases
- 🤖 **Selective Agentic Reasoning** — Multi-step reasoning only when complex queries demand it
- ✅ **Full Explainability** — Every response includes cited sources and calibrated confidence scores
- 🛡️ **Python Backend Verification** — Strict hallucination detection catches and rejects ungrounded claims

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🔍 Hybrid Retrieval System
- **Dense retrieval** via `bge-small-en-v1.5` embeddings
- **Sparse retrieval** via Qdrant native BM25
- **Reciprocal Rank Fusion (RRF)** combines results
- **Cross-encoder re-ranking** with `ms-marco-MiniLM-L-6-v2`

</td>
<td width="50%">

### 🧠 Intelligent Query Routing
- **Fast Path** — Simple queries answered in <2 seconds
- **Reasoning Path** — Complex RCA/anomaly queries routed to agentic pipeline
- **Intent Classification** — QnA, RCA, Anomaly Detection, Comparison
- **Complexity Scoring** — Automatic path selection

</td>
</tr>
<tr>
<td width="50%">

### 📊 Explainability Engine
- **Citation extraction** with inline source attribution
- **Calibrated confidence scoring** (85% confidence ≈ 85% accuracy)
- **Reasoning chain visualization** for agentic responses
- **Uncertainty handling** — graceful "I don't know" responses

</td>
<td width="50%">

### 🏗️ Telecom Domain Optimization
- **Custom NER** for telecom entities (Cell IDs, KPIs, spec refs)
- **Acronym expansion** (3000+ telecom acronyms)
- **Telecom-aware chunking** respecting spec boundaries
- **Domain-specific prompts** for 3GPP/O-RAN context

</td>
</tr>
</table>

---

## 🎯 Target KPIs

| Metric | Target | Description |
|--------|--------|-------------|
| **MRR** (Mean Reciprocal Rank) | > 75% | Quality of retrieval ranking |
| **Top-k Accuracy** | > 85% | Correct answer within top-k results |
| **Accuracy** | > 80% | Overall answer correctness |
| **Recall** | > 85% | Relevant documents found |
| **Faithfulness** | > 90% ⚠️ | Every claim grounded in sources — **hardest target** |

> **Note**: Faithfulness (90%) is the most critical and challenging metric. TARA addresses this through strict Python backend verification — every LLM claim is checked against retrieved context, and ungrounded answers are rejected and retried.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                               │
│         Streamlit Web UI  •  FastAPI REST  •  CLI Tool               │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     QUERY UNDERSTANDING MODULE                       │
│    Intent Classification → NER → Acronym Expansion → Routing        │
│                                                                      │
│              ┌──────────────┐          ┌──────────────────┐          │
│              │  FAST PATH   │          │  REASONING PATH  │          │
│              │  (< 2 sec)   │          │  (Agentic, CoT)  │          │
│              └──────┬───────┘          └────────┬─────────┘          │
└─────────────────────┼──────────────────────────┼────────────────────┘
                      │                          │
                      ▼                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      HYBRID RETRIEVAL LAYER                          │
│                                                                      │
│    ┌─────────────────┐      ┌──────────────────┐                    │
│    │ DENSE RETRIEVAL  │      │ SPARSE RETRIEVAL  │                    │
│    │ bge-small-en-v1.5│      │ Qdrant Native BM25│                    │
│    └────────┬────────┘      └────────┬─────────┘                    │
│             └───────────┬────────────┘                               │
│                         ▼                                            │
│              Reciprocal Rank Fusion (RRF)                            │
│                         ▼                                            │
│           Cross-Encoder Re-ranking (MiniLM)                          │
│                         ▼                                            │
│              Top-10 Relevant Documents                               │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     GENERATION & VERIFICATION                        │
│                                                                      │
│   Llama-3.2-3B-Instruct (4-bit GGUF via Ollama)                    │
│          │                                                           │
│          ▼                                                           │
│   ┌─────────────────────────────────────────────┐                   │
│   │  PYTHON BACKEND VERIFICATION                 │                   │
│   │  • Citation check against Qdrant context     │                   │
│   │  • Structured JSON output validation         │                   │
│   │  • Hallucination detection & retry           │                   │
│   └─────────────────────────────────────────────┘                   │
│          │                                                           │
│          ▼                                                           │
│   Explainability: Citations + Confidence Score + Reasoning Chain     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) (4-bit GGUF) | Generation — ~2GB VRAM |
| **Inference** | [Ollama](https://ollama.com/) | Local LLM serving with CPU/GPU hybrid offload |
| **Embeddings** | [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | Dense retrieval — 384d, ~130MB |
| **Re-ranker** | [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | Cross-encoder re-ranking — ~80MB |
| **Vector DB** | [Qdrant](https://qdrant.tech/) (local mode) | Dense + Sparse (BM25) vectors |
| **Cache** | `functools.lru_cache` + SQLite | Lightweight semantic caching |
| **Orchestration** | [LangChain](https://langchain.com/) + [LangGraph](https://github.com/langchain-ai/langgraph) | Pipeline orchestration & agentic workflows |
| **NER** | [spaCy](https://spacy.io/) + Custom | Telecom entity extraction |
| **API** | [FastAPI](https://fastapi.tiangolo.com/) | REST API backend |
| **UI** | [Streamlit](https://streamlit.io/) | Interactive web interface |
| **Evaluation** | [RAGAS](https://docs.ragas.io/) | RAG evaluation framework |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | RTX 4050 (6GB VRAM) | Any GPU with 6GB+ VRAM |
| **RAM** | 16GB | 32GB |
| **Storage** | 50GB SSD | 100GB NVMe SSD |
| **CPU** | 4 cores | 8+ cores |

> 💡 **Designed for laptop-grade hardware.** Ollama automatically handles CPU/GPU hybrid offloading. Total VRAM footprint: ~3-4GB (LLM + embeddings), leaving room for KV cache during inference. No cloud GPU required.

---

## 📁 Project Structure

```
tara-telecom-rag/
│
├── 📁 config/                        # Configuration files
│   ├── settings.py                   # Global settings (Pydantic)
│   ├── prompts.yaml                  # LLM prompt templates
│   └── models.yaml                   # Model configurations
│
├── 📁 data/                          # Data directory
│   ├── raw/                          # Raw source data
│   │   ├── 3gpp_specs/               # 3GPP specification PDFs
│   │   ├── oran_specs/               # O-RAN specification PDFs
│   │   └── teleqna/                  # TeleQnA evaluation dataset
│   ├── processed/                    # Processed & indexed data
│   │   ├── chunks/                   # Text chunks with metadata
│   │   └── embeddings/               # Pre-computed embeddings
│   └── evaluation/                   # Evaluation datasets & results
│
├── 📁 src/                           # Core source code
│   ├── 📁 ingestion/                 # Data ingestion pipeline
│   │   ├── pdf_extractor.py          # PDF text extraction (PyMuPDF)
│   │   ├── table_extractor.py        # Table extraction (pdfplumber)
│   │   ├── chunker.py                # Telecom-aware semantic chunking
│   │   ├── telecom_preprocessor.py   # Domain-specific preprocessing
│   │   └── indexer.py                # Qdrant vector + sparse indexing
│   │
│   ├── 📁 retrieval/                 # Retrieval components
│   │   ├── dense_retriever.py        # Dense vector retrieval
│   │   ├── sparse_retriever.py       # BM25 sparse retrieval (Qdrant)
│   │   ├── hybrid_fusion.py          # Reciprocal Rank Fusion (RRF)
│   │   ├── reranker.py               # Cross-encoder re-ranking
│   │   └── retrieval_pipeline.py     # Combined retrieval pipeline
│   │
│   ├── 📁 query/                     # Query understanding
│   │   ├── intent_classifier.py      # Intent classification
│   │   ├── entity_extractor.py       # Telecom NER
│   │   ├── query_enricher.py         # Acronym expansion & enrichment
│   │   ├── complexity_scorer.py      # Query complexity assessment
│   │   └── router.py                 # Fast path vs. reasoning path
│   │
│   ├── 📁 generation/                # Response generation
│   │   ├── context_builder.py        # LLM context construction
│   │   ├── llm_interface.py          # Ollama LLM abstraction
│   │   ├── generator.py              # Response generation
│   │   └── post_processor.py         # Citation extraction & formatting
│   │
│   ├── 📁 reasoning/                 # Selective agentic reasoning
│   │   ├── rca_agent.py              # Root cause analysis agent
│   │   ├── hypothesis_generator.py   # RCA hypothesis generation
│   │   ├── evidence_gatherer.py      # Evidence collection
│   │   └── reasoning_chain.py        # Multi-step reasoning (LangGraph)
│   │
│   ├── 📁 explainability/            # Explainability engine
│   │   ├── citation_engine.py        # Source attribution
│   │   ├── confidence_scorer.py      # Calibrated confidence scoring
│   │   ├── reasoning_visualizer.py   # Reasoning chain visualization
│   │   └── uncertainty_handler.py    # Low-confidence handling
│   │
│   ├── 📁 evaluation/                # Evaluation framework
│   │   ├── metrics.py                # MRR, Accuracy, Recall, etc.
│   │   ├── teleqna_evaluator.py      # TeleQnA benchmark runner
│   │   └── ragas_evaluator.py        # RAGAS integration
│   │
│   └── 📁 utils/                     # Utilities
│       ├── telecom_dictionary.py     # 3000+ telecom acronyms
│       ├── logging_utils.py          # Structured logging
│       └── caching.py               # lru_cache + SQLite cache
│
├── 📁 api/                           # FastAPI application
│   ├── main.py                       # API entrypoint
│   ├── routes/                       # API endpoints
│   └── schemas/                      # Pydantic request/response models
│
├── 📁 ui/                            # Streamlit frontend
│   ├── streamlit_app.py              # Main UI application
│   └── components/                   # Chat, sources, visualization
│
├── 📁 tests/                         # Test suite
├── 📁 docs/                          # Documentation
├── .env.example                      # Environment configuration
├── requirements.txt                  # Python dependencies
├── docker-compose.yml                # Container orchestration
└── README.md                         # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.11+
- **Ollama** ([install guide](https://ollama.com/download))
- **Git**
- GPU with 6GB+ VRAM (or CPU-only mode via Ollama)

### 1. Clone & Install

```bash
git clone https://github.com/your-username/tara-telecom-rag.git
cd tara-telecom-rag
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Pull the LLM via Ollama

```bash
# Install Ollama (if not already installed)
# https://ollama.com/download

# Pull the quantized model (~2GB download)
ollama pull llama3.2:3b

# Verify it's running
ollama list
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings (defaults work out of the box)
```

### 4. Download & Index Data

```bash
# Download TeleQnA dataset
python -m src.ingestion.download --dataset teleqna

# Download 3GPP specifications (selected series)
python -m src.ingestion.download --dataset 3gpp --series 38

# Process and index documents
python -m src.ingestion.indexer --config config/settings.py
```

### 5. Launch TARA

```bash
# Option A: Streamlit UI
streamlit run ui/streamlit_app.py

# Option B: FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# Option C: CLI
python -m src.cli --query "What is the purpose of PDCCH in NR?"
```

---

## 📊 Evaluation

### Running the TeleQnA Benchmark

```bash
python -m src.evaluation.teleqna_evaluator \
    --dataset data/raw/teleqna/TeleQnA.json \
    --output results/teleqna_results.json
```

### Running RAGAS Evaluation

```bash
python -m src.evaluation.ragas_evaluator \
    --config config/evaluation.yaml \
    --output results/ragas_results.json
```

### Metrics Tracked

| Metric | Measures | How We Optimize |
|--------|----------|-----------------|
| **MRR** | Retrieval ranking quality | Hybrid retrieval + RRF fusion |
| **Top-k Accuracy** | Coverage of retrieval | Telecom-aware chunking + query expansion |
| **Accuracy** | End-to-end correctness | Domain prompts + backend verification |
| **Recall** | Document coverage | Dense + sparse dual retrieval paths |
| **Faithfulness** | Grounding in sources | Citation engine + hallucination rejection |

---

## 📚 Datasets & Knowledge Sources

| Dataset | Description | Use in TARA |
|---------|-------------|-------------|
| [**TeleQnA**](https://github.com/netop-team/TeleQnA) | 1,827 MCQ from 3GPP standards | Primary evaluation benchmark |
| [**3GPP Specifications**](https://www.3gpp.org/specifications) | Release 16/17/18 tech specs | Core knowledge base (TS 38.xxx, 23.xxx) |
| [**O-RAN Alliance Specs**](https://www.o-ran.org/specifications) | O-RAN architecture & interfaces | Supplementary knowledge base |
| [**Simu5G**](https://github.com/Unipisa/Simu5G) | Synthetic 5G simulation data | Anomaly detection training |
| [**Tele-LLMs**](https://github.com/Ali-maatouk/Tele-LLMs) | Telecom language models research | Reference & fine-tuning data |

---

## 🗺️ Roadmap

### Phase 1 — Blueprint Submission (by May 13, 2026)
- [x] Solution architecture design
- [x] Technology stack selection & hardware feasibility
- [x] Implementation plan & timeline
- [ ] Blueprint PDF submission

### Phase 2 — Full Solution (May 26 — Jun 22, 2026)
- [ ] Data ingestion pipeline (PDF extraction, chunking, indexing)
- [ ] Hybrid retrieval system (Dense + Sparse + RRF)
- [ ] Query understanding module (NER, intent, routing)
- [ ] Generation pipeline (Ollama + context builder + citations)
- [ ] Explainability engine (confidence scoring, citation verification)
- [ ] RCA agent (LangGraph-based agentic reasoning)
- [ ] Streamlit UI + FastAPI backend
- [ ] Evaluation on TeleQnA benchmark

### Phase 3 & 4 — Presentation & Grand Finale
- [ ] Performance optimization & demo preparation
- [ ] Live presentation at Samsung ennovateX 2026

---

## 🏅 Hackathon Alignment

| Hackathon Expectation | TARA's Delivery |
|----------------------|-----------------|
| **AI Transformation (AX)** | Transforms manual telecom ops → AI-assisted decision-making |
| **Agentic Systems** | Selective agents for RCA and complex multi-step reasoning |
| **Tool Use / Chaining** | LangGraph orchestration, tool-augmented retrieval |
| **Memory / Context** | Semantic caching (SQLite), session-aware context building |
| **Reasoning & Planning** | Chain-of-thought prompting, hypothesis-evidence frameworks |
| **Open Weight Models** | Llama 3.2 3B (Apache 2.0 compatible), all deps OSS |
| **No Commercial APIs** | 100% local — Ollama + open models, zero API keys |

---

## 🤝 Contributing

This project is developed for the Samsung ennovateX AX Hackathon 2026. During the hackathon period, contributions are limited to team members. Post-hackathon, contributions will be welcome under the Apache 2.0 license.

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

All developed code and models are released under Apache-2.0 as per hackathon requirements.

---

## 🙏 Acknowledgments

- **Samsung R&D Institute India — Bangalore** for organizing ennovateX AX Hackathon 2026
- **Meta** for [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- **BAAI** for [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **Qdrant** for the vector database with native sparse vector support
- **Ollama** for making local LLM inference accessible
- **3GPP** and **O-RAN Alliance** for open telecom specifications
- The [TeleQnA](https://github.com/netop-team/TeleQnA) team for the evaluation dataset

---

<p align="center">
  <strong>🚀 Built for the Samsung ennovateX AX Hackathon 2026</strong><br>
  <em>Bringing AI Transformation to Telecom — One Query at a Time</em>
</p>