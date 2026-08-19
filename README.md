# Verifiable Local RAG (Offline Document QA & Fact-Checker)

An Offline, Hallucination-Free Retrieval-Augmented Generation (RAG) platform featuring Sentence-Level Fact-Checking, Deterministic Fallback Vector Engine, and Page-Aware Citation Verification. Built with Python, FastAPI, Microsoft Foundry Local SDK, and SQLite.

**Author:** Muhammed Beşir Kesen

---

## Key Features

- Microsoft Foundry Local SDK Integration: Runs local LLMs (Qwen2.5-0.5B / Phi-4) fully offline without external API dependencies.
- Fault-Tolerant Deterministic Fallback Engine: Features a custom hash-based vector engine that ensures application continuity even if SDK runtime is absent.
- Zero-Hallucination Fact-Checker: Evaluates model responses sentence-by-sentence against source chunks using Jaccard word-overlap matching (0.0% - 100.0% confidence score).
- Query Relevance Filter: Prevents false positive citations by verifying meaningful keyword intersections.
- Markdown Table Parsing (pdfplumber): Extracts tabular data from PDFs as Markdown matrices to preserve numerical context.
- Dual-Language UI (TR / EN): Compact custom web interface with instant language switching.

---

## System Architecture

```
[PDF / TXT Document]
        │
        ▼
[Parser & Table Extractor] ── (pdfplumber / Markdown Table Matrix)
        │
        ▼
[Chunker & Overlap Engine] ── (Page-Aware Metadata)
        │
        ▼
[Vector Embedding] ────────── (Microsoft Foundry SDK / Fallback Hash 384D)
        │
        ▼
[SQLite Vector Store] ─────── (JSON Serialized Vector Storage)
        │
        ▼
[Smart Retriever] ─────────── (Query Relevance Check + Cosine Similarity)
        │
        ▼
[Local LLM (Qwen2.5)] ─────── (Strict Zero-Hallucination System Prompt)
        │
        ▼
[Fact-Checker Verifier] ───── (Sentence-Level Citation Matching & Scoring)
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10 or higher.
- pip package manager.

### 2. Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run Application

Launch the local web app:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000 in your browser.

---

## Testing & Verification

Run the automated offline test suite covering target document retrieval, table extraction, and hallucination checks:

```bash
python -m notes.test_suite
```

---

## Project Structure

```
verifiable-local-rag/
├── api.py                  # FastAPI backend
├── static/                 # Custom web UI (HTML/CSS/JS)
├── src/
│   ├── database.py         # SQLite Vector Store & Cosine Similarity
│   ├── ingest.py           # PDF/TXT Parser & Markdown Table Extractor
│   ├── retriever.py        # Smart Retriever & Query Relevance Filter
│   ├── llm.py              # Microsoft Foundry Local SDK Client & Fallback Engine
│   └── verifier.py         # Sentence-Level Jaccard Fact-Checker
├── data/                   # Uploaded documents (local only)
├── notes/                  # Local notes (not pushed to GitHub)
└── requirements.txt        # Python Dependencies
```

---

## License

Distributed under the MIT License.

Copyright (c) 2026 Muhammed Beşir Kesen
