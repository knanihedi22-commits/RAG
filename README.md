# RAG
A Retrieval-Augmented Generation (RAG) pipeline that analyzes a startup's data room and produces a structured **solvency and liquidity assessment** for investment due diligence.

The project simulates the work of a venture-capital analyst reviewing **FallahTech SARL**, a Tunisian AgriTech startup requesting a **3,000,000 TND Series A** round. It ingests the company's financial documents (PDF and Excel), indexes them in a vector database, and uses an LLM to answer analytical questions with **source and page citations** and a final **risk verdict** (Low / Medium / High).

---

## Features

- **Multi-format ingestion**: extracts text and tables from PDFs (`pdfplumber`) and every sheet of Excel workbooks (`openpyxl`).
- **Table-aware extraction**: tables are appended to page text under a `[TABLEAU]` marker so figures are not lost.
- **Paragraph-aware chunking**: 1,200-character chunks with 240-character overlap.
- **Hosted embeddings**: `BAAI/bge-small-en-v1.5` through the Hugging Face Inference API (no local model install).
- **Persistent vector store**: ChromaDB with cosine similarity; the index is reused between runs.
- **Grounded generation**: `Llama-3.1-70B-Instruct` (temperature 0.1) answers only from retrieved excerpts.
- **Traceable answers**: every figure must cite its source file and page.
- **Built-in financial reasoning rules**: the system prompt encodes current-ratio thresholds and trend checks to reduce numerical hallucinations.

---

## How it works

```
 Data room (PDF / XLSX)
          │
          ▼
  Extraction  ──►  Chunking  ──►  Embeddings (HF API)  ──►  ChromaDB
                                                               │
 Question  ──►  Embed query  ──►  Top-k retrieval  ◄───────────┘
                                       │
                                       ▼
                     Prompt (system rules + context + question)
                                       │
                                       ▼
                         Llama-3.1-70B-Instruct (HF API)
                                       │
                                       ▼
              [Analyse] → [Signaux d'alerte] → [Verdict]
```

### Liquidity rules embedded in the prompt

| Current ratio (current assets / current liabilities) | Interpretation | Risk |
|---|---|---|
| > 2 | Comfortable liquidity | Low |
| 1 – 2 | Acceptable liquidity | Medium |
| 0.5 – 1 | Cash-flow tension | High |
| < 0.5 | Imminent insolvency | Critical |

The model is also instructed to flag sharp ratio declines (e.g. 9.43 → 1.56) as a trend warning, even when the absolute level remains acceptable.

---

## Project structure

```
.
├── systpr.py                    # Full pipeline: ingest → embed → retrieve → generate
├── .env                         # HF_TOKEN (not committed)
├── DataRoom_FallahTech_PDF/     # Source documents (PDF / XLSX)
└── chroma_db/                   # Persistent vector index (auto-created)
```

---

## Getting started

### Prerequisites

- Python 3.10+
- A [Hugging Face](https://huggingface.co/) account and access token
- Access granted to the gated model `meta-llama/Llama-3.1-70B-Instruct`

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

pip install pdfplumber chromadb openpyxl python-dotenv huggingface_hub
```

### Configuration

Create a `.env` file at the project root:

```env
HF_TOKEN=hf_your_token_here
```

Place your documents in `DataRoom_FallahTech_PDF/` (sub-folders are scanned recursively).

### Run

```bash
python systpr.py
```

The script runs three stages:

1. **Ingestion** — reads all `.pdf` and `.xlsx` files and builds chunks.
2. **Embedding & indexation** — embeds chunks in batches of 16 and stores them in ChromaDB.
3. **Test questions** — runs three predefined analyses and prints answers with retrieved sources and latency.

---

## Configuration reference

| Parameter | Default | Description |
|---|---|---|
| `LLM_MODEL` | `meta-llama/Llama-3.1-70B-Instruct` | Generation model |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `LLM_TEMPERATURE` | `0.1` | Low value for factual consistency |
| `MAX_NEW_TOKENS` | `1024` | Max answer length |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `240` | Chunking settings (characters) |
| `top_k` | `6` | Number of chunks retrieved per question |
| `DATA_ROOT` | `DataRoom_FallahTech_PDF` | Input folder |
| `CHROMA_DIR` | `chroma_db` | Vector store location |

---

## Example questions

1. How did the current ratio evolve between 2023 and 2025, and what is the short-term repayment capacity?
2. Stress-test: could the company repay its suppliers immediately given its cash position?
3. Does the structural solvency justify a 3 M TND Series A at a 12 M TND pre-money valuation?

Each answer follows the structure **[Analyse] → [Signaux d'alerte] → [Verdict]**, with citations such as *(Bilan_2025.pdf, p.3)*.

---

## Known limitations & roadmap

- **Embedding language**: `bge-small-en-v1.5` is English-focused while the documents and prompts are in French. A multilingual model (e.g. `BAAI/bge-m3` or `multilingual-e5`) would likely improve retrieval.
- **Index reuse check**: the collection is reused if the chunk *count* matches; changed content with the same count will not trigger re-indexing.
- **Prompt hardcoding**: example ratios (9.43 → 1.56) and company details are embedded in the prompt and test questions. Generalize them for other companies.
- **Scope**: only task T1 (solvency & liquidity) is implemented.
- **Ideas**: hybrid search (BM25 + vectors), re-ranking, automated evaluation of answer faithfulness, a CLI or web UI.

---

## Disclaimer

This project is an educational / prototype tool. Its output is not financial advice and should not replace professional due diligence.
