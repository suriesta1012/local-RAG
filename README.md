# local-RAG

A fully local RAG system — no cloud APIs, no external services, no Docker required.

## Stack

| Component | Library | Why |
|-----------|---------|-----|
| API | FastAPI | Async, auto-docs at `/docs` |
| Vector store | ChromaDB | In-process, persists to disk, zero setup |
| Keyword search | BM25 (`rank-bm25`) | Exact term matching for hybrid retrieval |
| Fusion | Reciprocal Rank Fusion | Merges BM25 + vector rankings without score calibration |
| Embeddings | `all-MiniLM-L6-v2` | Fast bi-encoder for indexing & search |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision re-scoring after fusion |
| LLM | Ollama (`llama3.2`) | Fully local inference |

## How it works

```
PDF upload
   └─► two-pass semantic chunking
          └─► bi-encoder embed ──► ChromaDB

Query
   ├─► BM25 keyword search  ──┐
   └─► ChromaDB vector search ┤
                              ▼
                   Reciprocal Rank Fusion (RRF)
                              │
                   Cross-encoder reranker
                              │
                   Ollama LLM (stuffed context)
                              │
                   answer + source citations
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install and start Ollama  →  https://ollama.com
ollama pull llama3.2

# 3. Run the API
python app.py
# Open http://localhost:8000/docs
```

ChromaDB creates `./chroma_db/` automatically on first run. No Docker needed.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Status |
| GET | `/health` | Readiness probe; shows document count |
| POST | `/upload` | Ingest a PDF (form-data `file` field) |
| POST | `/query` | Hybrid search → rerank → LLM answer |
| POST | `/eval` | Evaluate against a list of Q&A samples |
| POST | `/eval/auto` | Auto-generate test Q&A and evaluate |
| DELETE | `/reset` | Wipe the knowledge base |

## Usage examples

### Upload a PDF
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@report.pdf"
```

### Query (hybrid retrieval + rerank)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?", "top_k": 20, "top_n": 5}'
```

Response includes `rrf_score`, `bm25_rank`, `vector_rank`, and `rerank_score` per source chunk — useful for debugging retrieval quality.

### Evaluate with your own Q&A pairs
```bash
curl -X POST http://localhost:8000/eval \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {
        "question": "What is the main conclusion?",
        "expected_answer": "The study found that..."
      },
      {
        "question": "Who are the authors?",
        "expected_answer": "Smith and Jones"
      }
    ],
    "save_results": true
  }'
```

`expected_answer` is optional but enables the `context_recall` metric.

### Auto-evaluate (no test set needed)
```bash
curl -X POST http://localhost:8000/eval/auto \
  -H "Content-Type: application/json" \
  -d '{"count": 10, "save_results": true}'
```

The LLM generates 10 Q&A pairs from random chunks in the knowledge base, then evaluates the full pipeline against them. Results saved to `./eval_results/`.

## Evaluation metrics

| Metric | How it's computed |
|--------|-------------------|
| **faithfulness** | LLM-as-judge: are all answer claims grounded in the retrieved context? |
| **answer_relevancy** | Cosine similarity between query embedding and answer embedding |
| **context_precision** | LLM-as-judge: what fraction of retrieved chunks were actually useful? |
| **context_recall** | LLM-as-judge: did the context contain enough info to produce the expected answer? |
| **latency_ms** | End-to-end wall-clock time for retrieval + reranking + generation |
| **overall_score** | Arithmetic mean of the above (recall excluded when no expected answer) |

All metrics run **fully locally** via Ollama — no OpenAI API key needed.

## Configuration

All settings are env-var overridable (see `config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB data directory |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Bi-encoder for embedding |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for reranking |
| `LLM_MODEL` | `llama3.2` | Ollama model name |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks |
| `RETRIEVAL_TOP_K` | `20` | Candidates fetched before reranking |
| `RERANK_TOP_N` | `5` | Chunks passed to LLM after reranking |
| `BM25_WEIGHT` | `0.4` | RRF weight for BM25 results |
| `VECTOR_WEIGHT` | `0.6` | RRF weight for vector results |
| `EVAL_OUTPUT_DIR` | `./eval_results` | Where eval JSON+CSV are saved |
| `EVAL_AUTO_QA_COUNT` | `10` | Q&A pairs for auto-evaluation |

## File structure

```
local-RAG/
├── app.py                 — FastAPI application + all endpoints
├── config.py              — All settings, env-var driven
├── document_processor.py  — Two-pass semantic PDF chunking
├── embeddings.py          — Cached bi-encoder embeddings
├── vectorstore.py         — Cached ChromaDB client
├── hybrid_retriever.py    — BM25 + vector fusion via RRF
├── reranker.py            — Cross-encoder reranking
├── llm.py                 — Cached Ollama LLM
├── evaluator.py           — RAG eval (faithfulness, relevancy, precision, recall)
└── requirements.txt       — Pinned dependencies
```
