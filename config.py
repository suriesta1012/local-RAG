# config.py
import os

# --- ChromaDB ---
# Data is persisted in a local folder; no external service required.
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag-documents")

# --- Embedding model ---
# all-MiniLM-L6-v2 is fast and small; swap for a larger model for better recall.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Reranker model ---
# Cross-encoder reranker; runs locally, no API key needed.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- LLM (via Ollama) ---
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

# --- Chunking ---
# CHUNK_SIZE: target token count per chunk (not characters).
# CHUNK_OVERLAP: how many tokens overlap between adjacent chunks.
# RETRIEVAL_TOP_K: number of candidates fetched from the vector store before reranking.
# RERANK_TOP_N: final number of chunks passed to the LLM after reranking.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 20))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 5))

# --- Hybrid retrieval (BM25 + vector) ---
# BM25_WEIGHT + VECTOR_WEIGHT should sum to 1.0.
# Raise BM25_WEIGHT for keyword-heavy domains (legal, medical codes).
# Raise VECTOR_WEIGHT for semantic / paraphrase-heavy queries.
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", 0.4))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", 0.6))

# --- RAG evaluation ---
# Directory where eval results (JSON + CSV) are saved.
EVAL_OUTPUT_DIR = os.getenv("EVAL_OUTPUT_DIR", "./eval_results")
# Number of LLM-generated Q&A pairs to create when auto-building a test set.
EVAL_AUTO_QA_COUNT = int(os.getenv("EVAL_AUTO_QA_COUNT", 10))
