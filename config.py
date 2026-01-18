# config.py
import os

# OpenSearch configuration
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "rag-documents")

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM model (Ollama)
LLM_MODEL = "llama2"

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200