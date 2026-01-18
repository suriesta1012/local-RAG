# local-RAG
Local RAG system using FastAPI, Ollama, and OpenSearch.

## Features
- Upload PDF documents
- Text chunking
- Sentence transformer embeddings
- OpenSearch vector store
- Ollama LLM for question answering

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Start OpenSearch locally (e.g., via Docker)
3. Start Ollama and pull the model: `ollama pull llama2`
4. Run the app: `python app.py`

## API Endpoints
- POST /upload: Upload a PDF file
- POST /query: Query the knowledge base
