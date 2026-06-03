# app.py
"""
Local RAG API  v3.0

Endpoints
---------
GET  /              — status
GET  /health        — readiness probe
POST /upload        — ingest a PDF
POST /query         — hybrid retrieve → rerank → LLM answer
POST /eval          — evaluate a list of Q&A samples
POST /eval/auto     — auto-generate Q&A from the knowledge base and evaluate
DELETE /reset       — wipe the vector store
"""

import os
import shutil
import tempfile
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from config import RERANK_TOP_N, RETRIEVAL_TOP_K
from document_processor import load_and_chunk_pdf
from evaluator import EvalReport, EvalSample, Evaluator
from hybrid_retriever import HybridRetriever
from langchain.schema import Document
from llm import get_llm
from reranker import rerank
from vectorstore import get_vectorstore

app = FastAPI(
    title="Local RAG System",
    description=(
        "Fully local RAG: PDF ingestion · hybrid BM25+vector retrieval · "
        "cross-encoder reranking · Ollama LLM · built-in evaluation."
    ),
    version="3.0.0",
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question to answer.")
    top_k: Optional[int] = Field(
        default=RETRIEVAL_TOP_K, ge=1, le=100,
        description="Candidates to fetch (BM25 + vector each) before reranking.",
    )
    top_n: Optional[int] = Field(
        default=RERANK_TOP_N, ge=1, le=20,
        description="Chunks passed to the LLM after cross-encoder reranking.",
    )


class SourceChunk(BaseModel):
    page_content: str
    source: Optional[str] = None
    page: Optional[int] = None
    rerank_score: Optional[float] = None
    rrf_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    retrieval_mode: str = "hybrid (BM25 + vector + rerank)"
    sources: List[SourceChunk]


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    message: str


class HealthResponse(BaseModel):
    status: str
    vectorstore: str
    llm_model: str
    collection_count: int


class EvalSampleRequest(BaseModel):
    question: str = Field(..., min_length=1)
    expected_answer: Optional[str] = None


class EvalRequest(BaseModel):
    samples: List[EvalSampleRequest] = Field(..., min_items=1)
    save_results: bool = Field(default=True, description="Write JSON+CSV to eval_results/")


class EvalAutoRequest(BaseModel):
    count: int = Field(default=10, ge=1, le=50,
                       description="Number of Q&A pairs to auto-generate.")
    save_results: bool = True


class EvalResultResponse(BaseModel):
    question: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: Optional[float]
    latency_ms: float
    overall_score: float
    sources: List[str]


class EvalReportResponse(BaseModel):
    sample_count: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: Optional[float]
    avg_latency_ms: float
    avg_overall_score: float
    results: List[EvalResultResponse]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc_to_source(doc: Document) -> SourceChunk:
    return SourceChunk(
        page_content=doc.page_content,
        source=doc.metadata.get("source"),
        page=doc.metadata.get("page"),
        rerank_score=doc.metadata.get("rerank_score"),
        rrf_score=doc.metadata.get("rrf_score"),
        bm25_rank=doc.metadata.get("bm25_rank"),
        vector_rank=doc.metadata.get("vector_rank"),
    )


def _report_to_response(report: EvalReport) -> EvalReportResponse:
    recall = report.avg_context_recall
    return EvalReportResponse(
        sample_count=len(report.results),
        avg_faithfulness=round(report.avg_faithfulness, 4),
        avg_answer_relevancy=round(report.avg_answer_relevancy, 4),
        avg_context_precision=round(report.avg_context_precision, 4),
        avg_context_recall=round(recall, 4) if recall == recall else None,  # NaN → None
        avg_latency_ms=round(report.avg_latency_ms, 1),
        avg_overall_score=round(report.avg_overall, 4),
        results=[
            EvalResultResponse(
                question=r.question,
                answer=r.answer,
                faithfulness=round(r.faithfulness, 4),
                answer_relevancy=round(r.answer_relevancy, 4),
                context_precision=round(r.context_precision, 4),
                context_recall=round(r.context_recall, 4) if r.expected_answer else None,
                latency_ms=round(r.latency_ms, 1),
                overall_score=round(r.overall_score, 4),
                sources=r.sources,
            )
            for r in report.results
        ],
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
async def root():
    return {"message": "Local RAG System v3 is running. See /docs for the API."}


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health():
    """Readiness probe — also shows current document count in the collection."""
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
        return HealthResponse(
            status="ok",
            vectorstore="chromadb",
            llm_model=get_llm().model,
            collection_count=count,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {exc}")


@app.post("/upload", response_model=UploadResponse, tags=["Ingestion"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF into the knowledge base.
    Chunks are embedded and stored in ChromaDB.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = load_and_chunk_pdf(tmp_path)
        if not chunks:
            raise HTTPException(status_code=422, detail="No text could be extracted.")

        for chunk in chunks:
            chunk.metadata["source"] = file.filename

        get_vectorstore().add_documents(chunks)

        return UploadResponse(
            filename=file.filename,
            chunks_added=len(chunks),
            message=f"Ingested '{file.filename}' as {len(chunks)} chunks.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    finally:
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse, tags=["Retrieval"])
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base using hybrid retrieval.

    Pipeline:
      1. BM25 keyword search over all stored chunks (exact term matching).
      2. ChromaDB vector search (semantic similarity).
      3. Reciprocal Rank Fusion (RRF) merges both ranked lists.
      4. Cross-encoder reranker re-scores the fused top-k candidates.
      5. Top-n chunks are stuffed into the Ollama LLM prompt.
    """
    vs = get_vectorstore()
    if vs._collection.count() == 0:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base is empty. Upload at least one document first.",
        )

    try:
        retriever = HybridRetriever.from_vectorstore()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Step 1+2+3 — hybrid retrieval via RRF
    candidates: List[Document] = retriever.retrieve(request.query, k=request.top_k)
    if not candidates:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    # Step 4 — cross-encoder rerank
    top_docs = rerank(request.query, candidates, top_n=request.top_n)

    # Step 5 — LLM generation
    context = "\n\n---\n\n".join(doc.page_content for doc in top_docs)
    prompt = (
        "Use only the context below to answer the question. "
        "If the answer is not in the context, say \"I don't know.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {request.query}\n\nAnswer:"
    )

    try:
        answer = get_llm().invoke(prompt)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"LLM error (is Ollama running?): {exc}")

    return QueryResponse(
        answer=answer,
        sources=[_doc_to_source(doc) for doc in top_docs],
    )


@app.post("/eval", response_model=EvalReportResponse, tags=["Evaluation"])
async def evaluate(request: EvalRequest):
    """
    Evaluate the RAG pipeline against a list of questions.

    Provide `expected_answer` to enable the context_recall metric.
    Results are optionally saved to `eval_results/` as JSON + CSV.

    Metrics returned:
      - **faithfulness**: are all answer claims supported by the context?
      - **answer_relevancy**: is the answer on-topic for the question?
      - **context_precision**: what fraction of retrieved chunks were useful?
      - **context_recall**: did the context contain enough to answer? (requires expected_answer)
    """
    vs = get_vectorstore()
    if vs._collection.count() == 0:
        raise HTTPException(status_code=404, detail="Knowledge base is empty.")

    samples = [
        EvalSample(question=s.question, expected_answer=s.expected_answer)
        for s in request.samples
    ]

    try:
        evaluator = Evaluator()
        report = evaluator.run(samples, save=request.save_results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")

    return _report_to_response(report)


@app.post("/eval/auto", response_model=EvalReportResponse, tags=["Evaluation"])
async def evaluate_auto(request: EvalAutoRequest):
    """
    Auto-generate Q&A pairs from the knowledge base using the LLM,
    then evaluate the full RAG pipeline against them.

    Useful when you don't have a hand-labelled test set.
    """
    vs = get_vectorstore()
    if vs._collection.count() == 0:
        raise HTTPException(status_code=404, detail="Knowledge base is empty.")

    try:
        evaluator = Evaluator()
        report = evaluator.run_auto(count=request.count, save=request.save_results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Auto-evaluation failed: {exc}")

    return _report_to_response(report)


@app.delete("/reset", tags=["Ingestion"])
async def reset_knowledge_base():
    """Wipe the entire vector store collection. Protect this in production."""
    try:
        vs = get_vectorstore()
        vs._client.delete_collection(vs._collection.name)
        get_vectorstore.cache_clear()
        return {"message": "Knowledge base cleared successfully."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
