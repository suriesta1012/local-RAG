# hybrid_retriever.py

import math
from typing import List

from langchain.schema import Document
from rank_bm25 import BM25Okapi

from config import BM25_WEIGHT, RETRIEVAL_TOP_K, VECTOR_WEIGHT
from vectorstore import get_vectorstore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokeniser for BM25."""
    return text.lower().split()


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a document at position `rank` (1-indexed)."""
    return 1.0 / (k + rank)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combines BM25 and ChromaDB vector search via Reciprocal Rank Fusion.

    The retriever must be re-built (or updated) whenever new documents
    are added to the vector store. Call `HybridRetriever.from_vectorstore()`
    to build it from the current ChromaDB collection, or pass documents
    directly to the constructor.

    Args:
        documents: All Document objects currently in the knowledge base.
        bm25_weight:   Weight applied to BM25 RRF scores   (default: BM25_WEIGHT).
        vector_weight: Weight applied to vector RRF scores (default: VECTOR_WEIGHT).
    """

    def __init__(
        self,
        documents: List[Document],
        bm25_weight: float = BM25_WEIGHT,
        vector_weight: float = VECTOR_WEIGHT,
    ):
        self.documents = documents
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        tokenised = [_tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenised)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_vectorstore(
        cls,
        bm25_weight: float = BM25_WEIGHT,
        vector_weight: float = VECTOR_WEIGHT,
    ) -> "HybridRetriever":
        """
        Build a HybridRetriever from all documents currently stored in
        ChromaDB.  This fetches the raw texts so BM25 can index them.
        """
        vs = get_vectorstore()
        raw = vs._collection.get(include=["documents", "metadatas"])

        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(raw["documents"], raw["metadatas"])
        ]

        if not docs:
            raise ValueError(
                "Vector store is empty — upload at least one document before "
                "building the hybrid retriever."
            )

        return cls(docs, bm25_weight=bm25_weight, vector_weight=vector_weight)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = RETRIEVAL_TOP_K) -> List[Document]:
        """
        Return the top-k documents fused from BM25 and vector search.

        Steps:
          1. BM25 scores all documents; we take the top-k ranked by BM25.
          2. ChromaDB similarity search returns its own top-k.
          3. RRF merges both ranked lists into a single score per document.
          4. Documents are sorted by descending RRF score; top-k returned.
        """
        fetch_k = min(k * 3, len(self.documents))  # fetch more than needed; RRF will trim

        # --- BM25 ranking ---
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        bm25_ranked_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:fetch_k]

        # --- Vector ranking ---
        vs = get_vectorstore()
        vector_hits: List[Document] = vs.similarity_search(query, k=fetch_k)

        # Build a content → rank map for each method
        bm25_rank: dict[str, int] = {}
        for rank, idx in enumerate(bm25_ranked_indices, start=1):
            key = self.documents[idx].page_content
            bm25_rank[key] = rank

        vector_rank: dict[str, int] = {}
        for rank, doc in enumerate(vector_hits, start=1):
            vector_rank[doc.page_content] = rank

        # --- RRF fusion ---
        # Collect all unique documents from both result sets
        seen: dict[str, Document] = {}
        for idx in bm25_ranked_indices:
            d = self.documents[idx]
            seen[d.page_content] = d
        for d in vector_hits:
            seen[d.page_content] = d

        rrf_scores: dict[str, float] = {}
        for content, doc in seen.items():
            bm25_contrib = self.bm25_weight * _rrf_score(bm25_rank.get(content, fetch_k + 1))
            vec_contrib = self.vector_weight * _rrf_score(vector_rank.get(content, fetch_k + 1))
            rrf_scores[content] = bm25_contrib + vec_contrib

        ranked = sorted(seen.values(), key=lambda d: rrf_scores[d.page_content], reverse=True)

        # Store retrieval scores in metadata for debugging / eval
        for doc in ranked[:k]:
            doc.metadata["rrf_score"] = round(rrf_scores[doc.page_content], 6)
            doc.metadata["bm25_rank"] = bm25_rank.get(doc.page_content)
            doc.metadata["vector_rank"] = vector_rank.get(doc.page_content)

        return ranked[:k]
