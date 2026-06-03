# document_processor.py



import re
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


# Separators ordered from coarsest to finest so the splitter tries to
# break at natural boundaries first (double newline = paragraph, single
# newline = line, sentence-ending punctuation, then words).
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


def _clean_text(text: str) -> str:
    """Remove hyphenated line-breaks and normalise whitespace."""
    # Re-join words broken across lines (e.g. "infor-\nmation" → "information")
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Collapse multiple blank lines to one paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_and_chunk_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF and return a list of well-formed Document chunks.

    Raises:
        ValueError: if the PDF contains no extractable text.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    if not pages:
        raise ValueError("PDF appears to be empty or could not be parsed.")

    # Clean raw text on every page before chunking
    for page in pages:
        page.page_content = _clean_text(page.page_content)

    # Filter out pages with no usable content (scanned pages, cover images, etc.)
    pages = [p for p in pages if len(p.page_content.strip()) > 50]

    if not pages:
        raise ValueError(
            "No extractable text found. The PDF may be scanned/image-based."
        )

    # First pass — coarse split that respects paragraph & sentence boundaries
    coarse_splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=CHUNK_SIZE * 4,       # generous first-pass window
        chunk_overlap=CHUNK_OVERLAP * 2,
        length_function=len,
    )
    coarse_chunks = coarse_splitter.split_documents(pages)

    # Second pass — bring oversized chunks down to CHUNK_SIZE
    fine_splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = fine_splitter.split_documents(coarse_chunks)

    # Enrich metadata for citation purposes
    for i, chunk in enumerate(chunks):
        chunk.metadata.setdefault("chunk_index", i)
        chunk.metadata.setdefault("source", file_path)

    return chunks
