from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langchain.chains import RetrievalQA
import tempfile
import os
import shutil

from config import OPENSEARCH_INDEX
from vectorstore import get_vectorstore
from llm import get_llm
from document_processor import load_and_chunk_pdf

app = FastAPI(title="Local RAG System", description="A local RAG system using Ollama and OpenSearch")

# Initialize components
vectorstore = get_vectorstore()
llm = get_llm()

# Create retriever
retriever = vectorstore.as_retriever()

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
    
    try:
        # Load and chunk PDF
        chunks = load_and_chunk_pdf(temp_path)
        
        # Add to vectorstore
        vectorstore.add_documents(chunks)
        
        return JSONResponse(content={"message": f"Document {file.filename} processed and added to knowledge base"})
    
    finally:
        # Clean up temp file
        os.unlink(temp_path)

@app.post("/query")
async def query_knowledge_base(query: str):
    try:
        result = qa_chain.run(query)
        return JSONResponse(content={"answer": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Local RAG System API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)