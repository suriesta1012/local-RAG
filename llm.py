# llm.py
from langchain_community.llms.ollama import Ollama
from config import LLM_MODEL

def get_llm():
    return Ollama(model=LLM_MODEL)