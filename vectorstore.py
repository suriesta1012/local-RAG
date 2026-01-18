# vectorstore.py
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch
from embeddings import get_embeddings
from config import OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_INDEX

def get_vectorstore():
    embeddings = get_embeddings()
    client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=None,  # Add auth if needed
        use_ssl=False,
        verify_certs=False,
    )
    return OpenSearchVectorSearch(
        opensearch_url=f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}",
        index_name=OPENSEARCH_INDEX,
        embedding_function=embeddings,
        opensearch_client=client,
    )