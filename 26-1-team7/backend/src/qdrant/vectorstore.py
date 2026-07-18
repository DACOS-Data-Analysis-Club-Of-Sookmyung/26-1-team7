from langchain_qdrant import QdrantVectorStore

from ..config import settings
from src.qdrant.client import get_qdrant_client
from src.qdrant.embeddings import get_embeddings


def get_vectorstore() -> QdrantVectorStore:
    client = get_qdrant_client(timeout=20)
    embeddings = get_embeddings()

    return QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding=embeddings,
    )