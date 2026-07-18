from qdrant_client import QdrantClient
from ..config import settings


def get_qdrant_client(timeout: int = 30) -> QdrantClient:
    if not settings.QDRANT_URL:
        raise ValueError("QDRANT_URL이 .env에 없습니다.")

    if not settings.QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY가 .env에 없습니다.")

    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=timeout,
    )