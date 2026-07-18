from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ..config import settings


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY가 .env에 없습니다.")

    return GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
    )