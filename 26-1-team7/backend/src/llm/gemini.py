from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import settings


def get_gemini_llm() -> ChatGoogleGenerativeAI:
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY가 .env에 없습니다.")

    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.2,
    )