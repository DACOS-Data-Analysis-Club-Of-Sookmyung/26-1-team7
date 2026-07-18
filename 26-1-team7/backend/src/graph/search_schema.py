# schemas/search_schema.py
# Qdrant 검색 tool들이 공유하는 입력 스키마

from pydantic import BaseModel, Field

# 검색 결과 관련도 최소 기준 (score_threshold 기본값)
# 컬렉션 성격에 따라 다르게 튜닝 - 정확도 테스트하면서 조정
DEFAULT_SCORE_THRESHOLD = 0.5


class SearchInput(BaseModel):
    query: str = Field(description="검색할 사용자 질문 또는 검색어")
    limit: int = Field(default=5, description="반환할 문서 개수 (기본 5)")
    filters: dict = Field(
        default=None,
        description=(
            "payload 필드 기준 필터링 조건 (선택). "
            "예: {\"category\": \"보증금반환\"}. "
            "특정 카테고리로 범위를 좁히고 싶을 때만 사용."
        ),
    )
    score_threshold: float = Field(
        default=DEFAULT_SCORE_THRESHOLD,
        description="이 값보다 관련도 점수가 낮은 문서는 결과에서 제외 (기본값 사용 권장)",
    )