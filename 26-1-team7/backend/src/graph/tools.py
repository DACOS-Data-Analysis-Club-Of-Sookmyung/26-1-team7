# 검색 전용: 사용자 질문을 임베딩해서 Qdrant에서 유사 문서 검색
# LangGraph 에이전트에서 바로 tool로 물릴 수 있도록 @tool 래퍼 포함
# (그래프/에이전트 조립은 별도 파일에서 진행)

# ----------------------------------------------------------------------
# 사용자 질문
#   → embed_query() [task 프리픽스]
#   → search() [Qdrant 유사도 검색 + 선택적 필터]
#   → format_results() [LLM 프롬프트용 텍스트 변환]
#   → (tool 래퍼 통과 시) 에이전트가 바로 소비
# ----------------------------------------------------------------------

import os
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient

from ..qdrant.client import get_qdrant_client   # tools.py 위치에 맞게 상대경로 조정
from .search_schema import SearchInput, DEFAULT_SCORE_THRESHOLD
from langchain_core.tools import tool

genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
qdrant_client = get_qdrant_client()

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
VECTOR_DIM = 3072

CASE_COLLECTION = "kor_case_vec"
GUIDELINE_COLLECTION = "kor_guideline_vec"

# 검색 결과 관련도 최소 기준 (score_threshold 기본값)
# 컬렉션 성격에 따라 다르게 튜닝 - 정확도 테스트하면서 조정
DEFAULT_SCORE_THRESHOLD = 0.5


def embed_query(query: str, task: str = "question answering") -> list[float]:
    """사용자 질문 임베딩 - 검색용 프리픽스 사용
    task 옵션 예: "question answering", "fact checking", "search result"
    """
    content = f"task: {task} | query: {query}"
    result = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=content,
        config={"output_dimensionality": VECTOR_DIM}
    )
    return result.embeddings[0].values


def search(
    query: str,
    collection_name: str,
    limit: int = 5,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    filters: dict = None,
    task: str = "question answering",
):
    query_vector = embed_query(query, task=task)

    query_filter = None
    if filters:
        # filters 딕셔너리가 있으면 Qdrant의 Filter, FieldCondition, MatchValue로 카테고리만 거를 수 있음
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        conditions = []
        for k, v in filters.items():
            if isinstance(v, list):
                if not v:
                    continue
                conditions.append(FieldCondition(key=k, match=MatchAny(any=v)))
            else:
                conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))

        if conditions:
            query_filter = Filter(must=conditions)

    # 유사도 검색 수행
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        score_threshold=score_threshold,
        query_filter=query_filter,
    )
    #  Qdrant의 ScoredPoint 객체 리스트 (각각 id, score, payload 포함)
    return results.points


def format_results(points) -> str:
    """검색 결과를 챗봇 컨텍스트로 넣기 좋게 텍스트로 포맷"""
    if not points:
        return "관련 문서를 찾지 못했습니다."

    chunks = []
    for i, p in enumerate(points, 1):
        title = p.payload.get("title", "제목없음")
        text = p.payload.get("text", "")
        chunks.append(f"[{i}] {title} (score={p.score:.3f})\n{text}")
    return "\n\n".join(chunks)


# ----------------------------------------------------------------------
# LangGraph / LangChain 호환 tool
# 에이전트 그래프에서는 이 tool 객체를 그대로 tools 리스트에 넣어서 쓰면 됨
#   from search import case_search_tool, guideline_search_tool
#   llm.bind_tools([case_search_tool, guideline_search_tool])
#   또는 ToolNode([case_search_tool, guideline_search_tool])
# ----------------------------------------------------------------------


@tool("search_case_precedents", args_schema=SearchInput)
def case_search_tool(
    query: str,
    limit: int = 5,
    filters: dict = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> str:
    """전세사기 관련 판례, 판결문, 계약조항, 결정문을 검색합니다.
    사용자가 자신의 상황과 비슷한 판례나 법적 근거, 계약 조항 해석이
    필요할 때 사용하세요. (예: "보증금 못 받았는데 판례 있나요",
    "이 계약 조항이 불리한 건가요")
    """
    # 판례/조항은 사실관계 매칭 성격이 강해 fact checking 프리픽스 사용
    points = search(
        query,
        CASE_COLLECTION,
        limit=limit,
        score_threshold=score_threshold,
        filters=filters,
        task="fact checking",
    )
    return format_results(points)


@tool("search_prevention_guides", args_schema=SearchInput)
def guideline_search_tool(
    query: str,
    limit: int = 5,
    filters: dict = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> str:
    """전세사기 예방법, 피해 발생 시 대처 방법, 정부/LH 지원제도,
    신고 및 지원센터 안내 등 가이드 정보를 검색합니다.
    사용자가 예방/대응 방법이나 지원 제도를 물어볼 때 사용하세요.
    (예: "전세사기 예방하려면 뭘 확인해야 하나요",
    "피해자 지원 받으려면 어디로 연락해야 하나요")
    """
    # 가이드는 일반 질의응답 성격이라 question answering 프리픽스 유지
    points = search(
        query,
        GUIDELINE_COLLECTION,
        limit=limit,
        score_threshold=score_threshold,
        filters=filters,
        task="question answering",
    )
    return format_results(points)
 


if __name__ == "__main__":
    user_question = "전세보증금 못 받았는데 어떻게 해야 하나요"
    points = search(user_question, CASE_COLLECTION, limit=3, task="fact checking")
 
    print(format_results(points))
 
    # tool 인터페이스 테스트 (LangGraph에서 호출되는 방식과 동일)
    print("\n--- tool 호출 테스트 ---")
    print(case_search_tool.invoke({"query": user_question, "limit": 3}))
 
    # filters 사용 예시
    print("\n--- 카테고리 필터링 테스트 ---")
    print(case_search_tool.invoke({
        "query": user_question,
        "limit": 3,
        "filters": {"category": "보증금반환"},
    }))