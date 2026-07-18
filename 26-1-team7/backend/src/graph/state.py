from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RetrievedDoc(TypedDict):
    content: str
    source: Literal["case", "guideline"]
    metadata: dict


class JeonseAgentState(TypedDict, total=False):
    # --- 입력 (main.py가 invoke 시 넘겨줌) ---
    question: str
    session_id: str

    # --- 전처리 파이프라인 ---
    normalized_query: str
    route: Literal["case", "guideline", "both", "general"]
    route_label: str
    route_reason: str
    search_mode: str

    # --- 규칙 기반 라우터가 계산한 세부 분류 + Qdrant 필터용 knowledge_type 목록 ---
    fine_route: str
    knowledge_types: list[str]

    # --- agent <-> tools 루프 ---
    messages: Annotated[list[BaseMessage], add_messages]
    retrieval_count: int
    sources: list[RetrievedDoc]

    # --- 후처리 ---
    followup_questions: list[str]
    error: str