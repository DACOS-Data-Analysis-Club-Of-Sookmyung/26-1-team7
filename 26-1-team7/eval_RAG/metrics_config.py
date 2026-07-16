"""
judge LLM / embeddings / metrics 설정 (Ragas v0.4 API 기준)

ragas==0.4.3부터 evaluate() 대신 개별 metric.ascore()를 직접 호출하는 방식을 권장.
LangchainLLMWrapper 대신 llm_factory()를 사용 (deprecated 경고 방지).

* app/state.py(JeonseAgentState)와는 직접적인 연관이 없는 모듈이라 로직 변경 없음.
"""

import os
import sys
import types


from dotenv import load_dotenv
 
# eval/ 기준으로 두 단계 위(D:\...\26-1-team7\.env)에 .env가 있음
load_dotenv(os.path.join("..", "..", ".env"))

# ---------------------------------------------------------------------------
# ragas==0.4.3의 알려진 버그 우회 shim
#
# ragas/llms/base.py가 여전히
#   from langchain_community.chat_models.vertexai import ChatVertexAI
# 라는 옛날 경로를 import하는데, 최신 langchain-community(0.4+)에서는
# ChatVertexAI가 langchain-google-vertexai 패키지로 옮겨가면서 이 경로가
# 사라졌음 (https://github.com/vibrantlabsai/ragas/issues/2745).
#
# langchain-community를 다운그레이드하면 langgraph/langchain-google-genai/
# langchain-qdrant 등 나머지 최신 스택과 버전 충돌이 나서 그 방법은 못 쓰고,
# 대신 ragas가 그 모듈을 import하기 "전에" sys.modules에 가짜 모듈을
# 미리 등록해서 ImportError 자체가 안 나게 만든다.
#
# 실제로 VertexAI를 호출하는 코드가 아니라서(judge LLM은 Gemini API만 사용),
# 여기 채워넣는 ChatVertexAI는 인스턴스화되지 않는 껍데기(shim)일 뿐이고,
# GCP 인증/과금과는 전혀 무관하다.
# ---------------------------------------------------------------------------
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _shim = types.ModuleType("langchain_community.chat_models.vertexai")
    try:
        # 혹시 설치돼 있으면 진짜 클래스를 연결 (없어도 무방)
        from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
    except ImportError:
        class _ChatVertexAI:  # pragma: no cover - 실제 호출되지 않는 더미
            def __init__(self, *args, **kwargs):
                raise NotImplementedError(
                    "이 프로젝트는 VertexAI를 사용하지 않습니다. "
                    "ragas import 호환을 위한 shim 클래스입니다."
                )
    _shim.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _shim

from google import genai
import instructor  
from ragas.llms.base import InstructorLLM 
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRecall,
)

JUDGE_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되어 있지 않습니다.")
    return genai.Client(api_key=api_key)


def get_judge_llm(client: genai.Client | None = None):
    client = client or get_client()
    # google provider는 llm_factory가 async를 지원 안 해서 우회
    async_instructor_client = instructor.from_genai(client, use_async=True)
    return InstructorLLM(
        client=async_instructor_client,
        model=JUDGE_MODEL,
        provider="google",
    )


def get_judge_embeddings(client: genai.Client | None = None):
    client = client or get_client()
    # embeddings는 원래 sync client를 그대로 써야 함 (aembed_text가 내부에서
    # ThreadPoolExecutor로 sync 호출을 감싸서 async처럼 흉내내는 구조)
    return GoogleEmbeddings(client=client, model=EMBEDDING_MODEL)


def build_metrics() -> dict:
    """
    평가에 사용할 metric 인스턴스들을 딕셔너리로 반환.

    - faithfulness: 답변이 retrieved context에서 실제로 뒷받침되는가 (환각 체크)
    - response_relevancy: 답변이 질문에 실제로 답하고 있는가
    - context_precision: 검색된 청크 중 관련 있는 게 상위 랭킹에 있는가 (reference 없이도 계산 가능)
    - context_recall: 정답에 필요한 정보가 다 검색됐는가 (ground_truth 필요)
    """
    client = get_client()
    llm = get_judge_llm(client)
    embeddings = get_judge_embeddings(client)

    return {
        "faithfulness": Faithfulness(llm=llm),
        "response_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithoutReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }