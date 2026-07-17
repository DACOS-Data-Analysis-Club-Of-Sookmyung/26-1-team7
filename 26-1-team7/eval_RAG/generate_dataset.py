"""
2단계: qa_with_results.csv 생성

qa_seed.csv의 질문들을 실제 LangGraph 파이프라인(build_graph_builder)에
하나씩 태워서, 실제 검색된 contexts와 생성된 answer를 채운다.

입력: eval_RAG/data/qa_seed.csv
      (question, ground_truth, category, intent, followup_questions, source, source_id)
      -- followup_questions/source/source_id는 alias 후보 시절 메타데이터라
      실행에는 안 쓰이지만, source/source_id는 결과 추적용으로 출력에 그대로 실어간다.
출력: eval_RAG/data/qa_with_results.csv
      (question, contexts, answer, ground_truth, category, intent, route,
       followup_questions, error, source, source_id)
      -- 여기서 followup_questions는 seed의 것이 아니라 파이프라인 실행 중
      에이전트가 실제로 생성한 후속 질문임 (덮어써짐).

주의 (JeonseAgentState 기준으로 확인함, app/state.py):
- State에는 answer 필드가 없다. 최종 답변은 messages(add_messages)의 마지막
  AIMessage.content 에서 꺼내야 한다.
- retrieved_documents가 아니라 sources 필드다. sources는 langchain Document가
  아니라 RetrievedDoc(TypedDict) 형태라서 doc.page_content가 아니라
  doc["content"]로 접근해야 한다.
- session_id도 State 입력 필드로 정의돼 있어서, 백엔드가 session_id로 뭔가
  분기하는 로직이 있다면 eval 때도 채워주는 게 안전하다 (thread_id와는 별개).
- route / followup_questions는 ragas 채점엔 안 쓰이지만, 나중에 "라우팅이
  잘못돼서 점수가 낮은 건지" 디버깅할 때 필요해서 같이 저장해둔다.
- seed의 source / source_id는 "이 질문이 guide/contract_clause/quick_start_manual
  중 어디서 왔는지" 추적하려고 결과에도 그대로 실어둔다 (에러/폴백 분석할 때 유용).

content 파싱 관련:
- 최신 Gemini 통합은 message.content가 순수 문자열이 아니라
  [{"type": "text", "text": "..."}] 처럼 구조화된 블록 리스트로 오는 경우가 있고,
  tool-calling 컨텍스트 유지를 위한 {"type": "thinking", "signature": "..."} 같은
  블록이 섞여 있기도 하다. 이걸 그대로 str()로 찍어서 answer에 넣으면
  ragas 채점(faithfulness, response_relevancy)에 잡음이 섞여 점수가 부정확해지므로
  type == "text"인 블록만 골라 텍스트를 추출한다.

병렬 처리:
- graph.ainvoke()를 지원하므로 asyncio + Semaphore로 동시 실행 개수를 제한해서
  병렬 처리한다. 무제한 병렬은 Gemini rate limit / Qdrant 동시 연결 /
  SqliteSaver 동시 쓰기(락 경합) 문제를 일으킬 수 있어서 상한을 둔다.
  429(rate limit) 에러가 나면 MAX_CONCURRENCY를 낮춰서 재시도할 것.
"""

import asyncio
import sys

import pandas as pd

# eval/ 에서 실행하는 걸 기준으로, backend(= src의 부모)를 sys.path에 추가.
# graph/tools.py가 "from ..qdrant.client import ..." 처럼 src 레벨까지 거슬러
# 올라가는 상대import를 쓰고 있어서, src 자체를 top-level 패키지로 만들면 안 되고
# src의 부모(backend)를 path에 넣어 "src.graph.graph" 형태로 import해야 한다.
sys.path.append("../backend")
from src.graph.graph import build_graph_builder  # noqa: E402

INPUT_PATH = "data/qa_seed.csv"
OUTPUT_PATH = "data/qa_with_results.csv"

# CSV에는 리스트를 그대로 저장할 수 없어서 구분자로 join.
# 실제 텍스트에 나올 가능성이 거의 없는 문자열로 선택. run_eval.py의 CONTEXT_SEP과 동일해야 함.
CONTEXT_SEP = "\n<<<CTX_SEP>>>\n"

# 동시에 몇 개까지 그래프를 돌릴지. 너무 높이면 rate limit / DB 락 경합 위험,
# 너무 낮으면 순차 처리랑 다를 게 없음. 5~10 사이에서 시작해서 조정 추천.
MAX_CONCURRENCY = 10


def _extract_text_from_content_block(block) -> str:
    """LangChain 메시지의 content 블록 하나에서 순수 텍스트만 뽑아낸다.

    최신 Gemini 통합은 content가 str이 아니라
    [{"type": "text", "text": "..."}] 또는
    [{"type": "thinking", "thinking": "...", "signature": "..."}] 같은
    구조화된 블록 리스트로 오는 경우가 있다.
    thinking/signature 블록은 모델 내부 추론/컨텍스트 유지용 메타데이터라
    사용자에게 보여줄 답변이 아니므로, type == "text"인 블록만 골라낸다.
    """
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        # text 타입 블록만 답변으로 인정. thinking/signature 등은 무시.
        if block.get("type") == "text" and "text" in block:
            return block["text"]
        return ""
    return ""


def extract_answer(result: dict) -> str:
    """State에 answer 필드가 없으므로 messages 마지막 항목에서 답변 텍스트를 꺼낸다.
    content가 문자열이면 그대로, 구조화된 리스트/블록이면 text 타입만 이어붙인다.
    """
    messages = result.get("messages", [])
    if not messages:
        return ""

    content = messages[-1].content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_extract_text_from_content_block(b) for b in content]
        return "".join(p for p in parts if p)
    # 예상 못한 타입이면 최후의 수단으로 문자열 변환 (여기 걸리면 로그로 확인 필요)
    return str(content)


def extract_contexts(result: dict) -> list[str]:
    """sources는 RetrievedDoc(TypedDict) 리스트라 doc["content"]로 접근한다."""
    sources = result.get("sources", [])
    return [doc["content"] for doc in sources if doc.get("content")]


async def run_one(graph, i: int, row: dict, semaphore: asyncio.Semaphore, total: int) -> dict:
    async with semaphore:
        # 평가용 세션은 매번 새 thread_id로 격리 (SqliteSaver가 이전 대화 히스토리와 안 섞이도록)
        thread_id = f"eval-{i}"
        config = {"configurable": {"thread_id": thread_id}}

        inputs = {
            "question": row["question"],
            "session_id": thread_id,  # State 입력 필드; 백엔드 분기 로직 대비해서 채워줌
        }

        # seed 단계 메타데이터 (source/source_id) -- 실행 성공/실패와 무관하게 그대로 실어감
        source = row.get("source", "")
        source_id = row.get("source_id", "")

        try:
            result = await graph.ainvoke(inputs, config=config)
        except Exception as e:
            print(f"[에러] row {i} ('{str(row['question'])[:30]}...') 실행 실패: {e}")
            return {
                "question": row["question"],
                "contexts": "",
                "answer": "",
                "ground_truth": row.get("ground_truth", ""),
                "category": row.get("category", ""),
                "intent": row.get("intent", ""),
                "route": "",
                "followup_questions": "",
                "error": str(e),
                "source": source,
                "source_id": source_id,
            }

        contexts = extract_contexts(result)
        answer = extract_answer(result)
        followups = result.get("followup_questions", [])

        row_result = {
            "question": row["question"],
            "contexts": CONTEXT_SEP.join(contexts),
            "answer": answer,
            "ground_truth": row.get("ground_truth", ""),
            "category": row.get("category", ""),
            "intent": row.get("intent", ""),
            "route": result.get("route", ""),
            "followup_questions": " | ".join(followups) if followups else "",
            "error": "",
            "source": source,
            "source_id": source_id,
        }

        print(f"진행: {i + 1}/{total} 완료")
        return row_result


async def run_pipeline_on_questions(df: pd.DataFrame) -> pd.DataFrame:
    graph = build_graph_builder().compile()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    total = len(df)

    tasks = [
        run_one(graph, i, row.to_dict(), semaphore, total)
        for i, row in df.iterrows()
    ]

    # gather는 입력 순서를 그대로 유지해서 반환하므로, 완료 순서와 무관하게
    # 원본 df 순서와 동일한 결과 리스트를 받는다.
    rows = await asyncio.gather(*tasks)
    return pd.DataFrame(rows)


async def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"{len(df)}개 질문에 대해 파이프라인 실행 시작... (동시 실행 {MAX_CONCURRENCY}개)")

    result_df = await run_pipeline_on_questions(df)

    n_errors = (result_df["error"] != "").sum()
    if n_errors:
        print(f"\n⚠️  {n_errors}개 행에서 에러 발생 -- error 컬럼 확인 필요")

    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n완료: {OUTPUT_PATH} ({len(result_df)}행)")


if __name__ == "__main__":
    asyncio.run(main())