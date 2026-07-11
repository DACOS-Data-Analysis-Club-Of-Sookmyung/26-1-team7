import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from .graph.graph import build_graph_builder
from .graph.nodes import _extract_text

logger = logging.getLogger("uvicorn.error")


class ChatRequest(BaseModel):
    session_id: str
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    graph_builder = build_graph_builder()
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as memory:
        app.state.graph = graph_builder.compile(checkpointer=memory)
        yield


app = FastAPI(
    title="Jeonse Chatbot API",
    description="전세사기 예방 챗봇 FastAPI 서버",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def health_check():
    graph_ready = getattr(app.state, "graph", None) is not None
    return {
        "message": "Jeonse chatbot API is running",
        "status": "ok" if graph_ready else "not_ready",
        "graph_ready": graph_ready,
    }


# 👇 새로 추가하는 엔드포인트
@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """체크포인터에 저장된 해당 세션(thread_id)의 대화 기록을 복원해서 반환.

    주의: sources / followup_questions는 그래프 state 상 "마지막 턴 값"만
    남아있으므로, 가장 마지막 assistant 메시지에만 붙여서 반환한다.
    """
    config = {"configurable": {"thread_id": session_id}}

    try:
        state_snapshot = app.state.graph.get_state(config)
    except Exception:
        logger.exception("history 조회 실패: session_id=%s", session_id)
        return {"session_id": session_id, "messages": []}

    if not state_snapshot or not state_snapshot.values:
        # 아직 한 번도 대화한 적 없는 session_id (신규 세션)
        return {"session_id": session_id, "messages": []}

    values = state_snapshot.values
    all_messages = values.get("messages", [])
    sources = values.get("sources", [])
    followup_questions = values.get("followup_questions", [])
    route = values.get("route", "")
    route_label = values.get("route_label", "")
    search_mode = values.get("search_mode", "")

    history: list[dict] = []

    for msg in all_messages:
        if isinstance(msg, HumanMessage):
            history.append({
                "role": "user",
                "content": _extract_text(msg.content),
            })

        elif isinstance(msg, AIMessage):
            text = _extract_text(msg.content)
            has_tool_calls = bool(getattr(msg, "tool_calls", None))

            # tool 호출만 있고 실제 답변 텍스트가 없는 중간 메시지는
            # 화면에 보여줄 필요 없으므로 건너뜀
            if has_tool_calls and not text.strip():
                continue

            history.append({
                "role": "assistant",
                "content": text,
                "route": "",
                "route_label": "",
                "search_mode": "",
                "sources": [],
                "followup_questions": [],
            })

        # SystemMessage / ToolMessage는 화면에 보여줄 필요 없으므로 건너뜀

    # 가장 마지막 assistant 메시지에만 현재 state의 메타정보를 붙여준다
    for item in reversed(history):
        if item["role"] == "assistant":
            item["route"] = route
            item["route_label"] = route_label
            item["search_mode"] = search_mode
            item["sources"] = sources
            item["followup_questions"] = followup_questions
            break

    return {"session_id": session_id, "messages": history}


@app.post("/chat")
async def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        return {
            "session_id": request.session_id,
            "question": message,
            "answer": "질문을 입력해 주세요.",
            "route": "empty",
            "sources": [],
            "followup_questions": [],
        }

    config = {"configurable": {"thread_id": request.session_id}}

    try:
        result = app.state.graph.invoke(
            {
                "question": message,
                "session_id": request.session_id,
                "messages": [{"role": "user", "content": message}],
            },
            config=config,
        )

        if result.get("error"):
            return {
                "session_id": request.session_id,
                "question": message,
                "answer": result.get("answer") or "답변을 생성하지 못했습니다.",
                "route": result.get("route", "error"),
                "sources": [],
                "followup_questions": [],
            }

        last_message = result["messages"][-1]

        return {
            "session_id": request.session_id,
            "question": message,
            "answer": _extract_text(last_message.content),
            "route": result.get("route", "general"),
            "route_label": result.get("route_label", ""),
            "route_reason": result.get("route_reason", ""),
            "search_mode": result.get("search_mode", ""),
            "sources": result.get("sources", []),
            "followup_questions": result.get("followup_questions", []),
        }

    except Exception:
        logger.exception("chat graph invoke failed")
        return {
            "session_id": request.session_id,
            "question": message,
            "answer": "죄송합니다. 답변 생성 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "route": "error",
            "sources": [],
            "followup_questions": [],
        }