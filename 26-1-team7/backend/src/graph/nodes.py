import uuid

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode

from .tools import case_search_tool, guideline_search_tool
from .state import JeonseAgentState, RetrievedDoc
from .router import route_question
from .prompts import SYSTEM_PROMPT_TEMPLATE, FOLLOWUP_PROMPT_TEMPLATE


load_dotenv()

MAX_RETRIEVALS = 5

# tools.py에서 @tool("search_case_precedents", ...)
# @tool("search_prevention_guides", ...)로 이름을 명시했기 때문에
# 실제 tool_call.name은 아래 값으로 들어온다.
CASE_TOOL_NAME = "search_case_precedents"
GUIDELINE_TOOL_NAME = "search_prevention_guides"

# ToolNode가 실행할 수 있는 전체 tool 목록
tools = [case_search_tool, guideline_search_tool]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# route별로 LLM에게 노출할 tool을 제한한다.
case_llm = llm.bind_tools([case_search_tool])
guideline_llm = llm.bind_tools([guideline_search_tool])


FINE_TO_COARSE_ROUTE = {
    "case_law": "case",
    "contract_clause": "case",
    "prevention": "guideline",
    "support": "guideline",
    "glossary": "guideline",
    "general": "both",
}


def _extract_text(content) -> str:
    """AIMessage.content가 str이면 그대로, list 형태이면
    text 타입 블록만 골라 이어붙여 순수 텍스트로 변환한다.

    Gemini 2.5 계열 모델은 thinking/signature가 섞인
    리스트 형태의 content를 반환할 수 있다.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)

        return "".join(parts)

    return str(content)


def normalize_query_node(state: JeonseAgentState) -> dict:
    """main.py가 invoke 시 넘기는 state["question"]을 정리해서
    normalized_query로 만들고, agent 루프의 최초 HumanMessage를 채운다.

    체크포인터로 이어지는 세션에서 이번 턴에만 사용하는
    sources, followup_questions 등의 상태를 초기화한다.
    """
    query = state["question"].strip()
    normalized = " ".join(query.split())

    return {
        "normalized_query": normalized,
        "messages": [HumanMessage(content=normalized)],
        "retrieval_count": 0,
        "sources": [],
        "followup_questions": [],
        "error": "",
    }


def route_question_node(state: JeonseAgentState) -> dict:
    """규칙 기반 라우터로 질문을 분류하고 검색 대상을 정한다.

    - case_law, contract_clause: 판례 컬렉션
    - prevention, support, glossary: 가이드라인 컬렉션
    - general: 판례와 가이드라인 컬렉션 모두
    """
    result = route_question(state["normalized_query"])
    coarse_route = FINE_TO_COARSE_ROUTE.get(result.route, "both")

    return {
        "route": coarse_route,
        "route_label": result.label,
        "route_reason": result.reason,
        "fine_route": result.route,
        "knowledge_types": result.knowledge_types,
        "search_mode": "agentic_tool_calling",
    }


def _make_general_broad_search_message(query: str) -> AIMessage:
    """general 질문의 첫 검색에서 판례와 가이드라인 tool을 동시에 호출한다."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": CASE_TOOL_NAME,
                "args": {
                    "query": query,
                    "limit": 5,
                },
                "id": f"call_case_{uuid.uuid4().hex}",
                "type": "tool_call",
            },
            {
                "name": GUIDELINE_TOOL_NAME,
                "args": {
                    "query": query,
                    "limit": 5,
                },
                "id": f"call_guideline_{uuid.uuid4().hex}",
                "type": "tool_call",
            },
        ],
    )


def agent_node(state: JeonseAgentState) -> dict:
    """대화 맥락과 route에 따라 tool 호출 또는 최종 답변을 생성한다.

    - case_law, contract_clause:
      판례 검색 tool만 LLM에게 노출한다.

    - prevention, support, glossary:
      가이드라인 검색 tool만 LLM에게 노출한다.

    - general:
      첫 검색에서는 판례와 가이드라인 tool을 동시에 호출한다.
      검색 결과가 돌아온 뒤에는 두 결과를 바탕으로 최종 답변을 생성한다.
    """
    messages = state["messages"]

    if not any(isinstance(m, SystemMessage) for m in messages):
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            route=state.get("route", "both")
        )
        messages = [SystemMessage(content=system_prompt)] + messages

    route = state.get("route", "both")
    retrieval_count = state.get("retrieval_count", 0)

    if route == "case":
        response = case_llm.invoke(messages)

    elif route == "guideline":
        response = guideline_llm.invoke(messages)

    elif route == "both":
        if retrieval_count == 0:
            # general 질문의 첫 검색에서는 두 tool을 동시에 실행
            response = _make_general_broad_search_message(
                state["normalized_query"]
            )
        else:
            # 두 tool의 검색 결과를 받은 이후에는 최종 답변 생성
            response = llm.invoke(messages)

    else:
        response = llm.invoke(messages)

    return {"messages": [response]}


def _extract_sources_from_tool_messages(
    tool_messages: list[ToolMessage],
    request_message: AIMessage,
) -> list[RetrievedDoc]:
    """ToolMessage에서 실제 호출된 tool 이름을 확인해
    source를 case 또는 guideline으로 구분한다.
    """
    tool_name_by_call_id = {
        call["id"]: call["name"]
        for call in (request_message.tool_calls or [])
    }

    docs: list[RetrievedDoc] = []

    for tm in tool_messages:
        tool_name = tool_name_by_call_id.get(
            tm.tool_call_id,
            "",
        )

        source = (
            "case"
            if tool_name == CASE_TOOL_NAME
            else "guideline"
        )

        docs.append(
            RetrievedDoc(
                content=str(tm.content),
                source=source,
                metadata={
                    "tool_call_id": tm.tool_call_id,
                    "tool_name": tool_name,
                },
            )
        )

    return docs


def tools_node_wrapper(state: JeonseAgentState) -> dict:
    """agent가 요청한 tool_calls를 실행한다.

    모든 검색에서 knowledge_type, category 등 payload 필터를 제거하고
    선택된 컬렉션 내부를 임베딩 유사도만으로 검색한다.
    """
    ai_message = state["messages"][-1]

    patched_tool_calls = []

    for tool_call in ai_message.tool_calls or []:
        patched_args = dict(tool_call.get("args") or {})

        # LLM이 filters를 생성했더라도 검색 직전에 모두 제거한다.
        patched_args.pop("filters", None)

        patched_tool_calls.append(
            {
                **tool_call,
                "args": patched_args,
            }
        )

    patched_ai_message = ai_message.model_copy(
        update={"tool_calls": patched_tool_calls}
    )

    tool_node = ToolNode(tools)

    result = tool_node.invoke(
        {
            **state,
            "messages": [patched_ai_message],
        }
    )

    new_tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    new_sources = _extract_sources_from_tool_messages(
        new_tool_messages,
        patched_ai_message,
    )

    return {
        **result,
        "retrieval_count": state.get("retrieval_count", 0) + 1,
        "sources": state.get("sources", []) + new_sources,
    }


def route_after_agent(state: JeonseAgentState) -> str:
    """agent 노드 실행 직후 tools 또는 후속 질문 생성으로 이동한다.

    다음 두 조건을 모두 만족하면 tools 노드로 이동한다.

    1. 마지막 AIMessage에 tool_calls가 존재함
    2. retrieval_count가 MAX_RETRIEVALS보다 작음
    """
    last_message = state["messages"][-1]
    has_tool_calls = bool(
        getattr(last_message, "tool_calls", None)
    )

    if (
        has_tool_calls
        and state.get("retrieval_count", 0) < MAX_RETRIEVALS
    ):
        return "tools"

    return "generate_followup_questions"


def generate_followup_questions_node(
    state: JeonseAgentState,
) -> dict:
    """최종 답변과 검색 출처를 바탕으로 후속 질문 2~3개를 생성한다."""
    last_message = state["messages"][-1]
    answer = _extract_text(last_message.content)
    sources = state.get("sources", [])

    if sources:
        sources_summary = "\n".join(
            f"- [{source['source']}] {source['content'][:200]}"
            for source in sources
        )
    else:
        sources_summary = "(검색된 문서 없음)"

    prompt = FOLLOWUP_PROMPT_TEMPLATE.format(
        answer=answer,
        sources_summary=sources_summary,
    )

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    response_text = _extract_text(response.content)

    followups = [
        line.strip("-• ").strip()
        for line in response_text.splitlines()
        if line.strip()
    ]

    return {
        "followup_questions": followups,
    }