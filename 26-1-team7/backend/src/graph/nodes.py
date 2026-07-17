from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode

from .tools import case_search_tool, guideline_search_tool
from .state import JeonseAgentState, RetrievedDoc
from .router import route_question          # 규칙 기반 라우터
from .prompts import SYSTEM_PROMPT_TEMPLATE, FOLLOWUP_PROMPT_TEMPLATE  # 프롬프트는 여기서 import


load_dotenv()

MAX_RETRIEVALS = 5
# tools.py에서 @tool("search_case_precedents", ...) / @tool("search_prevention_guides", ...)
# 로 이름을 명시했기 때문에, 실제 tool_call.name은 이 값들로 들어온다.
CASE_TOOL_NAME = "search_case_precedents"
GUIDELINE_TOOL_NAME = "search_prevention_guides"

tools = [case_search_tool, guideline_search_tool]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools)

FINE_TO_COARSE_ROUTE = {
    "case_law": "case",
    "contract_clause": "case",
    "prevention": "guideline",
    "support": "guideline",
    "glossary": "general",
    "general": "general",
}


def _extract_text(content) -> str:
    """AIMessage.content가 str이면 그대로, list(블록 형태)면
    text 타입 블록만 골라 이어붙여서 순수 텍스트로 변환한다.
    Gemini 2.5류 모델은 thinking/signature가 섞인 리스트를 반환할 수 있다.
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
    체크포인터로 이어지는 세션에서 '이번 턴 한정' 필드도 여기서 리셋한다.
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

FINE_TO_COARSE_ROUTE = {
    "case_law": "case",
    "contract_clause": "case",
    "prevention": "guideline",
    "support": "guideline",
    "glossary": "general",
    "general": "general",
}


def route_question_node(state: JeonseAgentState) -> dict:
    """규칙 기반 라우터로 분류. LLM 호출 없이 즉시 계산되며,
    knowledge_types는 이후 tools_node_wrapper에서 검색 필터로 강제 적용된다.
    """
    result = route_question(state["normalized_query"])
    coarse_route = FINE_TO_COARSE_ROUTE.get(result.route, "both")

    return {
        "route": coarse_route,
        "route_label": result.label,
        "route_reason": result.reason,
        "fine_route": result.route,
        "knowledge_types": result.knowledge_types,
        "search_mode": "no_search" if coarse_route == "general" else "agentic_tool_calling",
    }


def agent_node(state: JeonseAgentState) -> dict:
    """LLM이 대화 맥락과 system prompt를 보고
    tool을 호출할지, 최종 답변을 낼지 스스로 판단하는 노드.
    """
    messages = state["messages"]

    if not any(isinstance(m, SystemMessage) for m in messages):
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(route=state.get("route", "both"))
        messages = [SystemMessage(content=system_prompt)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _extract_sources_from_tool_messages(
    tool_messages: list[ToolMessage],
    request_message: AIMessage,
) -> list[RetrievedDoc]:
    """ToolMessage들에서 실제 호출된 tool 이름(search_case_precedents /
    search_prevention_guides)을 보고 source(case/guideline)를 붙인다.
    """
    tool_name_by_call_id = {
        call["id"]: call["name"] for call in (request_message.tool_calls or [])
    }

    docs: list[RetrievedDoc] = []
    for tm in tool_messages:
        tool_name = tool_name_by_call_id.get(tm.tool_call_id, "")
        source = "case" if tool_name == CASE_TOOL_NAME else "guideline"
        docs.append(
            RetrievedDoc(
                content=str(tm.content),
                source=source,
                metadata={"tool_call_id": tm.tool_call_id, "tool_name": tool_name},
            )
        )
    return docs


def _inject_knowledge_type_filter(tool_call_args: dict, knowledge_types: list[str]) -> dict:
    """LLM이 만든 tool_call 인자에 knowledge_types 필터를 강제로 덮어쓴다.
    LLM이 filters를 안 주거나 다르게 줘도 이 필터가 우선 적용된다.
    """
    if not knowledge_types:
        return tool_call_args  # 필터 없이 검색 (예: general, glossary)

    new_args = dict(tool_call_args)
    existing_filters = dict(new_args.get("filters") or {})
    existing_filters["knowledge_type"] = knowledge_types  # 실제 payload 필드명으로 조정
    new_args["filters"] = existing_filters
    return new_args


# 실험 플래그: 문서 수가 적은 현재 단계에서는 knowledge_type 필터가
# 정답 문서까지 걸러버리는 경우가 많아 일단 꺼둔다.
# (예: "처벌받으면 보증금 자동으로 받나요" -> case_law로 라우팅돼
#  prevention/support 쪽 정답 문서가 후보군에서 아예 제외되는 문제)
# route/knowledge_types는 계속 계산해서 로깅·후속질문 생성 등에는 활용한다.
ENABLE_KNOWLEDGE_TYPE_FILTER = False


def tools_node_wrapper(state: JeonseAgentState) -> dict:
    """agent가 요청한 tool_calls를 실행한다.
    ENABLE_KNOWLEDGE_TYPE_FILTER=True일 때만 route 분류 결과로 만든
    knowledge_types 필터를 강제 주입하고, False면 필터 없이 임베딩
    유사도만으로 넓게 검색한다.
    """
    ai_message = state["messages"][-1]
    knowledge_types = state.get("knowledge_types", [])

    if ENABLE_KNOWLEDGE_TYPE_FILTER:
        patched_tool_calls = [
            {**tc, "args": _inject_knowledge_type_filter(tc["args"], knowledge_types)}
            for tc in (ai_message.tool_calls or [])
        ]
        patched_ai_message = ai_message.model_copy(update={"tool_calls": patched_tool_calls})
    else:
        patched_ai_message = ai_message

    tool_node = ToolNode(tools)
    result = tool_node.invoke({**state, "messages": [patched_ai_message]})

    new_tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    new_sources = _extract_sources_from_tool_messages(new_tool_messages, ai_message)

    return {
        **result,
        "retrieval_count": state.get("retrieval_count", 0) + 1,
        "sources": state.get("sources", []) + new_sources,
    }


def route_after_agent(state: JeonseAgentState) -> str:
    """agent 노드 실행 직후, tools로 갈지 후속질문 생성으로 갈지 결정.

    두 조건을 모두 만족해야 "tools"로 라우팅:
    1. 마지막 메시지에 tool_calls가 있는지
    2. retrieval_count가 MAX_RETRIEVALS 미만인지
    """
    last_message = state["messages"][-1]
    has_tool_calls = bool(getattr(last_message, "tool_calls", None))

    if has_tool_calls and state.get("retrieval_count", 0) < MAX_RETRIEVALS:
        return "tools"
    return "generate_followup_questions"


def generate_followup_questions_node(state: JeonseAgentState) -> dict:
    """agent 루프가 끝난 마지막 AIMessage(답변)와 sources를 바탕으로
    후속 질문 후보를 생성한다.
    """
    last_message = state["messages"][-1]
    answer = _extract_text(last_message.content)
    sources = state.get("sources", [])

    if sources:
        sources_summary = "\n".join(
            f"- [{s['source']}] {s['content'][:200]}" for s in sources
        )
    else:
        sources_summary = "(검색된 문서 없음)"

    prompt = FOLLOWUP_PROMPT_TEMPLATE.format(answer=answer, sources_summary=sources_summary)
    response = llm.invoke([HumanMessage(content=prompt)])

    followups = [
        line.strip("-• ").strip()
        for line in response.content.strip().splitlines()
        if line.strip()
    ]

    return {"followup_questions": followups}