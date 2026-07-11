from langgraph.graph import END, StateGraph

from .nodes import (
    agent_node,
    generate_followup_questions_node,
    normalize_query_node,
    route_after_agent,
    route_question_node,
    tools_node_wrapper,
)
from .state import JeonseAgentState


def build_graph_builder() -> StateGraph:
    """
    서버 시작할 때(lifespan) 딱 한 번만 그래프를 만들기 위해
    그래프 생성 로직을 함수 안에 넣고 호출하는 시점을 직접 정할 수 있도록 한다.
    compile되지 않은 StateGraph builder를 리턴.
    checkpointer(SqliteSaver)는 with 블록이 필요하기 때문에
    실제 compile()은 이 함수를 호출하는 쪽(main.py의 lifespan)에서 한다.
    """
    graph_builder = StateGraph(JeonseAgentState)

    graph_builder.add_node("normalize_query", normalize_query_node)
    graph_builder.add_node("route_question", route_question_node)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", tools_node_wrapper)
    graph_builder.add_node("generate_followup_questions", generate_followup_questions_node)

    graph_builder.set_entry_point("normalize_query")
    graph_builder.add_edge("normalize_query", "route_question")
    graph_builder.add_edge("route_question", "agent")

    graph_builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "generate_followup_questions": "generate_followup_questions",
        },
    )
    graph_builder.add_edge("tools", "agent")

    graph_builder.add_edge("generate_followup_questions", END)

    return graph_builder