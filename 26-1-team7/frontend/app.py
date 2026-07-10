import requests
import streamlit as st
from datetime import datetime

API_URL = "http://localhost:8000/chat"

# 자주 나오는 상황을 빠른 시작 버튼으로 제시
QUICK_START_OPTIONS = [
    "전세 계약이 끝났는데 집주인이 보증금을 안 돌려줘요",
    "전입신고/확정일자를 늦게 했는데 괜찮은 건가요",
    "집이 경매로 넘어간다는데 저는 어떻게 되나요",
]

st.set_page_config(
    page_title="USIM - 전세사기 피해자 지원 챗봇",
    page_icon="🏠",
    layout="wide",
)

# ── 세션 상태 초기화 ────────────────────────────────────────
if "sessions" not in st.session_state:
    st.session_state.sessions = {
        "chat_1": {
            "title": "새로운 대화",
            "messages": [],
        },
    }

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "chat_1"

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 1


# ── API 호출 & 콜백 ────────────────────────────────────────
def ask_api(question: str, thread_id: str) -> dict:
    response = requests.post(
        API_URL,
        json={"question": question, "thread_id": thread_id},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def set_pending_question(question: str):
    st.session_state.pending_question = question


# ── 렌더링 헬퍼 (참고한 문서 보기 / 이어서 물어볼 수 있어요) ────────────
def render_sources(sources: list[dict]):
    if not sources:
        return

    with st.expander("📚 참고한 문서 보기"):
        for i, source in enumerate(sources, start=1):
            title = source.get("title", "제목 없음")
            knowledge_type = source.get("knowledge_type", "")
            doc_source = source.get("doc_source", "")
            category = source.get("category", "")
            stage = source.get("stage", "")
            case_id = source.get("case_id", "")

            st.markdown(f"**{i}. {title}**")
            st.caption(f"{knowledge_type} · {doc_source}")

            if category:
                st.write(f"- 분류: {category}")
            if stage:
                st.write(f"- 단계: {stage}")
            if case_id:
                st.write(f"- 사건번호/식별자: {case_id}")

            st.divider()


def render_followup_questions(followup_questions: list[str], chat_id: str, message_index: int):
    if not followup_questions:
        return

    st.markdown("#### 💬 이어서 물어볼 수 있어요")

    for i, followup in enumerate(followup_questions):
        st.button(
            followup,
            key=f"followup_{chat_id}_{message_index}_{i}",
            on_click=set_pending_question,
            args=(followup,),
        )


def render_assistant_message(message: dict, chat_id: str, message_index: int):
    route_label = message.get("route_label", "")
    route = message.get("route", "")
    search_mode = message.get("search_mode", "")

    if route or search_mode:
        st.caption(f"분류: {route_label or route} · 검색 방식: {search_mode}")

    st.markdown(message["content"])

    render_sources(message.get("sources", []))
    render_followup_questions(
        message.get("followup_questions", []),
        chat_id,
        message_index,
    )


# ── 사이드바: 대화 목록 ────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💬 대화 목록")

    if st.button("➕ 새 대화 시작", use_container_width=True):
        st.session_state.chat_counter += 1
        new_id = f"chat_{st.session_state.chat_counter}"
        st.session_state.sessions[new_id] = {
            "title": "새로운 대화",
            "messages": [],
        }
        st.session_state.current_chat = new_id

    st.markdown("---")

    for chat_id, chat_data in st.session_state.sessions.items():
        if not chat_data["messages"]:
            # 아직 메시지가 없는 빈 대화는 목록에 표시하지 않음
            continue
        is_active = chat_id == st.session_state.current_chat
        if is_active:
            st.button(
                f"🔵 {chat_data['title']}",
                key=f"select_{chat_id}",
                use_container_width=True,
                type="primary",
            )
        else:
            if st.button(chat_data["title"], key=f"select_{chat_id}", use_container_width=True):
                st.session_state.current_chat = chat_id

    st.markdown("---")
    st.caption("전세사기 피해자 지원 RAG 챗봇")
    st.caption(f"{datetime.now().strftime('%Y.%m.%d')}")

# ── 상단 헤더: 팀 이름 ────────────────────────────────────────
header_left, header_right = st.columns([4, 1])
with header_left:
    st.markdown("## 🏠 전세사기 피해자 법률 상담 챗봇")
with header_right:
    st.markdown(
        "<div style='text-align:right; padding-top:14px; font-size:22px; font-weight:700; color:#4A6FA5;'>USIM</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

current_chat_id = st.session_state.current_chat
current = st.session_state.sessions[current_chat_id]

# ── 1) 질문 먼저 캡처 (렌더링보다 먼저! chat_input은 매 실행마다 반드시 호출) ──
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
else:
    question = st.chat_input("전세사기 관련 궁금한 점을 입력해주세요...")

# ── 2) 화면 렌더링: 질문도 없고 메시지도 없을 때만 인삿말 ────────
if question is None and len(current["messages"]) == 0:
    st.markdown(
        """
        ### 안녕하세요, USIM 챗봇입니다 👋

        전세사기 피해를 겪고 계신 분들을 위해 **유사 판례를 제시**하고
        **상황별 해결 방안**을 안내해드리는 법률 상담 챗봇입니다.

        - 📄 전세사기 관련 법률 데이터 및 실제 판례 기반 답변
        - ⚖️ 임차권등기명령, 보증금 반환 소송 등 절차 안내
        - 🔍 상황을 설명해주시면 유사 사례를 찾아드립니다

        아래 입력창에 궁금한 내용을 입력해 주세요.
        """
    )

    st.write("자주 있는 상황이면 아래에서 골라서 바로 시작할 수 있어요.")
    cols = st.columns(len(QUICK_START_OPTIONS))
    for i, (col, option) in enumerate(zip(cols, QUICK_START_OPTIONS)):
        with col:
            st.button(
                option,
                key=f"quickstart_{current_chat_id}_{i}",
                use_container_width=True,
                on_click=set_pending_question,
                args=(option,),
            )
else:
    for idx, message in enumerate(current["messages"]):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message, current_chat_id, idx)
            else:
                st.markdown(message["content"])

# ── 3) 질문 처리: API 호출 → 답변/출처/후속질문 저장 ──────────────
if question:
    current["messages"].append({"role": "user", "content": question})

    if current["title"] == "새로운 대화":
        current["title"] = question[:20]

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("관련 문서를 찾고 답변을 생성하는 중입니다..."):
                result = ask_api(question, current_chat_id)

            answer = result.get("answer", "답변을 가져오지 못했습니다.")

            assistant_message = {
                "role": "assistant",
                "content": answer,
                "route": result.get("route", ""),
                "route_label": result.get("route_label", ""),
                "search_mode": result.get("search_mode", ""),
                "sources": result.get("sources", []),
                "followup_questions": result.get("followup_questions", []),
            }

            current["messages"].append(assistant_message)

            render_assistant_message(
                assistant_message,
                current_chat_id,
                len(current["messages"]) - 1,
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "FastAPI 서버에 연결할 수 없습니다. "
                "먼저 터미널에서 `uvicorn api.main:app --reload`를 실행했는지 확인하세요."
            )

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")