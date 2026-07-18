from dataclasses import dataclass

from qdrant_client.models import FieldCondition, Filter, MatchValue


@dataclass
class RouteResult:
    route: str
    label: str
    reason: str
    knowledge_types: list[str]


ROUTE_INFO = {
    "prevention": {
        "label": "계약 전 예방/확인",
        "reason": "계약 전 확인, 등기부등본, 근저당, 보증보험 등 예방 질문",
        # 현재 업로드 데이터에서 일부 예방 문서가 support로 들어간 경우가 있어서 둘 다 열어둠
        "knowledge_types": ["prevention", "support"],
    },
    "support": {
        "label": "피해 발생 후 지원/신고",
        "reason": "피해 발생, 신고, 상담센터, 보증금 미반환, 경·공매 지원 질문",
        "knowledge_types": ["support"],
    },
    "case_law": {
        "label": "판례/형사/법률 판단",
        "reason": "사기죄, 처벌, 판례, 고소, 형사 판단 질문",
        "knowledge_types": ["case_law", "legal"],
    },
    "contract_clause": {
        "label": "계약서 조항/특약 검토",
        "reason": "계약서 문구, 특약, 불리한 조항 관련 질문",
        "knowledge_types": ["contract_clause", "prevention"],
    },
    "glossary": {
        "label": "용어 설명",
        "reason": "전세 관련 용어의 뜻이나 개념 설명 질문",
        # 용어 데이터는 아직 Qdrant에 따로 안 올렸으므로 필터 없이 검색
        "knowledge_types": [],
    },
    "general": {
        "label": "일반 질문",
        "reason": "특정 route로 분류하기 어려운 일반 질문",
        "knowledge_types": [],
    },
}


def normalize_text(text: str) -> str:
    return text.lower().replace(" ", "")


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(normalize_text(keyword) in text for keyword in keywords)


def make_route_result(route: str) -> RouteResult:
    info = ROUTE_INFO[route]

    return RouteResult(
        route=route,
        label=info["label"],
        reason=info["reason"],
        knowledge_types=info["knowledge_types"],
    )

def route_question(question: str) -> RouteResult:
    text = normalize_text(question)

    # 0순위: 용어 설명
    # "확정일자가 뭐야?", "임차권등기명령 뜻이 뭐야?" 같은 질문은
    # support/prevention보다 먼저 glossary로 보내야 함
    if contains_any(
        text,
        [
            "뜻",
            "뭐야",
            "뭔가요",
            "정의",
            "의미",
            "설명해",
            "개념",
            "용어",
        ],
    ):
        return make_route_result("glossary")

    # 1순위: 법률 판단/판례
    if contains_any(
        text,
        [
            "사기죄",
            "죄가되",
            "죄가돼",
            "형사",
            "고소",
            "고발",
            "처벌",
            "판례",
            "판결",
            "법원",
            "기망",
            "편취",
            "수사",
        ],
    ):
        return make_route_result("case_law")

    # 2순위: 계약서 조항/특약
    if contains_any(
        text,
        [
            "특약",
            "계약서문구",
            "계약조항",
            "조항",
            "문구봐",
            "이문구",
            "불리한조항",
            "독소조항",
            "약관",
        ],
    ):
        return make_route_result("contract_clause")

    # 3순위: 피해 발생 후 지원
    if contains_any(
        text,
        [
            "당했",
            "피해",
            "신고",
            "지원",
            "센터",
            "상담",
            "연락",
            "보증금안",
            "안돌려",
            "못돌려",
            "못받",
            "반환안",
            "경매",
            "공매",
            "임차권등기",
            "긴급주거",
            "대출",
            "lh",
            "hug",
            "전세피해",
        ],
    ):
        return make_route_result("support")

    # 4순위: 계약 전 예방/확인
    if contains_any(
        text,
        [
            "계약전",
            "확인",
            "등기부",
            "근저당",
            "선순위",
            "체납",
            "전입신고",
            "확정일자",
            "보증보험",
            "깡통",
            "신탁",
            "중개사",
            "임대인확인",
            "소유자",
            "예방",
            "안전",
        ],
    ):
        return make_route_result("prevention")

    return make_route_result("general")


def build_qdrant_filter(route_result: RouteResult):
    knowledge_types = route_result.knowledge_types

    if not knowledge_types:
        return None

    conditions = [
        FieldCondition(
            key="metadata.knowledge_type",
            match=MatchValue(value=knowledge_type),
        )
        for knowledge_type in knowledge_types
    ]

    if len(conditions) == 1:
        return Filter(must=conditions)

    return Filter(should=conditions)