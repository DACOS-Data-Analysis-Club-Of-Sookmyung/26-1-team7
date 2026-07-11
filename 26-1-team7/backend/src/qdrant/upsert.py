# 적재 전용: 판례/가이드라인 문서를 임베딩해서 Qdrant에 upsert
import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai
from qdrant_client.models import PointStruct
from .client import get_qdrant_client
from ..config import settings

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
VECTOR_DIM = 3072
EMBED_BATCH_SIZE = 100
CASE_COLLECTION = "kor_case_vec"
GUIDELINE_COLLECTION = "kor_guideline_vec"

qdrant_client = get_qdrant_client()
genai_client = genai.Client(api_key=settings.GOOGLE_API_KEY)


def embed_documents_batch(texts: list[str], titles: list[str] = None) -> list[list[float]]:
    """
    문서(판례/가이드라인) 배치 임베딩 - 인덱싱용 프리픽스 사용
    texts, titles는 같은 길이의 리스트 (titles 없으면 전부 'none' 처리)
    한 번의 API 호출로 최대 EMBED_BATCH_SIZE개까지 처리
    """
    
    if titles is None:
        titles = [None] * len(texts)
 
    contents = [
        f"title: {title or 'none'} | text: {text}"
        for text, title in zip(texts, titles)
    ]
 
    result = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=contents,
        config={"output_dimensionality": VECTOR_DIM}
    )
    return [emb.values for emb in result.embeddings]


def _clean(val) -> str:
    """nan/None을 빈 문자열로 정리"""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


# ----------------------------------------------------------------------
# 1) intent_전체_필터링_최종.csv → kor_case_vec (판례/계약조항)
# ----------------------------------------------------------------------

def build_case_document(row: dict) -> dict:
    """
    CSV 한 행(row)을 title/text/meta로 변환
    title: "데이터구분 - 제목(없으면 핵심 식별자)"
    text : 본문 내용 1 + 2 + 키워드
    """
    doc_type = _clean(row.get("데이터 구분"))
    category = _clean(row.get("대분류"))
    case_id = _clean(row.get("핵심 식별자"))
    heading = _clean(row.get("제목/항목")) or case_id

    title = f"{doc_type} - {heading}" if doc_type else heading

    body_parts = [_clean(row.get("본문 내용 1")), _clean(row.get("본문 내용 2"))]
    text = "\n".join(p for p in body_parts if p)

    keyword = _clean(row.get("키워드 / 비고"))
    if keyword:
        text += f"\n[키워드] {keyword}"

    # intent_json은 검색 결과 payload에 구조화된 의도 정보로 활용
    intent_raw = _clean(row.get("intent_json"))
    try:
        intent = json.loads(intent_raw) if intent_raw else {}
    except json.JSONDecodeError:
        intent = {}

    return {
        "title": title,
        "text": text,
        "meta": {
            "doc_type": doc_type,
            "category": category,
            "case_id": case_id,
            "intent": intent,
        },
    }


def load_case_documents_from_csv(csv_path: str) -> list[dict]:
    df = pd.read_csv(csv_path)
    documents = []
    for idx, row in df.iterrows():
        doc = build_case_document(row.to_dict())
        if not doc["text"]:  # 본문 없는 행은 skip
            continue
        doc["id"] = idx  # 정수 id (필요하면 case_id 기반 UUID로 교체 가능)
        documents.append(doc)
    return documents



# ----------------------------------------------------------------------
# 2) 예방_및_지원.xlsx → kor_guideline_vec (예방/대응 가이드)
# ----------------------------------------------------------------------


def build_guideline_document(row: dict, extra_cols: list = None) -> dict:
    """
    CSV 한 행(row)을 title/text/meta로 변환
    title: "대분류 > 제목/항목" (계층 구조 유지)
    text : 본문 내용 + 셀이 밀려 넘어간 unnamed 컬럼 내용까지 병합
    """
    doc_type = _clean(row.get("데이터 구분"))
    category = _clean(row.get("대분류"))
    item_title = _clean(row.get("제목/항목"))
    source = _clean(row.get("본문 내용"))
    summary = _clean(row.get("summary"))
    keywords = _clean(row.get("키워드 / 비고"))

    # 원본 파일에서 탭 등으로 인해 본문 내용 뒤 unnamed 컬럼에 내용이
    # 밀려 들어간 행이 일부 있어, 있으면 이어붙인다 (공백 join)
    if extra_cols:
        spill = " ".join(_clean(row.get(c)) for c in extra_cols if _clean(row.get(c)))
        if spill:
            source = f"{source} {spill}".strip()

    title_parts = [p for p in (category, item_title) if p]
    title = " > ".join(title_parts)

    return {
        "title": title,
        "text": source,
        "meta": {
            "doc_type": doc_type,
            "category": category,
            "summary": summary,
            "keywords": keywords,
        },
    }


def load_guideline_documents_from_csv(csv_path: str) -> list[dict]:
    df = pd.read_csv(csv_path)
    extra_cols = [c for c in df.columns if str(c).startswith("Unnamed")]

    documents = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        doc = build_guideline_document(row_dict, extra_cols=extra_cols)
        if not doc["text"]:
            continue
        doc["id"] = idx
        documents.append(doc)
    return documents



def upsert_documents(
    documents: list[dict],
    collection_name: str,
    embed_batch_size: int = EMBED_BATCH_SIZE,
    sleep_sec: float = 0.5,
):
    """
    documents: [{"id": 1, "title": "...", "text": "...", "meta": {...}}, ...]
    - id: 정수 또는 UUID (문자열 자유형식 불가)
    - embed_batch_size 단위로 묶어서 embed_content 한 번에 호출 (배치 임베딩)
    - 배치 사이 sleep_sec만큼 대기 (rate limit 방지)
    """
    total = len(documents)
    for i in range(0, total, embed_batch_size):
        batch = documents[i:i + embed_batch_size]
 
        texts = [doc["text"] for doc in batch]
        titles = [doc.get("title") for doc in batch]
 
        vectors = embed_documents_batch(texts, titles)
 
        points = [
            PointStruct(
                id=doc["id"],
                vector=vector,
                payload={
                    "title": doc.get("title"),
                    "text": doc["text"],
                    **doc.get("meta", {})
                }
            )
            for doc, vector in zip(batch, vectors)
        ]
 
        qdrant_client.upsert(collection_name=collection_name, points=points)
        print(f"[{collection_name}] {i + len(batch)}/{total} 적재 완료")
 
        time.sleep(sleep_sec)  # 배치 간 API rate limit 방지
 
    print(f"✅ '{collection_name}' 전체 적재 완료 (총 {total}건)")
    
    

# backend 폴더에서 python -m src.qdrant.upsert로 실행
if __name__ == "__main__":
    # 1) 판례/계약조항 CSV 적재
    case_docs = load_case_documents_from_csv("data/intent_전체_필터링_최종.csv")
    print(f"판례/계약조항 CSV에서 {len(case_docs)}건 로드")
    upsert_documents(case_docs, CASE_COLLECTION)

    # 2) 예방/지원 가이드 csv 적재
    guideline_docs = load_guideline_documents_from_csv("data/guide-전세사기_안내가이드.csv")
    print(f"예방/지원 가이드 CSV에서 {len(guideline_docs)}건 로드")
    upsert_documents(guideline_docs, GUIDELINE_COLLECTION)