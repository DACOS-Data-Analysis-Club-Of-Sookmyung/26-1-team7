"""
qa_seed.csv 생성 (통합본)

- 기존에 만든 csv(가이드 91개 alias[0] + 계약조항 임대차 30개 샘플) 결과를 그대로 유지
- 여기에 프론트 "빠른 시작" 질문 3개를 무조건 포함 (원본 alias와 정확히 일치하는 문장이
  없어서 MANUAL_GROUND_TRUTH로 수동 매핑된 기존 항목의 ground_truth/category/intent를 재사용)
- 카테고리별 캡(MAX_PER_CATEGORY) 로직은 제거 -- 데이터가 적은 상황에서 캡을 걸면
  오히려 표본이 과하게 깎이므로, "빠른시작 질문 무조건 포함 + 중복 질문 제거"만 적용

출력 컬럼은 qa_with_results.csv 포맷에 맞춤:
  question, contexts, answer, ground_truth, category, intent, route,
  followup_questions, error, source, source_id
  (contexts/answer/route/error는 실제 파이프라인을 태워야 채워지므로 빈 값으로 둠)
"""

import json
import random
import pandas as pd

random.seed(42)

GUIDE_CSV = "data/guide-전세사기_안내가이드.csv"
INTENT_CSV = "data/intent_전체_필터링_최종.csv"
OUTPUT_PATH = "data/qa_seed.csv"

# 프론트 "빠른 시작" 버튼 질문 - 카테고리/캡 로직과 무관하게 반드시 포함
MUST_INCLUDE_QUESTIONS = [
    "전세 계약이 끝났는데 집주인이 보증금을 안 돌려줘요",
    "전입신고/확정일자를 늦게 했는데 괜찮은 건가요",
    "집이 경매로 넘어간다는데 저는 어떻게 되나요",
]

# 위 3개는 원본 alias와 정확히 일치하는 문장이 없어서, 같은 주제를 다루는
# 기존 항목의 ground_truth/category/intent를 그대로 매핑해서 사용.
MANUAL_GROUND_TRUTH = {
    "전세 계약이 끝났는데 집주인이 보증금을 안 돌려줘요": {
        "ground_truth": "임대인이 보증금 반환을 신규 세입자 구하기와 연계하지 않도록 하고, 미반환 시 법적 대응이 가능하도록 반환 기한을 구체적으로 정해야 한다는 안내.",
        "category": "사기예방·계약가이드",
        "intent": "보증금반환_기한설정_특약안내",
    },
    "전입신고/확정일자를 늦게 했는데 괜찮은 건가요": {
        "ground_truth": "근저당 말소·보증금 반환 미이행 시 계약 해지 및 위약금 부과 조항, 계약 후 확정일자를 반드시 받아야 한다는 안내.",
        "category": "사기예방·계약가이드",
        "intent": "계약해지_위약금_확정일자_안내",
    },
    "집이 경매로 넘어간다는데 저는 어떻게 되나요": {
        "ground_truth": "피해 임차인이 거주 중인 주택이 경매·공매에 부쳐질 경우, 해당 주택을 우선적으로 매수할 수 있는 권한을 부여한다는 우선매수권 안내.",
        "category": "특별법지원(일반피해자)",
        "intent": "우선매수권_행사_안내",
    },
}

RESULT_COLUMNS = [
    "question", "contexts", "answer", "ground_truth", "category",
    "intent", "route", "followup_questions", "error", "source", "source_id",
]


def parse_alias(x):
    if pd.isna(x):
        return []
    try:
        parsed = json.loads(x)
        if isinstance(parsed, list):
            return [str(p).strip() for p in parsed if str(p).strip()]
    except Exception:
        pass
    return [s.strip() for s in str(x).split(",") if s.strip()]


def extract_intent_text(x):
    try:
        d = json.loads(x)
        return d.get("intent") or d.get("primary_intent") or ""
    except Exception:
        return ""


def build_rows():
    guide = pd.read_csv(GUIDE_CSV)
    intent = pd.read_csv(INTENT_CSV)

    rows = []

    # ---------- 1) guide 파일: 91개 전부 사용 ----------
    for _, r in guide.iterrows():
        aliases = parse_alias(r["alias"])
        if not aliases:
            continue
        intent_val = extract_intent_text(r["intent_json"])
        ground_truth = r["summary"] if pd.notna(r.get("summary")) else r.get("본문 내용", "")

        question = aliases[0]
        followups = aliases[1:4]

        rows.append({
            "question": question,
            "contexts": "",
            "answer": "",
            "ground_truth": ground_truth,
            "category": r["대분류"],
            "intent": intent_val,
            "route": "",
            "followup_questions": " | ".join(followups),
            "error": "",
            "source": "guide",
            "source_id": r["핵심 식별자"],
        })

    # ---------- 2) intent 파일: 계약조항(임대차 관련) 30개 샘플 ----------
    contract = intent[intent["데이터 구분"] == "계약조항"].copy()
    contract = contract[contract["핵심 식별자"].astype(str).str.contains("임대차계약", na=False)]

    contract_shuffled = contract.sample(frac=1, random_state=42).reset_index(drop=True)
    contract_sample = contract_shuffled.drop_duplicates(subset="intent_json", keep="first").reset_index(drop=True)
    target_n = 30
    if len(contract_sample) > target_n:
        contract_sample = contract_sample.sample(target_n, random_state=42).reset_index(drop=True)

    for _, r in contract_sample.iterrows():
        aliases = parse_alias(r["alias"])
        if not aliases:
            continue
        intent_val = extract_intent_text(r["intent_json"])
        body = str(r.get("본문 내용 1", "") or "")
        body2 = str(r.get("본문 내용 2", "") or "")
        ground_truth = (body + " " + body2).strip()

        question = aliases[0]
        followups = aliases[1:4]

        rows.append({
            "question": question,
            "contexts": "",
            "answer": "",
            "ground_truth": ground_truth,
            "category": "사기예방·계약가이드",
            "intent": intent_val,
            "route": "",
            "followup_questions": " | ".join(followups),
            "error": "",
            "source": "contract_clause",
            "source_id": r["핵심 식별자"],
        })

    df = pd.DataFrame(rows)

    # ---------- 3) 빠른 시작 질문 3개 무조건 포함 ----------
    must_rows = []
    existing_questions = set(df["question"])
    for q in MUST_INCLUDE_QUESTIONS:
        if q in existing_questions:
            # 이미 alias 후보 안에 동일 문장이 있으면 그대로 둠 (중복 방지 위해 스킵)
            print(f"ℹ️  '{q}' 은 이미 후보에 포함되어 있어 그대로 둠")
            continue
        manual = MANUAL_GROUND_TRUTH[q]
        must_rows.append({
            "question": q,
            "contexts": "",
            "answer": "",
            "ground_truth": manual["ground_truth"],
            "category": manual["category"],
            "intent": manual["intent"],
            "route": "",
            "followup_questions": "",
            "error": "",
            "source": "quick_start_manual",
            "source_id": "",
        })
        print(f"✅ 빠른시작 질문 '{q}' 추가 (category={manual['category']})")

    if must_rows:
        df = pd.concat([df, pd.DataFrame(must_rows)], ignore_index=True)

    # ---------- 4) 중복 질문 제거 (캡 로직 없음) ----------
    df = df.drop_duplicates(subset="question", keep="first").reset_index(drop=True)

    return df[RESULT_COLUMNS]


def main():
    df = build_rows()
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n=== 완료: {OUTPUT_PATH} ({len(df)}행) ===")
    print(df["category"].value_counts())
    print("\n포함 여부 확인 (빠른 시작 질문 3개):")
    for q in MUST_INCLUDE_QUESTIONS:
        print(f"  - {'✅' if q in set(df['question']) else '❌'} {q}")


if __name__ == "__main__":
    main()