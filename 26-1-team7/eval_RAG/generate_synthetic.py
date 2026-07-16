"""
1단계: qa_seed.csv 생성

원본 CSV(guide-전세사기_안내가이드.csv, intent_전체_필터링_최종.csv)의
alias 컬럼(이미 검증된 자연어 질문)을 우선 추출하고,
alias 커버리지가 얕은 카테고리는 Ragas TestsetGenerator로 보강한다.

입력: 원본 CSV 2개
출력: eval/datasets/qa_seed.csv  (question, ground_truth, category, intent, source)
      -- 이 파일에는 실제 챗봇 답변/검색결과가 없다. 순수 "질문지" 상태.

* app/state.py(JeonseAgentState)와는 직접적인 연관이 없는 모듈이라 로직 변경 없음.
"""

import os
import ast
import json
import pandas as pd

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
GUIDE_CSV = "./data/guide-전세사기_안내가이드.csv"
INTENT_CSV = "./data/intent_전체_필터링_최종.csv"
OUTPUT_PATH = "./data/qa_seed.csv"

MAX_PER_CATEGORY = 10       # 카테고리별 대표 질문 캡 (쏠림 방지)
SYNTHETIC_TARGET_CATEGORIES = ["국세징수법", "형사", "국세기본법"]  # alias 적은 카테고리 -> LLM 보강 대상
SYNTHETIC_SIZE_PER_CATEGORY = 8
USE_SYNTHETIC = False        # True로 바꾸면 TestsetGenerator까지 실행 (API 비용 발생, GOOGLE_API_KEY 필요)


# ---------------------------------------------------------------------------
# alias 파싱 (두 가지 포맷 존재: 파이썬 리스트 문자열 / 콤마구분 문자열)
# ---------------------------------------------------------------------------
def parse_alias(x) -> list[str]:
    if pd.isna(x):
        return []
    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, list):
            return [str(p).strip() for p in parsed if str(p).strip()]
    except Exception:
        pass
    return [s.strip() for s in str(x).split(",") if s.strip()]


def extract_intent_text(x) -> str:
    try:
        d = json.loads(x)
        return d.get("intent") or d.get("primary_intent") or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 1) guide CSV -> 후보 질문
# ---------------------------------------------------------------------------
def load_guide_candidates(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        ground_truth = row["summary"] if pd.notna(row.get("summary")) else row["본문 내용"]
        intent_text = extract_intent_text(row["intent_json"])
        for q in parse_alias(row["alias"]):
            rows.append({
                "question": q,
                "ground_truth": ground_truth,
                "category": row["대분류"],
                "intent": intent_text,
                "source": "guide",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2) intent CSV -> 후보 질문 (전체 스코프, 필터링 없음)
# ---------------------------------------------------------------------------
def load_intent_candidates(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["intent_text"] = df["intent_json"].apply(extract_intent_text)

    rows = []
    for _, row in df.iterrows():
        body1 = row["본문 내용 1"] if pd.notna(row["본문 내용 1"]) else ""
        body2 = row["본문 내용 2"] if pd.notna(row["본문 내용 2"]) else ""
        ground_truth = (body1 + " " + body2).strip()
        for q in parse_alias(row["alias"]):
            rows.append({
                "question": q,
                "ground_truth": ground_truth,
                "category": row["대분류"],
                "intent": row["intent_text"],
                "source": "intent",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3) 카테고리별 캡을 씌운 균형잡힌 서브셋 만들기
# ---------------------------------------------------------------------------
def stratify_and_cap(df: pd.DataFrame, max_per_category: int, seed: int = 42) -> pd.DataFrame:
    # intent당 대표 질문 1개만 남기기 (중복/편중 방지)
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    one_per_intent = shuffled.drop_duplicates(subset="intent", keep="first").reset_index(drop=True)

    # 카테고리별로 최대 max_per_category개까지만
    reshuffled = one_per_intent.sample(frac=1, random_state=seed).reset_index(drop=True)
    reshuffled["_rank_in_cat"] = reshuffled.groupby("category").cumcount()
    capped = reshuffled[reshuffled["_rank_in_cat"] < max_per_category].drop(columns="_rank_in_cat")
    return capped.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4) (선택) Ragas TestsetGenerator로 부족한 카테고리 보강
#    -- 대분류 단위로 KG를 따로 만들어서 무관한 주제끼리 엮이는 것 방지
# ---------------------------------------------------------------------------
def generate_synthetic_for_categories(
    intent_csv_path: str,
    target_categories: list[str],
    size_per_category: int,
) -> pd.DataFrame:
    from google import genai
    from ragas.llms import llm_factory
    from ragas.embeddings import GoogleEmbeddings
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import default_transforms, apply_transforms
    from ragas.testset import TestsetGenerator
    from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    generator_llm = llm_factory("gemini-2.5-flash", provider="google", client=client)
    generator_embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")

    df_intent = pd.read_csv(intent_csv_path)
    df_intent["본문"] = (
        df_intent["본문 내용 1"].fillna("") + " " + df_intent["본문 내용 2"].fillna("")
    )

    all_results = []
    for cat in target_categories:
        texts = df_intent.loc[df_intent["대분류"] == cat, "본문"].dropna().tolist()
        texts = [t for t in texts if t.strip()]
        if len(texts) < 3:
            print(f"[스킵] '{cat}' 카테고리는 텍스트가 너무 적어요 ({len(texts)}개)")
            continue

        kg = KnowledgeGraph()
        for text in texts:
            kg.nodes.append(Node(type=NodeType.DOCUMENT, properties={"page_content": text}))

        trans = default_transforms(documents=[], llm=generator_llm, embedding_model=generator_embeddings)
        apply_transforms(kg, trans)

        generator = TestsetGenerator(
            llm=generator_llm, embedding_model=generator_embeddings, knowledge_graph=kg
        )
        # 짧고 서로 독립적인 조항 데이터라 single-hop 위주로 생성
        query_distribution = [(SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0)]
        testset = generator.generate(testset_size=size_per_category, query_distribution=query_distribution)

        result_df = testset.to_pandas()
        result_df["category"] = cat
        result_df["source"] = "synthetic"
        all_results.append(result_df)
        print(f"[{cat}] {len(result_df)}개 합성 질문 생성 완료 -- 반드시 검토 후 사용하세요")

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    guide_candidates = load_guide_candidates(GUIDE_CSV)
    intent_candidates = load_intent_candidates(INTENT_CSV)
    all_candidates = pd.concat([guide_candidates, intent_candidates], ignore_index=True)

    print(f"전체 alias 후보: {len(all_candidates)}개")

    balanced = stratify_and_cap(all_candidates, MAX_PER_CATEGORY)
    print(f"카테고리별 캡 적용 후: {len(balanced)}개")
    print(balanced["category"].value_counts())

    final_df = balanced

    if USE_SYNTHETIC:
        synthetic_df = generate_synthetic_for_categories(
            INTENT_CSV, SYNTHETIC_TARGET_CATEGORIES, SYNTHETIC_SIZE_PER_CATEGORY
        )
        if not synthetic_df.empty:
            # ragas testset 컬럼명(user_input, reference 등)을 우리 스키마에 맞춰 정리
            synthetic_df = synthetic_df.rename(
                columns={"user_input": "question", "reference": "ground_truth"}
            )
            synthetic_df["intent"] = synthetic_df["category"] + "_synthetic"
            keep_cols = ["question", "ground_truth", "category", "intent", "source"]
            synthetic_df = synthetic_df[[c for c in keep_cols if c in synthetic_df.columns]]

            print("\n⚠️  합성 질문은 자동 생성된 것이니, qa_seed.csv에 병합하기 전에 꼭 검토하세요.")
            final_df = pd.concat([balanced, synthetic_df], ignore_index=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n완료: {OUTPUT_PATH} ({len(final_df)}행)")


if __name__ == "__main__":
    main()