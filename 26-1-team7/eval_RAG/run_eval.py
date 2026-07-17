"""
3단계: 채점

qa_with_results.csv (question, contexts, answer, ground_truth)를 읽어서
Ragas 메트릭(faithfulness, response_relevancy, context_precision, context_recall)으로
점수를 매기고 eval/results/eval_latest.csv에 저장한다.
"""

import asyncio
import os
from datetime import datetime

import pandas as pd

from metrics_config import build_metrics

INPUT_PATH = "data/qa_with_results.csv"
OUTPUT_PATH = "results/eval_latest.csv"
CONTEXT_SEP = "\n<<<CTX_SEP>>>\n"  # generate_dataset.py에서 쓴 구분자와 동일해야 함

CONCURRENCY = 5  # 동시에 처리할 행 개수


async def evaluate_row(row: dict, metrics: dict) -> dict:
    result = {
        "question": row["question"],
        "category": row.get("category", ""),
        "intent": row.get("intent", ""),
        "route": row.get("route", ""),
        "followup_questions": row.get("followup_questions", ""),
    }

    contexts_val = row.get("contexts")
    answer_val = row.get("answer")

    if (
        pd.isna(contexts_val)
        or pd.isna(answer_val)
        or not str(contexts_val).strip()
        or not str(answer_val).strip()
    ):
        result["skipped"] = "contexts 또는 answer가 비어있어 스킵"
        return result

    contexts = str(contexts_val).split(CONTEXT_SEP)

    try:
        faith = await metrics["faithfulness"].ascore(
            user_input=row["question"],
            response=row["answer"],
            retrieved_contexts=contexts,
        )
        result["faithfulness"] = faith.value

        relevancy = await metrics["response_relevancy"].ascore(
            user_input=row["question"],
            response=row["answer"],
        )
        result["response_relevancy"] = relevancy.value

        if row.get("ground_truth") and not pd.isna(row.get("ground_truth")):
            precision = await metrics["context_precision"].ascore(
                user_input=row["question"],
                retrieved_contexts=contexts,
                reference=str(row["ground_truth"]),
            )
            result["context_precision"] = precision.value

            recall = await metrics["context_recall"].ascore(
                user_input=row["question"],
                retrieved_contexts=contexts,
                reference=str(row["ground_truth"]),
            )
            result["context_recall"] = recall.value

        result["skipped"] = ""
    except Exception as e:
        # 개별 행에서 에러가 나도 전체가 죽지 않도록 방어
        result["skipped"] = f"평가 중 에러: {e}"

    return result


async def evaluate_row_with_limit(
    row: dict, metrics: dict, semaphore: asyncio.Semaphore, index: int, total: int
) -> dict:
    async with semaphore:
        res = await evaluate_row(row, metrics)
        print(f"평가 진행: {index}/{total}")
        return res


async def main():
    df = pd.read_csv(INPUT_PATH)
    if "error" in df.columns:
        n_before = len(df)
        df = df[df["error"].fillna("") == ""].reset_index(drop=True)
        if n_before != len(df):
            print(f"파이프라인 실행 에러난 {n_before - len(df)}개 행 제외하고 평가 진행")

    metrics = build_metrics()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    tasks = [
        evaluate_row_with_limit(row.to_dict(), metrics, semaphore, i + 1, len(df))
        for i, row in df.iterrows()
    ]
    rows = await asyncio.gather(*tasks)

    result_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    score_cols = ["faithfulness", "response_relevancy", "context_precision", "context_recall"]
    score_cols = [c for c in score_cols if c in result_df.columns]

    print(f"\n=== 전체 평균 ===")
    print(result_df[score_cols].mean())

    print(f"\n=== 카테고리별 평균 ===")
    print(result_df.groupby("category")[score_cols].mean())

    print(f"\n완료: {OUTPUT_PATH} ({len(result_df)}행)")


if __name__ == "__main__":
    asyncio.run(main())