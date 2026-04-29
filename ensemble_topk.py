"""
Top-k candidate ensemble.

Some models are more valuable as candidate generators than as direct Top-1
predictors. This layer merges their ranked candidates and compresses them into a
single auxiliary source for the final ensemble.
"""

import pandas as pd


TOPK_SOURCES = {
    "models/v2_lgbm/top10_v2.csv": 0.5,
    "models/v5_ranker/top16_v5.csv": 0.3,
    "models/v5_ranker/top256_v5.csv": 0.2,
}

OUTPUT_CSV = "topk_ensemble.csv"
SCORE_COLUMN = "score"


total_weight = sum(TOPK_SOURCES.values())
normalized_weights = {
    csv_path: weight / total_weight
    for csv_path, weight in TOPK_SOURCES.items()
}

frames = []

for csv_path, weight in normalized_weights.items():
    df = pd.read_csv(csv_path)
    required_columns = {"user_id", "book_id", SCORE_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing_columns)}")

    score = df[SCORE_COLUMN]
    if score.max() == score.min():
        df["normalized_score"] = 0.5
    else:
        df["normalized_score"] = (score - score.min()) / (score.max() - score.min())

    df = df[["user_id", "book_id", "normalized_score"]].copy()
    df["source_weight"] = weight
    frames.append(df)


combined = pd.concat(frames, ignore_index=True)
combined["weighted_score"] = combined["normalized_score"] * combined["source_weight"]

ranked = (
    combined.groupby(["user_id", "book_id"], as_index=False)["weighted_score"]
    .sum()
    .sort_values(["user_id", "weighted_score", "book_id"], ascending=[True, False, True])
)

top1 = ranked.groupby("user_id", as_index=False).head(1)
top1 = top1[["user_id", "book_id", "weighted_score"]]
top1.to_csv(OUTPUT_CSV, index=False)

print("=" * 60)
print("Top-k auxiliary ensemble")
print("=" * 60)
print(f"Sources: {len(TOPK_SOURCES)}")
print(f"Users: {top1['user_id'].nunique()}")
print(f"Output: {OUTPUT_CSV}")
