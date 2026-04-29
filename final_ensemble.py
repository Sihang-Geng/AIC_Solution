"""
Final ensemble: weighted voting with deterministic arbitration.

This is the main entry point of the project. Each model module produces a
candidate CSV with two required columns: user_id and book_id. The final layer
does not retrain models; it only aggregates their recommendations.
"""

from collections import Counter, defaultdict

import pandas as pd


# The order is part of the arbitration rule. If two books receive the same
# weighted score for one user, the candidate supported by the earlier source
# keeps priority.
MODEL_OUTPUTS = [
    ("f3_lgbm", "models/f3_lgbm/final_f3.csv", 1.1),
    ("f1_lgbm", "models/f1_lgbm/final_f1.csv", 0.5),
    ("topk_ensemble", "topk_ensemble.csv", 0.5),
    ("v5_base16", "models/v5_ranker/final_v5_16.csv", 0.5),
    ("dspos2", "models/dspos2/final_dspos2.csv", 0.5),
    ("gnn_bert", "models/gnn_bert/final_gnn_bert.csv", 0.5),
    ("mix_lgbm", "models/mix_lgbm/final_mix_lgbm.csv", 1.0),
    ("stable_v5", "stable_v5.csv", 1.7),
    ("v2_lgbm", "models/v2_lgbm/final_v2.csv", 1.8),
    ("legacy_mix23", "models/legacy_reuse/mix23/semifinal_mix23.csv", 1.1),
    ("legacy_f1", "models/legacy_reuse/f1/semifinal_f1.csv", 0.5),
    ("legacy_dspos2", "models/dspos2/semifinal_dspos2.csv", 0.5),
    ("legacy_gnn_bert", "models/gnn_bert/semifinal_gnn_bert.csv", 0.7),
]

OUTPUT_CSV = "submission.csv"


all_predictions = []

for source_index, (source_name, csv_path, weight) in enumerate(MODEL_OUTPUTS):
    df = pd.read_csv(csv_path)
    df = df[["user_id", "book_id"]].drop_duplicates()
    df["source_index"] = source_index
    df["source_name"] = source_name
    df["weight"] = weight
    all_predictions.extend(df.to_dict("records"))


# votes[user_id][book_id] = [weighted_score, best_source_priority]
votes = defaultdict(lambda: defaultdict(lambda: [0.0, float("inf")]))

for row in all_predictions:
    user_id = row["user_id"]
    book_id = row["book_id"]
    source_index = row["source_index"]

    votes[user_id][book_id][0] += row["weight"]
    votes[user_id][book_id][1] = min(votes[user_id][book_id][1], source_index)


submission_rows = []
selected_sources = []

for user_id, candidate_books in votes.items():
    ranked_books = sorted(candidate_books.items(), key=lambda item: (-item[1][0], item[1][1]))
    book_id, (weighted_score, source_index) = ranked_books[0]

    submission_rows.append({"user_id": user_id, "book_id": book_id})
    selected_sources.append({"source_index": source_index, "weighted_score": weighted_score})


submission = pd.DataFrame(submission_rows).sort_values("user_id").reset_index(drop=True)
submission.to_csv(OUTPUT_CSV, index=False)


print("=" * 60)
print("Final weighted ensemble")
print("=" * 60)
print(f"Sources: {len(MODEL_OUTPUTS)}")
print(f"Users: {len(votes)}")
print(f"Output: {OUTPUT_CSV}")

score_distribution = Counter(round(item["weighted_score"], 2) for item in selected_sources)
print("\nSelected score distribution:")
for score, count in sorted(score_distribution.items()):
    print(f"  score={score}: {count}")
