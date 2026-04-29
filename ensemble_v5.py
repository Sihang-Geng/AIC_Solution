"""
Stable V5 ensemble.

The V5 family contains several variants with different embedding dimensions and
candidate depths. A recommendation that repeatedly appears across these variants
is treated as a stable signal and sent to the final ensemble.
"""

from collections import Counter, defaultdict

import pandas as pd


V5_OUTPUTS = [
    "models/v5_ranker/final_v5_32.csv",
    "models/v5_ranker/final_v5_64.csv",
    "models/v5_ranker/final_v5_16.csv",
    "models/v5_ranker/final_v5_256.csv",
    "models/v5_ranker/final_v5_128.csv",
    "models/v5_ranker/semifinal_v5_64.csv",
    "models/v5_ranker/semifinal_v5_32.csv",
    "models/v5_ranker/semifinal_v5_128.csv",
    "models/v5_ranker/semifinal_v5_16.csv",
    "models/v5_ranker/semifinal_v5_512.csv",
]

OUTPUT_CSV = "stable_v5.csv"
MIN_VOTES = 6


all_predictions = []
model_sets = []

for source_index, csv_path in enumerate(V5_OUTPUTS):
    df = pd.read_csv(csv_path)
    df = df[["user_id", "book_id"]].drop_duplicates()
    df["source_index"] = source_index

    all_predictions.extend(df.to_dict("records"))
    model_sets.append(set(zip(df["user_id"], df["book_id"])))


votes = defaultdict(lambda: defaultdict(lambda: [0, float("inf")]))

for row in all_predictions:
    user_id = row["user_id"]
    book_id = row["book_id"]
    source_index = row["source_index"]

    votes[user_id][book_id][0] += 1
    votes[user_id][book_id][1] = min(votes[user_id][book_id][1], source_index)


stable_rows = []
selected_sources = []

for user_id, candidate_books in votes.items():
    ranked_books = sorted(candidate_books.items(), key=lambda item: (-item[1][0], item[1][1]))
    ranked_books = [item for item in ranked_books if item[1][0] >= MIN_VOTES]

    if ranked_books:
        book_id, (vote_count, source_index) = ranked_books[0]
        stable_rows.append({"user_id": user_id, "book_id": book_id})
        selected_sources.append({"source_index": source_index, "votes": vote_count})


stable_output = pd.DataFrame(stable_rows).sort_values("user_id").reset_index(drop=True)
stable_output.to_csv(OUTPUT_CSV, index=False)


print("=" * 60)
print("Stable V5 ensemble")
print("=" * 60)
print(f"V5 variants: {len(V5_OUTPUTS)}")
print(f"Vote threshold: {MIN_VOTES}")
print(f"Stable users: {len(stable_output)}")
print(f"Output: {OUTPUT_CSV}")

vote_distribution = Counter(item["votes"] for item in selected_sources)
print("\nSelected vote distribution:")
for vote_count, count in sorted(vote_distribution.items()):
    print(f"  votes={vote_count}: {count}")
