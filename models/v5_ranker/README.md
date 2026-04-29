# V5 Ranker

Ranking model family with multiple parameter variants.

## Role

The V5 family is used to mine stable recommendations. Different embedding
dimensions and candidate depths produce slightly different outputs; candidates
that repeatedly appear across variants are treated as high-confidence signals.

## Main File

- `train_v5_ranker.py`

## Typical Outputs

- `final_v5_16.csv`
- `final_v5_32.csv`
- `final_v5_64.csv`
- `final_v5_128.csv`
- `final_v5_256.csv`
- `top16_v5.csv`
- `top256_v5.csv`

These files feed `ensemble_v5.py` and `ensemble_topk.py`.
