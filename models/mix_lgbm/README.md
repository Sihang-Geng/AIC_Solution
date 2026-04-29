# Mixed LightGBM Baseline

Lightweight mixed-feature recommendation model.

## Role

This module provides a fast and stable baseline candidate source for the final
ensemble. It is used as one of the backbone signals rather than a standalone
final decision layer.

## Main File

- `train_mix_lgbm.py`

## Expected Output

- `final_mix_lgbm.csv`

Required columns:

```text
user_id,book_id
```
