# V2 LightGBM

Compact LightGBM-based candidate generator.

## Role

This module is a low-cost auxiliary model. In the final layer it receives a
relatively high fusion weight because it contributed useful complementary
signals during late-stage validation.

## Main File

- `train_v2_lgbm.py`

## Expected Outputs

- `final_v2.csv`
- `top10_v2.csv`
