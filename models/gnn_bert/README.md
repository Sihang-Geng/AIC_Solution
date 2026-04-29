# GNN-BERT Feature Model

Graph neural network model with expanded text and behavior features.

## Role

This module captures higher-capacity representation signals. It is more
expensive than the lightweight models, so it is used as a complementary
high-capacity source in the final ensemble.

## Main File

- `train_gnn_bert.py`

## Included Artifacts

- `model.pth`
- `training_curve.png`

## Expected Outputs

- `final_gnn_bert.csv`
- `semifinal_gnn_bert.csv`
