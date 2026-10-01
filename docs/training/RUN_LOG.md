# Neo Training Run Log

## Run: neo-v0.1

- Method: RLCD-inspired soft-target training from scratch.
- Dataset: `neo-synthetic-v1`.
- Dataset repository: `tokenaii/neo-dataset`.
- Total decisions: 1,000,000.
- Split: 800,178 train / 99,889 calibration / 99,933 test.
- Backbone: randomly initialized compact bidirectional Transformer encoder.
- Training script: `scripts/train_neo.py`.
- Hardware: one NVIDIA RTX PRO 6000 Blackwell Server Edition.
- Precision: bfloat16 autocast.
- Run status: in progress at the time this log was created.

The test split is reserved. It must not be used to select checkpoints,
thresholds, or calibration temperature.
