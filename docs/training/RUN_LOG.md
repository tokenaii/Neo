# Neo Training Run Log

## English training run

- Method: RLCD-inspired soft-target training from scratch.
- Dataset: `neo-synthetic`.
- Dataset repository: `tokenaii/Neo-dataset`.
- Total decisions: 1,000,000.
- Split: 800,178 train / 99,889 calibration / 99,933 test.
- Backbone: randomly initialized compact bidirectional Transformer encoder.
- Training script: `scripts/train_neo.py`.
- Hardware: one NVIDIA RTX PRO 6000 Blackwell Server Edition.
- Precision: bfloat16 autocast.
- Run status: completed successfully.
- Steps: 12,503 (one epoch over the 800,178-record training split).
- Final checkpoint: `model.safetensors` after the verified final transformation.
- Final training loss sample: approximately 1.13 at step 12,500.
- Load/inference smoke test: passed; the tool-router example returned typed
  probabilities and correctly marked the low-confidence result for review.
- Model release: https://huggingface.co/tokenaii/Neo/commit/e47ce606942b413569d0c5899a42cac5d7111396

The test split is reserved. It must not be used to select checkpoints,
thresholds, or calibration temperature.
