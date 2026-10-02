# Neo Training Run Log

## Active English training run

- Method: RLCD-inspired soft-target training from scratch.
- Dataset: `tokenaii/Neo-dataset`.
- Records: 400,000.
- Decisions: 1,000,000.
- Split: 799,882 train / 100,138 calibration / 99,980 test decisions.
- Backbone: randomly initialized bidirectional Transformer encoder.
- Tokenizer: standard English `bert-base-uncased`.
- Hardware: AWS GPU server.
- Precision: bfloat16 autocast.
- Expected epochs: 20.
- Expected optimizer steps: 15,640.
- Batch size: 1,024.
- Choice loss weight: 8.0; score and noul loss weights: 1.0.
- Checkpoint format: `model.safetensors`.
- Current observed step: 7,000 / 15,640.
- Latest observed loss: 1.6072139739990234.
- Run status: in progress.

The live log is written on the AWS server at
`/mnt/opet-data/neo/logs/neo-english-train.log`. It records the epoch, step,
and loss throughout the run. The complete history will be copied into this
repository after the run and evaluation finish.

The test split is reserved. It must not be used to select checkpoints,
thresholds, or calibration temperature.
