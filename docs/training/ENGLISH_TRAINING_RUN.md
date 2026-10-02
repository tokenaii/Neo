# English-Only Training Run

## Scope

The active run trains Neo exclusively on the regenerated English corpus.

## Inputs

- Data: `/mnt/opet-data/neo/data/generated-english/neo-english.jsonl`
- Generator: `neo-english-synthetic`
- Records: 400,000
- Decisions: 1,000,000
- Split: 799,882 train / 100,138 calibration / 99,980 test

## Configuration

- 20 epochs.
- Batch size 1,024.
- 15,640 expected optimizer steps.
- `bert-base-uncased` tokenizer.
- bfloat16 autocast.
- Choice loss weight 8.0; score and noul weights 1.0.
- Automatic post-training evaluation enabled.

## Publication policy

The checkpoint is not replaced on the Hub until the held-out benchmark is
complete and reviewed. Benchmark artifacts remain local until publication is
explicitly approved.
