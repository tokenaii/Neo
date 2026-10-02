---
license: other
license_name: tokenai-neo-model-license-v3.0
license_link: https://huggingface.co/tokenaii/Neo/blob/main/licenses/MODEL_LICENSE.md
library_name: pytorch
pipeline_tag: text-classification
tags:
- system-one
- decision-model
- rlcd
- tool-routing
- en
---

# NEO - Decision Model

<p align="center"><img src="assets/neo-cover.png" alt="NEO - Decision Model" width="360"></p>

## Ownership and contact

TokenAI is a non-profit startup founded in 2025 by Assem Sabry, based in
Alexandria, Egypt. The Neo model, source code, training data, documentation,
and related materials are owned by TokenAI.

Contact: **info@tokenai.llc** · https://tokenai.llc

## Summary

Neo is a compact TokenAI System One decision model for selecting actions from a
declared candidate set. It reads English application context and a typed
decision schema, then returns calibrated outputs for `choice`, `score`, and
`noul` in one encoder forward pass. It is not a conversational or free-form
text-generation model.

Canonical model: `tokenaii/Neo` — https://huggingface.co/tokenaii/Neo

Training dataset: `tokenaii/Neo-dataset` — https://huggingface.co/datasets/tokenaii/Neo-dataset

## Model architecture

- Bidirectional Transformer encoder, initialized and trained from scratch.
- Approximately 110 million parameters.
- 8 Transformer layers; hidden size 512; 8 attention heads.
- Intermediate size 2048; maximum context 1024 tokens.
- Typed heads: `choice`, `score`, and `noul`.
- Choice head: up to 32 candidate options per question.
- Score head: up to 5 ordered levels.
- Noul head: binary probability for a yes/no or escalation decision.
- A request can contain up to 8 independent decision questions.

## Input and output contract

The input is English text plus a predefined decision schema and candidate
options. The output contains typed probabilities and the selected index or
level. Applications should apply their own confidence thresholds, abstention
rules, validation, logging, and human review. Neo does not execute tools and
does not replace authorization or policy enforcement.

## Intended uses

Non-commercial research and education for tool routing, workflow selection,
request classification, department routing, escalation detection, validation
gates, and selecting the next action in an agent pipeline.

## Out-of-scope uses

The license prohibits commercial, monetized, paid, sponsored, client-facing,
production-business, or financially beneficial use. Do not use Neo for
unsupervised medical, legal, financial, employment, housing, admissions,
insurance, credit, safety-critical, government-benefit, law-enforcement, or
irreversible decisions.

## Training data and procedure

The English-only corpus contains 400,000 synthetic records and 1,000,000
independent decisions: 799,882 training, 100,138 calibration, and 99,980 test
decisions. The catalog contains 20 tools; records commonly present one target
and five distractors.

The training run uses RLCD-inspired weighted soft-target training from scratch,
20 epochs, approximately 15,640 optimizer steps, batch size 1024, bfloat16,
learning rate `0.0002`, choice loss weight `8.0`, and score/noul weights `1.0`.
The generator is identified as `neo-english-synthetic-v1` in the project
manifests. The final benchmark report is produced after training and is not
claimed by this card until verified.

## Tokenizer

Neo uses the standard English `bert-base-uncased` tokenizer and vocabulary. A
custom tokenizer was not built for this release, so no custom tokenizer is
claimed or published under `tokenizer/`. The Transformer parameters are Neo's
own randomly initialized and trained weights; the tokenizer vocabulary is not
the model architecture.

## Release status

The previous model weights were removed from this repository before the final
English-only training release. The current repository contains documentation,
configuration, tokenizer metadata, and licenses while the English training and
automatic evaluation pipeline complete. Do not infer benchmark accuracy from
the architecture or data counts. Verified results will be published only after
the local evaluation artifacts are reviewed.

## Repository layout

- `licenses/MODEL_LICENSE.md` — full model license.
- `licenses/NOTICE.md` — attribution and ownership notice.
- `assets/neo-cover.png` — repository artwork.
- `docs/MODEL_SPECIFICATION.md` — technical specification.
- `tokenizer/` — reserved for a custom tokenizer only if one is actually built.
- `training_config.json` — reproducibility metadata when a release includes it.

## License and attribution

Use is governed by [the TokenAI Neo Model License v3.0](licenses/MODEL_LICENSE.md).
Redistribution, renaming, rebranding, white-labeling, Derivative Models,
Commercial Use, and Financial Benefit are prohibited without written permission
from TokenAI. Every permitted downstream report or model must state:

> This work was trained using the TokenAI Neo Decision Model Dataset:
> `tokenaii/Neo-dataset`.

Permission requests must be sent to **info@tokenai.llc**. The complete legal
terms, trademark policy, notice-and-takedown process, patent reservation,
contribution/CLA rules, and dependency obligations are in the linked license.

## Limitations

Neo is trained on synthetic English decision records and may fail on unseen
schemas, ambiguous contexts, distribution shifts, adversarial options, or
languages other than English. Calibration and accuracy are task-dependent.
Always validate with held-out, task-specific data and add human oversight for
high-impact workflows.
