---
license: other
license_name: tokenai-neo-model-license
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

Complete source repository: [github.com/tokenaii/Neo](https://github.com/tokenaii/Neo).
It contains the complete project source code, training code, synthetic-data
generation code, evaluation and benchmark scripts, configurations, examples,
documentation, licenses, and reproducibility materials.

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

## Training records

Training code, configurations, live logs, loss history, checkpoints metadata,
evaluation reports, and benchmark history are maintained only in the GitHub
source repository:

https://github.com/tokenaii/Neo/tree/main/docs/training

This Hugging Face model repository does not publish the training history or
benchmark archive.

## Tokenizer

Neo uses the standard English `bert-base-uncased` tokenizer and vocabulary,
published under `tokenizer/` for repository organization. A custom tokenizer
was not built. The Transformer parameters are Neo's own randomly initialized
and trained weights; the tokenizer vocabulary is not the model architecture.

## Release status

The English-only training run is complete and the verified
`model.safetensors` checkpoint is published in this repository. Training logs,
loss history, evaluation reports, and benchmark records remain exclusively in
the GitHub source repository. Do not infer benchmark accuracy from the
architecture or data counts.

## Repository layout

- `licenses/MODEL_LICENSE.md` — full model license.
- `licenses/NOTICE.md` — attribution and ownership notice.
- `assets/neo-cover.png` — repository artwork.
- `docs/MODEL_SPECIFICATION.md` — technical specification.
- `model.safetensors` — verified model weights.
- `tokenizer/` — standard English `bert-base-uncased` tokenizer files.

## License and attribution

Use is governed by [the TokenAI Neo Model License](licenses/MODEL_LICENSE.md).
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
