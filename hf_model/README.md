---
license: other
license_name: tokenai-neo-model-license-v2.0
license_link: https://huggingface.co/tokenaii/Neo/blob/main/licenses/MODEL_LICENSE.md
library_name: pytorch
tags:
- system-one
- decision-model
- rlcd
- tool-routing
---

# NEO - Decision Model

<p align="center"><img src="assets/neo-cover.png" alt="NEO" width="360"></p>

TokenAI is a non-profit startup founded in 2025 by Assem Sabry. The Neo model,
source code, training data, documentation, and related materials are owned by
TokenAI. Contact: info@tokenaia.llc · https://tokenai.llc

Neo is a TokenAI System One decision model. It reads application state and a
declared decision schema and returns typed probabilities for `choice`, `score`,
and `noul` questions in one forward pass. It does not generate free-form
conversation text.

Canonical reference: `tokenaii/Neo` — https://huggingface.co/tokenaii/Neo

Dataset: https://huggingface.co/datasets/tokenaii/Neo-dataset

Training method: RLCD-inspired soft-target training from scratch on synthetic
decision records. See the project repository for code, manifests, run logs,
and evaluation protocols.

Language: English only. The released model and training corpus are intended
for English inference only.

The English-only release uses an English BERT tokenizer and a randomly
initialized bidirectional Transformer encoder with typed decision heads.

## Repository layout

- `pytorch_model.bin`: model weights.
- `tokenizer.json` and `tokenizer_config.json`: `bert-base-uncased` tokenizer
  files used by the model.
- `training_config.json`: reproducible training arguments.
- `licenses/`: canonical model license and notices.
- `assets/`: project artwork.
- `docs/`: technical specification and usage documentation.

The tokenizer is not custom-built. Neo uses the standard English
`bert-base-uncased` vocabulary and tokenizer behavior; the Transformer weights
are initialized and trained for Neo from scratch.

The model must remain identified as **TokenAI Neo** and must keep the canonical
reference `tokenaii/Neo`. Rebranding, renaming, white-labeling, redistributing,
or publishing a copy or derivative checkpoint under another identity is
prohibited. Do not use the dataset without preserving its `tokenaii/Neo`
reference. All use is subject to [licenses/MODEL_LICENSE.md](licenses/MODEL_LICENSE.md).

Commercial, monetized, paid, sponsored, advertising-supported, production,
client, or financially beneficial use of the model or its outputs is forbidden.
Any permitted use must state that the model was trained using
`tokenaii/Neo-dataset`.

## License

This model is released under the TokenAI Neo Model License v2.0. Commercial
use, redistribution, rebranding, renaming, and derivative checkpoint
publication are prohibited. Keep the `tokenaii/Neo` reference visible.

The model is not certified for medical, legal, financial, safety-critical, or
autonomous decisions without independent validation and human oversight.
