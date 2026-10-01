---
license: other
license_name: tokenai-neo-model-license-v1.0
license_link: https://huggingface.co/tokenaii/neo/blob/main/MODEL_LICENSE.md
library_name: pytorch
tags:
- system-one
- decision-model
- rlcd
- tool-routing
---

# NEO - Decision Model

<p align="center"><img src="neo-cover.png" alt="NEO" width="360"></p>

TokenAI is a non-profit startup founded in 2025 by Assem Sabry. The Neo model,
source code, training data, documentation, and related materials are owned by
TokenAI. Contact: info@tokenaia.llc · https://tokenai.llc

Neo is a TokenAI System One decision model. It reads application state and a
declared decision schema and returns typed probabilities for `choice`, `score`,
and `noul` questions in one forward pass. It does not generate free-form
conversation text.

Canonical reference: `tokenaii/neo` — https://huggingface.co/tokenaii/neo

Dataset: https://huggingface.co/datasets/tokenaii/neo-dataset

Training method: RLCD-inspired soft-target training from scratch on synthetic
decision records. See the project repository for code, manifests, run logs,
and evaluation protocols.

The model must remain identified as **TokenAI Neo** and must keep the canonical
reference `tokenaii/neo`. Rebranding, renaming, white-labeling, redistributing,
or publishing a copy or derivative checkpoint under another identity is
prohibited. Do not use the dataset without preserving its `tokenaii/neo`
reference. All use is subject to [MODEL_LICENSE.md](MODEL_LICENSE.md).

## License

This model is released under the TokenAI Neo Model License v1.0. Do not
redistribute, rebrand, rename, or publish a derivative checkpoint without
written permission from TokenAI. Keep the `tokenaii/neo` reference visible.

The model is not certified for medical, legal, financial, safety-critical, or
autonomous decisions without independent validation and human oversight.
