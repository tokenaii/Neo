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

# Neo

Neo is a TokenAI System One decision model. It reads application state and a
declared decision schema and returns typed probabilities for `choice`, `score`,
and `noul` questions in one forward pass. It does not generate free-form
conversation text.

Canonical reference: `tokenaii/neo` — https://huggingface.co/tokenaii/neo

Dataset: https://huggingface.co/datasets/tokenaii/neo-dataset

Training method: RLCD-inspired soft-target training from scratch on synthetic
decision records. See the project repository for code, manifests, run logs,
and evaluation protocols.

## License

This model is released under the TokenAI Neo Model License v1.0. Do not
redistribute, rebrand, rename, or publish a derivative checkpoint without
written permission from TokenAI. Keep the `tokenaii/neo` reference visible.

The model is not certified for medical, legal, financial, safety-critical, or
autonomous decisions without independent validation and human oversight.
