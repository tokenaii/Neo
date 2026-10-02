# NEO - Decision Model

<p align="center"><img src="assets/neo-cover.png" alt="NEO" width="360"></p>

TokenAI is a non-profit startup founded in 2025 by Assem Sabry, based in Alexandria, Egypt. The Neo model,
source code, training data, documentation, and related project materials are
owned by TokenAI. Contact: info@tokenai.llc · https://tokenai.llc

## Official repositories

- [Model repository on Hugging Face](https://huggingface.co/tokenaii/Neo) —
  published model artifacts, tokenizer files, model documentation, technical
  specifications, and legal notices.
- [Dataset repository on Hugging Face](https://huggingface.co/datasets/tokenaii/Neo-dataset)
  — the complete English synthetic training and evaluation dataset, organized
  data shards, manifest, provenance, and dataset legal files.
- [Complete source repository on GitHub](https://github.com/tokenaii/Neo) —
  the complete project source code, training code, data-generation code,
  evaluation and benchmark scripts, configurations, examples, documentation,
  licenses, and reproducibility materials.

Neo is a **System One Model**: a probabilistic decision model that reads an application state and answers typed questions without generating free-form text.

## Training Method: RLCD

Neo uses **Reinforcement Learning for Calibrated Decisions (RLCD)** as its primary training method. The current English-only run is trained from scratch on synthetic decision records rather than customer-specific fine-tuning or LoRA adapters.

RLCD is treated as a training objective and data-generation loop, not as a claim that Neo reproduces TypeSafe AI's private implementation. TypeSafe AI's Jev architecture, sampler, data, and exact RLCD recipe are not public. Neo will use an independently reproducible design with documented assumptions.

## Model Category: System One Models

Neo belongs to the **System One Models** category. It returns typed decisions and calibrated probabilities for software to consume directly. It is not a conversational text-generation model.

Neo must remain identified as **TokenAI Neo** and referenced as `tokenaii/Neo`.
Rebranding, renaming, white-labeling, or publishing a copy under another
identity is prohibited. The same attribution requirement applies to the
dataset and every derived product that uses it. See [MODEL_LICENSE.md](MODEL_LICENSE.md)
and [DATASET_LICENSE.md](DATASET_LICENSE.md).

## Decision interface

Neo supports three decision primitives:

- `choice`: select one option from a declared set.
- `score`: place the state on an ordered rubric.
- `noul`: estimate the probability that a declared proposition is true.

The complete technical specification is documented in
[docs/MODEL_SPECIFICATION.md](docs/MODEL_SPECIFICATION.md), and the active
English-only training configuration is [configs/neo-english.yaml](configs/neo-english.yaml).

One request may contain multiple independent questions. All questions share the encoded state and are answered in one forward pass.

## Training plan

### Phase 0 — Reproducible CPU and synthetic smoke test

- Validate the schema, tokenizer, collator, masking, loss, checkpoint save/load, and API on CPU.
- Generate 2,000 individual decisions from deterministic templates.
- Use 80% training, 10% calibration, and 10% test splits.
- No GPU or production-service changes.

### Phase 1 — English from-scratch release

- Architecture: compact bidirectional Transformer encoder with question-conditioned decision heads.
- Target size: approximately 110M learned parameters with an English tokenizer.
- Context: 1,024 tokens.
- Decision labels: up to 1,000,000 individual decisions for the first full run.
- Records: approximately 350,000–450,000 states, with multiple questions per state where useful.
- Questions: up to 8 per request.
- Choice: up to 32 options.
- Score: 3–5 ordered levels.
- Noul: binary probability.
- Language: English only.

### Phase 2 — RLCD calibration and robustness

For each synthetic state/question pair, generate a target distribution rather than only a hard label. The training objective combines:

1. supervised decision cross-entropy;
2. proper scoring loss for probability quality;
3. consistency under option permutation;
4. consistency under paraphrased instructions;
5. abstention pressure on ambiguous or unknowable cases.

The calibration split is never used for weight updates. A positive temperature is fitted after training on the calibration split and frozen for evaluation and serving.

### Phase 3 — Held-out evaluation

Neo must be tested on:

- known decision families;
- unseen combinations of state and question wording;
- unseen tools and option names;
- English-only prompts and labels;
- reordered options;
- ambiguous and unknowable states;
- multi-question requests;
- tool routing with approximately 20 tools.

Required metrics are accuracy, macro-F1, negative log-likelihood, Brier score, expected calibration error, selective accuracy, abstention rate, latency, and throughput.

After every completed training run, `scripts/post_training_eval.sh` runs the
held-out benchmark, latency test, and tool-router smoke test automatically.
Results remain in the configured local results directory until explicitly
approved for publication.

### Phase 4 — Scale decision

Only after the from-scratch prototype passes the held-out gates will we decide whether to scale the same architecture to approximately 350M–500M parameters. The 1,000,000-decision run is a first research release, not a claim that Neo is a general-purpose Jev replacement.

## Proposed architecture

```text
state tokens + question schema + option descriptions
                         |
                  shared Transformer encoder
                         |
                 question-conditioned pooling
                         |
       choice head | score head | noul head | abstain head
                         |
             masked probabilities + confidence
```

The model is a neural network with learned parameters, but its output space is defined by the caller at inference time. The number of training decisions is not the number of model parameters.

## Data policy

The dataset is synthetic and generated from tracked schemas, templates, tool descriptions, paraphrases, perturbations, and controlled ambiguity. Every record retains provenance, generator identity, schema hash, split, and validation status. External datasets will be added only after license review.

## Server policy

The AWS host is shared with existing model services. Neo uses an isolated workspace under `/mnt/opet-data/neo`. Preparation does not touch unrelated services. H-Preview may be paused only during an explicitly authorized GPU handoff, with its container and watchdog state recorded for restoration. GPU training starts only after a fresh preflight check.

## Initial acceptance gates

- 100% schema and checkpoint round-trip tests.
- No illegal option returned by the inference API.
- Option-order invariance test passes.
- Calibration artifact is bound to the checkpoint hash.
- Held-out accuracy materially exceeds the majority baseline.
- Unknowable cases do not produce unjustified high-confidence decisions.
- A real 20-tool routing evaluation is reported separately from synthetic scores.
