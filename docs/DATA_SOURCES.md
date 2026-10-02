# Neo Data Sources and Generation Plan

## Policy

Neo's first RLCD run will use a synthetic training corpus generated from versioned tool schemas, decision templates, controlled paraphrases, ambiguity transforms, and deterministic validators. Public datasets are source material for schema design, sanity checks, and held-out evaluation unless their license and provenance are explicitly admitted into the training manifest.

## Candidate public sources

| Source | What it provides | Intended use | Status |
|---|---|---|---|
| `lockon/ToolACE` | 11,300 synthetic multi-turn tool-use conversations; Apache-2.0 card | Tool taxonomy, conversation patterns, evaluation seed | Admitted for inspection; do not copy rows into training without manifest review |
| `Salesforce/xlam-function-calling-60k` | 60,000 single-turn function-calling examples; gated source, CC-BY-4.0 | Function/tool schema coverage and optional licensed evaluation | Requires access and attribution review |
| `PolyAI/banking77` | 77 English intent classes; CC-BY-4.0 | Intent-routing benchmark and label-shape reference | Evaluation/seed only |

## Neo-generated training corpus

The first full run will contain up to 1,000,000 individual decisions, approximately:

- 350,000–450,000 synthetic states;
- 20-tool catalogs plus distractor tools;
- Choice, Score, and Noul in balanced proportions;
- English-only state text, question wording, labels, and tool descriptions;
- direct, paraphrased, ambiguous, contradictory, and unknowable states;
- option-order permutations and question paraphrases;
- deterministic legality checks for tool names and arguments;
- teacher distributions retained as soft targets only when the validator accepts them.

The dataset manifest will record generator version, schema hash, source attribution, license decision, split, and deduplication hash for every record.

## Split policy

- 800,000 decisions: training.
- 100,000 decisions: calibration only.
- 100,000 decisions: final test only.

The final test includes unseen tool names, unseen tool combinations, reordered options, and unknowable cases. No calibration temperature or threshold may be fitted on the final test.

## Why not train directly on every public row?

The goal is a reusable RLCD decision model, not a memorizer of one function-calling dataset. Directly mixing public rows would also make license, provenance, duplicate leakage, and teacher-style artifacts harder to control. Public data therefore informs the generator and benchmarks; Neo's primary training records are generated and validated by this repository.
