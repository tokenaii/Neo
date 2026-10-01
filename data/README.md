# Neo Data Layout

- `raw/`: inspected external references; never silently mixed into training.
- `generated/`: small smoke and development corpora.
- `generated-large/`: the 1,000,000-decision first-run corpus.
- `processed/`: validated or flattened training views.
- `manifests/`: hashes, provenance, license decisions, and split records.

The canonical released dataset is `tokenaii/neo-dataset`.
