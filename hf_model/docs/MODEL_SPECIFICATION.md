# Neo Model Specification

Neo is an English-only System One decision model.

- Bidirectional Transformer encoder.
- 41,391,654 trainable parameters (approximately 41.4M).
- 8 layers, hidden size 512, 8 attention heads.
- Intermediate size 2048.
- Maximum context 1,024 tokens.
- Tokenizer: `bert-base-uncased`.
- Decision heads: `choice`, `score`, and `noul`.
- Choice head: up to 32 options.
- Score head: up to 5 ordered levels.
- Noul head: binary probability.

The model is trained from scratch; the tokenizer is standard pretrained
English vocabulary, not a custom tokenizer.

Training data and configuration are documented in the source repository:

https://github.com/tokenaii/Neo/blob/main/docs/MODEL_SPECIFICATION.md
