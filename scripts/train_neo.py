#!/usr/bin/env python3
"""Train the first Neo System One encoder from scratch on generated records."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import BertConfig, BertModel, BertTokenizerFast


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def score_distribution(size: int, target: int) -> list[float]:
    values = [0.0] * size
    values[target] = 0.8
    if target > 0:
        values[target - 1] += 0.1
    if target + 1 < size:
        values[target + 1] += 0.1
    total = sum(values)
    return [value / total for value in values]


def flatten_records(path: Path, split: str) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["metadata"].get("split") != split:
                continue
            state = record["state"]
            for question_id, question in record["questions"].items():
                label = record["labels"][question_id]
                criteria = question.get("criteria", {})
                if isinstance(criteria, dict):
                    options = list(criteria)
                    descriptions = [criteria[key] for key in options]
                else:
                    options = list(criteria)
                    descriptions = options
                prompt = (
                    "State language: " + state["language"] + "\n"
                    "State: " + state["text"] + "\n"
                    "Available tools: " + ", ".join(state["available_tools"]) + "\n"
                    "Question type: " + question["type"] + "\n"
                    "Question: " + question["instructions"] + "\n"
                    "Options: " + " | ".join(f"{key}: {value}" for key, value in zip(options, descriptions))
                )
                if question["type"] == "choice":
                    target = options.index(label["value"])
                    target_probs = [label["probabilities"].get(option, 0.0) for option in options]
                    yield {"text": prompt, "kind": "choice", "target": target, "probs": target_probs}
                elif question["type"] == "score":
                    target = options.index(label["value"])
                    yield {"text": prompt, "kind": "score", "target": target, "probs": score_distribution(len(options), target)}
                else:
                    probability = float(label["probability"])
                    yield {"text": prompt, "kind": "noul", "target": probability, "probs": [1.0 - probability, probability]}


class DecisionStream(IterableDataset):
    def __init__(self, path: Path, split: str):
        self.path = path
        self.split = split

    def __iter__(self):
        return iter(flatten_records(self.path, self.split))


class NeoModel(nn.Module):
    def __init__(self, vocab_size: int, max_position: int = 1024):
        super().__init__()
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,
            max_position_embeddings=max_position,
            type_vocab_size=1,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.encoder = BertModel(config, add_pooling_layer=False)
        self.choice = nn.Linear(config.hidden_size, 32)
        self.score = nn.Linear(config.hidden_size, 5)
        self.noul = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, kinds=None, token_type_ids=None):
        if kinds is None:
            kinds = ["choice"] * input_ids.shape[0]
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state[:, 0]
        output = torch.zeros((len(kinds), 32), device=hidden.device)
        for index, kind in enumerate(kinds):
            if kind == "choice":
                output[index] = self.choice(hidden[index])
            elif kind == "score":
                output[index, :5] = self.score(hidden[index])
            else:
                value = self.noul(hidden[index]).squeeze()
                output[index, :2] = torch.stack((-value, value))
        return output


def collate(batch, tokenizer, max_length):
    encoded = tokenizer([item["text"] for item in batch], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    return encoded, [item["kind"] for item in batch], [item["probs"] for item in batch]


def loss_for(logits, kinds, probs, kind_weights=None):
    kind_weights = kind_weights or {"choice": 1.0, "score": 1.0, "noul": 1.0}
    losses = []
    weights = []
    for index, kind in enumerate(kinds):
        target = torch.tensor(probs[index], device=logits.device, dtype=logits.dtype)
        if kind == "choice":
            prediction = logits[index, :len(target)]
        elif kind == "score":
            prediction = logits[index, :len(target)]
        else:
            prediction = logits[index, :2]
        losses.append(-(target * torch.log_softmax(prediction.float(), dim=-1)).sum())
        weights.append(kind_weights.get(kind, 1.0))
    weighted_losses = torch.stack(losses)
    weight_tensor = torch.tensor(weights, device=weighted_losses.device, dtype=weighted_losses.dtype)
    return (weighted_losses * weight_tensor).sum() / weight_tensor.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--choice-weight", type=float, default=8.0)
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--noul-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the training run")
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)
    model = NeoModel(len(tokenizer)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    dataset = DecisionStream(args.data, "train")
    step = 0
    model.train()
    for epoch in range(args.epochs):
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda b: collate(b, tokenizer, args.max_length))
        for encoded, kinds, probs in loader:
            encoded = {key: value.cuda(non_blocking=True) for key, value in encoded.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(encoded["input_ids"], encoded["attention_mask"], kinds)
                loss = loss_for(
                    logits,
                    kinds,
                    probs,
                    {"choice": args.choice_weight, "score": args.score_weight, "noul": args.noul_weight},
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 100 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": float(loss.detach().cpu())}), flush=True)
            if step % 5000 == 0:
                checkpoint = args.output / f"checkpoint-{step}"
                checkpoint.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint / "pytorch_model.bin")
                tokenizer.save_pretrained(checkpoint)
    torch.save(model.state_dict(), args.output / "pytorch_model.bin")
    tokenizer.save_pretrained(args.output)
    (args.output / "training_config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "steps": step, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
