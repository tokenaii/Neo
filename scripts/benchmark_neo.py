#!/usr/bin/env python3
"""Evaluate Neo on the held-out synthetic test split without changing weights."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import BertTokenizerFast
from safetensors.torch import load_file

from scripts.train_neo import NeoModel


def items(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["metadata"].get("split") != "test":
                continue
            state = record["state"]
            for question_id, question in record["questions"].items():
                label = record["labels"][question_id]
                criteria = question.get("criteria", {})
                options = list(criteria) if isinstance(criteria, dict) else list(criteria)
                descriptions = list(criteria.values()) if isinstance(criteria, dict) else options
                text = (
                    "State language: " + state["language"] + "\n"
                    "State: " + state["text"] + "\n"
                    "Available tools: " + ", ".join(state["available_tools"]) + "\n"
                    "Question type: " + question["type"] + "\n"
                    "Question: " + question["instructions"] + "\n"
                    "Options: " + " | ".join(f"{key}: {value}" for key, value in zip(options, descriptions))
                )
                if question["type"] == "noul":
                    target_probs = [1.0 - float(label["probability"]), float(label["probability"])]
                    target = int(float(label["probability"]) >= 0.5)
                else:
                    target = options.index(label["value"])
                    if question["type"] == "choice":
                        target_probs = [label["probabilities"].get(option, 0.0) for option in options]
                    else:
                        target_probs = [0.0] * len(options)
                        target_probs[target] = 1.0
                yield {
                    "text": text, "kind": question["type"], "target": target,
                    "target_probs": target_probs, "options": options,
                    "available_tools": state["available_tools"],
                }


def ece(confidences, correct, bins=10):
    total = len(correct)
    if not total:
        return 0.0
    result = 0.0
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        selected = [i for i, value in enumerate(confidences) if lo <= value < hi or (index == bins - 1 and value == hi)]
        if selected:
            result += len(selected) / total * abs(sum(correct[i] for i in selected) / len(selected) - sum(confidences[i] for i in selected) / len(selected))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizerFast.from_pretrained(args.checkpoint)
    model = NeoModel(len(tokenizer)).to(device).eval()
    model.load_state_dict(load_file(str(args.checkpoint / "model.safetensors"), device=device))
    records = list(items(args.data))
    per_kind = defaultdict(lambda: {"n": 0, "correct": 0, "nll": 0.0, "brier": 0.0, "conf": [], "ok": []})
    tool_route = {"n": 0, "correct": 0, "predicted_use": 0, "actual_use": 0, "conf": [], "ok": []}

    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        encoded = tokenizer([row["text"] for row in batch], padding=True, truncation=True, max_length=256, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        kinds = [row["kind"] for row in batch]
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(encoded["input_ids"], encoded["attention_mask"], kinds, encoded.get("token_type_ids"))
        for index, row in enumerate(batch):
            kind = row["kind"]
            size = 2 if kind == "noul" else len(row["options"])
            probs = torch.softmax(logits[index, :size].float(), dim=-1).cpu().tolist()
            target = row["target"]
            prediction = max(range(size), key=probs.__getitem__)
            is_correct = int(prediction == target)
            target_probs = row["target_probs"]
            nll = -sum(target_probs[j] * math.log(max(probs[j], 1e-12)) for j in range(size))
            brier = sum((probs[j] - target_probs[j]) ** 2 for j in range(size))
            bucket = per_kind[kind]
            bucket["n"] += 1
            bucket["correct"] += is_correct
            bucket["nll"] += nll
            bucket["brier"] += brier
            bucket["conf"].append(max(probs))
            bucket["ok"].append(is_correct)
            if kind == "choice":
                predicted_name = row["options"][prediction]
                actual_name = row["options"][target]
                predicted_use = predicted_name != "none" and predicted_name in row["available_tools"]
                actual_use = actual_name != "none" and actual_name in row["available_tools"]
                tool_route["n"] += 1
                tool_route["correct"] += is_correct
                tool_route["predicted_use"] += int(predicted_use)
                tool_route["actual_use"] += int(actual_use)
                tool_route["conf"].append(max(probs))
                tool_route["ok"].append(is_correct)

    result = {"checkpoint": str(args.checkpoint), "device": device, "test_decisions": len(records), "by_kind": {}}
    for kind, bucket in per_kind.items():
        result["by_kind"][kind] = {
            "n": bucket["n"], "accuracy": bucket["correct"] / bucket["n"],
            "nll": bucket["nll"] / bucket["n"], "brier": bucket["brier"] / bucket["n"],
            "ece_10bin": ece(bucket["conf"], bucket["ok"]),
        }
    result["tool_routing"] = {
        "n": tool_route["n"], "accuracy": tool_route["correct"] / tool_route["n"],
        "predicted_tool_use_rate": tool_route["predicted_use"] / tool_route["n"],
        "actual_tool_use_rate": tool_route["actual_use"] / tool_route["n"],
        "ece_10bin": ece(tool_route["conf"], tool_route["ok"]),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
