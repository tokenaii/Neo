#!/usr/bin/env python3
"""Extended post-training evaluation for Neo."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import BertTokenizerFast

from scripts.train_neo import NeoModel


def rows(path: Path, split: str):
    with path.open(encoding="utf-8") as handle:
        for record in handle:
            record = json.loads(record)
            if record["metadata"].get("split") != split:
                continue
            state = record["state"]
            for question_id, question in record["questions"].items():
                label = record["labels"][question_id]
                criteria = question.get("criteria", {})
                options = list(criteria) if isinstance(criteria, dict) else list(criteria)
                descriptions = list(criteria.values()) if isinstance(criteria, dict) else options
                yield {
                    "text": (
                        "State language: " + state["language"] + "\n"
                        "State: " + state["text"] + "\n"
                        "Available tools: " + ", ".join(state["available_tools"]) + "\n"
                        "Question type: " + question["type"] + "\n"
                        "Question: " + question["instructions"] + "\n"
                        "Options: " + " | ".join(f"{key}: {value}" for key, value in zip(options, descriptions))
                    ),
                    "option_descriptions": descriptions,
                    "kind": question["type"],
                    "options": options,
                    "available_tools": state["available_tools"],
                    "target": (int(float(label["probability"]) >= 0.5)
                               if question["type"] == "noul" else options.index(label["value"])),
                }


def predict(model, tokenizer, data, device, batch_size=128, temperature=1.0):
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            encoded = tokenizer([x["text"] for x in batch], padding=True, truncation=True,
                                max_length=256, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            kinds = [x["kind"] for x in batch]
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(encoded["input_ids"], encoded["attention_mask"], kinds,
                               encoded.get("token_type_ids"))
            for i, row in enumerate(batch):
                size = 2 if row["kind"] == "noul" else len(row["options"])
                values = (logits[i, :size].float() / temperature).cpu()
                probs = torch.softmax(values, dim=-1).tolist()
                output.append({**row, "logits": values.tolist(), "probs": probs,
                               "prediction": max(range(size), key=probs.__getitem__),
                               "confidence": max(probs)})
    return output


def ece(conf, correct, bins=10):
    total = len(correct)
    result = 0.0
    for b in range(bins):
        selected = [i for i, x in enumerate(conf) if b / bins <= x < (b + 1) / bins or (b == bins - 1 and x == 1)]
        if selected:
            result += len(selected) / total * abs(sum(correct[i] for i in selected) / len(selected) - sum(conf[i] for i in selected) / len(selected))
    return result


def metrics(data):
    correct = [int(x["prediction"] == x["target"]) for x in data]
    conf = [x["confidence"] for x in data]
    nll = 0.0
    brier = 0.0
    for x in data:
        p = max(x["probs"][x["target"]], 1e-12)
        nll -= math.log(p)
        brier += sum((p_i - (1.0 if j == x["target"] else 0.0)) ** 2 for j, p_i in enumerate(x["probs"]))
    labels = sorted(set(x["target"] for x in data))
    f1 = []
    for label in labels:
        tp = sum(x["prediction"] == label and x["target"] == label for x in data)
        fp = sum(x["prediction"] == label and x["target"] != label for x in data)
        fn = sum(x["prediction"] != label and x["target"] == label for x in data)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    counts = Counter(x["target"] for x in data)
    majority = max(counts.values()) / len(data)
    selective = {}
    for threshold in (0.5, 0.7, 0.9):
        kept = [i for i, c in enumerate(conf) if c >= threshold]
        selective[str(threshold)] = {
            "coverage": len(kept) / len(data),
            "selective_accuracy": sum(correct[i] for i in kept) / len(kept) if kept else 0.0,
            "abstention_rate": 1 - len(kept) / len(data),
        }
    return {"n": len(data), "accuracy": sum(correct) / len(data), "macro_f1": sum(f1) / len(f1),
            "majority_baseline": majority, "nll": nll / len(data), "brier": brier / len(data),
            "ece_10bin": ece(conf, correct), "selective": selective}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizerFast.from_pretrained(args.checkpoint)
    model = NeoModel(len(tokenizer)).to(device)
    model.load_state_dict(load_file(str(args.checkpoint / "model.safetensors"), device=device))
    calibration = list(rows(args.data, "calibration"))
    test = list(rows(args.data, "test"))
    raw_calibration = predict(model, tokenizer, calibration, device, args.batch_size)
    raw = predict(model, tokenizer, test, device, args.batch_size)
    by_kind = {kind: metrics([x for x in raw if x["kind"] == kind]) for kind in ("choice", "score", "noul")}
    temps = [0.50 + i * 0.01 for i in range(251)]
    def calibration_nll(t):
        total = 0.0
        for x in raw_calibration:
            m = max(x["logits"])
            denom = sum(math.exp((z - m) / t) for z in x["logits"])
            total -= math.log(max(math.exp((x["logits"][x["target"]] - m) / t) / denom, 1e-12))
        return total
    best_t = min(temps, key=calibration_nll)
    calibrated = predict(model, tokenizer, test, device, args.batch_size, best_t)
    choice = [x for x in raw if x["kind"] == "choice"]
    tool = {"n": len(choice), "accuracy": sum(x["prediction"] == x["target"] for x in choice) / len(choice),
            "predicted_tool_use_rate": sum(x["options"][x["prediction"]] != "none" and x["options"][x["prediction"]] in x["available_tools"] for x in choice) / len(choice),
            "actual_tool_use_rate": sum(x["options"][x["target"]] != "none" and x["options"][x["target"]] in x["available_tools"] for x in choice) / len(choice)}
    permuted = []
    for x in choice:
        order = list(reversed(range(len(x["options"]))))
        y = dict(x); y["options"] = [x["options"][i] for i in order]
        y["target"] = order.index(x["target"])
        prefix = x["text"].rsplit("Options: ", 1)[0]
        y["text"] = prefix + "Options: " + " | ".join(
            f"{x['options'][i]}: {x['option_descriptions'][i]}" for i in order
        )
        permuted.append(y)
    perm = predict(model, tokenizer, permuted, device, args.batch_size)
    invariance = sum(choice[i]["options"][choice[i]["prediction"]] == perm[i]["options"][perm[i]["prediction"]] for i in range(len(choice))) / len(choice)
    result = {"device": device, "test_decisions": len(raw), "by_kind": by_kind,
              "calibrated_temperature": best_t,
              "calibrated_by_kind": {k: metrics([x for x in calibrated if x["kind"] == k]) for k in ("choice", "score", "noul")},
              "tool_routing": tool,
              "option_order_invariance": invariance,
              "unseen_tool_evaluation": {"status": "not_available", "reason": "No separately labeled unseen-tool split was provided."},
              "external_generalization": {"status": "not_run", "reason": "No external evaluation corpus was authorized or staged."}}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
