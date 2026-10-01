#!/usr/bin/env python3
"""Measure Neo inference latency and throughput on a fixed representative batch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import BertTokenizerFast

from scripts.benchmark_neo import items
from scripts.train_neo import NeoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=100)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizerFast.from_pretrained(args.checkpoint)
    model = NeoModel(len(tokenizer)).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint / "pytorch_model.bin", map_location=device))
    sample = list(items(args.data))[:args.batch_size]
    encoded = tokenizer([row["text"] for row in sample], padding=True, truncation=True, max_length=256, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    kinds = [row["kind"] for row in sample]
    with torch.inference_mode():
        for _ in range(10):
            model(encoded["input_ids"], encoded["attention_mask"], kinds, encoded.get("token_type_ids"))
        if device == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.batches):
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                model(encoded["input_ids"], encoded["attention_mask"], kinds, encoded.get("token_type_ids"))
        if device == "cuda": torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {
        "device": device, "batch_size": len(sample), "batches": args.batches,
        "total_decisions": len(sample) * args.batches,
        "elapsed_seconds": elapsed,
        "batch_latency_ms": elapsed / args.batches * 1000,
        "decision_latency_ms": elapsed / (len(sample) * args.batches) * 1000,
        "decisions_per_second": len(sample) * args.batches / elapsed,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
