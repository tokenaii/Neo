"""Run the Neo tool-routing use case against a local checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import BertTokenizerFast

from scripts.train_neo import NeoModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
    model = NeoModel(len(tokenizer)).cuda().eval()
    weights = checkpoint / "pytorch_model.bin"
    model.load_state_dict(torch.load(weights, map_location="cuda"))
    tools = json.loads((Path(__file__).parent / "tool_catalog.json").read_text())
    options = [item["name"] for item in tools]
    prompt = "State: " + args.message + "\nQuestion: Which available tool should handle this request?\nOptions: "
    prompt += " | ".join(f"{item['name']}: {item['description']}" for item in tools)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    encoded = {key: value.cuda() for key, value in encoded.items()}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model.encoder(**encoded).last_hidden_state[:, 0]
        probabilities = torch.softmax(model.choice(logits)[0, :len(options)].float(), dim=-1)
    result = {"tool": options[int(probabilities.argmax())], "probabilities": dict(zip(options, probabilities.tolist()))}
    result["requires_review"] = max(result["probabilities"].values()) < 0.90
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
