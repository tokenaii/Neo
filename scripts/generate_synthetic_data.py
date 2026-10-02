#!/usr/bin/env python3
"""Generate a deterministic, synthetic Neo decision corpus.

The generated records are labels for a decision model, not chat completions.
No public dataset rows are copied into this corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


TOOLS = [
    ("search", "Search public information", ["search", "find", "lookup"]),
    ("calculator", "Perform arithmetic", ["calculate", "sum", "compute"]),
    ("browser", "Interact with a web page", ["website", "page", "browse"]),
    ("code", "Write or inspect code", ["code", "bug", "program"]),
    ("file_read", "Read a local file", ["file", "document", "read"]),
    ("file_write", "Write or update a local file", ["save", "write", "update"]),
    ("database", "Query structured records", ["database", "record", "query"]),
    ("email", "Send or manage email", ["email", "mail", "message"]),
    ("calendar", "Create or inspect calendar events", ["calendar", "meeting", "schedule"]),
    ("weather", "Retrieve weather information", ["weather", "rain", "forecast"]),
    ("maps", "Find a place or route", ["map", "route", "directions"]),
    ("image", "Analyze an image", ["image", "photo", "picture"]),
    ("audio", "Transcribe or analyze audio", ["audio", "voice", "recording"]),
    ("translate", "Translate text", ["translate", "language", "translation"]),
    ("summarize", "Summarize supplied content", ["summarize", "short", "summary"]),
    ("classify", "Classify or label content", ["classify", "category", "label"]),
    ("moderate", "Check content safety", ["safe", "policy", "moderate"]),
    ("retrieve_memory", "Retrieve relevant saved context", ["memory", "remember", "context"]),
    ("notify", "Send a notification", ["notify", "alert", "notification"]),
    ("human_review", "Escalate to a human reviewer", ["human", "review", "escalate"]),
]

LANGUAGES = ("en",)
LEVELS = ("low", "medium", "high", "critical")


def stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)


def choose_tool(text: str) -> str:
    lowered = text.lower()
    scores = {
        name: sum(1 for keyword in keywords if keyword.lower() in lowered)
        for name, _, keywords in TOOLS
    }
    best, score = max(scores.items(), key=lambda item: item[1])
    return best if score else "none"


def tool_map(names: list[str]) -> dict[str, str]:
    descriptions = {name: description for name, description, _ in TOOLS}
    return {name: descriptions[name] for name in names if name in descriptions}


def make_state(rng: random.Random, language: str, target: str, index: int) -> tuple[str, list[str]]:
    name, description, keywords = next(item for item in TOOLS if item[0] == target)
    templates = [
        f"Please {keywords[0]} the information related to my request.",
        f"I need help with a {name.replace('_', ' ')} task and want the next safe step.",
        f"Can you handle this request using the appropriate capability: {description.lower()}?",
    ]
    text = templates[index % len(templates)]
    if index % 17 == 0:
        text += " The request is ambiguous; ask for clarification if needed."
    if index % 29 == 0:
        text += " Do not take an external action yet."
    return text, [name]


def distribution(options: list[str], target: str, confidence: float) -> dict[str, float]:
    if target not in options:
        target = options[0]
    remaining = max(0.0, 1.0 - confidence)
    others = [option for option in options if option != target]
    share = remaining / len(others) if others else 0.0
    result = {option: round(share, 8) for option in others}
    result[target] = round(confidence, 8)
    return result


def make_record(index: int, seed: int) -> dict:
    rng = random.Random(seed + index)
    language = "en"
    target = TOOLS[index % len(TOOLS)][0]
    state, _ = make_state(rng, language, target, index)
    names = [target]
    names.extend(tool[0] for tool in rng.sample([tool for tool in TOOLS if tool[0] != target], 5))
    if index % 11 == 0:
        names[-1] = "none"
        choice_target = "none"
    else:
        choice_target = target
    if "none" in names:
        criteria = {name: ("No tool is appropriate" if name == "none" else tool_map([name])[name]) for name in names}
    else:
        criteria = tool_map(names)
    confidence = 0.62 if index % 17 == 0 else (0.82 if index % 29 == 0 else 0.96)
    questions = {
        "next_action": {
            "type": "choice",
            "instructions": "Which available action is the correct next step? Choose none if no tool is appropriate.",
            "criteria": criteria,
        }
    }
    labels = {
        "next_action": {
            "value": choice_target,
            "probabilities": distribution(list(criteria), choice_target, confidence),
        }
    }
    if index % 2 == 0:
        urgency = "high" if index % 13 == 0 else ("medium" if index % 5 == 0 else "low")
        questions["urgency"] = {
            "type": "score",
            "instructions": "Rate the urgency of the request.",
            "criteria": list(LEVELS),
        }
        labels["urgency"] = {"value": urgency, "index": list(LEVELS).index(urgency)}
    urgent = index % 7 == 0 or index % 29 == 0
    questions["needs_review"] = {
        "type": "noul",
        "instructions": "Is human review needed before taking the next action?",
    }
    labels["needs_review"] = {"value": urgent, "probability": 0.9 if urgent else 0.06}
    return {
        "id": f"neo-english-synth-{index:07d}",
        "state": {"language": language, "text": state, "available_tools": names},
        "questions": questions,
        "labels": labels,
        "metadata": {"generator": "neo-english-synthetic-v1", "seed": seed, "source": "generated"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=400000)
    parser.add_argument("--decisions", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--output", type=Path, default=Path("data/generated/neo-english.jsonl"))
    args = parser.parse_args()
    if args.decisions < args.records * 2 or args.decisions > args.records * 3:
        raise SystemExit("For this generator, decisions must be between 2x and 3x records")
    records = [make_record(index, args.seed) for index in range(args.records)]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    split_at_1 = int(len(records) * 0.8)
    split_at_2 = int(len(records) * 0.9)
    counts = {"train": 0, "calibration": 0, "test": 0}
    with args.output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            split = "train" if index < split_at_1 else ("calibration" if index < split_at_2 else "test")
            record["metadata"]["split"] = split
            counts[split] += len(record["labels"])
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "generator": "neo-english-synthetic-v1",
        "records": len(records),
        "decisions": counts,
        "total_decisions": sum(counts.values()),
        "seed": args.seed,
        "output": str(args.output),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
