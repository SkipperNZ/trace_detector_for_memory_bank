"""Freeze the complete student experiment before teacher labeling or model fitting."""

from pathlib import Path
import json
from memory_trace.student_data import freeze_data
from memory_trace.io import read_jsonl, write_jsonl


if __name__ == "__main__":
    out = Path("runs/sentiment-students-v1")
    if (out / "data/design.json").exists():
        raise ValueError("Student design already frozen")
    prior = json.loads(
        Path("runs/sentiment-tuning-v1/fewshot-provenance.json").read_text(encoding="utf-8")
    )
    ids = {r["input_id"] for r in prior["examples"]}
    write_jsonl(
        out / "reviewed-fewshot.jsonl",
        [r for r in read_jsonl("runs/sentiment/examples.jsonl") if r["input_id"] in ids],
    )
    report = freeze_data(
        Path("runs/sentiment/examples.jsonl"),
        [
            Path("runs/sentiment/pilot.jsonl"),
            Path("runs/sentiment-tuning-v1/holdout.jsonl"),
            out / "reviewed-fewshot.jsonl",
        ],
        Path("configs/judge.local.json"),
        Path("runs/sentiment-students-v1/data"),
    )
    retained_ids = {r["input_id"] for r in read_jsonl(out / "data/examples.jsonl")}
    write_jsonl(
        out / "data/source_labels.jsonl",
        [
            r
            for r in read_jsonl("runs/sentiment/source_labels.jsonl")
            if r["input_id"] in retained_ids
        ],
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("retained_messages", "splits", "exclusion_counts", "near_duplicate_pairs")
            },
            indent=2,
        )
    )
