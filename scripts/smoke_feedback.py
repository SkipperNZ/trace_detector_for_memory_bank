"""Opt-in real E5/backprop/serialization test on explicitly synthetic fixtures."""

import argparse
import gc
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from memory_trace.feedback import TASKS, CLASSES, feedback_input
from memory_trace.feedback_data import read
from memory_trace.feedback_models import fit_candidate, select_feedback, FeedbackPredictor
from memory_trace.feedback_report import evaluate_feedback, report_feedback
from memory_trace.io import digest, write_json, write_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("runs/feedback-synthetic-smoke"))
    a = p.parse_args()
    base = a.base
    if (base / "design.json").exists():
        raise ValueError("Use a new synthetic output directory")
    policy = read("configs/feedback-study.json")
    policy.update(
        epochs=1,
        C=[1],
        balanced=[False],
        bootstrap_replicates=20,
        latency_sample_messages=3,
        effective_batch_size=4,
    )
    policy["encoders"] = policy["encoders"][:1]
    examples = {}
    labels = []
    for split, n in (("train", 12), ("calibration", 6), ("test", 6)):
        rows = []
        for i in range(n):
            text = (
                "Please add a download button.",
                "You ignored my requirement again! Fix your patch.",
                "That is still not right.",
            )[i % 3] + f" Fixture {split} item {i}."
            ex = feedback_input(
                {
                    "text": text,
                    "tenant_id": "SYNTHETIC_FIXTURE_ONLY",
                    "session_id": f"{split}-{i}",
                    "message_id": str(i),
                }
            )
            ex["split"] = split
            rows.append(ex)
            labels.append(
                {
                    "input_id": ex["input_id"],
                    "task": "feedback",
                    "judge_version": "synthetic-only",
                    "label_source": "llm_judge",
                    "synthetic": True,
                    "signals": {t: {"label": CLASSES[i % 3]} for t in TASKS},
                    "reason_tags": [],
                }
            )
        examples[split] = rows
        write_jsonl(base / f"data/{split}.jsonl", rows)
    design = {
        "judge_version": "synthetic-only",
        "policy": policy,
        "policy_hash": digest(policy),
        "synthetic": True,
        "exclusions": {},
        "splits": {
            s: {"hash": digest(r), "messages": len(r), "sessions": len(r)}
            for s, r in examples.items()
        },
    }
    write_json(base / "design.json", design)
    write_jsonl(base / "labels.jsonl", labels)
    write_json(
        base / "labels-frozen.json",
        {"judge_version": "synthetic-only", "labels_hash": digest(labels)},
    )
    write_json(
        base / "census.json",
        {
            "messages": 24,
            "canonical_requests": 24,
            "labelled_messages": 24,
            "unlabelled_messages": 0,
            "reason_tags": {},
            "tasks": {
                t: {
                    "counts": dict.fromkeys(CLASSES, 8),
                    "yes_fraction_all_messages": 1 / 3,
                    "sessions_with_yes": 8,
                }
                for t in TASKS
            },
        },
    )
    fit_candidate(base, "tfidf")
    fit_candidate(base, "e5")
    # Verify both neural serializations even if the small TF-IDF wins this fixture.
    for name in ("e5-frozen", "e5-finetuned"):
        model = FeedbackPredictor(base / f"models/{name}")
        predictions = model.predict(examples["test"])
        assert len(predictions) == 6
        for r in predictions:
            for t in TASKS:
                assert abs(sum(r["signals"][t]["probabilities"].values()) - 1) < 1e-6
        del model
        gc.collect()
    select_feedback(base)
    evaluate_feedback(base)
    report_feedback(base, base / "export")
    write_json(
        base / "smoke-complete.json",
        {"synthetic": True, "real_e5_backprop": True, "both_heads_and_serializations": True},
    )
    print("Synthetic feedback smoke passed.")


if __name__ == "__main__":
    main()
