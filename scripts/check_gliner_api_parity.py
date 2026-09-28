"""Compare the benchmark scoring adapter to native GLiNER classify_text on calibration."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/gliner25-decide-v1"))
    parser.add_argument("--source", type=Path, default=Path("runs/feedback-v1"))
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from memory_trace.feedback import TASKS, CLASSES, student_inputs
    from memory_trace.feedback_data import read
    from memory_trace.feedback_models import load_partition
    from memory_trace.gliner_feedback import DecidePredictor, probabilities
    from memory_trace.io import write_json

    torch.set_num_threads(8)
    manifest = read(args.out / "manifest.json")
    predictor = DecidePredictor(manifest["snapshot"], manifest["policy"], device="cpu")
    rows, _ = load_partition(args.source, "calibration")
    rendered = student_inputs(rows)
    # A fixed set of short calibration inputs plus synthetic boundary cases; no test selection.
    texts = [r["context"][0]["text"] for r in rendered if len(r["context"][0]["text"]) < 600][:12]
    texts += [
        "Your patch crashes on startup; please fix it.",
        "You ignored my instructions again! Fix the wrong file you changed.",
        "This answer is useless.",
        "That is still not right.",
        "Please implement JSON export.",
        "Great, now add a dark theme.",
        "I am frustrated with my internet provider.",
        "You used red although I requested blue.",
    ]
    checks = []
    for schema_name, schema in manifest["policy"]["schemas"].items():
        native_schema = {
            task: {"labels": spec["labels"], "prompt": spec["instruction"]}
            for task, spec in schema.items()
        }
        records = predictor.predict(texts, schema_name)
        if any(row["windows"] != 1 for row in records):
            raise ValueError("Native one-pass parity must use single-window inputs")
        predictions = probabilities(records, "mean").argmax(axis=2)
        mismatches = 0
        native_counts = {t: {c: 0 for c in CLASSES} for t in TASKS}
        for text, prediction in zip(texts, predictions, strict=True):
            native = predictor.classifier.model.classify_text(text, native_schema)
            for j, task in enumerate(TASKS):
                native_counts[task][native[task]] += 1
                mismatches += native[task] != CLASSES[int(prediction[j])]
        checks.append(
            {
                "schema": schema_name,
                "messages": len(texts),
                "task_decisions": len(texts) * len(TASKS),
                "label_mismatches": int(mismatches),
                "native_predicted_counts": native_counts,
            }
        )
    result = {
        "passed": all(r["label_mismatches"] == 0 for r in checks),
        "checks": checks,
        "scope": "Public Classifier scoring adapter vs native model.classify_text, using calibration and synthetic inputs. Checks serialization/decoding, not semantic accuracy.",
    }
    write_json(args.out / "native-api-parity.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise ValueError("Adapter disagrees with native GLiNER API")


if __name__ == "__main__":
    main()
