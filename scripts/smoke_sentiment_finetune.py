"""Real encoder/backprop/save-load plumbing on SYNTHETIC fixtures, not research data."""

import os
import argparse

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path
import json
import numpy as np
import torch
from memory_trace.dataset import convert
from memory_trace.io import digest, write_json, write_jsonl
from memory_trace.sentiment import provenance, fit_head, save_student, load_student, predict_student
from memory_trace.sentiment_encoder import SentimentEncoder
from memory_trace.student_study import freeze_study
from memory_trace.student_finetune import finetune

torch.set_num_threads(8)
parser = argparse.ArgumentParser()
parser.add_argument("--output", default="runs/student-finetune-synthetic-smoke")
base = Path(parser.parse_args().output)
if (base / "finetune-selection.json").exists():
    raise ValueError("Synthetic smoke already completed")
source = json.loads(
    Path("runs/sentiment-students-v1/encoder-source.json").read_text(encoding="utf-8")
)
policy = json.loads(Path("configs/student-study.json").read_text(encoding="utf-8"))
policy["teacher_version"] = "synthetic-fixture-only"
policy["finetune"] = {
    **policy["finetune"],
    "epochs": 1,
    "document_batch_size": 3,
    "gradient_accumulation_steps": 1,
}
examples = {}
labels = []
for split, count in (("train", 12), ("calibration", 6), ("test", 0)):
    rows = []
    for i in range(count):
        text = (
            "This is broken and frustrating.",
            "Please open the configuration file.",
            "Excellent work, thank you!",
        )[i % 3]
        ex, _ = convert(
            [
                {
                    "id": f"{split}-{i}",
                    "source_dataset": "SYNTHETIC_FIXTURE_NOT_RESEARCH",
                    "session_id": f"{split}-{i}",
                    "content_text": text,
                    "sentiment_label": ("NEGATIVE", "NEUTRAL", "POSITIVE")[i % 3],
                }
            ]
        )
        ex[0]["split"] = split
        rows.append(ex[0])
        labels.append(
            {
                "input_id": ex[0]["input_id"],
                "task": "sentiment",
                "label_source": "llm_judge",
                "judge_version": "synthetic-fixture-only",
                "sentiment_label": ("NEGATIVE", "NEUTRAL", "POSITIVE")[i % 3],
                "synthetic": True,
            }
        )
    examples[split] = rows
    write_jsonl(base / f"data/{split}.jsonl", rows)
write_jsonl(base / "teacher/labels.jsonl", labels)
write_json(
    base / "teacher/summary.json",
    {"unattempted_calls": 0, "judge_version": "synthetic-fixture-only", "synthetic": True},
)
write_json(
    base / "data/design.json",
    {
        "judge_version": "synthetic-fixture-only",
        "splits": {s: {"hash": digest(rows)} for s, rows in examples.items()},
        "synthetic": True,
    },
)
write_json(base / "encoder-source.json", source)
write_json(base / "policy.json", policy)
freeze_study(base, base / "policy.json")
encoder = SentimentEncoder(source["model"], source["revision"], device="cpu")
vectors, _ = encoder.encode(examples["train"])
head = fit_head(vectors, [i % 3 for i in range(12)], c=1.0)
save_student(
    base / "models/encoder-all",
    kind="encoder",
    training=provenance(examples["train"], labels),
    head=head,
    encoder=encoder.config,
)
write_json(base / "baseline-selection.json", {"finetune_required": True, "synthetic": True})
del encoder
result = finetune(base, device="cpu")
artifact = load_student(base / "models/finetuned-all")
config = artifact["encoder"]
loaded = SentimentEncoder(
    config["model"],
    config["revision"],
    device="cpu",
    finetuned_path=base / "models/finetuned-all/encoder",
    finetuned_sha=config["finetuned_sha256"],
)
predictions, _ = predict_student(examples["calibration"], artifact, encoder=loaded)
assert all(abs(sum(p["probabilities"].values()) - 1) < 1e-9 for p in predictions)
saved = [
    json.loads(line)
    for line in (base / "calibration/finetuned-all.jsonl").read_text(encoding="utf-8").splitlines()
]
np.testing.assert_allclose(
    [list(p["probabilities"].values()) for p in predictions],
    [p["probabilities"] for p in saved],
    rtol=1e-5,
    atol=1e-6,
)
write_json(
    base / "smoke-result.json",
    {
        "synthetic_only": True,
        "backpropagation": True,
        "safetensors_reload_probabilities_match": True,
        "duration_seconds": result["total_seconds"],
    },
)
print(
    "SYNTHETIC plumbing check passed: backward, save, checksum, reload, identical probabilities.",
    flush=True,
)
