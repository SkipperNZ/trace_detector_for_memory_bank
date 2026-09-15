"""Exercise selection, evaluation and report export on SYNTHETIC fixtures only."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
from pathlib import Path
import torch
from memory_trace.io import digest, write_json, write_jsonl
from memory_trace.sentiment import make_sentiment_input, load_student
from memory_trace.sentiment_encoder import SentimentEncoder, cache_embeddings
from memory_trace.student_study import freeze_study, fit_baselines, freeze_winner
from memory_trace.student_evaluation import evaluate_study
from memory_trace.student_report import render_report

torch.set_num_threads(8)
base = Path("runs/student-study-synthetic-smoke")
if base.exists():
    raise ValueError("Synthetic full-study fixture already exists")
source = json.loads(Path("runs/sentiment-students-v1/encoder-source.json").read_text())
policy = json.loads(Path("configs/student-study.json").read_text())
policy.update(
    teacher_version="SYNTHETIC-MOCK-NOT-QWEN",
    learning_curve_messages=[18],
    tfidf_C=[1.0],
    encoder_C=[1.0],
    class_weight_balanced=[False],
    bootstrap_replicates=50,
    prevalence_replicates=100,
)
splits, labels, calls = {}, [], []
for split, count in (("train", 18), ("calibration", 9), ("test", 6)):
    rows = []
    for i in range(count):
        text = (
            "Broken code, a bad bug, frustrating failure.",
            "Please open the configuration file.",
            "Excellent result, wonderful work, thank you!",
        )[i % 3]
        row = make_sentiment_input(
            text, tenant_id="synthetic-test", session_id=f"{split}-{i}", message_id=f"{split}-{i}"
        )
        row["split"] = split
        rows.append(row)
        label = ("NEGATIVE", "NEUTRAL", "POSITIVE")[i % 3]
        missing = split == "test" and i == 5
        if not missing:
            labels.append(
                {
                    "input_id": row["input_id"],
                    "task": "sentiment",
                    "label_source": "llm_judge",
                    "sentiment_label": label,
                    "judge_version": policy["teacher_version"],
                    "synthetic": True,
                }
            )
        calls.append(
            {
                "input_id": row["input_id"],
                "runtime_status": "error" if missing else "ok",
                "answer": None if missing else {"label": label, "reason": "Synthetic fixture"},
                "attempts": [{"elapsed_seconds": 0.02, "response": {"usage": {}}}],
                "synthetic": True,
            }
        )
    write_jsonl(base / f"data/{split}.jsonl", rows)
    splits[split] = {"hash": digest(rows), "messages": count, "sessions": count}
write_jsonl(base / "teacher/labels.jsonl", labels)
write_jsonl(base / "teacher/calls.jsonl", calls)
write_json(
    base / "teacher/summary.json",
    {"unattempted_calls": 0, "judge_version": policy["teacher_version"]},
)
write_json(base / "cache-reuse.json", [])
write_json(base / "encoder-source.json", source)
write_json(base / "policy.json", policy)
write_json(
    base / "data/design.json",
    {
        "judge_version": policy["teacher_version"],
        "splits": splits,
        "original_messages": 33,
        "retained_messages": 33,
        "exclusion_counts": {},
        "synthetic": True,
    },
)
freeze_study(base, base / "policy.json")
encoder = SentimentEncoder(source["model"], source["revision"], device="cpu")
from memory_trace.io import read_jsonl

for split in ("train", "calibration"):
    cache_embeddings(
        read_jsonl(base / f"data/{split}.jsonl"), encoder, base / f"embeddings/{split}"
    )
del encoder
baseline = fit_baselines(base)
assert not baseline["finetune_required"], "Separable synthetic fixture should not need fine-tuning"
selection = freeze_winner(base)
evaluation = evaluate_study(base, device="cpu")
assert evaluation["selection_hash"] == digest(selection)
assert all(r["known_teacher_test_messages"] == 5 for r in evaluation["models"])
assert any(
    r["category"] == "teacher_unresolved" for r in read_jsonl(base / "test-disagreements.jsonl")
)
render_report(base, report_path=base / "synthetic-report.md", model_output=base / "portable")
assert load_student(base / "portable")["model_version"] == selection["winner"]["model_version"]
write_json(
    base / "smoke-result.json",
    {"synthetic_only": True, "selection_evaluation_report_export": "passed"},
)
print(
    "SYNTHETIC study plumbing passed, including a missing teacher answer and portable export.",
    flush=True,
)
