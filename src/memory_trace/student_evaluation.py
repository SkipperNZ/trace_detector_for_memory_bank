"""One final evaluation after the student, hyperparameters, and routing policies freeze."""

import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .dataset import SENTIMENTS
from .io import digest, read_jsonl, write_json, write_jsonl
from .sentiment import load_student, head_logits, softmax, validate_prediction_inputs
from .sentiment_encoder import SentimentEncoder
from .sentiment_metrics import (
    classification,
    clustered_intervals,
    routing_metrics,
    prevalence_experiment,
)
from .student_data import text_of
from .student_study import read_json, load_study, split_targets


def attempt_records(call):
    return [a for record in call.get("previous_runs", []) + [call] for a in record["attempts"]]


def teacher_summary(base):
    import numpy as np

    calls = read_jsonl(Path(base) / "teacher/calls.jsonl")
    total_tokens = Counter()
    attempt_seconds = []
    for call in calls:
        for attempt in attempt_records(call):
            attempt_seconds.append(attempt["elapsed_seconds"])
            response = attempt.get("response") or {}
            total_tokens.update(
                {
                    k: v
                    for k, v in response.get("usage", {}).items()
                    if k in {"prompt_tokens", "completion_tokens", "total_tokens"}
                    and isinstance(v, int)
                }
            )
    labels = read_jsonl(Path(base) / "teacher/labels.jsonl")
    by_id = {r["input_id"]: r["sentiment_label"] for r in labels}
    counts = {}
    for split in ("train", "calibration", "test"):
        examples = read_jsonl(Path(base) / f"data/{split}.jsonl")
        counts[split] = {
            "eligible": len(examples),
            "labelled": sum(ex["input_id"] in by_id for ex in examples),
            "labels": dict(
                Counter(by_id[ex["input_id"]] for ex in examples if ex["input_id"] in by_id)
            ),
        }
    reuse = read_json(Path(base) / "cache-reuse.json")
    return {
        "calls": len(calls),
        "attempts_including_original_cached_and_retries": len(attempt_seconds),
        "reused_calls": len(reuse),
        "call_status_counts": dict(Counter(c["runtime_status"] for c in calls)),
        "tokens": dict(total_tokens),
        "sum_attempt_seconds_including_reused": float(sum(attempt_seconds)),
        "p50_attempt_seconds": float(np.quantile(attempt_seconds, 0.5)),
        "p95_attempt_seconds": float(np.quantile(attempt_seconds, 0.95)),
        "splits": counts,
    }


def evaluate_study(base, *, device="cpu"):
    import numpy as np
    import psutil
    import torch
    from .sentiment import TfidfFeatures

    base = Path(base)
    policy = load_study(base)
    selection = read_json(base / "selection.json")
    if not selection["selected_before_test"] or selection["study_policy_hash"] != digest(policy):
        raise ValueError("Freeze the final student and routing policy first")
    if (base / "evaluation.json").exists():
        result = read_json(base / "evaluation.json")
        if result["selection_hash"] != digest(selection):
            raise ValueError("Evaluation already belongs to a different selection")
        return result
    start = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "selection_hash": digest(selection),
        "selected_at": selection["created_at"],
    }
    write_json(base / "evaluation-start.json", start)
    torch.set_num_threads(8)
    examples = read_jsonl(base / "data/test.jsonl")
    known, y, _, missing = split_targets(base, "test", policy)
    known_ids = {ex["input_id"] for ex in known}
    valid_indices = [i for i, ex in enumerate(examples) if ex["input_id"] in known_ids]
    y_by_id = {ex["input_id"]: value for ex, value in zip(known, y)}
    calls = {r["input_id"]: r for r in read_jsonl(base / "teacher/calls.jsonl")}
    teacher_seconds = {
        ex["input_id"]: sum(a["elapsed_seconds"] for a in attempt_records(calls[ex["input_id"]]))
        for ex in examples
    }
    process = psutil.Process()
    reports = []
    probabilities_by_name = {}
    baseline = read_json(base / "baseline-selection.json")
    candidates = list(baseline["candidates"])
    if baseline["finetune_required"]:
        candidates.append(read_json(base / "finetune-selection.json")["winner"])
    # All candidates were fixed using calibration. Test metrics cannot choose a winner.
    for candidate in candidates:
        started = time.perf_counter()
        artifact = load_student(base / candidate["path"])
        if artifact["model_version"] != candidate["model_version"]:
            raise ValueError("Candidate artifact changed after calibration")
        validate_prediction_inputs(examples, artifact)
        encoder = None
        extractor = None
        if artifact["kind"] == "tfidf":
            extractor = TfidfFeatures(artifact["features"])
        elif artifact["kind"] in {"encoder", "finetuned"}:
            config = artifact["encoder"]
            encoder = SentimentEncoder(
                config["model"],
                config["revision"],
                device=device,
                batch_size=policy.get("encoder_batch_size", 32),
                prefix=config.get("prefix"),
                max_length=config.get("max_length"),
                finetuned_path=(
                    base / candidate["path"] / "encoder"
                    if artifact["kind"] == "finetuned"
                    else None
                ),
                finetuned_sha=config.get("finetuned_sha256"),
            )
            if encoder.config != config:
                raise ValueError("Evaluation encoder differs from candidate")
        load_seconds = time.perf_counter() - started

        def score(rows):
            if artifact["kind"] == "majority":
                vectors = np.zeros((len(rows), 1))
            elif extractor is not None:
                vectors = extractor.transform([text_of(ex) for ex in rows])
            else:
                vectors, _ = encoder.encode(rows)
            return softmax(
                head_logits(vectors, artifact["head"]), artifact["calibration"]["temperature"]
            )

        # Warm up separately, then measure real text preprocessing and model inference.
        score(examples[: min(3, len(examples))])
        started = time.perf_counter()
        probabilities = score(examples)
        batch_seconds = time.perf_counter() - started
        latency = []
        rss = [process.memory_info().rss]
        for ex in sorted(examples, key=lambda ex: digest([42, "latency-order", ex["input_id"]])):
            started = time.perf_counter()
            score([ex])
            latency.append(time.perf_counter() - started)
            rss.append(process.memory_info().rss)
        metrics = classification(y, probabilities[valid_indices])
        interval = clustered_intervals(
            y,
            probabilities[valid_indices],
            [ex["group_id"] for ex in known],
            replicates=policy["bootstrap_replicates"],
            seed=policy["seed"],
        )
        record = {
            "name": candidate["name"],
            "kind": candidate["kind"],
            "size": candidate["size"],
            "path": candidate["path"],
            "model_version": artifact["model_version"],
            "selected_on_calibration": candidate["name"] == selection["winner"]["name"],
            "training_messages": candidate["training_messages"],
            "artifact_bytes": candidate["artifact_bytes"],
            "eligible_test_messages": len(examples),
            "known_teacher_test_messages": len(known),
            "teacher_missing_input_ids": missing,
            "student_missing": 0,
            "metrics": metrics,
            "intervals": interval,
            "joint_agreement_all_eligible": float(
                (probabilities[valid_indices].argmax(axis=1) == np.asarray(y)).sum() / len(examples)
            ),
            "runtime": {
                "device": encoder.device if encoder else "cpu",
                "torch_threads": 8,
                "model_load_seconds": load_seconds,
                "batch_seconds": batch_seconds,
                "batch_messages_per_second": len(examples) / batch_seconds,
                "single_message_p50_seconds": float(np.quantile(latency, 0.5)),
                "single_message_p95_seconds": float(np.quantile(latency, 0.95)),
                "single_message_mean_seconds": float(np.mean(latency)),
                "observed_process_rss_max_bytes": max(rss),
                "single_message_samples": len(latency),
                "notes": "Warm timings include text tokenization/features and head, exclude disk/model load. RSS includes Python and loaded libraries; not isolated model memory.",
            },
        }
        reports.append(record)
        probabilities_by_name[candidate["name"]] = probabilities
        write_jsonl(
            base / f"test-predictions/{candidate['name']}.jsonl",
            [
                {
                    "input_id": ex["input_id"],
                    "sentiment_label": SENTIMENTS[int(p.argmax())],
                    "probabilities": p.tolist(),
                    "model_version": artifact["model_version"],
                }
                for ex, p in zip(examples, probabilities)
            ],
        )
        write_json(base / "evaluation-progress.json", reports)
        print(
            f"Test {candidate['name']}: agreement {metrics['agreement']:.4f}, macro-F1 {metrics['macro_f1']:.4f}",
            flush=True,
        )
        del encoder, extractor
    winner = next(r for r in reports if r["selected_on_calibration"])
    winner_probabilities = probabilities_by_name[winner["name"]]
    scores = winner_probabilities[:, 0]
    source_path = base / "data/source_labels.jsonl"
    source_labels = (
        {r["input_id"]: r["sentiment_label"] for r in read_jsonl(source_path)}
        if source_path.exists()
        else {}
    )
    disagreements = []
    for ex, probabilities in zip(examples, winner_probabilities):
        predicted = int(probabilities.argmax())
        expected = y_by_id.get(ex["input_id"])
        if expected is None or predicted != expected:
            answer = calls[ex["input_id"]].get("answer") or {}
            disagreements.append(
                {
                    "input_id": ex["input_id"],
                    "source_row_id": ex["target_event_id"],
                    "group_id": ex["group_id"],
                    "text": text_of(ex),
                    "student_label": SENTIMENTS[predicted],
                    "teacher_label": SENTIMENTS[expected] if expected is not None else None,
                    "teacher_reason": answer.get("reason"),
                    "source_label": source_labels.get(ex["input_id"]),
                    "probabilities": probabilities.tolist(),
                    "category": (
                        "teacher_unresolved" if expected is None else "student_teacher_disagreement"
                    ),
                }
            )
    write_jsonl(base / "test-disagreements.jsonl", disagreements)
    # Conservative online overhead: mean single-message inference, not batched throughput.
    student_seconds = winner["runtime"]["single_message_mean_seconds"] * len(examples)
    routing = []
    prevalence = []
    for routing_policy in selection["routing_policies"]:
        for audit in [0.0] + selection["audit_rates"]:
            routing.append(
                routing_metrics(
                    examples,
                    y_by_id,
                    scores,
                    routing_policy,
                    teacher_seconds,
                    student_seconds=student_seconds,
                    audit_rate=audit,
                )
            )
        for audit in selection["audit_rates"]:
            prevalence.append(
                {
                    "target_calibration_recall": routing_policy["target_calibration_recall"],
                    "audit_rate": audit,
                    **prevalence_experiment(
                        examples,
                        y_by_id,
                        scores,
                        routing_policy,
                        audit_rate=audit,
                        replicates=policy["prevalence_replicates"],
                        seed=policy["seed"],
                    ),
                }
            )
    result = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "selection_hash": digest(selection),
        "selection": selection,
        "teacher": teacher_summary(base),
        "models": reports,
        "routing": routing,
        "prevalence": prevalence,
        "scope": "Three-class agreement with a fixed Qwen teacher; no human gold. Final test was excluded from fitting, hyperparameters, temperature and threshold selection.",
        "deployment_mode": "shadow",
    }
    write_json(base / "evaluation.json", result)
    return result
