"""Reproducible student study. Model selection is completed before final evaluation."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .io import digest, read_jsonl, write_json, write_jsonl
from .sentiment import (
    TfidfFeatures,
    targets,
    provenance,
    fit_head,
    head_logits,
    fit_temperature,
    softmax,
    save_student,
    load_student,
)
from .sentiment_metrics import classification, nested_training_sets, choose_thresholds
from .student_data import text_of


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze_study(base, policy_path):
    base = Path(base)
    policy = read_json(policy_path)
    design = read_json(base / "data/design.json")
    encoder = read_json(base / "encoder-source.json")
    if policy["teacher_version"] != design["judge_version"]:
        raise ValueError("Study teacher does not match data design")
    if (policy["encoder_model"], policy["encoder_revision"]) != (
        encoder["model"],
        encoder["revision"],
    ):
        raise ValueError("Study encoder differs from pinned source")
    frozen = {
        "policy": policy,
        "policy_hash": digest(policy),
        "data_design_hash": digest(design),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_used_for_selection": False,
    }
    target = base / "study-design.json"
    if target.exists():
        old = read_json(target)
        if (
            old["policy_hash"] != frozen["policy_hash"]
            or old["data_design_hash"] != frozen["data_design_hash"]
        ):
            raise ValueError("Study already frozen with different design")
        return old
    write_json(target, frozen)
    return frozen


def load_study(base, *, require_labels=True):
    base = Path(base)
    frozen = read_json(base / "study-design.json")
    if digest(frozen["policy"]) != frozen["policy_hash"]:
        raise ValueError("Frozen study policy checksum mismatch")
    design = read_json(base / "data/design.json")
    if digest(design) != frozen["data_design_hash"]:
        raise ValueError("Frozen data design changed")
    for split in ("train", "calibration", "test"):
        if digest(read_jsonl(base / f"data/{split}.jsonl")) != design["splits"][split]["hash"]:
            raise ValueError("Frozen split changed")
    if require_labels:
        summary = read_json(base / "teacher/summary.json")
        if (
            summary["unattempted_calls"]
            or summary["judge_version"] != frozen["policy"]["teacher_version"]
        ):
            raise ValueError("Complete frozen teacher run before fitting")
    return frozen["policy"]


def split_targets(base, split, policy):
    examples = read_jsonl(Path(base) / f"data/{split}.jsonl")
    ids = {ex["input_id"] for ex in examples}
    # Only the named partition's targets are returned to a training/calibration stage.
    labels = [r for r in read_jsonl(Path(base) / "teacher/labels.jsonl") if r["input_id"] in ids]
    valid, y, missing = targets(examples, labels, judge_version=policy["teacher_version"])
    return valid, y, labels, missing


def cached_vectors(base, split, selected):
    import numpy as np

    folder = Path(base) / f"embeddings/{split}"
    manifest = read_json(folder / "manifest.json")
    examples = read_jsonl(Path(base) / f"data/{split}.jsonl")
    if digest(examples) != manifest["inputs_hash"]:
        raise ValueError("Embedding cache input mismatch")
    vectors = np.load(folder / "vectors.npy", allow_pickle=False)
    if digest(vectors.tolist()) != manifest["vectors_hash"]:
        raise ValueError("Embedding checksum mismatch")
    indices = {key: i for i, key in enumerate(manifest["input_ids"])}
    return vectors[[indices[ex["input_id"]] for ex in selected]], manifest


def fit_baselines(base):
    import numpy as np

    base = Path(base)
    policy = load_study(base)
    if (base / "baseline-selection.json").exists():
        return read_json(base / "baseline-selection.json")
    train, train_y, labels, missing_train = split_targets(base, "train", policy)
    calibration, cal_y, _, missing_cal = split_targets(base, "calibration", policy)
    y_by_id = {ex["input_id"]: y for ex, y in zip(train, train_y)}
    sets = nested_training_sets(train, policy["learning_curve_messages"], policy["seed"])
    write_json(
        base / "learning-curve-design.json",
        {
            name: {
                "input_ids": [ex["input_id"] for ex in rows],
                "groups": sorted({ex["group_id"] for ex in rows}),
                "messages": len(rows),
            }
            for name, rows in sets.items()
        },
    )
    candidates = []
    counts = np.bincount(train_y, minlength=3)
    prior = (counts + 1) / (counts.sum() + 3)
    majority_head = {"coef": [[0.0], [0.0], [0.0]], "intercept": np.log(prior).tolist()}
    majority_probs = np.tile(prior, (len(cal_y), 1))
    write_jsonl(
        base / "calibration/majority-all.jsonl",
        [
            {"input_id": ex["input_id"], "probabilities": p.tolist()}
            for ex, p in zip(calibration, majority_probs)
        ],
    )
    model = save_student(
        base / "models/majority-all",
        kind="majority",
        training=provenance(train, labels),
        head=majority_head,
    )
    candidates.append(
        {
            "name": "majority-all",
            "kind": "majority",
            "size": "all",
            "training_messages": len(train),
            "calibration": classification(cal_y, majority_probs),
            "path": "models/majority-all",
            "model_version": model["model_version"],
            "artifact_bytes": (base / "models/majority-all/model.json").stat().st_size,
        }
    )
    cached_train, embedding_manifest = cached_vectors(base, "train", train)
    cached_cal, calibration_manifest = cached_vectors(base, "calibration", calibration)
    if embedding_manifest["encoder"] != calibration_manifest["encoder"]:
        raise ValueError("Train and calibration embedding serializers differ")
    encoder_config = embedding_manifest["encoder"]
    if (encoder_config["model"], encoder_config["revision"]) != (
        policy["encoder_model"],
        policy["encoder_revision"],
    ):
        raise ValueError("Cached encoder differs from frozen study policy")
    for key in ("prefix", "max_length"):
        if f"encoder_{key}" in policy and encoder_config[key] != policy[f"encoder_{key}"]:
            raise ValueError("Cached encoder preprocessing differs from frozen study policy")
    lookup = {ex["input_id"]: i for i, ex in enumerate(train)}
    for size, rows in sets.items():
        y = [y_by_id[ex["input_id"]] for ex in rows]
        training = provenance(rows, labels)
        for kind in ("tfidf", "encoder"):
            print(f"Fit {kind}, group-preserving training size {len(rows)} ({size})", flush=True)
            started = time.perf_counter()
            features = None
            if kind == "tfidf":
                extractor = TfidfFeatures()
                vectors = extractor.fit_transform([text_of(ex) for ex in rows])
                cal_vectors = extractor.transform([text_of(ex) for ex in calibration])
                features = extractor.state()
            else:
                vectors = cached_train[[lookup[ex["input_id"]] for ex in rows]]
                cal_vectors = cached_cal
            feature_seconds = time.perf_counter() - started
            trials = []
            best = None
            for c in policy[f"{kind}_C"]:
                for balanced in policy["class_weight_balanced"]:
                    started = time.perf_counter()
                    head = fit_head(vectors, y, c=c, balanced=balanced)
                    logits = head_logits(cal_vectors, head)
                    metrics = classification(cal_y, softmax(logits))
                    trial = {
                        "C": c,
                        "balanced": balanced,
                        "calibration": metrics,
                        "head_fit_seconds": time.perf_counter() - started,
                        "iterations": head["iterations"],
                    }
                    trials.append(trial)
                    rank = (metrics["macro_f1"], metrics["agreement"], -c, not balanced)
                    if best is None or rank > best[0]:
                        best = (rank, head, logits, trial)
            _, head, logits, trial = best
            calibration_fit = fit_temperature(logits, cal_y)
            probabilities = softmax(logits, calibration_fit["temperature"])
            name = f"{kind}-{size}"
            artifact = save_student(
                base / f"models/{name}",
                kind=kind,
                training=training,
                head=head,
                features=features,
                encoder=embedding_manifest["encoder"] if kind == "encoder" else None,
                calibration=calibration_fit,
            )
            record = {
                "name": name,
                "kind": kind,
                "size": size,
                "training_messages": len(rows),
                "training_sessions": len(training["training_groups"]),
                "calibration": classification(cal_y, probabilities),
                "temperature": calibration_fit,
                "path": f"models/{name}",
                "model_version": artifact["model_version"],
                "artifact_payload_bytes": (base / f"models/{name}/model.json").stat().st_size,
                "backbone_bytes": (
                    embedding_manifest["encoder_disk_bytes"] if kind == "encoder" else 0
                ),
                "artifact_bytes": (base / f"models/{name}/model.json").stat().st_size
                + (embedding_manifest["encoder_disk_bytes"] if kind == "encoder" else 0),
                "feature_seconds": feature_seconds,
                "trials": trials,
            }
            candidates.append(record)
            write_json(base / "baseline-candidates.json", candidates)
            write_jsonl(
                base / f"calibration/{name}.jsonl",
                [
                    {"input_id": ex["input_id"], "probabilities": p.tolist()}
                    for ex, p in zip(calibration, probabilities)
                ],
            )
            print(f"Selected {name}: macro-F1 {record['calibration']['macro_f1']:.4f}", flush=True)
    eligible = [r for r in candidates if r["size"] == "all"]
    winner = max(
        eligible,
        key=lambda r: (
            r["calibration"]["macro_f1"],
            r["calibration"]["agreement"],
            -r["artifact_bytes"],
        ),
    )
    should_finetune = policy.get("finetune", {}).get("force", False) or (
        winner["calibration"]["macro_f1"] < 0.90
        and winner["calibration"]["macro_f1"] >= candidates[0]["calibration"]["macro_f1"] + 0.10
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "winner": winner,
        "candidates": candidates,
        "finetune_required": should_finetune,
        "missing_train_teacher_labels": missing_train,
        "missing_calibration_teacher_labels": missing_cal,
        "policy_hash": digest(policy),
        "test_used": False,
    }
    write_json(base / "baseline-selection.json", result)
    return result


def freeze_winner(base):
    base = Path(base)
    policy = load_study(base)
    if (base / "selection.json").exists():
        return read_json(base / "selection.json")
    baseline = read_json(base / "baseline-selection.json")
    candidates = [r for r in baseline["candidates"] if r["size"] == "all"]
    if baseline["finetune_required"]:
        tuned = read_json(base / "finetune-selection.json")
        candidates.append(tuned["winner"])
    winner = max(
        candidates,
        key=lambda r: (
            r["calibration"]["macro_f1"],
            r["calibration"]["agreement"],
            -r["artifact_bytes"],
        ),
    )
    examples, y, _, _ = split_targets(base, "calibration", policy)
    cal = read_jsonl(base / f"calibration/{winner['name']}.jsonl")
    predictions = {r["input_id"]: r["probabilities"] for r in cal}
    scores = [predictions[ex["input_id"]][0] for ex in examples]
    policies = choose_thresholds(y, scores, recalls=policy["recall_targets"])
    artifact = load_student(base / winner["path"])
    if artifact["model_version"] != winner["model_version"]:
        raise ValueError("Selected student artifact changed")
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "winner": winner,
        "candidates": candidates,
        "routing_policies": policies,
        "audit_rates": policy["audit_rates"],
        "selected_before_test": True,
        "test_used_for_selection": False,
        "study_policy_hash": digest(policy),
        "teacher_version": policy["teacher_version"],
    }
    write_json(base / "selection.json", result)
    return result
