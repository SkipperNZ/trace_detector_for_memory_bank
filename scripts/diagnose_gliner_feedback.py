"""Exploratory calibration-only GLiNER ablations; never replaces the frozen test benchmark."""

import argparse
import copy
import gc
import os
import sys
import time
from pathlib import Path


def main():
    # Third-party model loading prints Unicode even with a redirected Windows console.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/gliner25-diagnosis-v1"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import torch
    from gliner2.classification import ClassificationSchema
    from sklearn.metrics import average_precision_score
    from memory_trace.feedback import CLASSES, TASKS, student_inputs
    from memory_trace.feedback_data import read
    from memory_trace.feedback_models import load_partition, task_metrics
    from memory_trace.gliner_feedback import DecidePredictor, split_complete, now, yes_metrics
    from memory_trace.io import digest, read_jsonl, write_json

    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = read("runs/gliner25-decide-v1/manifest.json")
    policy = manifest["policy"]
    probes = [
        "This is a terrible answer. You keep ignoring my instructions.",
        "Your answer is useless and I am disappointed in your work.",
        "You used red although I requested blue. Please correct it.",
        "Your patch crashes on startup; please fix it.",
        "The bug is still present after your fix.",
        "You forgot to add the tests I requested.",
        "Please implement JSON export.",
        "Great work, thank you!",
        "I am frustrated with my internet provider.",
        "Please help debug a crash in a library I downloaded.",
        "Change the color to blue.",
        "I hate this answer. It does not do what I asked.",
    ]
    names = {
        "dissatisfaction": dict(
            zip(CLASSES, ["no complaint", "complaint about assistant", "ambiguous complaint"])
        ),
        "correction": dict(
            zip(
                CLASSES,
                [
                    "new request without correction",
                    "correction of assistant error",
                    "ambiguous correction",
                ],
            )
        ),
    }
    variants = [
        {"name": "original", "descriptions": True},
        {"name": "separate_heads", "descriptions": True, "separate": True},
        {"name": "semantic_names", "descriptions": True, "rename": True},
        {"name": "no_descriptions", "descriptions": False},
        {"name": "semantic_names_no_descriptions", "descriptions": False, "rename": True},
        {"name": "binary_without_unclear", "descriptions": True, "binary": True},
    ]
    frozen = {
        "created_at": now(),
        "model_fingerprint": manifest["fingerprint"],
        "variants": variants,
        "semantic_names": names,
        "synthetic_probes": probes,
        "scope": "Exploratory calibration-only diagnostic; no new test evaluation, no weight training.",
    }
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        old = read(protocol_path)
        if {k: v for k, v in old.items() if k != "created_at"} != {
            k: v for k, v in frozen.items() if k != "created_at"
        }:
            raise ValueError("Diagnostic protocol changed; use a new output directory")
    else:
        write_json(protocol_path, frozen)
    rows, y = load_partition("runs/feedback-v1", "calibration")
    texts = [r["context"][0]["text"] for r in student_inputs(rows)]
    predictor = DecidePredictor(manifest["snapshot"], policy, device=args.device)

    def encode_many(texts, definitions):
        schema = ClassificationSchema()
        for task, spec in definitions.items():
            schema.single(
                task, spec["labels"], instruction=spec.get("instruction"), activation="softmax"
            )
        compiled = predictor.classifier.compile_schema(schema)

        def length(text):
            return predictor.classifier.model.processor.collate_fn_inference(
                [(text, compiled.build())], max_len=None
            ).input_ids.shape[1]

        limit = min(1024, length("") + 512)
        results, windows = [], []
        for start in range(0, len(texts), 16):
            current = texts[start : start + 16]
            planned = [split_complete(t, length, limit) for t in current]
            flat = [(i, text, size) for i, chunks in enumerate(planned) for text, size in chunks]
            flat.sort(key=lambda r: r[2])
            scores = predictor.classifier.batch_score(
                [r[1] for r in flat], compiled, config=predictor.config
            )
            owned = [{task: [] for task in definitions} for _ in current]
            for (owner, _, _), score in zip(flat, scores, strict=True):
                for task, spec in definitions.items():
                    owned[owner][task].append(
                        [score.logit(task, label) for label in spec["labels"]]
                    )
            for item in owned:
                normalized = {}
                for task, values in item.items():
                    logits = np.mean(values, axis=0)
                    if not np.isfinite(logits).all():
                        raise ValueError("Nonfinite diagnostic logits")
                    exponent = np.exp(logits - logits.max())
                    normalized[task] = (exponent / exponent.sum()).tolist()
                results.append(normalized)
            windows.extend(map(len, planned))
        return results, windows

    outputs = {}
    for variant in variants:
        name = variant["name"]
        path = args.out / f"{name}.json"
        if path.exists():
            outputs[name] = read(path)
            continue
        start = time.monotonic()
        print("DIAGNOSE", name, len(texts), flush=True)
        definitions = copy.deepcopy(policy["schemas"]["compact"])
        classes = CLASSES[:2] if variant.get("binary") else CLASSES
        for task, definition in definitions.items():
            new_labels = {
                names[task][c] if variant.get("rename") else c: definition["labels"][c]
                for c in classes
            }
            definition["labels"] = new_labels if variant["descriptions"] else list(new_labels)
        p = np.zeros((len(texts) + len(probes), 2, 3))
        window_counts = []
        groups = [{t: definitions[t]} for t in TASKS] if variant.get("separate") else [definitions]
        for group in groups:
            scores, counts = encode_many(texts + probes, group)
            window_counts.append(counts)
            for i, score in enumerate(scores):
                for task, values in score.items():
                    p[i, TASKS.index(task), : len(values)] = values
        cal = p[: len(texts)]
        diagnostics = {}
        for j, task in enumerate(TASKS):
            mask = y[:, j] >= 0
            if variant.get("binary"):
                mask &= y[:, j] != 2
            diagnostics[task] = {
                "average_precision_yes": float(
                    average_precision_score(y[mask, j] == 1, cal[mask, j, 1])
                ),
                "yes_at_half": yes_metrics(y[mask, j], cal[mask, j, 1], 0.5),
                "mean_max_probability": float(cal[mask, j].max(axis=1).mean()),
                "predicted_counts": {
                    c: int((cal[mask, j].argmax(axis=1) == k).sum()) for k, c in enumerate(CLASSES)
                },
            }
        result = {
            "name": name,
            "definitions": definitions,
            "seconds": time.monotonic() - start,
            "calibration": task_metrics(y, cal) if not variant.get("binary") else None,
            "diagnostics": diagnostics,
            "synthetic_predictions": p[len(texts) :].tolist(),
            "windows": {
                "messages_with_multiple_windows": int(
                    sum(max(c[i] for c in window_counts) > 1 for i in range(len(texts)))
                )
            },
        }
        np.savez_compressed(args.out / f"{name}-local-predictions.npz", probabilities=cal, labels=y)
        write_json(path, result)
        outputs[name] = result
        print(
            "DONE",
            name,
            round(result["seconds"], 1),
            result["calibration"]["mean_macro_f1"] if result["calibration"] else "binary",
            flush=True,
        )

    controls = {}
    for domain in ("review_sentiment", "agent_handoff", "document_type"):
        path = args.out / f"control-{domain}.json"
        if path.exists():
            controls[domain] = read(path)
            continue
        inputs = read_jsonl(args.out / f"{domain}.jsonl")
        # Each row supplies candidate labels only; true_label never reaches the encoder.
        grouped = {}
        for row in inputs:
            definitions = {
                t["task"]: {"labels": t["labels"]} for t in row["output"]["classifications"]
            }
            key = digest(definitions)
            grouped.setdefault(key, (definitions, []))[1].append(row)
        correct, total, maximum = 0, 0, []
        for definitions, items in grouped.values():
            scores, _ = encode_many([r["input"] for r in items], definitions)
            for row, score in zip(items, scores, strict=True):
                for task in row["output"]["classifications"]:
                    prediction = task["labels"][int(np.argmax(score[task["task"]]))]
                    correct += prediction in task["true_label"]
                    total += 1
                    maximum.append(max(score[task["task"]]))
        result = {
            "correct": int(correct),
            "decisions": total,
            "agreement": correct / total,
            "mean_max_probability": float(np.mean(maximum)),
            "scope": "Public development subset only; not the held-out advertised benchmark.",
        }
        write_json(path, result)
        controls[domain] = result
        print("CONTROL", domain, result, flush=True)
    write_json(
        args.out / "summary.json",
        {
            "scope": frozen["scope"],
            "model_revision": policy["revision"],
            "calibration_messages": len(rows),
            "variants": outputs,
            "official_development_controls": controls,
            "completed_at": now(),
        },
    )
    del predictor
    gc.collect()
    print("DIAGNOSIS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
