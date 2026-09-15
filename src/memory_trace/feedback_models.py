"""Portable two-head feedback classifiers; selection uses calibration only."""

import gc
import time
from datetime import datetime, timezone
from pathlib import Path

from .feedback import TASKS, CLASSES, student_inputs
from .feedback_data import read
from .io import digest, index_rows, read_jsonl, write_json, write_jsonl
from .sentiment import TfidfFeatures, head_logits, fit_temperature, softmax
from .sentiment_encoder import SentimentEncoder, cache_embeddings, encoder_directory_hash
from .sentiment_metrics import classification


def load_partition(base, split):
    import numpy as np

    base = Path(base)
    design, frozen = read(base / "design.json"), read(base / "labels-frozen.json")
    labels = read_jsonl(base / "labels.jsonl")
    if (
        digest(labels) != frozen["labels_hash"]
        or frozen["judge_version"] != design["judge_version"]
    ):
        raise ValueError("Frozen feedback labels changed")
    rows = read_jsonl(base / f"data/{split}.jsonl")
    if digest(rows) != design["splits"][split]["hash"]:
        raise ValueError("Frozen feedback partition changed")
    by_id = index_rows(labels)
    y = np.full((len(rows), len(TASKS)), -1, dtype=int)
    for i, ex in enumerate(rows):
        label = by_id.get(ex["input_id"])
        if label is None:
            continue
        if (
            label["task"] != "feedback"
            or label["judge_version"] != design["judge_version"]
            or label["label_source"] != "llm_judge"
        ):
            raise ValueError("Feedback targets have incompatible provenance")
        for j, task in enumerate(TASKS):
            if task in label["signals"]:
                y[i, j] = CLASSES.index(label["signals"][task]["label"])
    return rows, y


def fit_feedback_head(vectors, y, *, c=1, balanced=False):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y)
    mask = y >= 0
    if not mask.any():
        raise ValueError("No observed labels for a feedback task")
    y = y[mask]
    observed = sorted(set(y.tolist()))
    coef = np.zeros((3, vectors.shape[1]))
    intercept = np.full(3, -30.0)
    if len(observed) == 1:
        intercept[observed[0]] = 0
        iterations = []
    else:
        model = LogisticRegression(
            C=c,
            class_weight="balanced" if balanced else None,
            max_iter=1000,
            tol=1e-5,
            random_state=42,
        )
        model.fit(vectors[mask], y)
        if len(observed) == 2:
            # Symmetric logits preserve sklearn's binary sigmoid probabilities.
            for sign, label in zip((-0.5, 0.5), model.classes_):
                coef[label] = sign * model.coef_[0]
                intercept[label] = sign * model.intercept_[0]
        else:
            coef[model.classes_] = model.coef_
            intercept[model.classes_] = model.intercept_
        iterations = model.n_iter_.tolist()
    return {
        "coef": coef.tolist(),
        "intercept": intercept.tolist(),
        "C": c,
        "balanced": balanced,
        "observed_classes": [CLASSES[i] for i in observed],
        "iterations": iterations,
    }


def task_metrics(y, probabilities):
    result = {}
    for i, task in enumerate(TASKS):
        known = y[:, i] >= 0
        if not known.any():
            raise ValueError(f"No evaluation labels for {task}")
        result[task] = {
            **classification(y[known, i], probabilities[known, i], classes=CLASSES),
            "missing_teacher_labels": int((~known).sum()),
        }
    return {
        "tasks": result,
        "mean_macro_f1": sum(m["macro_f1"] for m in result.values()) / len(TASKS),
        "mean_yes_f1": sum(m["per_class"]["yes"]["f1"] for m in result.values()) / len(TASKS),
    }


def calibrate(logits, y):
    import numpy as np

    temperatures, probabilities = {}, []
    for j, task in enumerate(TASKS):
        known = y[:, j] >= 0
        temperatures[task] = fit_temperature(logits[known, j], y[known, j])
        probabilities.append(softmax(logits[:, j], temperatures[task]["temperature"]))
    return temperatures, np.stack(probabilities, axis=1)


def save_model(base, name, kind, rows, heads, calibration, *, features=None, encoder=None):
    base = Path(base)
    design = read(base / "design.json")
    if any(ex["split"] != "train" for ex in rows):
        raise ValueError("Only train rows may enter model provenance")
    value = {
        "format": "feedback-student-v1",
        "kind": kind,
        "tasks": TASKS,
        "classes": CLASSES,
        "heads": heads,
        "temperature": calibration,
        "features": features,
        "encoder": encoder,
        "input_serialization": "feedback-visible-context-v1",
        "mode": "shadow",
        "training": {
            "inputs_hash": digest(rows),
            "input_ids": sorted(ex["input_id"] for ex in rows),
            "group_ids": sorted({ex["group_id"] for ex in rows}),
            "judge_version": design["judge_version"],
            "labels_hash": read(base / "labels-frozen.json")["labels_hash"],
            "policy_hash": design["policy_hash"],
        },
    }
    value["model_version"] = digest(value)
    write_json(base / f"models/{name}/model.json", value)
    return value


def load_model(path):
    value = read(Path(path) / "model.json")
    if value.get("format") != "feedback-student-v1" or value.get("model_version") != digest(
        {k: v for k, v in value.items() if k != "model_version"}
    ):
        raise ValueError("Feedback model checksum or format mismatch")
    if (
        value["tasks"] != list(TASKS)
        or value["classes"] != list(CLASSES)
        or value["input_serialization"] != "feedback-visible-context-v1"
    ):
        raise ValueError("Feedback model task/serializer mismatch")
    return value


class FeedbackPredictor:
    def __init__(self, path, *, device="cpu", batch_size=8):
        self.path = Path(path)
        self.artifact = a = load_model(path)
        self.extractor = TfidfFeatures(a["features"]) if a["kind"] == "tfidf" else None
        self.encoder = None
        if a["encoder"]:
            c = a["encoder"]
            bundled = self.path / "backbone-cache"
            if bundled.exists():
                manifest = read(self.path / "backbone-bundle.json")
                if manifest != {
                    "model": c["model"],
                    "revision": c["revision"],
                    "sha256": encoder_directory_hash(bundled, pretrained=True),
                }:
                    raise ValueError("Bundled backbone checksum mismatch")
            self.encoder = SentimentEncoder(
                c["model"],
                c["revision"],
                device=device,
                batch_size=batch_size,
                prefix=c["prefix"],
                max_length=c["max_length"],
                pretrained_path=bundled if bundled.exists() else None,
                finetuned_path=self.path / "encoder" if a["kind"] == "finetuned" else None,
                finetuned_sha=c.get("finetuned_sha256"),
            )
            if self.encoder.config != c:
                raise ValueError("Loaded encoder differs from feedback artifact")

    def probabilities(self, examples):
        import numpy as np

        if any(
            ex["input_id"] in self.artifact["training"]["input_ids"]
            or ex["group_id"] in self.artifact["training"]["group_ids"]
            for ex in examples
        ):
            raise ValueError("Prediction inputs overlap training groups")
        rendered = student_inputs(examples)
        if self.extractor is not None:
            vectors = self.extractor.transform([r["context"][0]["text"] for r in rendered])
        else:
            vectors, _ = self.encoder.encode(rendered)
        return np.stack(
            [
                softmax(
                    head_logits(vectors, self.artifact["heads"][t]),
                    self.artifact["temperature"][t]["temperature"],
                )
                for t in TASKS
            ],
            axis=1,
        )

    def predict(self, examples):
        probabilities = self.probabilities(examples)
        return [
            {
                "input_id": ex["input_id"],
                "task": "feedback",
                "mode": "shadow",
                "model_version": self.artifact["model_version"],
                "signals": {
                    t: {
                        "label": CLASSES[int(p[j].argmax())],
                        "probabilities": dict(zip(CLASSES, map(float, p[j]))),
                    }
                    for j, t in enumerate(TASKS)
                },
            }
            for ex, p in zip(examples, probabilities)
        ]


def record_candidate(base, name, artifact, metrics, *, extra=None, backbone_bytes=0):
    path = Path(base) / f"models/{name}"
    row = {
        "name": name,
        "kind": artifact["kind"],
        "path": f"models/{name}",
        "model_version": artifact["model_version"],
        "calibration": metrics,
        "artifact_bytes": sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        + backbone_bytes,
        **(extra or {}),
    }
    write_json(Path(base) / f"candidates/{name}.json", row)
    return row


def fit_candidate(base, name, *, device="cpu"):
    import numpy as np
    import torch

    base = Path(base)
    if (base / "selection.json").exists():
        raise ValueError("Model selection already frozen")
    policy = read(base / "design.json")["policy"]
    train, y = load_partition(base, "train")
    cal, cal_y = load_partition(base, "calibration")
    keep = (y >= 0).any(axis=1)
    train, y = [ex for ex, m in zip(train, keep) if m], y[keep]
    torch.set_num_threads(8)
    torch.manual_seed(policy["seed"])
    encoder = None
    features = None
    spec = None
    if name == "tfidf":
        extractor = TfidfFeatures()
        vectors = extractor.fit_transform([r["context"][0]["text"] for r in student_inputs(train)])
        cal_vectors = extractor.transform([r["context"][0]["text"] for r in student_inputs(cal)])
        features = extractor.state()
        candidate_name = name
    else:
        spec = next(s for s in policy["encoders"] if s["name"] == name)
        encoder = SentimentEncoder(
            spec["model"],
            spec["revision"],
            device=device,
            batch_size=spec["batch_size"],
            prefix=spec["prefix"],
            max_length=spec["max_length"],
        )
        vectors, _ = cache_embeddings(
            student_inputs(train), encoder, base / f"embeddings/{name}/train"
        )
        cal_vectors, _ = cache_embeddings(
            student_inputs(cal), encoder, base / f"embeddings/{name}/calibration"
        )
        candidate_name = name + "-frozen"
    heads, temperatures, probabilities, trials = {}, {}, [], {}
    for j, task in enumerate(TASKS):
        best = None
        trials[task] = []
        for c in policy["C"]:
            for balanced in policy["balanced"]:
                head = fit_feedback_head(vectors, y[:, j], c=c, balanced=balanced)
                logits = head_logits(cal_vectors, head)
                mask = cal_y[:, j] >= 0
                m = classification(cal_y[mask, j], softmax(logits[mask]), classes=CLASSES)
                trials[task].append(
                    {
                        "C": c,
                        "balanced": balanced,
                        "macro_f1": m["macro_f1"],
                        "yes_f1": m["per_class"]["yes"]["f1"],
                    }
                )
                rank = (m["macro_f1"], m["per_class"]["yes"]["f1"], -c, not balanced)
                if best is None or rank > best[0]:
                    best = rank, head, logits
        heads[task] = best[1]
        temperatures[task] = fit_temperature(best[2][mask], cal_y[mask, j])
        probabilities.append(softmax(best[2], temperatures[task]["temperature"]))
    metrics = task_metrics(cal_y, np.stack(probabilities, axis=1))
    artifact = save_model(
        base,
        candidate_name,
        "tfidf" if encoder is None else "frozen",
        train,
        heads,
        temperatures,
        features=features,
        encoder=encoder.config if encoder else None,
    )
    record_candidate(
        base,
        candidate_name,
        artifact,
        metrics,
        backbone_bytes=encoder.disk_bytes if encoder else 0,
        extra={"trials": trials},
    )
    print(candidate_name, "calibration macro-F1", metrics["mean_macro_f1"], flush=True)
    if encoder is not None:
        finetune_feedback(base, name, encoder, heads, train, y, cal, cal_y, policy, spec)


def finetune_feedback(base, name, encoder, initial_heads, train, y, cal, cal_y, policy, spec):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from contextlib import nullcontext

    base = Path(base)
    candidate_name = name + "-finetuned"
    if (base / f"candidates/{candidate_name}.json").exists():
        return
    if spec["gradient_checkpointing"]:
        encoder.model[0].auto_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if encoder.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    head = torch.nn.Linear(encoder.dimension, 6, device=encoder.device)
    with torch.no_grad():
        head.weight.copy_(
            torch.tensor([initial_heads[t]["coef"] for t in TASKS], device=encoder.device).reshape(
                6, encoder.dimension
            )
        )
        head.bias.copy_(
            torch.tensor(
                [initial_heads[t]["intercept"] for t in TASKS], device=encoder.device
            ).reshape(6)
        )
    windows = []
    for ex in student_inputs(train):
        w, _, weights, _ = encoder.windows([ex])
        windows.append(list(zip(w, weights)))
    micro = spec["microbatch"]
    accumulation = policy["effective_batch_size"] // micro
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.model.parameters(), "lr": policy["learning_rate"]},
            {"params": head.parameters(), "lr": policy["head_learning_rate"]},
        ],
        weight_decay=policy["weight_decay"],
    )
    updates = int(np.ceil(len(train) / policy["effective_batch_size"])) * policy["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0, 1 - step / max(updates, 1))
    )
    bf16 = encoder.device == "cuda" and torch.cuda.is_bf16_supported()
    started = time.perf_counter()
    best, trials = None, []
    for epoch in range(policy["epochs"]):
        encoder.model.train()
        head.train()
        order = np.random.default_rng(policy["seed"] + epoch).permutation(len(train))
        optimizer.zero_grad(set_to_none=True)
        for b, start in enumerate(range(0, len(order), micro)):
            indices = order[start : start + micro]
            ids, owners, weights = [], [], []
            for owner, i in enumerate(indices):
                for window, weight in windows[int(i)]:
                    ids.append(window)
                    owners.append(owner)
                    weights.append(weight)
            batch = encoder.model.tokenizer.pad(
                {"input_ids": ids}, padding=True, return_tensors="pt"
            )
            batch = {k: v.to(encoder.device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext():
                v = F.normalize(encoder.model(batch)["sentence_embedding"].float(), dim=1)
                docs = torch.zeros((len(indices), encoder.dimension), device=encoder.device)
                docs.index_add_(
                    0,
                    torch.tensor(owners, device=encoder.device),
                    v * torch.tensor(weights, device=encoder.device)[:, None],
                )
                logits = head(F.normalize(docs, dim=1)).reshape(-1, len(TASKS), 3)
                target = torch.tensor(y[indices], device=encoder.device)
                loss = F.cross_entropy(
                    logits.reshape(-1, 3), target.reshape(-1), ignore_index=-1, reduction="sum"
                )
            group_start = b // accumulation * policy["effective_batch_size"]
            group_labels = int(
                (y[order[group_start : group_start + policy["effective_batch_size"]]] >= 0).sum()
            )
            (loss / max(group_labels, 1)).backward()
            if (b + 1) % accumulation == 0 or start + micro >= len(order):
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.model.parameters()) + list(head.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        encoder.model.eval()
        head.eval()
        cal_vectors, _ = encoder.encode(student_inputs(cal))
        with torch.inference_mode():
            logits = (
                head(torch.tensor(cal_vectors, device=encoder.device))
                .reshape(-1, len(TASKS), 3)
                .float()
                .cpu()
                .numpy()
            )
        temperature, probabilities = calibrate(logits, cal_y)
        metrics = task_metrics(cal_y, probabilities)
        trials.append(
            {
                "epoch": epoch + 1,
                "calibration": metrics,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        write_json(base / f"{name}-finetune-trials.json", trials)
        rank = metrics["mean_macro_f1"], metrics["mean_yes_f1"]
        if best is None or rank > best[0]:
            encoder_path = base / f"models/{candidate_name}/encoder"
            encoder.model.save(str(encoder_path), safe_serialization=True, create_model_card=False)
            sha = encoder_directory_hash(encoder_path)
            coef = (
                head.weight.detach().float().cpu().numpy().reshape(len(TASKS), 3, encoder.dimension)
            )
            intercept = head.bias.detach().float().cpu().numpy().reshape(len(TASKS), 3)
            heads = {
                t: {"coef": coef[j].tolist(), "intercept": intercept[j].tolist()}
                for j, t in enumerate(TASKS)
            }
            artifact = save_model(
                base,
                candidate_name,
                "finetuned",
                train,
                heads,
                temperature,
                encoder={**encoder.config, "finetuned_sha256": sha},
            )
            best = rank, artifact, metrics, epoch + 1
        print(
            name, "epoch", epoch + 1, "calibration macro-F1", metrics["mean_macro_f1"], flush=True
        )
    record_candidate(
        base,
        candidate_name,
        best[1],
        best[2],
        extra={
            "selected_epoch": best[3],
            "training_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated() if encoder.device == "cuda" else None
            ),
            "dtype": "bf16 autocast" if bf16 else "float32",
        },
    )


def select_feedback(base):
    base = Path(base)
    if (base / "selection.json").exists():
        return read(base / "selection.json")
    candidates = [read(p) for p in sorted((base / "candidates").glob("*.json"))]
    expected = {"tfidf"} | {
        s["name"] + suffix
        for s in read(base / "design.json")["policy"]["encoders"]
        for suffix in ("-frozen", "-finetuned")
    }
    if {r["name"] for r in candidates} != expected:
        raise ValueError("Not all planned candidates have completed")
    winner = max(
        candidates,
        key=lambda r: (
            r["calibration"]["mean_macro_f1"],
            r["calibration"]["mean_yes_f1"],
            -r["artifact_bytes"],
        ),
    )
    selection = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "winner": winner,
        "candidates": candidates,
        "test_used": False,
        "design_hash": digest(read(base / "design.json")),
        "labels_hash": read(base / "labels-frozen.json")["labels_hash"],
    }
    write_json(base / "selection.json", selection)
    return selection
