"""Portable three-class students. Inputs are text only; targets are versioned teacher labels."""

import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path

from .dataset import SENTIMENTS
from .io import digest, index_rows, write_json
from .student_data import text_of


def make_sentiment_input(text, *, tenant_id, session_id, message_id):
    """Build an unlabelled message input without claiming public-dataset provenance."""
    if not isinstance(text, str):
        raise ValueError("Message text must be a string")
    if any(
        not isinstance(value, str) or not value for value in (tenant_id, session_id, message_id)
    ):
        raise ValueError("Tenant, session and message IDs must be nonempty strings")
    group = digest([tenant_id, session_id])
    identity = {
        "tenant_id": tenant_id,
        "trace_id": group,
        "branch_id": "unknown",
        "target_event_id": message_id,
        "metric_version": "message-sentiment-v1",
        "input_version": "sentiment-message-only-v1",
        "context": [{"event_id": message_id, "role": "user", "text": text}],
        "context_status": "complete",
        "context_scope": "message_only",
        "history_available": False,
    }
    return {
        **identity,
        "input_id": digest(identity),
        "group_id": group,
        "language": "unknown",
        "split": "inference",
    }


def targets(examples, annotations, *, judge_version=None):
    """Missing labels stay missing; reject mixed tasks, versions and duplicate identities."""
    by_id = index_rows(annotations)
    index_rows(examples)
    selected, y, missing = [], [], []
    versions = set()
    for ex in examples:
        row = by_id.get(ex["input_id"])
        if row is None:
            missing.append(ex["input_id"])
            continue
        if row.get("task") != "sentiment" or row.get("label_source") != "llm_judge":
            raise ValueError("Student targets must be sentiment labels from llm_judge")
        label, version = row.get("sentiment_label"), row.get("judge_version")
        if label not in SENTIMENTS or not isinstance(version, str) or not version:
            raise ValueError("Invalid teacher class or version")
        versions.add(version)
        if judge_version is not None and version != judge_version:
            raise ValueError("Teacher version differs from frozen design")
        text_of(ex)
        selected.append(ex)
        y.append(SENTIMENTS.index(label))
    if len(versions) > 1:
        raise ValueError("Student targets mix teacher versions")
    return selected, y, missing


def provenance(examples, annotations):
    if not examples or any(ex["split"] != "train" for ex in examples):
        raise ValueError("Student training accepts only train examples")
    versions = {ex["input_version"] for ex in examples}
    if len(versions) != 1:
        raise ValueError("Mixed student input versions")
    valid, _, missing = targets(examples, annotations)
    if missing:
        raise ValueError("Filter unlabelled train inputs explicitly before fitting")
    ids = {ex["input_id"] for ex in valid}
    used_labels = [r for r in annotations if r["input_id"] in ids]
    return {
        "training_messages": len(examples),
        "training_groups": sorted({ex["group_id"] for ex in examples}),
        "training_input_ids": sorted(ids),
        "training_inputs_hash": digest(examples),
        "training_labels_hash": digest(sorted(used_labels, key=lambda r: r["input_id"])),
        "training_label_source": "llm_judge",
        "judge_versions": sorted({r["judge_version"] for r in used_labels}),
        "input_version": next(iter(versions)),
        "python": sys.version,
        "environment": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")
        },
    }


def softmax(logits, temperature=1.0):
    import numpy as np

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    logits = np.asarray(logits, dtype=float) / temperature
    if logits.ndim != 2 or logits.shape[1] != 3 or not np.isfinite(logits).all():
        raise ValueError("Expected finite three-class logits")
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(logits, y):
    import numpy as np
    from scipy.optimize import minimize_scalar

    y = np.asarray(y, dtype=int)

    def loss(t):
        p = softmax(logits, t)
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())

    result = minimize_scalar(loss, bounds=(0.25, 4.0), method="bounded")
    temperature = float(result.x) if result.success and result.fun < loss(1.0) else 1.0
    return {
        "temperature": temperature,
        "calibration_nll_before": loss(1.0),
        "calibration_nll_after": loss(temperature),
    }


class TfidfFeatures:
    def __init__(self, state=None):
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        shared = dict(lowercase=True, sublinear_tf=True, dtype=np.float64)
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000, **shared)
        self.char = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2, max_features=60000, **shared
        )
        if state:
            import numpy as np

            for name in ("word", "char"):
                vectorizer = getattr(self, name)
                vectorizer.set_params(vocabulary=state[name]["vocabulary"])
                vectorizer.idf_ = np.asarray(state[name]["idf"])

    def fit_transform(self, texts):
        from scipy.sparse import hstack

        return hstack(
            [self.word.fit_transform(texts), self.char.fit_transform(texts)], format="csr"
        )

    def transform(self, texts):
        from scipy.sparse import hstack

        return hstack([self.word.transform(texts), self.char.transform(texts)], format="csr")

    def state(self):
        return {
            name: {
                "vocabulary": {k: int(v) for k, v in getattr(self, name).vocabulary_.items()},
                "idf": getattr(self, name).idf_.tolist(),
            }
            for name in ("word", "char")
        }


def fit_head(vectors, y, *, c=1.0, balanced=False):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    if set(y) != {0, 1, 2}:
        raise ValueError("Training requires all three sentiment classes")
    classifier = LogisticRegression(
        C=c, class_weight="balanced" if balanced else None, max_iter=1000, tol=1e-5, random_state=42
    )
    classifier.fit(vectors, np.asarray(y))
    if classifier.classes_.tolist() != [0, 1, 2]:
        raise ValueError("Unexpected class order")
    return {
        "coef": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
        "C": c,
        "balanced": balanced,
        "iterations": classifier.n_iter_.tolist(),
    }


def head_logits(vectors, head):
    import numpy as np

    return np.asarray(vectors @ np.asarray(head["coef"]).T) + np.asarray(head["intercept"])


def save_student(output, *, kind, training, head, features=None, encoder=None, calibration=None):
    artifact = {
        "format": "sentiment-student-v1",
        "kind": kind,
        "classes": list(SENTIMENTS),
        "training": training,
        "head": head,
        "features": features,
        "encoder": encoder,
        "calibration": calibration or {"temperature": 1.0},
        "default_mode": "shadow",
    }
    artifact["model_version"] = digest(artifact)
    write_json(Path(output) / "model.json", artifact)
    return artifact


def load_student(path):
    path = Path(path)
    artifact = json.loads(
        (path / "model.json" if path.is_dir() else path).read_text(encoding="utf-8")
    )
    if artifact.get("model_version") != digest(
        {k: v for k, v in artifact.items() if k != "model_version"}
    ):
        raise ValueError("Student artifact checksum mismatch")
    if artifact.get("format") != "sentiment-student-v1" or artifact.get("classes") != list(
        SENTIMENTS
    ):
        raise ValueError("Unsupported student artifact/classes")
    return artifact


def validate_prediction_inputs(examples, artifact):
    training = artifact["training"]
    groups = set(training["training_groups"])
    for ex in examples:
        text_of(ex)
        if ex["input_version"] != training["input_version"]:
            raise ValueError("Student input version differs from training")
        if ex["split"] != "train" and ex["group_id"] in groups:
            raise ValueError("Held-out student input overlaps a training session")


def predict_student(examples, artifact, *, encoder=None):
    import numpy as np

    validate_prediction_inputs(examples, artifact)
    started = time.perf_counter()
    if artifact["kind"] == "tfidf":
        vectors = TfidfFeatures(artifact["features"]).transform([text_of(ex) for ex in examples])
    elif artifact["kind"] in {"encoder", "finetuned"}:
        if encoder is None or encoder.config != artifact["encoder"]:
            raise ValueError("Load the exact encoder from the student artifact")
        vectors, _ = encoder.encode(examples)
    elif artifact["kind"] == "majority":
        vectors = np.zeros((len(examples), 1))
    else:
        raise ValueError("Unknown sentiment student kind")
    probabilities = softmax(
        head_logits(vectors, artifact["head"]), artifact["calibration"]["temperature"]
    )
    records = [
        {
            "input_id": ex["input_id"],
            "task": "sentiment",
            "sentiment_label": SENTIMENTS[int(np.argmax(p))],
            "probabilities": {label: float(p[j]) for j, label in enumerate(SENTIMENTS)},
            "negative_score": float(p[0]),
            "runtime_status": "ok",
            "model_version": artifact["model_version"],
            "mode": "shadow",
        }
        for ex, p in zip(examples, probabilities)
    ]
    return records, {"messages": len(examples), "elapsed_seconds": time.perf_counter() - started}


def train_student(
    examples, annotations, output, *, kind="tfidf", encoder=None, c=1.0, balanced=False
):
    if any(ex["split"] != "train" for ex in examples):
        raise ValueError("Student training accepts only train examples")
    valid, y, missing = targets(examples, annotations)
    features = None
    if kind == "tfidf":
        extractor = TfidfFeatures()
        vectors = extractor.fit_transform([text_of(ex) for ex in valid])
        features = extractor.state()
    elif kind == "encoder" and encoder is not None:
        vectors, _ = encoder.encode(valid)
    else:
        raise ValueError("Choose tfidf or encoder with a configured encoder")
    head = fit_head(vectors, y, c=c, balanced=balanced)
    artifact = save_student(
        output,
        kind=kind,
        training=provenance(valid, annotations),
        head=head,
        features=features,
        encoder=encoder.config if kind == "encoder" else None,
    )
    return {
        "model_version": artifact["model_version"],
        "training_messages": len(valid),
        "unlabelled_messages": len(missing),
    }
