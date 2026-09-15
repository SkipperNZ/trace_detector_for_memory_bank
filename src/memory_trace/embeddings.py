"""Optional frozen encoder + logistic head. Heavy imports happen only on explicit use."""

import importlib.metadata
import json
import re
import sys
from pathlib import Path

from .io import digest, write_json
from .metrics import validate_labels


def select_device(requested: str, torch) -> str:
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("device must be auto/cpu/mps/cuda")
    cuda = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps = mps_backend is not None and mps_backend.is_available()
    if requested == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if requested == "cuda" and not cuda or requested == "mps" and not mps:
        raise ValueError(f"Requested {requested} backend is unavailable; use --device cpu or auto")
    return requested


class FrozenEncoder:
    def __init__(
        self,
        model: str,
        revision: str,
        *,
        device: str = "auto",
        batch_size: int = 32,
        prefix: str = "query: ",
        local_files_only: bool = False,
    ):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be a pinned 40-character checkpoint commit SHA")
        if Path(model).exists():
            raise ValueError(
                "Use a Hugging Face repository ID with pinned revision, not a local path"
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ValueError("Install the ML extra: python -m pip install -e '.[ml]'") from exc
        self.device = select_device(device, torch)
        self.model = SentenceTransformer(
            model,
            revision=revision,
            device=self.device,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        self.batch_size = batch_size
        self.config = {
            "model": model,
            "revision": revision,
            "prefix": prefix,
            "serialization": "target-first-json-v1",
        }

    def encode(self, examples: list[dict]):
        import numpy as np

        texts, statuses = [], []
        # Target first also protects it if a third-party tokenizer changes its behavior.
        # Over-limit examples are not scored; no silent truncation is accepted.
        for example in examples:
            text = self.config["prefix"] + json.dumps(
                {
                    "target": example["context"][-1],
                    "history": example["context"][:-1],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            tokens = self.model.tokenizer(text, truncation=False, add_special_tokens=True)[
                "input_ids"
            ]
            status = example["context_status"]
            if len(tokens) > self.model.max_seq_length:
                status = "token_limit"
            texts.append(text)
            statuses.append(status)
        eligible = [i for i, status in enumerate(statuses) if status == "complete"]
        vectors = np.zeros((len(examples), self.model.get_embedding_dimension()))
        if eligible:
            vectors[eligible] = self.model.encode(
                [texts[i] for i in eligible],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                prompt="",
            )
        return vectors, statuses


def train(
    examples: list[dict],
    annotations: list[dict],
    encoder: FrozenEncoder,
    output: Path,
    *,
    label_source: str = "human",
):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    labels = validate_labels(examples, annotations, label_source=label_source)
    if not examples or any(ex["split"] != "train" for ex in examples):
        raise ValueError("Training accepts only examples assigned to train")
    metric_versions = {ex["metric_version"] for ex in examples}
    input_versions = {ex["input_version"] for ex in examples}
    if len(metric_versions) != 1 or len(input_versions) != 1:
        raise ValueError("Training cannot mix input or metric versions")
    vectors, statuses = encoder.encode(examples)
    eligible = [
        i
        for i, ex in enumerate(examples)
        if statuses[i] == "complete" and labels[ex["input_id"]]["label"] in {"yes", "no"}
    ]
    y = np.array([int(labels[examples[i]["input_id"]]["label"] == "yes") for i in eligible])
    if len(set(y.tolist())) != 2:
        raise ValueError("Training needs both yes and no with complete, in-budget context")
    head = LogisticRegression(max_iter=1000, random_state=42)
    head.fit(vectors[eligible], y)
    artifact = {
        "format": "frozen-logistic-v1",
        "encoder": encoder.config,
        "metric_version": next(iter(metric_versions)),
        "input_version": next(iter(input_versions)),
        "coef": head.coef_[0].tolist(),
        "intercept": float(head.intercept_[0]),
        "positive_label": "yes",
        "training_messages": len(eligible),
        "excluded_messages": len(examples) - len(eligible),
        "training_groups": sorted({digest([ex["tenant_id"], ex["group_id"]]) for ex in examples}),
        "training_input_hash": digest(sorted(ex["input_id"] for ex in examples)),
        "training_label_source": label_source,
        "training_labels_hash": digest(sorted(annotations, key=lambda row: row["input_id"])),
        "training_judge_versions": sorted(
            {row["judge_version"] for row in annotations if "judge_version" in row}
        ),
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scikit-learn", "sentence-transformers", "torch")
        },
        "python": sys.version,
        "training_device": encoder.device,
    }
    artifact["model_version"] = digest(artifact)
    write_json(output, artifact)
    return artifact


def predict(examples: list[dict], artifact: dict, encoder: FrozenEncoder):
    import numpy as np

    unsigned = {key: value for key, value in artifact.items() if key != "model_version"}
    if artifact.get("model_version") != digest(unsigned):
        raise ValueError("Model artifact checksum mismatch")
    if artifact["format"] != "frozen-logistic-v1" or encoder.config != artifact["encoder"]:
        raise ValueError("Encoder configuration does not match trained head")
    if any(
        ex["metric_version"] != artifact["metric_version"]
        or ex["input_version"] != artifact["input_version"]
        for ex in examples
    ):
        raise ValueError("Input/metric versions differ from training")
    training_groups = set(artifact["training_groups"])
    if any(
        ex["split"] != "train" and digest([ex["tenant_id"], ex["group_id"]]) in training_groups
        for ex in examples
    ):
        raise ValueError("Held-out input overlaps a training group")
    vectors, statuses = encoder.encode(examples)
    logits = vectors @ np.asarray(artifact["coef"]) + artifact["intercept"]
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -709, 709)))
    return [
        {
            "input_id": example["input_id"],
            "model_version": artifact["model_version"],
            "score": float(probabilities[i]) if statuses[i] == "complete" else None,
            "runtime_status": "ok" if statuses[i] == "complete" else "not_evaluated",
            "model_abstention": statuses[i] != "complete",
            "context_status": statuses[i],
        }
        for i, example in enumerate(examples)
    ]
