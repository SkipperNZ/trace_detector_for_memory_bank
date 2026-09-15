"""Pinned public sentiment corpus; source labels never enter model inputs."""

import hashlib
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from .io import digest, write_json, write_jsonl
from .prepare import split_for

DATASET_ID = "davanstrien/agent-trace-sentiment"
DATASET_REVISION = "434c8d7b267672cc4638bb5a80fb0da13d18d76d"
SENTIMENTS = ("NEGATIVE", "NEUTRAL", "POSITIVE")
MESSAGE_METRIC = "agent-dissatisfaction-message-v1"


def download(output: Path, revision: str = DATASET_REVISION) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Dataset revision must be a full commit SHA")
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for remote, local in (
        ("data/train-00000-of-00001.parquet", "train.parquet"),
        ("README.md", "dataset-card.md"),
    ):
        path = output / local
        url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{revision}/{remote}"
        temporary = path.with_suffix(path.suffix + ".partial")
        checksum = hashlib.sha256()
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                temporary.open("wb") as target,
            ):
                while chunk := response.read(1024 * 1024):
                    checksum.update(chunk)
                    target.write(chunk)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        files[local] = {"sha256": checksum.hexdigest(), "bytes": path.stat().st_size}
    manifest = {"dataset_id": DATASET_ID, "revision": revision, "files": files}
    write_json(output / "download-manifest.json", manifest)
    return manifest


def convert(rows: list[dict], *, revision: str = DATASET_REVISION, seed: int = 42):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Dataset revision must be a full commit SHA")
    examples, references, seen = [], [], set()
    for row in rows:
        key = str(row["id"])
        if key in seen:
            raise ValueError(f"Duplicate source row ID: {key}")
        seen.add(key)
        text, sentiment = row["content_text"], row["sentiment_label"]
        source, session = row["source_dataset"], row["session_id"]
        if not isinstance(text, str) or not isinstance(source, str) or not isinstance(session, str):
            raise ValueError("Dataset text/source/session must be strings")
        if sentiment not in SENTIMENTS:
            raise ValueError(f"Unknown source sentiment: {sentiment}")
        group = digest([source, session])
        identity = {
            "tenant_id": "public-agent-trace-sentiment",
            "trace_id": group,
            "branch_id": "unknown",
            "target_event_id": key,
            "metric_version": MESSAGE_METRIC,
            "input_version": "sentiment-message-only-v1",
            "context": [{"event_id": key, "role": "user", "text": text}],
            "context_status": "complete",
            "context_scope": "message_only",
            "history_available": False,
            "dataset_id": DATASET_ID,
            "dataset_revision": revision,
        }
        example = {
            **identity,
            "input_id": digest(identity),
            "group_id": group,
            "language": "unknown",
            "source_dataset": source,
            "split": split_for("public-agent-trace-sentiment", group, seed),
            "split_seed": seed,
        }
        examples.append(example)
        references.append(
            {
                "input_id": example["input_id"],
                "source_row_id": key,
                "sentiment_label": sentiment,
                "label_source": "dataset_llm",
                "dataset_id": DATASET_ID,
                "dataset_revision": revision,
            }
        )
    return examples, references


def pilot_sample(examples: list[dict], references: list[dict], per_class: int, seed: int):
    if per_class < 1:
        raise ValueError("per_class must be positive")
    by_id = {row["input_id"]: row for row in examples if row["split"] == "calibration"}
    strata = defaultdict(list)
    for ref in references:
        if ref["input_id"] in by_id:
            strata[ref["sentiment_label"]].append(ref["input_id"])
    selected, design = [], []
    for label in SENTIMENTS:
        ids = sorted(strata[label], key=lambda key: digest([seed, "judge-pilot-v1", key]))
        count = min(per_class, len(ids))
        for key in ids[:count]:
            selected.append(by_id[key])
            design.append(
                {
                    "input_id": key,
                    "stratum": label,
                    "population": len(ids),
                    "sample_size": count,
                    "inclusion_probability": count / len(ids),
                }
            )
    return selected, design


def import_parquet(
    path: Path,
    output: Path,
    *,
    per_class: int = 100,
    seed: int = 42,
    revision: str = DATASET_REVISION,
):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError(
            "Install the dataset extra: python -m pip install -e '.[dataset]'"
        ) from exc
    # No future session statistics, reasons or labels enter examples.jsonl.
    rows = pq.read_table(
        path, columns=["id", "source_dataset", "session_id", "content_text", "sentiment_label"]
    ).to_pylist()
    examples, references = convert(rows, revision=revision, seed=seed)
    pilot, design = pilot_sample(examples, references, per_class, seed)
    write_jsonl(output / "examples.jsonl", examples)
    write_jsonl(output / "source_labels.jsonl", references)
    for split in ("train", "calibration", "test"):
        write_jsonl(output / f"{split}.jsonl", [ex for ex in examples if ex["split"] == split])
    write_jsonl(output / "pilot.jsonl", pilot)
    write_jsonl(output / "pilot_design.jsonl", design)
    texts = defaultdict(set)
    for ex in examples:
        texts[digest(ex["context"][0]["text"])].add(ex["split"])
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_revision": revision,
        "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "messages": len(examples),
        "sessions": len({e["trace_id"] for e in examples}),
        "sources": len({e["source_dataset"] for e in examples}),
        "source_label_counts": dict(Counter(r["sentiment_label"] for r in references)),
        "split_counts": dict(Counter(e["split"] for e in examples)),
        "pilot_messages": len(pilot),
        "pilot_counts": dict(Counter(r["stratum"] for r in design)),
        "exact_texts_shared_across_splits": sum(len(s) > 1 for s in texts.values()),
        "seed": seed,
        "per_class": per_class,
        "context_scope": "message_only",
        "notes": [
            "Source sentiment labels are automatic, not human gold.",
            "Session splits support exploratory teacher agreement, not production validation.",
            "The pilot is class-balanced calibration data, not a prevalence sample.",
            "No assistant history or branch structure is reconstructed from this table.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    return manifest
