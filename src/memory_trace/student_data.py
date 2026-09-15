"""Freeze message-only student splits, removing exact and near-template leakage."""

import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .io import digest, index_rows, read_jsonl, write_json, write_jsonl
from .judge import messages, read_config


def text_of(example):
    if example.get("context_scope") != "message_only" or len(example["context"]) != 1:
        raise ValueError("Sentiment students require one message-only event")
    return example["context"][0]["text"]


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def duplicate_components(examples, *, near=True):
    """Use text alone; hashing proposes pairs, exact shingle Jaccard verifies them."""
    parent = list(range(len(examples)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i, j):
        a, b = root(i), root(j)
        parent[max(a, b)] = min(a, b)

    by_text = defaultdict(list)
    texts = [normalized(text_of(ex)) for ex in examples]
    for i, text in enumerate(texts):
        by_text[text].append(i)
    for ids in by_text.values():
        for i in ids[1:]:
            join(ids[0], i)
    near_pairs = []
    if near:
        import numpy as np
        from sklearn.feature_extraction.text import HashingVectorizer

        ids = [v[0] for text, v in by_text.items() if len(text) >= 80]
        if ids:
            vectorizer = HashingVectorizer(
                analyzer="char",
                ngram_range=(5, 5),
                n_features=2**19,
                alternate_sign=False,
                binary=True,
                norm="l2",
                dtype=np.float32,
            )
            matrix = vectorizer.transform([texts[i] for i in ids])
            shingles = {}

            def grams(i):
                if i not in shingles:
                    shingles[i] = {texts[i][k : k + 5] for k in range(len(texts[i]) - 4)}
                return shingles[i]

            for start in range(0, len(ids), 128):
                similarities = (matrix[start : start + 128] @ matrix.T).tocoo()
                for local_i, local_j, similarity in zip(
                    similarities.row, similarities.col, similarities.data
                ):
                    i_pos, j_pos = start + int(local_i), int(local_j)
                    if j_pos <= i_pos or similarity < 0.95:
                        continue
                    i, j = ids[i_pos], ids[j_pos]
                    if min(len(texts[i]), len(texts[j])) / max(len(texts[i]), len(texts[j])) < 0.85:
                        continue
                    a, b = grams(i), grams(j)
                    jaccard = len(a & b) / len(a | b)
                    if jaccard >= 0.85:
                        join(i, j)
                        near_pairs.append(
                            {
                                "left": examples[i]["input_id"],
                                "right": examples[j]["input_id"],
                                "jaccard": jaccard,
                            }
                        )
    return [root(i) for i in range(len(examples))], near_pairs


def clean_splits(examples, reviewed, config, *, near=True):
    index_rows(examples)
    reviewed_ids = {ex["input_id"] for ex in reviewed}
    reviewed_groups = {ex["group_id"] for ex in reviewed}
    components, near_pairs = duplicate_components(examples, near=near)
    seen_components = {c for ex, c in zip(examples, components) if ex["input_id"] in reviewed_ids}
    eligible, exclusions = [], []
    for ex, component in zip(examples, components):
        reason = None
        if not normalized(text_of(ex)):
            reason = "empty_text"
        elif (
            sum(
                len(m["content"]) for m in messages(ex, config.task, system_prompt=config.prompt[1])
            )
            > config.max_input_chars
        ):
            reason = "judge_input_limit"
        elif ex["split"] == "test" and ex["group_id"] in reviewed_groups:
            reason = "previously_reviewed_session"
        elif ex["split"] == "test" and component in seen_components:
            reason = "duplicate_of_reviewed_message"
        if reason:
            exclusions.append({"input_id": ex["input_id"], "split": ex["split"], "reason": reason})
        else:
            eligible.append((ex, component))
    priority = {"train": 0, "calibration": 1, "test": 2}
    chosen = {}
    for ex, component in eligible:
        chosen[component] = max(chosen.get(component, 0), priority[ex["split"]])
    kept = []
    for ex, component in eligible:
        if priority[ex["split"]] != chosen[component]:
            exclusions.append(
                {
                    "input_id": ex["input_id"],
                    "split": ex["split"],
                    "reason": "duplicate_in_higher_priority_split",
                }
            )
        else:
            kept.append(ex)
    groups = {s: {ex["group_id"] for ex in kept if ex["split"] == s} for s in priority}
    if any(groups[a] & groups[b] for a in priority for b in priority if a < b):
        raise ValueError("Session overlap between student splits")
    kept_ids = {ex["input_id"] for ex in kept}
    by_component = defaultdict(set)
    for ex, c in zip(examples, components):
        if ex["input_id"] in kept_ids:
            by_component[c].add(ex["split"])
    if any(len(s) > 1 for s in by_component.values()):
        raise ValueError("Duplicate component overlaps student splits")
    return kept, exclusions, near_pairs


def freeze_data(source: Path, reviewed_paths: list[Path], config_path: Path, output: Path):
    if (output / "design.json").exists():
        raise ValueError("Student design already frozen; use another output directory")
    examples = read_jsonl(source)
    reviewed = [ex for path in reviewed_paths for ex in read_jsonl(path)]
    config = read_config(config_path)
    kept, exclusions, pairs = clean_splits(examples, reviewed, config)
    for split in ("train", "calibration", "test"):
        subset = sorted(
            [ex for ex in kept if ex["split"] == split],
            key=lambda ex: digest([42, "student-order-v1", ex["input_id"]]),
        )
        write_jsonl(output / f"{split}.jsonl", subset)
    # Label train first so its completed checkpoints can support implementation checks.
    ordered = [
        ex for s in ("train", "calibration", "test") for ex in read_jsonl(output / f"{s}.jsonl")
    ]
    write_jsonl(output / "examples.jsonl", ordered)
    write_jsonl(output / "exclusions.jsonl", exclusions)
    write_jsonl(output / "near-duplicate-pairs.jsonl", pairs)
    manifest = {
        "format": "sentiment-student-design-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "judge_version": config.version,
        "source_hash": digest(examples),
        "inputs_hash": digest(ordered),
        "reviewed_ids": sorted({ex["input_id"] for ex in reviewed}),
        "reviewed_groups": sorted({ex["group_id"] for ex in reviewed}),
        "reviewed_files": [str(p) for p in reviewed_paths],
        "original_messages": len(examples),
        "retained_messages": len(kept),
        "splits": {
            s: {
                "messages": sum(ex["split"] == s for ex in kept),
                "sessions": len({ex["group_id"] for ex in kept if ex["split"] == s}),
                "hash": digest(read_jsonl(output / f"{s}.jsonl")),
            }
            for s in ("train", "calibration", "test")
        },
        "exclusion_counts": dict(Counter(ex["reason"] for ex in exclusions)),
        "near_duplicate_pairs": len(pairs),
        "duplicate_policy": "NFKC/casefold/whitespace exact; >=80 chars, hashed char-5 cosine>=0.95, length ratio>=0.85, verified shingle Jaccard>=0.85; transitive components. Retain component in test > calibration > train. Preserve within-split multiplicities. Exclude reviewed sessions/components from test.",
        "input_policy": "Full teacher message fits frozen max_input_chars including system/serialization; no source labels in selection or inputs; no class enrichment; no silent truncation.",
        "evaluation_scope": "Teacher agreement on eligible previously unseen sessions within an explored public corpus; not independent sentiment ground truth or production validation.",
    }
    write_json(output / "design.json", manifest)
    return manifest
