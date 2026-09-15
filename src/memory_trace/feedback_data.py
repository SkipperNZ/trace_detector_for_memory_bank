"""Immutable full-corpus census and separate leakage-controlled training partitions."""

import json
import os
import sqlite3
import time
import urllib.request
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .feedback import TASKS, CLASSES, feedback_config, feedback_input, canonical_key, student_inputs
from .io import canonical, digest, index_rows, read_jsonl, write_json, write_jsonl
from .judge import messages, run_judge, _NoRedirect
from .prepare import split_for
from .student_data import clean_splits


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_feedback(source, config_path, policy_path, base):
    base = Path(base)
    config = feedback_config(config_path)
    policy = read(policy_path)
    examples = [feedback_input(ex) for ex in read_jsonl(source)]
    if not examples:
        raise ValueError("Feedback corpus is empty")
    index_rows(examples)
    for ex in examples:
        if ex.get("split") not in {"train", "calibration", "test"}:
            ex["split"] = split_for(ex["tenant_id"], ex["group_id"], policy["seed"])
    design = {
        "format": "feedback-data-v1",
        "source_hash": digest(examples),
        "policy": policy,
        "policy_hash": digest(policy),
        "judge_version": config.version,
        "prompt": config.prompt[1],
        "classes": CLASSES,
        "tasks": TASKS,
        "census_messages": len(examples),
        "census_sessions": len({ex["group_id"] for ex in examples}),
        "test_scope": "New task labels on session-held-out portion of an already explored public corpus; not independent human gold or corporate validation.",
    }
    if (base / "design.json").exists():
        old = read(base / "design.json")
        if any(old[k] != design[k] for k in ("source_hash", "policy_hash", "judge_version")):
            raise ValueError("Frozen feedback design differs; use a new output directory")
        return old
    canonical_rows, mapping = {}, []
    for ex in sorted(examples, key=lambda ex: ex["input_id"]):
        key = canonical_key(ex)
        representative = canonical_rows.setdefault(key, ex)
        mapping.append(
            {"input_id": ex["input_id"], "canonical_input_id": representative["input_id"]}
        )
    canonical_examples = list(canonical_rows.values())
    eligible = [
        ex
        for ex in examples
        if sum(len(m["content"]) for m in messages(ex, config.task, system_prompt=config.prompt[1]))
        <= config.max_input_chars
    ]
    eligible_ids = {ex["input_id"] for ex in eligible}
    # Deduplication examines the complete student-visible prefix, not isolated snippets.
    kept_views, exclusions, pairs = clean_splits(
        student_inputs(eligible), [], replace(config, max_input_chars=10**9)
    )
    kept_ids = {ex["input_id"] for ex in kept_views}
    kept = [ex for ex in eligible if ex["input_id"] in kept_ids]
    exclusions += [
        {"input_id": ex["input_id"], "split": ex["split"], "reason": "judge_input_limit"}
        for ex in examples
        if ex["input_id"] not in eligible_ids
    ]
    pilot = sorted(
        [
            ex
            for ex in canonical_examples
            if ex["split"] == "train" and ex["input_id"] in eligible_ids
        ],
        key=lambda ex: digest([policy["seed"], "feedback-pilot", ex["input_id"]]),
    )[: policy["pilot_messages"]]
    write_jsonl(base / "data/examples.jsonl", examples)
    write_jsonl(base / "data/canonical.jsonl", canonical_examples)
    write_jsonl(base / "data/mapping.jsonl", mapping)
    write_jsonl(base / "data/pilot.jsonl", pilot)
    write_jsonl(base / "data/exclusions.jsonl", exclusions)
    write_jsonl(base / "data/near-duplicate-pairs.jsonl", pairs)
    design.update(
        created_at=datetime.now(timezone.utc).isoformat(),
        canonical_messages=len(canonical_examples),
        training_eligible_messages=len(kept),
        exclusions=dict(Counter(x["reason"] for x in exclusions)),
        splits={},
    )
    for split in ("train", "calibration", "test"):
        rows = sorted(
            [ex for ex in kept if ex["split"] == split],
            key=lambda ex: digest([policy["seed"], ex["input_id"]]),
        )
        write_jsonl(base / f"data/{split}.jsonl", rows)
        design["splits"][split] = {
            "messages": len(rows),
            "sessions": len({ex["group_id"] for ex in rows}),
            "hash": digest(rows),
        }
    write_json(base / "design.json", design)
    return design


def verify_deployment(config, output):
    token = os.environ.get(config.api_key_env) or (
        Path(config.api_key_file).read_text(encoding="utf-8-sig").strip()
        if config.api_key_file
        else ""
    )
    request = urllib.request.Request(
        config.base_url.rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + token} if token else {},
    )
    with urllib.request.build_opener(_NoRedirect()).open(request, timeout=20) as response:
        metadata = json.load(response)
    found = [m for m in metadata["data"] if m.get("id") == config.model]
    if len(found) != 1:
        raise ValueError("Expected Qwen model unavailable")
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "expected_model_available": True,
        "model_id": config.model,
    }
    write_json(output, result)
    return result


def label_feedback(base, config_path, *, pilot=False):
    base = Path(base)
    config = feedback_config(config_path)
    design = read(base / "design.json")
    if config.version != design["judge_version"]:
        raise ValueError("Feedback judge changed after design freeze")
    if digest(read_jsonl(base / "data/examples.jsonl")) != design["source_hash"]:
        raise ValueError("Census inputs changed")
    verify_deployment(config, base / "deployment.json")
    rows = read_jsonl(base / ("data/pilot.jsonl" if pilot else "data/canonical.jsonl"))
    out = base / ("pilot" if pilot else "teacher")
    out.mkdir(parents=True, exist_ok=True)
    if not pilot:
        if not read(base / "pilot-gate.json")["passed"]:
            raise ValueError("Pilot technical validity gate not met")
        # Exact requests, including IDs, are copied from the pilot journal.
        with sqlite3.connect(out / "calls.sqlite3") as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, record TEXT NOT NULL)"
            )
            for call in read_jsonl(base / "pilot/calls.jsonl"):
                if call["judge_version"] != config.version:
                    raise ValueError("Pilot judge differs")
                db.execute(
                    "INSERT OR IGNORE INTO calls VALUES (?,?)", (call["call_id"], canonical(call))
                )
            db.commit()
    started = time.perf_counter()

    def progress(done, total):
        if done == 16 or done % 100 == 0 or done == total:
            write_json(
                base / "label-progress.json",
                {
                    "stage": "pilot" if pilot else "teacher",
                    "completed": done,
                    "total": total,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )

    result = run_judge(rows, config, out, repeats=1, progress=progress)
    if result["failed_calls"] or result["unattempted_calls"]:
        result = run_judge(rows, config, out, repeats=1, retry_errors=True, progress=progress)
    if result["unattempted_calls"]:
        raise RuntimeError("Teacher run incomplete; resumable journal retained")
    if pilot:
        calls = read_jsonl(out / "calls.jsonl")
        seconds = [sum(a["elapsed_seconds"] for a in call["attempts"]) for call in calls]
        passed = (
            bool(rows)
            and result["successful_calls"] / len(rows)
            >= design["policy"]["pilot_min_valid_fraction"]
        )
        estimate = (
            (sum(seconds) / max(len(seconds), 1))
            * (design["canonical_messages"] - len(rows))
            / config.concurrency
        )
        gate = {
            "passed": passed,
            "technical_validity_only": True,
            "not_a_quality_gold": True,
            "summary": result,
            "mean_seconds_per_call": sum(seconds) / max(len(seconds), 1),
            "estimated_remaining_label_seconds": estimate,
            "conservative_remaining_label_seconds": estimate * 1.35,
            "estimated_total_remaining_seconds_with_training": estimate * 1.35 + 2400,
            "measured_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(base / "pilot-gate.json", gate)
        if not passed:
            raise RuntimeError(
                "Pilot validity below threshold; inspect local unresolved records before bulk calls"
            )
        return gate
    labels = index_rows(read_jsonl(out / "labels.jsonl"))
    expanded = []
    for link in read_jsonl(base / "data/mapping.jsonl"):
        if link["canonical_input_id"] in labels:
            expanded.append({**labels[link["canonical_input_id"]], **link})
    write_jsonl(base / "labels.jsonl", expanded)
    census = read_jsonl(base / "data/examples.jsonl")
    by_id = index_rows(expanded)
    stats = {
        "messages": len(census),
        "canonical_requests": len(rows),
        "labelled_messages": len(expanded),
        "unlabelled_messages": len(census) - len(expanded),
        "tasks": {},
        "reason_tags": dict(Counter(t for r in expanded for t in r["reason_tags"])),
        "teacher_summary": result,
        "labels_hash": digest(expanded),
    }
    for task in TASKS:
        counts = Counter(r["signals"][task]["label"] for r in expanded)
        unknown = len(census) - len(expanded)
        stats["tasks"][task] = {
            "counts": {c: counts[c] for c in CLASSES},
            "yes_fraction_all_messages": counts["yes"] / len(census),
            "yes_fraction_among_technical_successes": counts["yes"] / max(len(expanded), 1),
            "yes_fraction_bounds_with_unclear_and_missing": [
                counts["yes"] / len(census),
                (counts["yes"] + counts["unclear"] + unknown) / len(census),
            ],
            "sessions_with_yes": len(
                {
                    ex["group_id"]
                    for ex in census
                    if ex["input_id"] in by_id
                    and by_id[ex["input_id"]]["signals"][task]["label"] == "yes"
                }
            ),
        }
    stats["overlap"] = dict(
        Counter("/".join(r["signals"][t]["label"] for t in TASKS) for r in expanded)
    )
    write_json(base / "census.json", stats)
    write_json(
        base / "labels-frozen.json",
        {
            "judge_version": config.version,
            "labels_hash": digest(expanded),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return stats
