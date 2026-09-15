"""Bounded stability trial; valid new labels are merged into the main request journal."""

import argparse
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from memory_trace.feedback import feedback_config
from memory_trace.feedback_data import read, verify_deployment
from memory_trace.io import canonical, read_jsonl, write_json
from memory_trace.judge import run_judge


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("runs/feedback-v1"))
    p.add_argument("--config", type=Path, default=Path("configs/judge.local.json"))
    p.add_argument("--name", default="no-cuda-graphs-v1")
    p.add_argument("--messages", type=int, default=256)
    a = p.parse_args()
    if not a.name or Path(a.name).name != a.name or a.messages < 1:
        raise ValueError("Expected a simple trial name and a positive message count")
    cfg = feedback_config(a.config)
    if cfg.version != read(a.base / "design.json")["judge_version"]:
        raise ValueError("Teacher configuration differs from the frozen study")
    out = a.base / "runtime-validation" / a.name
    deadline = time.monotonic() + 180
    while True:
        try:
            verify_deployment(cfg, out / "deployment.json")
            break
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(10)
    main_journal = a.base / "teacher/calls.sqlite3"
    with closing(sqlite3.connect(main_journal)) as db:
        previous = {k: json.loads(v) for k, v in db.execute("SELECT id, record FROM calls")}
    if (out / "inputs.jsonl").exists():
        selected = read_jsonl(out / "inputs.jsonl")
    else:
        from memory_trace.io import write_jsonl

        done = {
            c["input_id"]
            for c in previous.values()
            if c["runtime_status"] in {"ok", "not_evaluated"}
        }
        selected = [
            r for r in read_jsonl(a.base / "data/canonical.jsonl") if r["input_id"] not in done
        ][: a.messages]
        write_jsonl(out / "inputs.jsonl", selected)
    if not selected:
        raise ValueError("No pending requests")
    print("Starting runtime stability trial:", len(selected), "pending requests", flush=True)
    started = time.perf_counter()
    result = run_judge(selected, cfg, out, repeats=1, retry_errors=True)
    if result["unattempted_calls"] and not result.get("stopped_due_to_transport"):
        # A persistently malformed first generation is not a server crash.
        result = run_judge(selected, cfg, out, repeats=1, retry_errors=True)
    records = read_jsonl(out / "calls.jsonl")
    # Never overwrite a successful result; retain all failed attempt history on replacement.
    with closing(sqlite3.connect(main_journal)) as db:
        for c in records:
            old = previous.get(c["call_id"])
            if old and old["runtime_status"] in {"ok", "not_evaluated"}:
                continue
            if old:
                c["previous_runs"] = old.get("previous_runs", []) + [
                    {k: v for k, v in old.items() if k != "previous_runs"}
                ]
            db.execute("INSERT OR REPLACE INTO calls VALUES (?,?)", (c["call_id"], canonical(c)))
        db.commit()
    elapsed = time.perf_counter() - started
    summary = {
        "teacher": result,
        "elapsed_seconds": elapsed,
        "mean_seconds_per_completed_call": sum(
            sum(attempt["elapsed_seconds"] for attempt in c["attempts"]) for c in records
        )
        / max(sum(bool(c["attempts"]) for c in records), 1),
        "technical_stability_only": True,
        "valid_labels_reused_in_main_journal": True,
    }
    write_json(out / "stability.json", summary)
    print(summary, flush=True)
    if result.get("stopped_due_to_transport") or result["unattempted_calls"]:
        raise RuntimeError(
            "Server failed during runtime stability trial; completed labels retained"
        )


if __name__ == "__main__":
    main()
