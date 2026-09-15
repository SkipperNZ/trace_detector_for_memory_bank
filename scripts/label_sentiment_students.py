"""Resume the frozen teacher run; reuse only byte-equivalent versioned requests."""

import json
import os
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from memory_trace.io import canonical, digest, read_jsonl, write_json
from memory_trace.judge import _NoRedirect, messages, read_config, run_judge


def main():
    base = Path("runs/sentiment-students-v1")
    output = base / "teacher"
    output.mkdir(parents=True, exist_ok=True)
    config = read_config(Path("configs/judge.local.json"))
    design = json.loads((base / "data/design.json").read_text(encoding="utf-8"))
    examples = read_jsonl(base / "data/examples.jsonl")
    if config.version != design["judge_version"] or digest(examples) != design["inputs_hash"]:
        raise ValueError("Frozen teacher or input design changed")
    token = os.environ.get(config.api_key_env)
    if not token and config.api_key_file:
        token = Path(config.api_key_file).read_text(encoding="utf-8-sig").strip()
    request = urllib.request.Request(
        config.base_url.rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + token} if token else {},
    )
    with urllib.request.build_opener(_NoRedirect()).open(request, timeout=20) as response:
        metadata = json.load(response)
    models = [m for m in metadata["data"] if m["id"] == config.model]
    if len(models) != 1:
        raise ValueError("Expected served model is unavailable")
    if not (base / "deployment.json").exists():
        write_json(
            base / "deployment.json",
            {
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "models": models,
                "weights_sha256": None,
            },
        )
    print("Verified model:", config.model, "frozen inputs:", len(examples), flush=True)
    by_id = {ex["input_id"]: ex for ex in examples}
    reused = []
    with sqlite3.connect(output / "calls.sqlite3") as db:
        db.execute("CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, record TEXT NOT NULL)")
        for path in sorted(Path("runs/sentiment-tuning-v1").glob("*/calls.jsonl")):
            for call in read_jsonl(path):
                ex = by_id.get(call["input_id"])
                if (
                    ex is None
                    or call["judge_version"] != config.version
                    or call["repeat"] != 0
                    or call["runtime_status"] != "ok"
                ):
                    continue
                if call["prompt_hash"] != digest(
                    messages(ex, config.task, system_prompt=config.prompt[1])
                ):
                    raise ValueError("Cached request prompt differs")
                if call["call_id"] != digest([ex["input_id"], config.version, 0]):
                    raise ValueError("Cached call identity differs")
                cursor = db.execute(
                    "INSERT OR IGNORE INTO calls VALUES (?, ?)", (call["call_id"], canonical(call))
                )
                if cursor.rowcount:
                    reused.append({"call_id": call["call_id"], "source": str(path)})
        db.commit()
    if not (base / "cache-reuse.json").exists():
        write_json(base / "cache-reuse.json", reused)
    started = time.perf_counter()

    def progress(done, total):
        if done == 1 or done % 25 == 0 or done == total:
            print(f"Teacher {done}/{total}; wall {time.perf_counter()-started:.1f}s", flush=True)

    result = run_judge(examples, config, output, repeats=1, progress=progress)
    if result["unattempted_calls"]:
        raise ValueError("Teacher canary failed; incomplete run retained for resume")
    if result["failed_calls"]:
        print("Retrying failed requests once, preserving attempts", flush=True)
        result = run_judge(
            examples, config, output, repeats=1, retry_errors=True, progress=progress
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
