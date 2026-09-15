"""Read the running study status without API calls or loading message text."""

import json
import sqlite3
import time
from pathlib import Path

base = Path("runs/sentiment-students-v1")
design = json.loads((base / "data/design.json").read_text(encoding="utf-8"))
database = base / "teacher/calls.sqlite3"
with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
    statuses = dict(
        connection.execute(
            "SELECT json_extract(record, '$.runtime_status'), COUNT(*) FROM calls GROUP BY 1"
        )
    )
result = {
    "planned": design["retained_messages"],
    "completed": sum(statuses.values()),
    "statuses": statuses,
}
if (base / "completion-progress.json").exists():
    completion = json.loads((base / "completion-progress.json").read_text(encoding="utf-8"))
    result["pipeline_status"] = completion["status"]
    result["stages"] = completion["stages"]
    waiting = completion.get("waiting_for_teacher")
    if waiting and not completion["stages"]:
        reused = len(json.loads((base / "cache-reuse.json").read_text(encoding="utf-8")))
        elapsed = time.time() - waiting["create_time"]
        rate = (result["completed"] - reused) / max(elapsed, 1)
        result["elapsed_minutes"] = round(elapsed / 60, 1)
        result["estimated_teacher_remaining_minutes"] = round(
            (result["planned"] - result["completed"]) / max(rate, 1e-9) / 60, 1
        )
print(json.dumps(result, ensure_ascii=True, indent=2))
