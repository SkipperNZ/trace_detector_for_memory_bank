"""Run one backbone comparison with reused labels, stage logs, and Qwen restoration.

The subprocess runner owns all stage transitions; it needs no LLM status polling.
Paths are arguments so the experiment also runs on Linux/macOS without Docker.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from memory_trace.io import write_json
from memory_trace.student_study import freeze_study, read_json


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("runs/sentiment-students-v1"))
    parser.add_argument("--base", type=Path, default=Path("runs/sentiment-bge-m3-v1"))
    parser.add_argument("--policy", type=Path, default=Path("configs/student-study-bge-m3.json"))
    parser.add_argument("--model-output", type=Path, default=Path("artifacts/sentiment-bge-m3-v1"))
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--qwen-compose-dir", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    base = args.base.resolve()
    source = args.source.resolve()
    if base == source or base.is_relative_to(source):
        raise ValueError("Use a separate experiment directory")
    base.mkdir(parents=True, exist_ok=True)
    if (base / "complete.json").exists():
        print("This frozen comparison has already completed.", flush=True)
        return
    reused = {}
    files = list((source / "data").glob("*.json*")) + [
        source / "teacher/labels.jsonl",
        source / "teacher/calls.jsonl",
        source / "teacher/summary.json",
        source / "cache-reuse.json",
    ]
    for src in files:
        relative = src.relative_to(source)
        dst = base / relative
        checksum = file_hash(src)
        if dst.exists() and file_hash(dst) != checksum:
            raise ValueError(f"Previously copied input changed: {relative}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        reused[relative.as_posix()] = checksum
    write_json(
        base / "reused-inputs.json",
        {
            "source_experiment": source.name,
            "file_sha256": reused,
            "new_teacher_calls": 0,
            "existing_test_previously_inspected": True,
        },
    )
    freeze_study(base, args.policy)
    state = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "stages": [],
    }
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    }

    def stage(name, command, cwd=root):
        record = {
            "stage": name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        state["stages"].append(record)
        write_json(base / "progress.json", state)
        print("START", name, flush=True)
        started = time.perf_counter()
        with (base / f"{name}.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(
            status="complete" if result.returncode == 0 else "error",
            exit_code=result.returncode,
            elapsed_seconds=time.perf_counter() - started,
        )
        write_json(base / "progress.json", state)
        print("END", name, record["status"], flush=True)
        if result.returncode:
            raise RuntimeError(f"Stage {name} failed; inspect {base / (name+'.log')}")

    stopped = False
    compose_dir = args.qwen_compose_dir.resolve() if args.qwen_compose_dir else None
    compose = ["docker", "compose", "--profile", "llama"]
    try:
        try:
            if args.device == "cuda" and compose_dir:
                if not (compose_dir / "docker-compose.yml").is_file():
                    raise ValueError("Compose file missing")
                running = subprocess.run(
                    compose + ["ps", "--status", "running", "--services"],
                    cwd=compose_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if "qwen38-llama" in running.stdout.splitlines():
                    stopped = True
                    write_json(
                        base / "qwen-runtime-control.json",
                        {"was_running": True, "restore_required": True},
                    )
                    stage(
                        "qwen-stop", compose + ["stop", "-t", "30", "qwen38-llama"], cwd=compose_dir
                    )
            stage(
                "embeddings",
                [
                    sys.executable,
                    "scripts/embed_sentiment_students.py",
                    "--base",
                    str(base),
                    "--device",
                    args.device,
                    "--verify",
                ],
            )
            for name in ("baselines", "finetune", "select"):
                stage(
                    name,
                    [
                        sys.executable,
                        "scripts/run_sentiment_study.py",
                        name,
                        "--base",
                        str(base),
                        "--device",
                        args.device,
                    ],
                )
        finally:
            if stopped:
                stage("qwen-restore", compose + ["start", "qwen38-llama"], cwd=compose_dir)
                write_json(
                    base / "qwen-runtime-control.json",
                    {
                        "was_running": True,
                        "restore_required": False,
                        "restore_command_succeeded": True,
                    },
                )
        stage(
            "evaluate",
            [
                sys.executable,
                "scripts/run_sentiment_study.py",
                "evaluate",
                "--base",
                str(base),
                "--device",
                "cpu",
            ],
        )
        stage(
            "report",
            [
                sys.executable,
                "scripts/compare_sentiment_backbones.py",
                "--base",
                str(base),
                "--reference",
                str(source),
                "--model-output",
                str(args.model_output),
            ],
        )
        state.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
        write_json(base / "complete.json", state)
    except Exception as exc:
        state.update(status="error", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(base / "progress.json", state)


if __name__ == "__main__":
    main()
