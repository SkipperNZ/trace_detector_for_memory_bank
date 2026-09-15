"""Finish this one-off experiment, optionally pausing and restoring the user's Qwen service.

This is not a scheduled automation. It waits for the already running teacher process,
then executes the remaining frozen stages. No system-specific path is hardcoded.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil
from memory_trace.io import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-pid", type=int)
    parser.add_argument("--qwen-compose-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root / "runs/sentiment-students-v1"
    state = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": [],
        "status": "running",
    }
    state_path = base / "completion-progress.json"
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "TOKENIZERS_PARALLELISM": "false",
    }

    def stage(name, command, cwd=root):
        print("START", name, flush=True)
        record = {
            "stage": name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        state["stages"].append(record)
        write_json(state_path, state)
        started = time.perf_counter()
        with (base / f"completion-{name}.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(
            status="complete" if result.returncode == 0 else "error",
            exit_code=result.returncode,
            elapsed_seconds=time.perf_counter() - started,
        )
        write_json(state_path, state)
        print("END", name, "exit", result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(f"Stage {name} failed; see its saved log")

    def study(name, device=None):
        command = [sys.executable, "scripts/run_sentiment_study.py", name]
        if device:
            command += ["--device", device]
        stage(name, command)

    stopped = False
    try:
        if args.teacher_pid:
            try:
                process = psutil.Process(args.teacher_pid)
                identity = process.create_time()
                if not any(
                    str(arg).replace("\\", "/").endswith("/label_sentiment_students.py")
                    for arg in process.cmdline()
                ):
                    raise ValueError("Provided PID is not the expected teacher script")
                state["waiting_for_teacher"] = {"pid": args.teacher_pid, "create_time": identity}
                write_json(state_path, state)
                print("Waiting for the active teacher run", args.teacher_pid, flush=True)
                while process.is_running() and process.create_time() == identity:
                    time.sleep(15)
            except psutil.NoSuchProcess:
                pass
        # Refresh exports and retry recorded errors using the current robust retry
        # implementation. Existing successful request IDs do not trigger API calls.
        stage("teacher-finalize", [sys.executable, "scripts/label_sentiment_students.py"])
        summary = json.loads((base / "teacher/summary.json").read_text(encoding="utf-8"))
        if summary["unattempted_calls"] or summary["completed_calls"] != summary["planned_calls"]:
            raise ValueError("Teacher run is incomplete")
        write_json(
            base / "teacher-complete.json",
            {"completed_at": datetime.now(timezone.utc).isoformat(), "summary": summary},
        )
        study("baselines")
        baseline = json.loads((base / "baseline-selection.json").read_text(encoding="utf-8"))
        if baseline["finetune_required"]:
            if args.device == "cuda" and args.qwen_compose_dir:
                directory = args.qwen_compose_dir.resolve()
                if not (directory / "docker-compose.yml").is_file():
                    raise ValueError("Qwen compose directory does not contain docker-compose.yml")
                status = subprocess.run(
                    [
                        "docker",
                        "compose",
                        "--profile",
                        "llama",
                        "ps",
                        "--status",
                        "running",
                        "--services",
                    ],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if "qwen38-llama" in status.stdout.splitlines():
                    # Restore even if stopping succeeds but a later stage fails.
                    stopped = True
                    write_json(
                        base / "qwen-runtime-control.json",
                        {
                            "compose_directory": str(directory),
                            "service": "qwen38-llama",
                            "was_running": True,
                            "restore_required": True,
                        },
                    )
                    stage(
                        "qwen-stop",
                        [
                            "docker",
                            "compose",
                            "--profile",
                            "llama",
                            "stop",
                            "-t",
                            "30",
                            "qwen38-llama",
                        ],
                        cwd=directory,
                    )
            study("finetune", args.device)
            if stopped:
                stage(
                    "qwen-restore",
                    ["docker", "compose", "--profile", "llama", "start", "qwen38-llama"],
                    cwd=args.qwen_compose_dir.resolve(),
                )
                stopped = False
                write_json(
                    base / "qwen-runtime-control.json",
                    {
                        "compose_directory": str(args.qwen_compose_dir.resolve()),
                        "service": "qwen38-llama",
                        "was_running": True,
                        "restore_required": False,
                        "restore_command_succeeded": True,
                    },
                )
        study("select")
        study("evaluate", "cpu")
        study("report")
        state.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
        write_json(base / "complete.json", state)
        print("Complete student study finished", flush=True)
    except Exception as exc:
        state.update(status="error", error_type=type(exc).__name__, error=str(exc))
        write_json(state_path, state)
        raise
    finally:
        if stopped:
            stage(
                "qwen-restore-after-error",
                ["docker", "compose", "--profile", "llama", "start", "qwen38-llama"],
                cwd=args.qwen_compose_dir.resolve(),
            )
            write_json(
                base / "qwen-runtime-control.json",
                {
                    "compose_directory": str(args.qwen_compose_dir.resolve()),
                    "service": "qwen38-llama",
                    "was_running": True,
                    "restore_required": False,
                    "restore_command_succeeded": True,
                },
            )


if __name__ == "__main__":
    main()
