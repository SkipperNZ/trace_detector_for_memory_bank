"""Autonomous labeling, training, evaluation and export. No LLM status polling."""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from memory_trace.feedback import feedback_config
from memory_trace.feedback_data import read, verify_deployment, remaining_estimate
from memory_trace.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("runs/sentiment/examples.jsonl"))
    p.add_argument("--base", type=Path, default=Path("runs/feedback-v1"))
    p.add_argument("--config", type=Path, default=Path("configs/judge.local.json"))
    p.add_argument("--policy", type=Path, default=Path("configs/feedback-study.json"))
    p.add_argument("--model-output", type=Path, default=Path("artifacts/feedback-v1"))
    p.add_argument("--report", type=Path)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--qwen-compose-dir", type=Path)
    p.add_argument("--stop-after-pilot", action="store_true")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = a.base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    if (base / "complete.json").exists():
        print("Study already complete:", base / "report.md", flush=True)
        return
    state = (
        read(base / "progress.json")
        if (base / "progress.json").exists()
        else {"stages": [], "started_at": datetime.now(timezone.utc).isoformat()}
    )
    state.update(status="running")
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
    common = [
        "--base",
        str(base),
        "--source",
        str(a.source.resolve()),
        "--config",
        str(a.config.resolve()),
        "--policy",
        str(a.policy.resolve()),
        "--model-output",
        str(a.model_output.resolve()),
        "--device",
        a.device,
    ]
    if a.report:
        common += ["--report", str(a.report.resolve())]

    def stage(name, command=None, *, cwd=root, resume=True):
        if resume and any(
            r["stage"] == name and r["status"] == "complete" for r in state["stages"]
        ):
            return
        command = command or [
            sys.executable,
            str(root / "scripts/feedback_workflow.py"),
            name,
            *common,
        ]
        row = {
            "stage": name,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
        }
        state["stages"].append(row)
        write_json(base / "progress.json", state)
        print("START", name, flush=True)
        started = time.perf_counter()
        with (base / (name + ".log")).open("a", encoding="utf-8") as log:
            process = subprocess.run(
                command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        row.update(
            status="complete" if process.returncode == 0 else "error",
            exit_code=process.returncode,
            elapsed_seconds=time.perf_counter() - started,
        )
        write_json(base / "progress.json", state)
        print("END", name, row["status"], flush=True)
        if process.returncode:
            raise RuntimeError(f"Stage {name} failed; see {base / (name+'.log')}")

    compose_dir = a.qwen_compose_dir.resolve() if a.qwen_compose_dir else None
    compose = ["docker", "compose", "--profile", "llama"]
    runtime_path = base / "qwen-runtime-control.json"
    stopped = runtime_path.exists() and read(runtime_path).get("restore_required", False)

    def restore():
        nonlocal stopped
        if stopped:
            if compose_dir is None:
                raise ValueError(
                    "Interrupted GPU run requires the original --qwen-compose-dir to restore Qwen"
                )
            stage(
                "qwen-restore", compose + ["start", "qwen38-llama"], cwd=compose_dir, resume=False
            )
            deadline = time.monotonic() + 300
            while True:
                try:
                    verify_deployment(feedback_config(a.config), base / "deployment-restored.json")
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "Qwen restore command succeeded but API readiness timed out"
                        )
                    time.sleep(15)
            stopped = False
            write_json(
                runtime_path, {"was_running": True, "restore_required": False, "api_verified": True}
            )

    try:
        restore()
        # Always validate source/config/policy on resume, even when prepare has completed.
        stage("prepare", resume=False)
        stage("pilot")
        eta = remaining_estimate(base)
        print(
            "Conservative remaining hours including training:",
            round(eta["conservative_remaining_seconds_with_training"] / 3600, 2),
            flush=True,
        )
        if a.stop_after_pilot:
            state["status"] = "pilot_complete"
            return
        stage("label")
        stage("tfidf")
        try:
            need_gpu = any(
                not any(
                    r["stage"] == s["name"] and r["status"] == "complete" for r in state["stages"]
                )
                for s in read(base / "design.json")["policy"]["encoders"]
            )
            if a.device == "cuda" and compose_dir and need_gpu:
                running = subprocess.run(
                    compose + ["ps", "--status", "running", "--services"],
                    cwd=compose_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if "qwen38-llama" in running.stdout.splitlines():
                    stopped = True
                    write_json(runtime_path, {"was_running": True, "restore_required": True})
                    stage(
                        "qwen-stop",
                        compose + ["stop", "-t", "30", "qwen38-llama"],
                        cwd=compose_dir,
                        resume=False,
                    )
            for spec in read(base / "design.json")["policy"]["encoders"]:
                stage(spec["name"])
            stage("select")
        finally:
            restore()
        stage("evaluate")
        stage("report")
        state.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
        write_json(base / "complete.json", state)
    except Exception as exc:
        state.update(status="error", error_type=type(exc).__name__, error=str(exc))
        write_json(base / "error.json", state)
        raise
    finally:
        write_json(base / "progress.json", state)


if __name__ == "__main__":
    main()
