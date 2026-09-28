"""Benchmark pinned GLiNER2.5-Decide on the existing frozen feedback labels."""

import argparse
import gc
import os
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("runs/feedback-v1"))
    p.add_argument("--out", type=Path, default=Path("runs/gliner25-decide-v1"))
    p.add_argument("--policy", type=Path, default=Path("configs/gliner25-feedback.json"))
    p.add_argument(
        "--report", type=Path, default=Path("docs/experiments/gliner25-decide-2026-09-28.md")
    )
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/hf"))
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--stage", choices=["all", "score", "latency", "report"], default="all")
    p.add_argument(
        "--download", action="store_true", help="Permit downloading the pinned public checkpoint"
    )
    p.add_argument("--qwen-compose-dir", type=Path)
    p.add_argument("--qwen-config", type=Path, default=Path("configs/judge.local.json"))
    a = p.parse_args()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if not a.download:
        os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from huggingface_hub import snapshot_download
    from memory_trace.feedback_data import read, verify_deployment
    from memory_trace.feedback import feedback_config
    from memory_trace.io import write_json
    from memory_trace.gliner_feedback import (
        DecidePredictor,
        build_manifest,
        calibrate_decide,
        evaluate_decide,
        measure_cpu,
        now,
        report_decide,
    )

    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    if a.stage == "report":
        report_decide(a.source, a.out, a.report)
        print("REPORT COMPLETE", a.report, flush=True)
        return
    policy = read(a.policy)
    snapshot = snapshot_download(
        policy["model"],
        revision=policy["revision"],
        cache_dir=str(a.cache_dir),
        local_files_only=not a.download,
        allow_patterns=["*.json", "model.safetensors"],
        token=False,
    )
    manifest = build_manifest(a.source, a.out, policy, snapshot)
    compose = ["docker", "compose", "--profile", "llama"]
    control_path = a.out / "qwen-runtime-control.json"
    control = read(control_path) if control_path.exists() else {}
    restore_required = control.get("restore_required", False)

    def command(args):
        if a.qwen_compose_dir is None:
            raise ValueError("Pass the original --qwen-compose-dir to restore Qwen")
        return subprocess.run(
            compose + args, cwd=a.qwen_compose_dir, check=True, capture_output=True, text=True
        )

    def restore():
        nonlocal restore_required
        if not restore_required:
            return
        print("RESTORE QWEN", flush=True)
        command(["start", "qwen38-llama"])
        deadline = time.monotonic() + 300
        while True:
            try:
                verify_deployment(feedback_config(a.qwen_config), a.out / "qwen-restored.json")
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Qwen restore API timeout")
                time.sleep(10)
        restore_required = False
        write_json(
            control_path, {"restore_required": False, "api_verified": True, "updated_at": now()}
        )

    predictor = None
    try:
        restore()
        if a.stage in {"all", "score"} and not (a.out / "evaluation.json").exists():
            if a.device == "cuda" and a.qwen_compose_dir:
                running = command(["ps", "--status", "running", "--services"])
                if "qwen38-llama" in running.stdout.splitlines():
                    verify_deployment(
                        feedback_config(a.qwen_config), a.out / "qwen-before-stop.json"
                    )
                    write_json(control_path, {"restore_required": True, "updated_at": now()})
                    restore_required = True
                    print("STOP QWEN FOR GPU BENCHMARK", flush=True)
                    command(["stop", "-t", "30", "qwen38-llama"])
            print("LOAD GLINER", a.device, flush=True)
            write_json(
                a.out / "execution.json",
                {
                    "scoring_device": a.device,
                    "dtype": "float32",
                    "threads": 8,
                    "started_at": now(),
                },
            )
            predictor = DecidePredictor(snapshot, policy, device=a.device)
            selection = calibrate_decide(predictor, a.source, a.out, manifest)
            evaluate_decide(predictor, a.source, a.out, manifest, selection)
        elif (a.out / "selection.json").exists():
            selection = read(a.out / "selection.json")
            if selection["manifest_hash"] != manifest["fingerprint"]:
                raise ValueError("Selection belongs to another benchmark")
    except Exception as exc:
        write_json(
            a.out / "error.json", {"at": now(), "type": type(exc).__name__, "error": str(exc)}
        )
        raise
    finally:
        predictor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        restore()
    if a.stage in {"all", "latency"}:
        print("CPU LATENCY: 100 FIXED INPUTS", flush=True)
        if not (a.out / "runtime.json").exists():
            predictor = DecidePredictor(snapshot, policy, device="cpu")
            measure_cpu(predictor, a.source, a.out, selection)
        report_decide(a.source, a.out, a.report)
        print("BENCHMARK COMPLETE", a.report, flush=True)


if __name__ == "__main__":
    main()
