"""Run a frozen stage. No stage chooses a model or threshold using final test results."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse
from pathlib import Path
from memory_trace.student_study import freeze_study, fit_baselines, freeze_winner

parser = argparse.ArgumentParser()
parser.add_argument(
    "stage", choices=["freeze", "baselines", "finetune", "select", "evaluate", "report"]
)
parser.add_argument("--base", type=Path, default=Path("runs/sentiment-students-v1"))
parser.add_argument("--policy", type=Path, default=Path("configs/student-study.json"))
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
args = parser.parse_args()
if args.stage == "freeze":
    result = freeze_study(args.base, args.policy)
elif args.stage == "baselines":
    result = fit_baselines(args.base)
elif args.stage == "select":
    result = freeze_winner(args.base)
elif args.stage == "finetune":
    from memory_trace.student_finetune import finetune

    result = finetune(args.base, device=args.device)
elif args.stage == "evaluate":
    from memory_trace.student_evaluation import evaluate_study

    result = evaluate_study(args.base, device=args.device)
else:
    from memory_trace.student_report import render_report

    result = render_report(args.base)
print("Completed study stage:", args.stage, flush=True)
