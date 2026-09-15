"""Individual resumable stages of the feedback study."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "pilot",
            "label",
            "tfidf",
            "e5",
            "bge-m3",
            "select",
            "evaluate",
            "report",
        ],
    )
    parser.add_argument("--base", type=Path, default=Path("runs/feedback-v1"))
    parser.add_argument("--source", type=Path, default=Path("runs/sentiment/examples.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/judge.local.json"))
    parser.add_argument("--policy", type=Path, default=Path("configs/feedback-study.json"))
    parser.add_argument("--model-output", type=Path, default=Path("artifacts/feedback-v1"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    a = parser.parse_args()
    if a.stage == "prepare":
        from memory_trace.feedback_data import prepare_feedback

        prepare_feedback(a.source, a.config, a.policy, a.base)
    elif a.stage in {"pilot", "label"}:
        from memory_trace.feedback_data import label_feedback

        result = label_feedback(a.base, a.config, pilot=a.stage == "pilot")
        if a.stage == "pilot":
            print(
                "Estimated remaining seconds, including training:",
                round(result["estimated_total_remaining_seconds_with_training"]),
                flush=True,
            )
    elif a.stage in {"tfidf", "e5", "bge-m3"}:
        from memory_trace.feedback_models import fit_candidate

        fit_candidate(a.base, a.stage, device=a.device)
    elif a.stage == "select":
        from memory_trace.feedback_models import select_feedback

        print("Selected:", select_feedback(a.base)["winner"]["name"], flush=True)
    elif a.stage == "evaluate":
        from memory_trace.feedback_report import evaluate_feedback

        evaluate_feedback(a.base)
    else:
        from memory_trace.feedback_report import report_feedback

        print(report_feedback(a.base, a.model_output, report_path=a.report), flush=True)


if __name__ == "__main__":
    main()
