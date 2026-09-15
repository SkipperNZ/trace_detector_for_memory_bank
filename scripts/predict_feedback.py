"""Run both feedback heads locally, without the Qwen teacher."""

import argparse
from pathlib import Path

from memory_trace.feedback import feedback_input
from memory_trace.io import read_jsonl, write_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    import torch
    from memory_trace.feedback_models import FeedbackPredictor

    torch.set_num_threads(8)
    rows = [feedback_input(r) for r in read_jsonl(a.input)]
    if not rows:
        raise ValueError("No input messages")
    predictor = FeedbackPredictor(a.model_dir, device=a.device, batch_size=a.batch_size)
    write_jsonl(a.out, predictor.predict(rows))
    print("Predicted", len(rows), "messages locally")


if __name__ == "__main__":
    main()
