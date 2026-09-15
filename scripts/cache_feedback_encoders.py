"""Download exactly the pinned public encoders before moving into an offline environment."""

import argparse
from pathlib import Path
from memory_trace.feedback_data import read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, default=Path("configs/feedback-study.json"))
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/hf"))
    a = p.parse_args()
    from huggingface_hub import snapshot_download

    for spec in read(a.policy)["encoders"]:
        snapshot_download(
            spec["model"],
            revision=spec["revision"],
            cache_dir=a.cache_dir,
            token=False,
            allow_patterns=[
                "*.json",
                "model.safetensors",
                "pytorch_model.bin",
                "sentencepiece.bpe.model",
                "1_Pooling/config.json",
            ],
            ignore_patterns=["onnx/*", "openvino/*"],
        )
        print("Cached", spec["name"], spec["revision"])


if __name__ == "__main__":
    main()
