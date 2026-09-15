"""Download pinned model/tokenizer artifacts; legacy PyTorch weights load weights_only."""

import json
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from memory_trace.io import write_json

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=Path, default=Path("runs/sentiment-students-v1"))
parser.add_argument(
    "--policy", type=Path, help="Create/validate encoder-source from a frozen policy"
)
args = parser.parse_args()
base = args.base
if args.policy:
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    source = {"model": policy["encoder_model"], "revision": policy["encoder_revision"]}
    if (base / "encoder-source.json").exists():
        assert source == json.loads((base / "encoder-source.json").read_text(encoding="utf-8"))
    write_json(base / "encoder-source.json", source)
else:
    source = json.loads((base / "encoder-source.json").read_text(encoding="utf-8"))
snapshot = snapshot_download(
    source["model"],
    revision=source["revision"],
    cache_dir="artifacts/hf",
    allow_patterns=[
        "*.json",
        "model.safetensors",
        "pytorch_model.bin",
        "sentencepiece.bpe.model",
        "README.md",
        "1_Pooling/config.json",
    ],
    ignore_patterns=["onnx/*", "openvino/*", ".eval_results/*"],
    token=False,
)
write_json(
    base / "encoder-cache.json",
    {
        "model": source["model"],
        "revision": source["revision"],
        "snapshot": str(Path(snapshot).resolve()),
    },
)
print("Cached pinned encoder:", source["model"], source["revision"], flush=True)
