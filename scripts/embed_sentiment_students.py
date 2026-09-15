"""Compute pinned embeddings and optionally verify the encoder on synthetic messages."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
import argparse
from pathlib import Path
import torch
from memory_trace.io import read_jsonl
from memory_trace.sentiment_encoder import SentimentEncoder, cache_embeddings
from memory_trace.student_study import load_study

torch.set_num_threads(8)
parser = argparse.ArgumentParser()
parser.add_argument("--base", type=Path, default=Path("runs/sentiment-students-v1"))
parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
parser.add_argument("--verify", action="store_true")
args = parser.parse_args()
base = args.base
source = json.loads((base / "encoder-source.json").read_text(encoding="utf-8"))
policy = load_study(base)
encoder = SentimentEncoder(
    source["model"],
    source["revision"],
    device=args.device,
    batch_size=policy.get("encoder_batch_size", 32),
    prefix=policy.get("encoder_prefix"),
    max_length=policy.get("encoder_max_length"),
)
if args.verify:
    import numpy as np
    from memory_trace.sentiment import make_sentiment_input
    from memory_trace.sentiment_encoder import token_windows
    from memory_trace.io import write_json

    texts = ["", "Fix this bug.", "Отлично, спасибо!", "Почему это не работает?", "  code", "!!!"]
    rows = [
        make_sentiment_input(t, tenant_id="synthetic", session_id=str(i), message_id=str(i))
        for i, t in enumerate(texts)
    ]
    vectors, _ = encoder.encode(rows)
    official = encoder.model.encode(
        [encoder.config["prefix"] + t for t in texts],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    np.testing.assert_allclose(vectors, official, atol=2e-5, rtol=2e-5)
    for text in texts:
        windows, _ = token_windows(
            encoder.model.tokenizer,
            text,
            max_length=encoder.config["max_length"],
            prefix=encoder.config["prefix"],
        )
        assert (
            windows[0][0] == encoder.model.tokenizer(encoder.config["prefix"] + text)["input_ids"]
        )
    # The unprefixed BGE path must preserve every token of a long document.
    if encoder.config["prefix"] == "":
        text = "Длинный текст. Long message. " * 600
        windows, _ = token_windows(
            encoder.model.tokenizer, text, max_length=encoder.config["max_length"], prefix=""
        )
        recovered = [t for ids, _ in windows for t in ids[1:-1]]
        assert recovered == encoder.model.tokenizer(text, add_special_tokens=False)["input_ids"]
    write_json(
        base / "encoder-verification.json",
        {
            "synthetic_only": True,
            "model": encoder.config,
            "parameter_count": sum(p.numel() for p in encoder.model.parameters()),
            "official_embedding_max_absolute_difference": float(np.max(np.abs(vectors - official))),
            "short_tokenization_matches": True,
            "full_text_preserved": True,
            "pooling": encoder.model[1].get_config_dict(),
        },
    )
# Test inputs need no embeddings until the selected models have been frozen.
for split in ("train", "calibration"):
    _, manifest = cache_embeddings(
        read_jsonl(base / f"data/{split}.jsonl"), encoder, base / f"embeddings/{split}"
    )
    print(split, json.dumps(manifest["runtime"]), flush=True)
