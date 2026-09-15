"""Independent E5 tokenization/embedding checks on synthetic inputs only."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
from pathlib import Path
import numpy as np
import torch
from memory_trace.io import write_json
from memory_trace.sentiment_encoder import SentimentEncoder, token_windows

torch.set_num_threads(8)
base = Path("runs/sentiment-students-v1")
source = json.loads((base / "encoder-source.json").read_text(encoding="utf-8"))
encoder = SentimentEncoder(source["model"], source["revision"], device="cpu")
texts = [
    "Please inspect this code.",
    "Отлично, спасибо!",
    "  x = 1\nreturn x",
    "I did not like that change.",
    "trailing space   ",
    "",
    "!!!",
]
for text in texts:
    windows, _ = token_windows(encoder.model.tokenizer, text)
    reference = encoder.model.tokenizer(
        "query: " + text, add_special_tokens=True, truncation=False
    )["input_ids"]
    assert len(windows) == 1 and windows[0][0] == reference, repr(text)
examples = [{"context_scope": "message_only", "context": [{"text": text}]} for text in texts]
actual, _ = encoder.encode(examples)
reference = encoder.model.encode(
    ["query: " + text for text in texts],
    prompt="",
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False,
)
np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-5)
long_text = "Here is a long synthetic example with important content. " * 250 + "CRITICAL END"
windows, count = token_windows(encoder.model.tokenizer, long_text)
full = encoder.model.tokenizer("query: " + long_text, add_special_tokens=False, truncation=False)
prefix = encoder.model.tokenizer("query:", add_special_tokens=False)["input_ids"]
recovered = prefix + [token for ids, _ in windows for token in ids[1 + len(prefix) : -1]]
assert recovered == full["input_ids"]
assert sum(weight for _, weight in windows) == count
assert all(len(ids) <= 512 for ids, _ in windows)
result = {
    "synthetic_only": True,
    "encoder": encoder.config,
    "short_inputs": len(texts),
    "official_tokenization_matches": True,
    "official_embeddings_max_abs_difference": float(np.abs(actual - reference).max()),
    "long_windows": len(windows),
    "long_content_preserved": True,
}
write_json(base / "encoder-verification.json", result)
print(json.dumps(result), flush=True)
