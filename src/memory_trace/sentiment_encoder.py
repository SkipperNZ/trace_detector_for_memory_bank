"""Complete-text E5/BGE-M3 embeddings with explicit token windows, on CPU/MPS/CUDA."""

import time
import hashlib
from pathlib import Path

from .embeddings import select_device
from .io import digest, write_json
from .student_data import text_of


def encoder_directory_hash(path, *, pretrained=False):
    path = Path(path)
    files = []
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.suffix in (
            {".json", ".safetensors", ".model", ".bin"}
            if pretrained
            else {".json", ".safetensors", ".model"}
        ):
            checksum = hashlib.sha256()
            with file.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    checksum.update(chunk)
            files.append([file.relative_to(path).as_posix(), checksum.hexdigest()])
    if not any(
        name.endswith(".safetensors") or (pretrained and name.endswith(".bin")) for name, _ in files
    ):
        raise ValueError("Fine-tuned encoder has no safetensors weights")
    return digest(files)


def token_windows(tokenizer, text, *, max_length=512, prefix="query: "):
    # Tokenize the complete prefixed text once. SentencePiece can merge the
    # trailing prefix space into the first content token; separate tokenization
    # would insert an extra space token, including for ordinary short messages.
    encoded = tokenizer(
        prefix + text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    boundary = len(prefix.rstrip())
    split = next(
        (i for i, (_, end) in enumerate(encoded["offset_mapping"]) if end > boundary),
        len(encoded["input_ids"]),
    )
    prefix_ids = encoded["input_ids"][:split]
    content = encoded["input_ids"][split:]
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False) - len(prefix_ids)
    if budget < 1:
        raise ValueError("Prefix leaves no content token budget")
    windows = []
    for start in range(0, max(len(content), 1), budget):
        part = content[start : start + budget]
        if (
            tokenizer.num_special_tokens_to_add(pair=False) != 2
            or tokenizer.cls_token_id is None
            or tokenizer.sep_token_id is None
        ):
            raise ValueError("This window serializer requires CLS/content/SEP tokens")
        ids = [tokenizer.cls_token_id] + prefix_ids + part + [tokenizer.sep_token_id]
        windows.append((ids, max(len(part), 1)))
    return windows, len(content)


class SentimentEncoder:
    def __init__(
        self,
        model,
        revision,
        *,
        device="cpu",
        batch_size=32,
        cache_folder="artifacts/hf",
        local_files_only=True,
        finetuned_path=None,
        finetuned_sha=None,
        pretrained_path=None,
        prefix=None,
        max_length=None,
    ):
        import re
        import torch
        from sentence_transformers import SentenceTransformer

        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Encoder revision must be a full checkpoint SHA")
        if batch_size < 1:
            raise ValueError("Encoder batch size must be positive")
        if model == "BAAI/bge-m3":
            default_prefix, default_length = "", 8192
        elif model.startswith("intfloat/multilingual-e5-"):
            default_prefix, default_length = "query: ", 512
        else:
            if prefix is None or max_length is None:
                raise ValueError("Unknown encoder requires explicit prefix and max_length")
            default_prefix, default_length = prefix, max_length
        prefix = default_prefix if prefix is None else prefix
        max_length = default_length if max_length is None else max_length
        if not isinstance(prefix, str) or not isinstance(max_length, int) or max_length < 3:
            raise ValueError("Expected a string prefix and an integer max_length >= 3")
        self.device = select_device(device, torch)
        self.batch_size = batch_size
        started = time.perf_counter()
        if finetuned_path and (
            not finetuned_sha or encoder_directory_hash(finetuned_path) != finetuned_sha
        ):
            raise ValueError("Fine-tuned encoder checksum mismatch")
        from huggingface_hub import snapshot_download

        # Resolve the pinned snapshot first. Some transformers versions otherwise probe
        # adapter_config on mutable main even when their parent requested local files.
        model_path = (
            finetuned_path
            or pretrained_path
            or snapshot_download(
                model,
                revision=revision,
                cache_dir=cache_folder,
                local_files_only=local_files_only,
                allow_patterns=[
                    "*.json",
                    "model.safetensors",
                    "pytorch_model.bin",
                    "sentencepiece.bpe.model",
                    "1_Pooling/config.json",
                ],
                ignore_patterns=["onnx/*", "openvino/*"],
                token=False,
            )
        )
        self.model = SentenceTransformer(
            str(model_path),
            device=self.device,
            cache_folder=cache_folder,
            local_files_only=local_files_only,
            trust_remote_code=False,
            model_kwargs={"weights_only": True},
        )
        self.model.eval()
        if max_length > self.model.max_seq_length:
            raise ValueError("Requested window exceeds the encoder's supported context")
        self.disk_bytes = sum(p.stat().st_size for p in Path(model_path).rglob("*") if p.is_file())
        self.load_seconds = time.perf_counter() - started
        self.config = {
            "model": model,
            "revision": revision,
            "prefix": prefix,
            "max_length": max_length,
            "serialization": "message-text-token-windows-v2",
            "pooling": (
                "nonoverlapping windows; repeated query prefix; content-token-weighted mean of normalized window embeddings; final L2 normalization"
                if prefix == "query: "
                else "nonoverlapping windows; configured prefix; content-token-weighted mean of normalized window embeddings; final L2 normalization"
            ),
        }
        if finetuned_sha:
            self.config["finetuned_sha256"] = finetuned_sha
        self.dimension = self.model.get_embedding_dimension()

    def windows(self, examples):
        result, owners, weights, token_counts = [], [], [], []
        for owner, ex in enumerate(examples):
            windows, count = token_windows(
                self.model.tokenizer,
                text_of(ex),
                prefix=self.config["prefix"],
                max_length=self.config["max_length"],
            )
            token_counts.append(count)
            for ids, weight in windows:
                result.append(ids)
                owners.append(owner)
                weights.append(weight)
        return result, owners, weights, token_counts

    def encode(self, examples, progress=None):
        import numpy as np
        import torch
        import torch.nn.functional as F

        started = time.perf_counter()
        windows, owners, weights, token_counts = self.windows(examples)
        vectors = np.zeros((len(examples), self.dimension), dtype=np.float32)
        # Sorting changes batching only; token ownership restores original order.
        order = sorted(range(len(windows)), key=lambda i: len(windows[i]))
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                indices = order[start : start + self.batch_size]
                features = self.model.tokenizer.pad(
                    {"input_ids": [windows[i] for i in indices]}, padding=True, return_tensors="pt"
                )
                features = {k: v.to(self.device) for k, v in features.items()}
                embedded = (
                    F.normalize(self.model(features)["sentence_embedding"], p=2, dim=1)
                    .float()
                    .cpu()
                    .numpy()
                )
                for i, vector in zip(indices, embedded):
                    vectors[owners[i]] += vector * weights[i]
                if progress:
                    progress(min(start + self.batch_size, len(order)), len(order))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors /= np.maximum(norms, 1e-12)
        return vectors, {
            "messages": len(examples),
            "windows": len(windows),
            "multiwindow_messages": int(
                sum(n > 1 for n in np.bincount(owners, minlength=len(examples)))
            ),
            "max_content_tokens": max(token_counts, default=0),
            "elapsed_seconds": time.perf_counter() - started,
            "device": self.device,
            "batch_size": self.batch_size,
            "truncated_messages": 0,
        }


def cache_embeddings(examples, encoder, output):
    import numpy as np
    import json

    output = Path(output)
    expected = {
        "input_ids": [ex["input_id"] for ex in examples],
        "inputs_hash": digest(examples),
        "encoder": encoder.config,
    }
    if (output / "manifest.json").exists():
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError("Embedding cache belongs to other inputs or encoder")
        vectors = np.load(output / "vectors.npy", allow_pickle=False)
        if digest(vectors.tolist()) != manifest["vectors_hash"]:
            raise ValueError("Embedding cache checksum mismatch")
        if manifest.get("encoder_disk_bytes") != encoder.disk_bytes:
            manifest["encoder_disk_bytes"] = encoder.disk_bytes
            write_json(output / "manifest.json", manifest)
        return vectors, manifest
    last = [0.0]

    def progress(done, total):
        now = time.perf_counter()
        if now - last[0] > 30 or done == total:
            print(f"Encoder windows {done}/{total}", flush=True)
            last[0] = now

    vectors, runtime = encoder.encode(examples, progress=progress)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "vectors.npy", vectors, allow_pickle=False)
    manifest = {
        **expected,
        "vectors_hash": digest(vectors.tolist()),
        "runtime": runtime,
        "encoder_disk_bytes": encoder.disk_bytes,
    }
    write_json(output / "manifest.json", manifest)
    return vectors, manifest
