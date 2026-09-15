"""Optional supervised encoder adaptation, selected solely on calibration."""

import math
import time
from datetime import datetime, timezone
from pathlib import Path

from .io import digest, write_json, write_jsonl
from .sentiment import load_student, save_student, fit_temperature, softmax, provenance
from .sentiment_encoder import SentimentEncoder, encoder_directory_hash
from .sentiment_metrics import classification
from .student_study import load_study, read_json, split_targets


def finetune(base, *, device="cpu"):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from contextlib import nullcontext

    base = Path(base)
    policy = load_study(base)
    baseline = read_json(base / "baseline-selection.json")
    if not baseline["finetune_required"]:
        result = {"performed": False, "reason": "Predeclared calibration condition not met"}
        write_json(base / "finetune-skipped.json", result)
        return result
    if (base / "finetune-selection.json").exists():
        return read_json(base / "finetune-selection.json")
    config = policy["finetune"]
    train, y, labels, _ = split_targets(base, "train", policy)
    calibration, cal_y, _, _ = split_targets(base, "calibration", policy)
    seed = policy["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(8)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    encoder = SentimentEncoder(
        policy["encoder_model"],
        policy["encoder_revision"],
        device=device,
        batch_size=policy.get("encoder_batch_size", 32),
        prefix=policy.get("encoder_prefix"),
        max_length=policy.get("encoder_max_length"),
    )
    if config.get("gradient_checkpointing", False):
        encoder.model[0].auto_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if encoder.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    initial = load_student(base / "models/encoder-all")
    head = torch.nn.Linear(encoder.dimension, 3, device=encoder.device)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(initial["head"]["coef"], device=encoder.device))
        head.bias.copy_(torch.tensor(initial["head"]["intercept"], device=encoder.device))
    train_windows = []
    for ex in train:
        windows, _, weights, _ = encoder.windows([ex])
        train_windows.append(list(zip(windows, weights)))
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.model.parameters(), "lr": config["learning_rate"]},
            {"params": head.parameters(), "lr": config["head_learning_rate"]},
        ],
        weight_decay=config["weight_decay"],
    )
    batch_size = config["document_batch_size"]
    accumulation = config["gradient_accumulation_steps"]
    batches_per_epoch = math.ceil(len(train) / batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / accumulation)
    total_updates = updates_per_epoch * config["epochs"]
    # Fixed linear decay, no test-driven schedule search.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1 - step / max(total_updates, 1))
    )
    use_bf16 = encoder.device == "cuda" and torch.cuda.is_bf16_supported()
    run_design = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy_hash": digest(policy),
        "initial_model_version": initial["model_version"],
        "seed": seed,
        "device": encoder.device,
        "dtype": "bf16 autocast" if use_bf16 else "float32",
        "loss": "unweighted cross entropy",
        "scheduler": "linear decay to zero",
        "train_messages": len(train),
        "gradient_checkpointing": config.get("gradient_checkpointing", False),
        "test_used": False,
    }
    write_json(base / "finetune-design.json", run_design)
    trials = []
    best = None
    output = base / "models/finetuned-all"
    started = time.perf_counter()
    for epoch in range(config["epochs"]):
        encoder.model.train()
        head.train()
        order = np.random.default_rng(seed + epoch).permutation(len(train))
        optimizer.zero_grad(set_to_none=True)
        losses = []
        last = time.perf_counter()
        for batch_number, start in enumerate(range(0, len(order), batch_size)):
            document_indices = order[start : start + batch_size]
            windows, owners, weights = [], [], []
            for owner, i in enumerate(document_indices):
                for ids, weight in train_windows[int(i)]:
                    windows.append(ids)
                    owners.append(owner)
                    weights.append(weight)
            features = encoder.model.tokenizer.pad(
                {"input_ids": windows}, padding=True, return_tensors="pt"
            )
            features = {k: v.to(encoder.device) for k, v in features.items()}
            owner_tensor = torch.tensor(owners, device=encoder.device, dtype=torch.long)
            weight_tensor = torch.tensor(weights, device=encoder.device, dtype=torch.float32)
            with (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            ):
                window_vectors = F.normalize(
                    encoder.model(features)["sentence_embedding"].float(), dim=1
                )
                document_vectors = torch.zeros(
                    (len(document_indices), encoder.dimension), device=encoder.device
                )
                document_vectors.index_add_(
                    0, owner_tensor, window_vectors * weight_tensor[:, None]
                )
                document_vectors = F.normalize(document_vectors, dim=1)
                logits = head(document_vectors)
                batch_y = torch.tensor([y[int(i)] for i in document_indices], device=encoder.device)
                loss = F.cross_entropy(logits, batch_y)
            # Weight documents, including the final short accumulation group, equally.
            group_start = (batch_number // accumulation) * accumulation * batch_size
            group_docs = min(accumulation * batch_size, len(train) - group_start)
            (loss * len(document_indices) / group_docs).backward()
            losses.append((float(loss.detach()), len(document_indices)))
            if (batch_number + 1) % accumulation == 0 or start + batch_size >= len(order):
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.model.parameters()) + list(head.parameters()),
                    config["gradient_clip_norm"],
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if time.perf_counter() - last >= 30 or start + batch_size >= len(order):
                print(
                    f"Fine-tune epoch {epoch+1}/{config['epochs']}, documents {min(start+batch_size,len(order))}/{len(order)}, loss {float(loss.detach()):.4f}",
                    flush=True,
                )
                last = time.perf_counter()
        encoder.model.eval()
        head.eval()
        cal_vectors, _ = encoder.encode(calibration)
        with torch.inference_mode():
            logits = head(torch.tensor(cal_vectors, device=encoder.device)).float().cpu().numpy()
        calibration_fit = fit_temperature(logits, cal_y)
        probabilities = softmax(logits, calibration_fit["temperature"])
        metrics = classification(cal_y, probabilities)
        trial = {
            "epoch": epoch + 1,
            "train_loss": sum(v * n for v, n in losses) / sum(n for _, n in losses),
            "calibration": metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        trials.append(trial)
        write_json(base / "finetune-trials.json", trials)
        rank = (metrics["macro_f1"], metrics["agreement"])
        if best is None or rank > best[0]:
            encoder.model.save(
                str(output / "encoder"), safe_serialization=True, create_model_card=False
            )
            checksum = encoder_directory_hash(output / "encoder")
            head_state = {
                "coef": head.weight.detach().cpu().float().tolist(),
                "intercept": head.bias.detach().cpu().float().tolist(),
            }
            artifact = save_student(
                output,
                kind="finetuned",
                training=provenance(train, labels),
                head=head_state,
                encoder={**encoder.config, "finetuned_sha256": checksum},
                calibration=calibration_fit,
            )
            record = {
                "name": "finetuned-all",
                "kind": "finetuned",
                "size": "all",
                "training_messages": len(train),
                "calibration": metrics,
                "temperature": calibration_fit,
                "path": "models/finetuned-all",
                "model_version": artifact["model_version"],
                "epoch": epoch + 1,
                "artifact_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
            }
            best = (rank, record)
            write_jsonl(
                base / "calibration/finetuned-all.jsonl",
                [
                    {"input_id": ex["input_id"], "probabilities": p.tolist()}
                    for ex, p in zip(calibration, probabilities)
                ],
            )
        print(
            f"Fine-tune epoch {epoch+1}: calibration macro-F1 {metrics['macro_f1']:.4f}", flush=True
        )
    result = {
        "performed": True,
        "winner": best[1],
        "trials": trials,
        "design": run_design,
        "total_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if encoder.device == "cuda" else None
        ),
        "test_used": False,
    }
    write_json(base / "finetune-selection.json", result)
    return result
