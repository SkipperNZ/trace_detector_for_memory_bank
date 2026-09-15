"""One cross-platform CLI; no shell scripts, service, GPU, or network required for core."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .io import digest, read_jsonl, write_json, write_jsonl, write_text
from .metrics import evaluate, prevalence
from .prepare import annotation_template, normalized_rows, prepare
from .routing import Policy, route


def prepare_artifacts(rows: list[dict], output: Path, max_chars: int, seed: int):
    normalized = normalized_rows(rows)
    examples = prepare(normalized, max_chars=max_chars, seed=seed)
    write_jsonl(output / "events.jsonl", normalized)
    write_jsonl(output / "examples.jsonl", examples)
    write_jsonl(output / "annotations.jsonl", annotation_template(examples))
    for split in ("train", "calibration", "test"):
        subset = [ex for ex in examples if ex["split"] == split]
        write_jsonl(output / f"{split}.jsonl", subset)
        write_jsonl(output / f"{split}.annotations.jsonl", annotation_template(subset))
    traces = {(row["tenant_id"], row["trace_id"]) for row in normalized}
    eligible = {(row["tenant_id"], row["trace_id"]) for row in examples}
    manifest = {
        "package_version": __version__,
        "normalized_events": len(normalized),
        "duplicate_deliveries_removed": len(rows) - len(normalized),
        "eligible_messages": len(examples),
        "eligible_traces": len(eligible),
        "no_eligible_user_traces": len(traces - eligible),
        "events_hash": digest(normalized),
        "inputs_hash": digest(examples),
        "seed": seed,
        "max_chars": max_chars,
        "splits": {
            s: sum(ex["split"] == s for ex in examples) for s in ("train", "calibration", "test")
        },
    }
    write_json(output / "manifest.json", manifest)
    return examples, manifest


def _policy(args):
    return Policy(
        mode=args.mode, threshold=args.threshold, audit_rate=args.audit_rate, seed=args.seed
    )


def _encoder(args, config=None):
    from .embeddings import FrozenEncoder

    config = config or {"model": args.model, "revision": args.revision, "prefix": args.prefix}
    return FrozenEncoder(
        config["model"],
        config["revision"],
        prefix=config["prefix"],
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Memory Bank offline trace research")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    from .dataset import DATASET_REVISION

    fetch = sub.add_parser("fetch-sentiment", help="Download the pinned public sentiment dataset")
    fetch.add_argument("--out", type=Path, default=Path("data/agent-trace-sentiment"))
    fetch.add_argument("--revision", default=DATASET_REVISION)
    importer = sub.add_parser(
        "import-sentiment", help="Convert sentiment parquet into blind message inputs"
    )
    importer.add_argument("parquet", type=Path)
    importer.add_argument("--out", type=Path, default=Path("runs/sentiment"))
    importer.add_argument("--revision", default=DATASET_REVISION)
    importer.add_argument("--per-class", type=int, default=100)
    importer.add_argument("--seed", type=int, default=42)
    plan = sub.add_parser(
        "judge-plan", help="Export exact judge messages without calling any endpoint"
    )
    plan.add_argument("examples", type=Path)
    plan.add_argument("--out", type=Path, required=True)
    plan_mode = plan.add_mutually_exclusive_group()
    plan_mode.add_argument(
        "--task", choices=["sentiment", "sentiment_and_complaint"], default="sentiment"
    )
    plan_mode.add_argument(
        "--config", type=Path, help="Use the exact prompt from a judge configuration"
    )
    judge = sub.add_parser("judge", help="Label inputs with the configured LLM judge, with resume")
    judge.add_argument("examples", type=Path)
    judge.add_argument("--config", type=Path, required=True)
    judge.add_argument("--out", type=Path, required=True)
    judge.add_argument("--repeats", type=int, default=2)
    judge.add_argument("--retry-errors", action="store_true")
    comparison = sub.add_parser(
        "judge-report", help="Measure source agreement and judge repeatability"
    )
    comparison.add_argument("examples", type=Path)
    comparison.add_argument("--calls", type=Path, required=True)
    comparison.add_argument("--source-labels", type=Path, required=True)
    comparison.add_argument("--out", type=Path, required=True)
    comparison.add_argument("--repeats", type=int, default=2)
    prep = sub.add_parser(
        "prepare", help="Normalize events, build causal inputs and annotation forms"
    )
    prep.add_argument("events", type=Path)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--max-chars", type=int, default=12000)
    prep.add_argument("--seed", type=int, default=42)

    demo = sub.add_parser("demo", help="Run a synthetic offline plumbing demonstration")
    demo.add_argument("--out", type=Path, default=Path("runs/demo"))
    routing = sub.add_parser("route", help="Compute trace routing and export human review queue")
    routing.add_argument("examples", type=Path)
    routing.add_argument("--predictions", type=Path)
    routing.add_argument("--out", type=Path, required=True)
    routing.add_argument("--mode", choices=["shadow", "cascade"], default="shadow")
    routing.add_argument("--threshold", type=float, default=0.5)
    routing.add_argument("--audit-rate", type=float, default=0.1)
    routing.add_argument("--seed", type=int, default=42)

    sampling = sub.add_parser("sample", help="Probability sample without an ML filter")
    sampling.add_argument("examples", type=Path)
    sampling.add_argument("--out", type=Path, required=True)
    sampling.add_argument("--rate", type=float, default=0.1)
    sampling.add_argument("--seed", type=int, default=42)

    for command in ("evaluate", "prevalence"):
        report = sub.add_parser(command)
        report.add_argument("examples", type=Path)
        report.add_argument("--decisions", type=Path, required=True)
        report.add_argument("--labels", type=Path, required=True)
        report.add_argument("--out", type=Path, required=True)
        report.add_argument("--label-source", choices=["human", "llm_judge"], default="llm_judge")
        if command == "evaluate":
            report.add_argument("--predictions", type=Path, required=True)

    training = sub.add_parser("train", help="Fit a logistic head on frozen embeddings (ML extra)")
    training.add_argument("examples", type=Path)
    training.add_argument("--labels", type=Path, required=True)
    training.add_argument("--label-source", choices=["human", "llm_judge"], default="llm_judge")
    training.add_argument("--model", default="intfloat/multilingual-e5-small")
    training.add_argument("--revision", required=True, help="Pinned Hugging Face commit SHA")
    training.add_argument("--prefix", default="query: ")
    prediction = sub.add_parser("predict", help="Score causal inputs using a trained frozen head")
    prediction.add_argument("examples", type=Path)
    prediction.add_argument("--head", type=Path, required=True)
    for command in (training, prediction):
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
        command.add_argument("--batch-size", type=int, default=32)
        command.add_argument("--local-files-only", action="store_true")
    sentiment_train = sub.add_parser(
        "train-sentiment", help="Fit a portable three-class student on Qwen labels"
    )
    sentiment_train.add_argument("examples", type=Path)
    sentiment_train.add_argument("--labels", type=Path, required=True)
    sentiment_train.add_argument("--kind", choices=["tfidf", "encoder"], default="tfidf")
    sentiment_train.add_argument("--model", default="intfloat/multilingual-e5-small")
    sentiment_train.add_argument("--revision", help="Full pinned encoder checkpoint SHA")
    sentiment_train.add_argument("--prefix", default=None, help="Model default unless overridden")
    sentiment_train.add_argument("--max-length", type=int, default=None, help="Token window size")
    sentiment_train.add_argument("--C", dest="c", type=float, default=1.0)
    sentiment_train.add_argument("--balanced", action="store_true")
    sentiment_predict = sub.add_parser(
        "predict-sentiment", help="Predict three sentiment classes and probabilities in shadow mode"
    )
    sentiment_predict.add_argument("examples", type=Path)
    sentiment_predict.add_argument("--model-dir", type=Path, required=True)
    for command in (sentiment_train, sentiment_predict):
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
        command.add_argument("--batch-size", type=int, default=32)
        command.add_argument("--cache-folder", default="artifacts/hf")
        command.add_argument("--allow-download", action="store_true")
    return parser


def run(args):
    if args.command == "fetch-sentiment":
        from .dataset import download

        return download(args.out, args.revision)
    if args.command == "import-sentiment":
        from .dataset import import_parquet

        return import_parquet(
            args.parquet, args.out, per_class=args.per_class, seed=args.seed, revision=args.revision
        )
    if args.command == "prepare":
        _, manifest = prepare_artifacts(
            read_jsonl(args.events), args.out, args.max_chars, args.seed
        )
        return manifest
    if args.command == "demo":
        from .demo import annotations_and_scores, events

        examples, _ = prepare_artifacts(events(), args.out, 12000, 42)
        labels, scores = annotations_and_scores(examples)
        decisions = route(examples, scores, Policy())
        report = evaluate(examples, scores, decisions, labels, synthetic=True)
        write_jsonl(args.out / "synthetic_labels.jsonl", labels)
        write_jsonl(args.out / "synthetic_predictions.jsonl", scores)
        write_jsonl(args.out / "decisions.jsonl", decisions)
        write_json(args.out / "report.json", report)
        return report
    examples = read_jsonl(args.examples)
    if args.command in {"train-sentiment", "predict-sentiment"}:
        from .sentiment import load_student, predict_student, train_student
        from .sentiment_encoder import SentimentEncoder

        artifact = load_student(args.model_dir) if args.command == "predict-sentiment" else None
        config = (
            artifact["encoder"]
            if artifact
            else {
                "model": args.model,
                "revision": args.revision,
                "prefix": args.prefix,
                "max_length": args.max_length,
            }
        )
        kind = artifact["kind"] if artifact else args.kind
        encoder = None
        if kind in {"encoder", "finetuned"}:
            if not config.get("revision"):
                raise ValueError("Encoder training requires --revision with a pinned SHA")
            encoder = SentimentEncoder(
                config["model"],
                config["revision"],
                device=args.device,
                batch_size=args.batch_size,
                cache_folder=args.cache_folder,
                prefix=config.get("prefix"),
                max_length=config.get("max_length"),
                local_files_only=not args.allow_download,
                finetuned_path=args.model_dir / "encoder" if kind == "finetuned" else None,
                finetuned_sha=config.get("finetuned_sha256"),
            )
        if artifact:
            predictions, runtime = predict_student(examples, artifact, encoder=encoder)
            write_jsonl(args.out, predictions)
            return runtime
        return train_student(
            examples,
            read_jsonl(args.labels),
            args.out,
            kind=kind,
            encoder=encoder,
            c=args.c,
            balanced=args.balanced,
        )
    if args.command == "judge-plan":
        from .judge import messages, prompt_spec, read_config

        config = read_config(args.config) if args.config else None
        version, prompt = config.prompt if config else prompt_spec(args.task)

        write_jsonl(
            args.out,
            [
                {
                    "input_id": ex["input_id"],
                    "prompt_version": version,
                    "messages": messages(
                        ex, config.task if config else args.task, system_prompt=prompt
                    ),
                }
                for ex in examples
            ],
        )
        return {"planned_inputs": len(examples), "network_requests": 0}
    if args.command == "judge":
        from .judge import read_config, run_judge

        def progress(done, total):
            if done == 1 or done % 10 == 0 or done == total:
                print(f"Judge calls: {done}/{total}", file=sys.stderr, flush=True)

        return run_judge(
            examples,
            read_config(args.config),
            args.out,
            repeats=args.repeats,
            retry_errors=args.retry_errors,
            progress=progress,
        )
    if args.command == "judge-report":
        from .judge_report import compare, render_markdown

        report, disagreements = compare(
            examples,
            read_jsonl(args.calls),
            read_jsonl(args.source_labels),
            expected_repeats=args.repeats,
        )
        write_json(args.out / "report.json", report)
        write_text(args.out / "report.md", render_markdown(report))
        write_jsonl(args.out / "disagreements.jsonl", disagreements)
        return report
    if args.command in {"route", "sample"}:
        if args.command == "sample":
            predictions = [
                {
                    "input_id": ex["input_id"],
                    "score": 0.0,
                    "runtime_status": "ok",
                    "model_abstention": False,
                    "model_version": "sampling-control-v1",
                }
                for ex in examples
            ]
            policy = Policy(mode="cascade", audit_rate=args.rate, seed=args.seed)
        else:
            predictions = read_jsonl(args.predictions) if args.predictions else []
            policy = _policy(args)
        decisions = route(examples, predictions, policy)
        write_jsonl(args.out / "decisions.jsonl", decisions)
        if args.command == "sample":
            write_jsonl(args.out / "control_scores.jsonl", predictions)
        routed_ids = {
            i for decision in decisions if decision["routed"] for i in decision["input_ids"]
        }
        write_jsonl(
            args.out / "review.jsonl",
            annotation_template([ex for ex in examples if ex["input_id"] in routed_ids]),
        )
        return {
            "traces": len(decisions),
            "routed": sum(d["routed"] for d in decisions),
            "candidate_routed": sum(d["would_route"] for d in decisions),
            "mode": policy.mode,
        }
    if args.command in {"evaluate", "prevalence"}:
        decisions = read_jsonl(args.decisions)
        labels = read_jsonl(args.labels)
        if args.command == "evaluate":
            report = evaluate(
                examples,
                read_jsonl(args.predictions),
                decisions,
                labels,
                label_source=args.label_source,
            )
        else:
            report = prevalence(examples, decisions, labels, label_source=args.label_source)
        write_json(args.out, report)
        return report
    if args.command == "train":
        from .embeddings import train

        head = train(
            examples,
            read_jsonl(args.labels),
            _encoder(args),
            args.out,
            label_source=args.label_source,
        )
        return {
            "model_version": head["model_version"],
            "training_messages": head["training_messages"],
            "excluded_messages": head["excluded_messages"],
        }
    if args.command == "predict":
        from .embeddings import predict

        artifact = json.loads(args.head.read_text(encoding="utf-8"))
        predictions = predict(examples, artifact, _encoder(args, artifact["encoder"]))
        write_jsonl(args.out, predictions)
        return {
            "messages": len(predictions),
            "scored": sum(p["score"] is not None for p in predictions),
        }
    raise ValueError("Unknown command")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    if args.command == "judge" and (result["failed_calls"] or result["unattempted_calls"]):
        return 3
    return 0
