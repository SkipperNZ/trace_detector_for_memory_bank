"""Run saved prompt candidates sequentially against one-slot local inference servers."""

import argparse
import json
from pathlib import Path

from memory_trace.io import read_jsonl, write_json, write_jsonl, write_text
from memory_trace.judge import read_config, run_judge
from memory_trace.judge_report import compare, render_markdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--root", type=Path, default=Path("runs/sentiment-tuning-v1"))
    parser.add_argument("--inputs", default="development")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    examples = read_jsonl(args.root / f"{args.inputs}.jsonl")
    references = read_jsonl("runs/sentiment/source_labels.jsonl")
    results = []
    for name in args.names:
        config = read_config(args.root / "configs" / f"{name}.json")
        output = args.root / (args.prefix + name)

        def progress(done, total):
            if done == 1 or done % 10 == 0 or done == total:
                print(f"{name}: {done}/{total}", flush=True)

        execution = run_judge(examples, config, output, repeats=1, progress=progress)
        report, disagreements = compare(
            examples, read_jsonl(output / "calls.jsonl"), references, expected_repeats=1
        )
        write_json(output / "comparison" / "report.json", report)
        write_jsonl(output / "comparison" / "disagreements.jsonl", disagreements)
        write_text(output / "comparison" / "report.md", render_markdown(report))
        row = {
            "candidate": name,
            "inputs": args.inputs,
            "completed": execution["labeled_messages"],
            "expected": len(examples),
            "agreement": report["sentiment_agreement_all_inputs_failures_as_nonagreement"],
            "macro_f1": report["sentiment_macro_f1_on_successful"],
            "elapsed_seconds": report["sum_attempt_elapsed_seconds"],
            "judge_version": config.version,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        if execution["unattempted_calls"]:
            raise SystemExit("Canary failed; inspect the saved response before further calls")
    write_json(args.root / f"{args.prefix}{args.inputs}-candidates.json", results)


if __name__ == "__main__":
    main()
