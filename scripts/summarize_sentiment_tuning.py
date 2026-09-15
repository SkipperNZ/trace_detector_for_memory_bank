"""Freeze a development winner before holdout, then summarize paired holdout results."""

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from memory_trace.dataset import SENTIMENTS
from memory_trace.io import digest, read_jsonl, write_json, write_jsonl, write_text
from memory_trace.judge import read_config

ROOT = Path("runs/sentiment-tuning-v1")
CANDIDATES = {
    "minimal": "minimal",
    "priority": "priority",
    "broad": "broad",
    "fewshot": "screen-fewshot",
    "content": "screen-content",
    "content-thinking": "screen-content-thinking",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze():
    if (ROOT / "selection.json").exists():
        raise ValueError("Selection is already frozen")
    if any((ROOT / name).exists() for name in ("holdout-baseline", "holdout-winner")):
        raise ValueError("Freeze the winner before running either holdout condition")
    rows = []
    for name, folder in CANDIDATES.items():
        report = read(ROOT / folder / "comparison/report.json")
        config = read_config(ROOT / "configs" / f"{name}.json")
        if read(ROOT / folder / "run.json")["judge_version"] != config.version:
            raise ValueError(f"Configuration changed after the run: {name}")
        rows.append(
            {
                "candidate": name,
                "folder": folder,
                "coverage": report["successful_primary_judgments"],
                "inputs": report["messages"],
                "agreement": report["sentiment_agreement_all_inputs_failures_as_nonagreement"],
                "macro_f1": report["sentiment_macro_f1_on_successful"],
                "elapsed_seconds": report["sum_attempt_elapsed_seconds"],
                "judge_version": config.version,
            }
        )
    eligible = [r for r in rows if r["coverage"] == r["inputs"] == 75]
    if not eligible:
        raise ValueError("No complete candidate to select")
    eligible.sort(key=lambda r: (-r["agreement"], -r["macro_f1"], r["elapsed_seconds"]))
    winner = eligible[0]
    config = read_config(ROOT / "configs" / f"{winner['candidate']}.json")
    holdout = read_jsonl(ROOT / "holdout.jsonl")
    design = read(ROOT / "design.json")
    assert digest(holdout) == design["holdout_hash"]
    write_json(ROOT / "configs/winner.json", asdict(config))
    selection = {
        "winner": winner,
        "all_candidates": rows,
        "holdout_hash": digest(holdout),
        "selected_before_holdout": True,
        "policy": design["selection"],
    }
    write_json(ROOT / "selection.json", selection)
    print(json.dumps(selection, indent=2))


def summarize():
    selection = read(ROOT / "selection.json")
    examples = read_jsonl(ROOT / "holdout.jsonl")
    assert digest(examples) == selection["holdout_hash"]
    keys = {row["input_id"] for row in examples}
    refs = {
        r["input_id"]: r["sentiment_label"]
        for r in read_jsonl("runs/sentiment/source_labels.jsonl")
    }
    predictions, reports, all_input_metrics = {}, {}, {}
    for name in ("baseline", "winner"):
        calls = read_jsonl(ROOT / f"holdout-{name}/calls.jsonl")
        predictions[name] = {
            r["input_id"]: r["answer"]["sentiment_label"]
            for r in calls
            if r["repeat"] == 0 and r["runtime_status"] == "ok"
        }
        if {r["input_id"] for r in calls if r["repeat"] == 0} != keys:
            raise ValueError(f"Holdout calls are missing for {name}")
        # Keep failed API outputs in the denominator; None is a runtime failure, not a class.
        predictions[name] = {key: predictions[name].get(key) for key in keys}
        reports[name] = read(ROOT / f"holdout-{name}/comparison/report.json")
        by_class = {}
        for label in SENTIMENTS:
            support = sum(refs[key] == label for key in keys)
            predicted = sum(predictions[name][key] == label for key in keys)
            tp = sum(refs[key] == predictions[name][key] == label for key in keys)
            by_class[label] = {
                "support": support,
                "recall": tp / support,
                "f1": 2 * tp / (support + predicted),
            }
        all_input_metrics[name] = {
            "per_class": by_class,
            "macro_f1": sum(r["f1"] for r in by_class.values()) / len(SENTIMENTS),
            "successful": sum(value is not None for value in predictions[name].values()),
        }
    assert (
        read(ROOT / "holdout-winner/run.json")["judge_version"]
        == selection["winner"]["judge_version"]
    )
    paired = Counter()
    changes = []
    for example in examples:
        key = example["input_id"]
        old, new = predictions["baseline"][key], predictions["winner"][key]
        before, after = old == refs[key], new == refs[key]
        paired[
            (
                "both_match"
                if before and after
                else (
                    "only_baseline_matches"
                    if before
                    else "only_winner_matches" if after else "neither_matches"
                )
            )
        ] += 1
        if old != new:
            changes.append(
                {
                    "input_id": key,
                    "source_sentiment": refs[key],
                    "baseline_sentiment": old,
                    "winner_sentiment": new,
                    "context": example["context"],
                }
            )
    # Resample pairs within each class. One input per session, so no within-session duplication.
    rng = random.Random(42)
    strata = [[key for key in sorted(keys) if refs[key] == label] for label in SENTIMENTS]
    deltas = {
        key: int(predictions["winner"][key] == refs[key])
        - int(predictions["baseline"][key] == refs[key])
        for key in keys
    }
    boot = sorted(
        sum(deltas[key] for group in strata for key in rng.choices(group, k=len(group))) / len(keys)
        for _ in range(10000)
    )
    result = {
        "source_reference": "dataset_llm",
        "selected_before_holdout": True,
        "winner": selection["winner"]["candidate"],
        "messages": len(keys),
        "paired_counts": dict(paired),
        "baseline_agreement": reports["baseline"][
            "sentiment_agreement_all_inputs_failures_as_nonagreement"
        ],
        "winner_agreement": reports["winner"][
            "sentiment_agreement_all_inputs_failures_as_nonagreement"
        ],
        "all_input_metrics": all_input_metrics,
        "agreement_gain": sum(deltas.values()) / len(keys),
        "gain_stratified_paired_bootstrap_95_interval": [boot[249], boot[9749]],
        "bootstrap_seed": 42,
        "bootstrap_replicates": 10000,
        "holdout_policy": "No prompt or candidate selection changes after holdout evaluation.",
    }
    write_json(ROOT / "holdout-comparison.json", result)
    write_jsonl(ROOT / "holdout-changed-predictions.jsonl", changes)
    lines = [
        "# Подбор рубрики Qwen: сравнение с автоматической разметкой",
        "",
        "Победитель выбран по 75 development-сообщениям до проверки на 120 других сессиях.",
        "",
        "| Вариант | Совпадение на development | Macro-F1 |",
        "|---|---:|---:|",
    ]
    for row in selection["all_candidates"]:
        lines.append(f"| {row['candidate']} | {row['agreement']:.1%} | {row['macro_f1']:.3f} |")
    lines += [
        "",
        f"Выбран **{result['winner']}**.",
        "",
        "## Отдельная проверочная выборка",
        "",
        "| Метрика | Прежний трёхклассовый промпт | Выбранный вариант |",
        "|---|---:|---:|",
        f"| Совпадение | {result['baseline_agreement']:.1%} | {result['winner_agreement']:.1%} |",
        f"| Macro-F1 на всех входах | {all_input_metrics['baseline']['macro_f1']:.3f} | {all_input_metrics['winner']['macro_f1']:.3f} |",
        f"| Получено меток | {all_input_metrics['baseline']['successful']}/120 | {all_input_metrics['winner']['successful']}/120 |",
    ]
    for label in SENTIMENTS:
        lines.append(
            f"| Recall {label} | {all_input_metrics['baseline']['per_class'][label]['recall']:.1%} | {all_input_metrics['winner']['per_class'][label]['recall']:.1%} |"
        )
    lines += [
        "",
        f"Прирост согласия: {result['agreement_gain'] * 100:.1f} п.п.; 95% интервал парного bootstrap по классам: {boot[249] * 100:.1f}…{boot[9749] * 100:.1f} п.п.",
        "",
        f"Стали совпадать с источником: {paired['only_winner_matches']}; перестали совпадать: {paired['only_baseline_matches']}.",
        "Ошибки API считаются несовпадениями; они не исключаются из знаменателя и не получают фиктивный класс.",
        "",
        "Это согласие с автоматическим источником, а не точность по человеческому эталону. "
        "Выборка сбалансирована, очищена от повторов пилота и содержит одно сообщение на сессию; "
        "она не описывает естественную частоту классов. Корпус ранее использовался для разведочного анализа, "
        "поэтому этот holdout проверяет текущий подбор, а не production-качество.",
        "",
    ]
    write_text(ROOT / "report.md", "\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "report"])
    args = parser.parse_args()
    freeze() if args.phase == "freeze" else summarize()
