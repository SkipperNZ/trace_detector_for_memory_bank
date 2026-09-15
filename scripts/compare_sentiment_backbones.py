"""Report a frozen BGE-M3 comparison and verify its separately exported artifact."""

import argparse
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from memory_trace.io import read_jsonl, write_json, write_text, digest
from memory_trace.student_study import read_json, split_targets
from memory_trace.sentiment import load_student
from memory_trace.sentiment_encoder import encoder_directory_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("runs/sentiment-bge-m3-v1"))
    parser.add_argument("--reference", type=Path, default=Path("runs/sentiment-students-v1"))
    parser.add_argument("--model-output", type=Path, default=Path("artifacts/sentiment-bge-m3-v1"))
    args = parser.parse_args()
    base, reference, output = args.base, args.reference, args.model_output
    result, prior = read_json(base / "evaluation.json"), read_json(reference / "evaluation.json")
    policy = read_json(base / "study-design.json")["policy"]
    assert read_json(base / "data/design.json") == read_json(reference / "data/design.json")
    assert digest(read_jsonl(base / "teacher/labels.jsonl")) == digest(
        read_jsonl(reference / "teacher/labels.jsonl")
    )
    selection = read_json(base / "selection.json")
    assert selection["created_at"] < read_json(base / "evaluation-start.json")["started_at"]
    winner = next(r for r in result["models"] if r["selected_on_calibration"])
    if output.exists():
        assert load_student(output)["model_version"] == winner["model_version"]
    else:
        shutil.copytree(base / winner["path"], output)
    artifact = load_student(output)
    if artifact["kind"] == "finetuned":
        assert encoder_directory_hash(output / "encoder") == artifact["encoder"]["finetuned_sha256"]
    write_json(
        output / "routing-policy.json",
        {
            "mode": "shadow",
            "model_version": artifact["model_version"],
            "policies": selection["routing_policies"],
            "audit_rates": selection["audit_rates"],
            "note": "Calibration targets are not production recall guarantees.",
        },
    )
    write_json(output / "study-provenance.json", read_json(base / "study-design.json"))
    write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "sentence-transformers",
                    "transformers",
                    "torch",
                    "huggingface-hub",
                    "safetensors",
                )
            },
        },
    )
    predictions = base / "exported-predictions.jsonl"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "memory_trace",
            "predict-sentiment",
            str(base / "data/test.jsonl"),
            "--model-dir",
            str(output),
            "--out",
            str(predictions),
            "--device",
            "cpu",
            "--batch-size",
            "8",
        ],
        check=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        stdout=subprocess.DEVNULL,
    )
    actual, expected = read_jsonl(predictions), read_jsonl(
        base / f"test-predictions/{winner['name']}.jsonl"
    )
    values = []
    for a, b in zip(actual, expected, strict=True):
        assert a["input_id"] == b["input_id"] and a["model_version"] == b["model_version"]
        values.append([a["probabilities"][s] for s in ("NEGATIVE", "NEUTRAL", "POSITIVE")])
    diff = float(
        np.max(np.abs(np.asarray(values) - np.asarray([r["probabilities"] for r in expected])))
    )
    assert diff < 1e-7
    write_json(
        base / "export-verification.json",
        {
            "messages": len(actual),
            "max_probability_difference": diff,
            "selection_precedes_evaluation": True,
            "same_data_and_teacher": True,
            "platform": platform.platform(),
        },
    )
    rows = []
    for label, evaluation, directory in (("E5", prior, reference), ("BGE-M3", result, base)):
        candidates = {r["name"]: r for r in evaluation["selection"]["candidates"]}
        for r in evaluation["models"]:
            if r["size"] != "all" or (label == "BGE-M3" and r["kind"] in {"tfidf", "majority"}):
                continue
            name = {
                "majority": "Always NEUTRAL",
                "tfidf": "TF-IDF",
                "encoder": f"{label} frozen",
                "finetuned": f"{label} fine-tuned",
            }[r["kind"]]
            rows.append(
                {
                    "label": name,
                    "calibration_macro_f1": candidates[r["name"]]["calibration"]["macro_f1"],
                    **r,
                }
            )
    known, y, _, _ = split_targets(base, "test", policy)
    old_name = prior["selection"]["winner"]["name"]
    old_predictions = {
        r["input_id"]: r for r in read_jsonl(reference / f"test-predictions/{old_name}.jsonl")
    }
    new_predictions = {r["input_id"]: r for r in expected}
    groups = sorted({ex["group_id"] for ex in known})
    matrices = []
    for predictions_by_id in (old_predictions, new_predictions):
        matrices.append(
            np.asarray(
                [
                    np.bincount(
                        [
                            3 * target
                            + int(np.argmax(predictions_by_id[ex["input_id"]]["probabilities"]))
                            for ex, target in zip(known, y)
                            if ex["group_id"] == group
                        ],
                        minlength=9,
                    ).reshape(3, 3)
                    for group in groups
                ]
            )
        )
    sampled_groups = np.random.default_rng(42).integers(0, len(groups), size=(5000, len(groups)))
    scores = []
    for group_matrices in matrices:
        sampled = group_matrices[sampled_groups].sum(axis=1)
        diagonal = np.diagonal(sampled, axis1=1, axis2=2)
        denominator = sampled.sum(axis=1) + sampled.sum(axis=2)
        f1 = np.divide(
            2 * diagonal,
            denominator,
            out=np.zeros_like(diagonal, dtype=float),
            where=denominator > 0,
        ).mean(axis=1)
        agreement = diagonal.sum(axis=1) / sampled.sum(axis=(1, 2))
        scores.append((f1, agreement))
    paired = {
        "method": "paired session bootstrap, 5000 replicates, seed 42; selected BGE-study candidate minus selected E5-study candidate",
        "sessions": len(groups),
        "macro_f1_delta_95_interval": np.quantile(
            scores[1][0] - scores[0][0], [0.025, 0.975]
        ).tolist(),
        "agreement_delta_95_interval": np.quantile(
            scores[1][1] - scores[0][1], [0.025, 0.975]
        ).tolist(),
    }
    write_json(
        base / "comparison.json",
        {
            "models": rows,
            "bge_winner": winner["name"],
            "existing_test_reused": True,
            "paired_comparison": paired,
        },
    )
    lines = [
        "# BGE-M3 как бэкбон студента Qwen",
        "",
        "Дата: 2026-09-15. Сравнение замороженных и дообученных dense-энкодеров.",
        "",
        "Использованы те же сообщения, разбиение по сессиям и готовые метки Qwen; новых вызовов учителя: 0. Обучение: 5 353 размеченных сообщения, calibration: 1 356, test: 148 (11 NEGATIVE, 128 NEUTRAL, 9 POSITIVE).",
        "",
        "| Модель | Calibration macro-F1 | Test agreement | Test macro-F1 | NEG recall | CPU p50, мс | CPU p95, мс | Размер, MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        m, t = r["metrics"], r["runtime"]
        lines.append(
            f"| {r['label']} | {r['calibration_macro_f1']:.4f} | {m['agreement']:.2%} | {m['macro_f1']:.4f} | {m['per_class']['NEGATIVE']['recall']:.2%} | {t['single_message_p50_seconds']*1000:.1f} | {t['single_message_p95_seconds']*1000:.1f} | {r['artifact_bytes']/2**20:.1f} |"
        )
    fine = read_json(base / "finetune-selection.json")
    old_fine = next(r for r in rows if r["label"] == "E5 fine-tuned")
    new_fine = next(r for r in rows if r["label"] == "BGE-M3 fine-tuned")
    peak = fine["peak_cuda_allocated_bytes"]
    peak_text = f"{peak/2**30:.2f} GiB" if peak is not None else "не использовалась"
    lines += [
        "",
        f"Выбор по calibration: **{winner['name']}**. Дообучение BGE-M3: {fine['total_seconds']:.1f} с; пик выделенной CUDA-памяти: {peak_text}.",
        "",
        f"Парный bootstrap по {len(groups)} сессиям, 5 000 повторов: 95% интервал разницы test macro-F1 (выбранная модель BGE-эксперимента минус выбранный E5) — [{paired['macro_f1_delta_95_interval'][0]:+.4f}, {paired['macro_f1_delta_95_interval'][1]:+.4f}]. Он описывает выборочную неопределённость на этом корпусе и не устраняет ограничение повторного использования test.",
        "",
        f"У дообученного BGE-M3 precision NEGATIVE {new_fine['metrics']['per_class']['NEGATIVE']['precision']:.1%} против {old_fine['metrics']['per_class']['NEGATIVE']['precision']:.1%} у E5; recall NEGATIVE {new_fine['metrics']['per_class']['NEGATIVE']['recall']:.1%} против {old_fine['metrics']['per_class']['NEGATIVE']['recall']:.1%}. F1 POSITIVE {new_fine['metrics']['per_class']['POSITIVE']['f1']:.3f} против {old_fine['metrics']['per_class']['POSITIVE']['f1']:.3f}. Поэтому одинаковое общее согласие скрывает различие ошибок между классами.",
        "",
        "Исследован один seed и один заранее заданный рецепт дообучения. Результат относится к этой настройке BGE-M3 и не устанавливает предел качества семейства моделей.",
        "",
        "## Зафиксированный протокол",
        "",
        f"- BGE-M3 revision: `{policy['encoder_revision']}`; CLS-пулинг, без префикса, dense-вектор 1024.",
        "- Окно 512 токенов для сравнения с E5. Весь текст сохраняется через неперекрывающиеся окна и взвешенное усреднение; нативный контекст BGE-M3 8192 здесь не исследуется.",
        "- Линейная голова: та же сетка C и весов классов. Три эпохи полного supervised fine-tuning, LR encoder 2e-5, LR head 2e-4, effective batch 32, seed 42; BGE microbatch 2 × accumulation 16, gradient checkpointing, CUDA bf16 autocast.",
        "- Эпоха, температура и пороги выбраны только по calibration. Архитектура, гиперпараметры и число эпох зафиксированы до запуска BGE-M3.",
        "- Test уже использовался для E5 и разбора ошибок. Это повторное описательное сравнение на существующем test, не новый независимый holdout. Гиперпараметры BGE-M3 по нему не подбирались.",
        "- Метрики означают совпадение с автоматическим Qwen, а не точность относительно человеческого gold. Всего 11 тестовых негативов; небольшая разница не доказывает превосходство.",
        "- CPU-время E5 взято из предыдущего запуска на той же машине, BGE измерено сейчас; это не синхронный микробенчмарк.",
        "",
        "## Маршрутизация выбранного BGE-эксперимента",
        "",
        "| Целевая полнота на calibration | Аудит | Ожидаемый охват NEGATIVE | Доля сообщений Qwen | Экономия последовательного времени |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in result["routing"]:
        lines.append(
            f"| {r['policy']['target_calibration_recall']:.0%} | {r['audit_rate']:.0%} | "
            f"{r['expected_coverage_with_audit']:.1%} | {r['expected_teacher_call_fraction']:.1%} | "
            f"{r['expected_cost_saving_fraction']:.1%} |"
        )
    # Save the full routing and prevalence data without assuming they transfer to production.
    lines += [
        "",
        "Численные сценарии маршрутизации и выборочного оценивания доли негатива находятся в `evaluation.json`. Их доверительные интервалы требуют осторожности при редких пропущенных негативах; режим остаётся shadow.",
        "",
        "## Воспроизведение",
        "",
        "```text",
        "python scripts/cache_sentiment_encoder.py --base runs/sentiment-bge-m3-v1 --policy configs/student-study-bge-m3.json",
        "python scripts/run_backbone_study.py --device cuda",
        f"python -m memory_trace predict-sentiment INPUT.jsonl --model-dir {output.as_posix()} --out predictions.jsonl --device cpu --batch-size 8",
        "```",
        "",
        "Опционально передайте `--qwen-compose-dir PATH`, чтобы временно остановить и затем восстановить сервис qwen38-llama. Для macOS используйте `--device mps`, для CPU — `--device cpu`. В этом сравнении фактически проверен Windows; Linux/macOS BGE-запуски здесь не выполнялись.",
        "",
        "Команда cache загружает зафиксированные веса в `artifacts/hf` и готовит `encoder-source.json`; `encoder-cache.json` содержит путь локального snapshot текущего запуска. Оригинальные PyTorch-веса читаются с weights_only=True без remote code; дообученные веса сохраняются в safetensors. Все результаты и экспорт лежат отдельно от E5.",
        "",
        "Проверки: синтетические токены/эмбеддинги сверены со штатным SentenceTransformer; экспортированная модель повторно предсказала все 148 сообщений через CLI, max |Δp| = "
        + f"{diff:.3g}.",
        "",
        "[BGE-M3 — карточка автора](https://huggingface.co/BAAI/bge-m3) · [E5 — предыдущий эксперимент](sentiment-students-2026-09-14.md)",
        "",
    ]
    text = "\n".join(lines)
    write_text(base / "report.md", text)
    write_text(Path("docs/experiments/sentiment-bge-m3-2026-09-15.md"), text)
    write_text(
        output / "README.md",
        "# Студент Qwen: BGE-M3 experiment\n\n"
        + f"Выбран по calibration: {winner['name']}. Классы NEGATIVE / NEUTRAL / POSITIVE, режим shadow.\n\n"
        + f"```text\npython -m memory_trace predict-sentiment INPUT.jsonl --model-dir {output.as_posix()} --out predictions.jsonl --device cpu --batch-size 8\n```\n\n"
        + "Python 3.11+, зависимости проекта `[ml]`. См. study-provenance.json и routing-policy.json. Старый E5-артефакт сохранён отдельно.\n",
    )
    print("BGE comparison and portable export verified.", flush=True)


if __name__ == "__main__":
    main()
