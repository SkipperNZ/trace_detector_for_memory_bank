"""Standalone model evaluation, corpus counts and a portable selected model."""

import gc
import importlib.metadata
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from .feedback import TASKS, CLASSES
from .feedback_data import read
from .feedback_models import FeedbackPredictor, load_model, load_partition, task_metrics
from .io import digest, read_jsonl, write_json, write_jsonl, write_text
from .sentiment_metrics import clustered_intervals


def evaluate_feedback(base):
    import numpy as np
    import torch

    base = Path(base)
    selection = read(base / "selection.json")
    design = read(base / "design.json")
    if selection["test_used"] or selection["design_hash"] != digest(design):
        raise ValueError("Freeze feedback model selection first")
    if (base / "evaluation.json").exists():
        result = read(base / "evaluation.json")
        if result["selection_hash"] != digest(selection):
            raise ValueError("Evaluation selection changed")
        return result
    write_json(
        base / "evaluation-start.json",
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "selected_at": selection["created_at"],
            "selection_hash": digest(selection),
        },
    )
    rows, y = load_partition(base, "test")
    _, train_y = load_partition(base, "train")
    majority = [
        int(np.bincount(train_y[train_y[:, j] >= 0, j], minlength=3).argmax())
        for j in range(len(TASKS))
    ]
    constant = np.tile(np.eye(3)[majority], (len(rows), 1, 1))
    torch.set_num_threads(8)
    latency_rows = sorted(rows, key=lambda ex: digest([42, "feedback-latency", ex["input_id"]]))[
        : design["policy"]["latency_sample_messages"]
    ]
    reports = []
    for candidate in selection["candidates"]:
        started = time.perf_counter()
        predictor = FeedbackPredictor(base / candidate["path"], device="cpu", batch_size=8)
        if predictor.artifact["model_version"] != candidate["model_version"]:
            raise ValueError("Candidate changed")
        load_seconds = time.perf_counter() - started
        predictor.probabilities(latency_rows[:3])
        started = time.perf_counter()
        probabilities = predictor.probabilities(rows)
        elapsed = time.perf_counter() - started
        latency = []
        for row in latency_rows:
            started = time.perf_counter()
            predictor.probabilities([row])
            latency.append(time.perf_counter() - started)
        metrics = task_metrics(y, probabilities)
        intervals = {}
        for j, task in enumerate(TASKS):
            mask = y[:, j] >= 0
            # Existing bootstrap treats index 0 as the positive-of-interest class.
            remap = np.asarray([1, 0, 2])
            ci = clustered_intervals(
                remap[y[mask, j]],
                probabilities[mask, j][:, [1, 0, 2]],
                [ex["group_id"] for ex, m in zip(rows, mask) if m],
                replicates=design["policy"]["bootstrap_replicates"],
                seed=42,
            )
            ci["yes_recall_95"] = ci.pop("negative_recall_95")
            intervals[task] = ci
        record = {
            "name": candidate["name"],
            "model_version": candidate["model_version"],
            "selected_on_calibration": candidate["name"] == selection["winner"]["name"],
            "calibration": candidate["calibration"],
            "test": metrics,
            "intervals": intervals,
            "artifact_bytes": candidate["artifact_bytes"],
            "runtime": {
                "device": "cpu",
                "threads": 8,
                "load_seconds": load_seconds,
                "batch_seconds": elapsed,
                "batch_messages_per_second": len(rows) / elapsed,
                "latency_sample_messages": len(latency),
                "p50_ms": float(np.quantile(latency, 0.5) * 1000),
                "p95_ms": float(np.quantile(latency, 0.95) * 1000),
            },
        }
        reports.append(record)
        write_jsonl(
            base / f"test-predictions/{candidate['name']}.jsonl",
            [
                {
                    "input_id": ex["input_id"],
                    "model_version": candidate["model_version"],
                    "signals": {
                        t: {
                            "label": CLASSES[int(p[j].argmax())],
                            "probabilities": dict(zip(CLASSES, map(float, p[j]))),
                        }
                        for j, t in enumerate(TASKS)
                    },
                }
                for ex, p in zip(rows, probabilities)
            ],
        )
        write_json(base / "evaluation-progress.json", reports)
        print(candidate["name"], "test mean macro-F1", metrics["mean_macro_f1"], flush=True)
        del predictor
        gc.collect()
    result = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "selection_hash": digest(selection),
        "models": reports,
        "test_messages": len(rows),
        "test_sessions": len({ex["group_id"] for ex in rows}),
        "majority_baseline": {
            "labels": dict(zip(TASKS, [CLASSES[j] for j in majority])),
            "test": task_metrics(y, constant),
        },
        "scope": "Agreement with frozen automatic Qwen labels, not independent gold. No Qwen calls during student inference.",
    }
    write_json(base / "evaluation.json", result)
    return result


def report_feedback(base, output, *, report_path=None):
    import numpy as np

    base, output = Path(base), Path(output)
    design, census, selection, evaluation = (
        read(base / f) for f in ("design.json", "census.json", "selection.json", "evaluation.json")
    )
    winner = selection["winner"]
    if output.exists():
        if load_model(output)["model_version"] != winner["model_version"]:
            raise ValueError("Export target contains another model")
    else:
        shutil.copytree(base / winner["path"], output)
    # Ship original frozen encoder weights as well, so every exported winner is portable.
    artifact = load_model(output)
    if artifact["kind"] == "frozen":
        from huggingface_hub import snapshot_download

        cfg = artifact["encoder"]
        snapshot = snapshot_download(
            cfg["model"],
            revision=cfg["revision"],
            cache_dir="artifacts/hf",
            local_files_only=True,
            token=False,
        )
        bundled = output / "backbone-cache"
        if not bundled.exists():
            shutil.copytree(snapshot, bundled)
        from .sentiment_encoder import encoder_directory_hash

        write_json(
            output / "backbone-bundle.json",
            {
                "model": cfg["model"],
                "revision": cfg["revision"],
                "sha256": encoder_directory_hash(bundled, pretrained=True),
            },
        )
    predictor = FeedbackPredictor(output, device="cpu", batch_size=8)
    rows, _ = load_partition(base, "test")
    predictions = predictor.predict(rows)
    expected = read_jsonl(base / f"test-predictions/{winner['name']}.jsonl")
    difference = 0.0
    for a, b in zip(predictions, expected, strict=True):
        if a["input_id"] != b["input_id"]:
            raise ValueError("Export prediction IDs differ")
        for t in TASKS:
            difference = max(
                difference,
                max(
                    abs(a["signals"][t]["probabilities"][c] - b["signals"][t]["probabilities"][c])
                    for c in CLASSES
                ),
            )
    if difference > 1e-7:
        raise ValueError("Exported probabilities differ")
    write_json(
        base / "export-verification.json",
        {
            "messages": len(rows),
            "probability_max_abs_difference": difference,
            "model_version": winner["model_version"],
        },
    )
    write_json(
        output / "environment.json",
        {
            "python_platform": platform.platform(),
            "packages": {
                p: importlib.metadata.version(p)
                for p in ("numpy", "scikit-learn", "torch", "sentence-transformers", "transformers")
            },
        },
    )
    write_json(output / "study-design.json", design)
    lines = [
        "# Разметка обратной связи об агенте",
        "",
        f"Завершено: {evaluation['completed_at']}. Классы каждой независимой задачи: no / yes / unclear.",
        "",
        "Недовольство — явная претензия к агенту. Коррекция — указание на ошибку его предыдущей работы, в том числе спокойное. Упоминания зацикливания и откатов — сообщения пользователя, не подтверждённые эпизоды действий агента.",
        "",
        f"Корпус: **{census['messages']} сообщений**; уникальных запросов с учётом повторного использования одинакового текста внутри tenant: {census['canonical_requests']}. Размечено {census['labelled_messages']}, технически не размечено {census['unlabelled_messages']}. Исходная кратность сохранена для подсчёта долей.",
        "",
        "| Признак | yes | no | unclear | yes / все сообщения | Сессии с yes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        s = census["tasks"][task]
        n = s["counts"]
        lines.append(
            f"| {task} | {n['yes']} | {n['no']} | {n['unclear']} | {s['yes_fraction_all_messages']:.2%} | {s['sessions_with_yes']} |"
        )
    lines += [
        "",
        "Неразмеченные записи не превращаются в no. Доля yes по всему корпусу — нижняя оценка при наличии unclear/пропусков. Полные границы и пересечения сигналов сохранены в census.json.",
        "",
        "Причины положительных меток (могут пересекаться): "
        + "; ".join(f"{k}: {v}" for k, v in sorted(census["reason_tags"].items()))
        + ".",
        "",
        "## Разбиение для моделей",
        "",
        "Подсчёт выполнен на всём корпусе. Для обучения отдельно исключены пустые/слишком длинные входы и пересечения точных/близких дубликатов между сессиями разных splits; исходные session splits сохранены. Метки не использовались для выбора split.",
        "",
        "| Split | Сообщения | Сессии |",
        "|---|---:|---:|",
    ]
    for s, r in design["splits"].items():
        lines.append(f"| {s} | {r['messages']} | {r['sessions']} |")
    lines += [
        "",
        f"Исключения из модельного исследования: {design['exclusions']}. Технически отсутствующие метки дополнительно маскируются в loss/метриках. Это публичный корпус пользовательских реплик; не проверка на корпоративных трейсах и не человеческий gold.",
        "",
        "## Самостоятельные модели",
        "",
        "У моделей общий текстовый бэкбон и две головы. Каждый вход кодируется один раз. Таблица показывает совпадение с фиксированным Qwen; при инференсе Qwen не вызывается.",
        "",
        "| Модель | Calibration mean macro-F1 | Test mean macro-F1 | Недовольство yes P / R / F1 | Коррекция yes P / R / F1 | CPU p50 / p95, мс | MiB |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for r in evaluation["models"]:
        a, b = (r["test"]["tasks"][t]["per_class"]["yes"] for t in TASKS)
        tag = " **(выбрана)**" if r["selected_on_calibration"] else ""
        lines.append(
            f"| {r['name']}{tag} | {r['calibration']['mean_macro_f1']:.4f} | {r['test']['mean_macro_f1']:.4f} | {a['precision']:.3f} / {a['recall']:.3f} / {a['f1']:.3f} | {b['precision']:.3f} / {b['recall']:.3f} / {b['f1']:.3f} | {r['runtime']['p50_ms']:.1f} / {r['runtime']['p95_ms']:.1f} | {r['artifact_bytes']/2**20:.1f} |"
        )
    baseline = evaluation.get("majority_baseline")
    if baseline:
        lines += [
            "",
            f"Контрольная модель, всегда выдающая самый частый train-класс ({baseline['labels']}): test mean macro-F1 {baseline['test']['mean_macro_f1']:.4f}. Macro-F1 во всех таблицах усредняется по трём фиксированным классам, включая отсутствующие (их F1 равен 0).",
        ]
    lines += [
        "",
        f"Выбрана **{winner['name']}** до теста, по среднему calibration macro-F1. Сетка C, class weights, 3 эпохи, seed 42 и гиперпараметры закреплены заранее в configs/feedback-study.json. Один seed не оценивает вариативность обучения. Полные accuracy, confusion matrices, support каждого класса и session-bootstrap интервалы — в evaluation.json.",
        "",
        "Окно энкодеров 512 токенов; весь переданный видимый контекст сохраняется через окна. E5 использует query-префикс и штатный пулинг, BGE-M3 — без префикса и с CLS. Теги причин — дополнительные метки учителя; модели обучены только на двух основных признаках.",
        "",
        f"Экспорт: `{output.as_posix()}`. Повторная загрузка и предсказания на {len(rows)} test-входах: max |Δp|={difference:.3g}.",
        "",
        "## Перенос в контур компании",
        "",
        "См. docs/feedback-guide.md: установка, локальный OpenAI-совместимый endpoint, входные JSONL и автономный запуск. Код не отправляет трейсы в OpenAI или GitHub; сообщения уходят только в настроенный judge endpoint. Все данные, API-ключи и веса исключены из Git.",
        "",
        "Корпоративные данные следует размечать новой версией запуска, сохранив реальные предыдущие ответы агента и tool-события. Проверка настоящих циклов/откатов требует событийных детекторов; наши reason_tags не заменяют такую проверку.",
        "",
    ]
    content = "\n".join(lines)
    if (base / "server-runtime-change.json").exists():
        runtime_change = read(base / "server-runtime-change.json")
        write_json(output / "server-runtime-change.json", runtime_change)
        content += "\n## Исполнение разметки\n\nПосле повторных CUDA-сбоев llama.cpp выключено ускорение CUDA Graphs (`GGML_CUDA_DISABLE_GRAPHS=1`). Веса, квантовка, промпт и параметры запросов сохранены; успешные ответы до переключения использованы из журнала. Изменение режима исполнения означает, что не все метки получены при идентичной конфигурации GPU-оптимизаций. Проверка стабильности сохранена в runtime-validation; это не независимая проверка смысловой точности.\n"
    write_text(base / "report.md", content)
    if report_path:
        write_text(report_path, content)
    write_text(
        output / "README.md",
        f"# Feedback model: {winner['name']}\n\nДва независимых признака: dissatisfaction / correction; классы no / yes / unclear. Режим shadow.\n\n```text\npython scripts/predict_feedback.py INPUT.jsonl --model-dir {output.as_posix()} --out predictions.jsonl --device cpu\n```\n\nВход и установка описаны в docs/feedback-guide.md. Учитель при инференсе не используется. Это модель воспроизведения фиксированных автоматических меток.\n",
    )
    return {"winner": winner["name"], "report": str(base / "report.md"), "export": str(output)}
