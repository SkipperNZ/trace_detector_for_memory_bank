"""Pinned, resumable GLiNER Decide benchmark against the frozen feedback study."""

import hashlib
import importlib.metadata
import json
import platform
import re
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from .feedback import CLASSES, TASKS, student_inputs
from .feedback_data import read
from .feedback_models import load_partition, task_metrics
from .io import digest, read_jsonl, write_json, write_jsonl, write_text
from .sentiment import softmax


def now():
    return datetime.now(timezone.utc).isoformat()


def split_complete(text, measure, limit):
    """Keep every source character and enforce the actual encoded sequence budget."""
    if limit <= 0:
        raise ValueError("A positive token budget is required")
    pending, output = [text], []
    while pending:
        piece = pending.pop()
        length = measure(piece)
        if length <= limit:
            output.append((piece, length))
            continue
        if len(piece) < 2:
            raise ValueError("Schema leaves no room for a text character")
        middle = len(piece) // 2
        boundaries = [m.end() for m in re.finditer(r"\s+", piece) if 0 < m.end() < len(piece)]
        cut = min(boundaries, key=lambda n: abs(n - middle)) if boundaries else middle
        # Avoid extremely uneven splits caused by one distant whitespace boundary.
        if not len(piece) // 4 <= cut <= 3 * len(piece) // 4:
            cut = middle
        pending.extend([piece[cut:], piece[:cut]])
    if "".join(piece for piece, _ in output) != text:
        raise AssertionError("Chunking changed source text")
    return output


def aggregate_logits(logits, mode):
    import numpy as np

    values = np.asarray(logits, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (len(TASKS), len(CLASSES)):
        raise ValueError("Expected windows x tasks x classes")
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Nonempty finite logits are required")
    if mode == "max":
        return values.max(axis=0)
    if mode == "mean":
        return values.mean(axis=0)
    raise ValueError("Unknown logit aggregation")


def probabilities(records, mode):
    import numpy as np

    logits = np.asarray([r["logits"][mode] for r in records], dtype=float)
    return np.stack([softmax(logits[:, j]) for j in range(len(TASKS))], axis=1)


def yes_metrics(y, scores, threshold):
    import numpy as np

    y, scores = np.asarray(y), np.asarray(scores)
    known = y >= 0
    positive, selected = y[known] == 1, scores[known] >= threshold
    tp = int((positive & selected).sum())
    fp = int((~positive & selected).sum())
    fn = int((positive & ~selected).sum())
    return {
        "threshold": float(threshold),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "selected_fraction": float(selected.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "messages": int(known.sum()),
    }


def select_recall_threshold(y, scores, target):
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    y, scores = np.asarray(y), np.asarray(scores)
    mask = y >= 0
    if not 0 < target <= 1 or not (y[mask] == 1).any():
        raise ValueError("A valid recall target and positive calibration labels are required")
    precision, recall, thresholds = precision_recall_curve(y[mask] == 1, scores[mask])
    eligible = np.flatnonzero(recall[:-1] >= target)
    index = max(eligible, key=lambda i: (precision[i], thresholds[i]))
    return {"target_recall": target, **yes_metrics(y, scores, float(thresholds[index]))}


class DecidePredictor:
    def __init__(self, snapshot, policy, *, device="cpu"):
        import torch
        from gliner2.classification import Classifier, ClassificationConfig, ClassificationSchema

        self.policy, self.device = policy, device
        self.config = ClassificationConfig(batch_size=policy["batch_size"], max_len=None)
        started = time.perf_counter()
        self.classifier = (
            Classifier.from_pretrained(str(snapshot), local_files_only=True)
            .to(device=device, dtype=torch.float32)
            .eval()
        )
        processor = self.classifier.model.processor
        # The library defaults to dummy records on malformed input. A benchmark must fail.
        processor.collate_fn_inference = partial(
            processor.collate_fn_inference, error_policy="raise"
        )
        self.schemas, self.limits, self.overheads = {}, {}, {}
        for name, definitions in policy["schemas"].items():
            schema = ClassificationSchema()
            for task in TASKS:
                schema.single(
                    task,
                    definitions[task]["labels"],
                    instruction=definitions[task]["instruction"],
                    activation="softmax",
                )
            self.schemas[name] = self.classifier.compile_schema(schema)
            overhead = self.encoded_length("", name)
            self.overheads[name] = overhead
            limit = min(policy["max_encoded_tokens"], overhead + policy["max_content_tokens"])
            if limit <= overhead + 8:
                raise ValueError("Classification schema consumes the whole token budget")
            self.limits[name] = limit
        self.load_seconds = time.perf_counter() - started
        self.parameter_count = sum(p.numel() for p in self.classifier.model.parameters())

    def encoded_length(self, text, schema_name):
        batch = self.classifier.model.processor.collate_fn_inference(
            [(text, self.schemas[schema_name].build())], max_len=None
        )
        return int(batch.input_ids.shape[1])

    def predict(self, texts, schema_name):
        import torch

        started = time.perf_counter()
        planned = [
            split_complete(
                text, lambda s: self.encoded_length(s, schema_name), self.limits[schema_name]
            )
            for text in texts
        ]
        flat = [
            (owner, text, size) for owner, chunks in enumerate(planned) for text, size in chunks
        ]
        order = sorted(range(len(flat)), key=lambda i: flat[i][2])
        scores = self.classifier.batch_score(
            [flat[i][1] for i in order],
            self.schemas[schema_name],
            config=self.config,
        )
        if len(scores) != len(flat):
            raise ValueError("GLiNER dropped a window")
        by_owner = [[] for _ in texts]
        for i, score in zip(order, scores, strict=True):
            if set(score.tasks) != set(TASKS):
                raise ValueError("GLiNER dropped or renamed a task")
            if any(set(score.tasks[t]) != set(CLASSES) for t in TASKS):
                raise ValueError("GLiNER changed the class inventory")
            by_owner[flat[i][0]].append([[score.logit(t, c) for c in CLASSES] for t in TASKS])
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return [
            {
                "logits": {
                    m: aggregate_logits(values, m).tolist() for m in self.policy["aggregations"]
                },
                "windows": len(values),
                "max_encoded_tokens": max(n for _, n in chunks),
                "text_characters": len(text),
                "truncated_characters": 0,
                "batch_seconds_per_message": elapsed / len(texts),
            }
            for text, values, chunks in zip(texts, by_owner, planned, strict=True)
        ]


def score_partition(predictor, rows, schema_name, out, manifest_hash, split):
    """Per-message SQLite commits survive interruption; model/schema/input hashes guard reuse."""
    out = Path(out)
    rendered = student_inputs(rows)
    texts = [r["context"][0]["text"] for r in rendered]
    ids = [
        digest([manifest_hash, schema_name, r["input_id"], digest(text)])
        for r, text in zip(rows, texts, strict=True)
    ]
    with closing(sqlite3.connect(out / "scores.sqlite3")) as db:
        db.execute("CREATE TABLE IF NOT EXISTS scores (id TEXT PRIMARY KEY, record TEXT NOT NULL)")
        cached = {
            key: json.loads(value) for key, value in db.execute("SELECT id, record FROM scores")
        }
        pending = [i for i, key in enumerate(ids) if key not in cached]
        started = time.perf_counter()
        size = predictor.policy["document_batch_size"]
        for begin in range(0, len(pending), size):
            indices = pending[begin : begin + size]
            values = predictor.predict([texts[i] for i in indices], schema_name)
            for i, value in zip(indices, values, strict=True):
                record = {"input_id": rows[i]["input_id"], "text_hash": digest(texts[i]), **value}
                db.execute("INSERT INTO scores VALUES (?, ?)", (ids[i], json.dumps(record)))
                cached[ids[i]] = record
            db.commit()
            completed = min(begin + size, len(pending))
            elapsed = time.perf_counter() - started
            write_json(
                out / "progress.json",
                {
                    "status": "running",
                    "split": split,
                    "schema": schema_name,
                    "completed": len(rows) - len(pending) + completed,
                    "total": len(rows),
                    "elapsed_seconds": elapsed,
                    "remaining_seconds_this_schema": elapsed
                    / completed
                    * (len(pending) - completed),
                    "updated_at": now(),
                },
            )
    return [cached[key] for key in ids]


def build_manifest(source, out, policy, snapshot):
    source, out, snapshot = Path(source), Path(out), Path(snapshot)
    out.mkdir(parents=True, exist_ok=True)
    design = read(source / "design.json")
    frozen = read(source / "labels-frozen.json")
    weights = snapshot / "model.safetensors"
    with weights.open("rb") as stream:
        weights_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    stable = {
        "adapter_version": "gliner-decide-complete-text-v1",
        "policy": policy,
        "source_design_hash": digest(design),
        "teacher": frozen,
        "weights_sha256": weights_hash,
        "checkpoint_config_hash": digest(read(snapshot / "config.json")),
        "packages": {
            name: importlib.metadata.version(name)
            for name in [
                "gliner2",
                "transformers",
                "torch",
                "numpy",
                "scikit-learn",
                "huggingface-hub",
            ]
        },
    }
    if stable["packages"]["gliner2"] != policy["gliner2_version"]:
        raise ValueError("Use the pinned gliner2 package version")
    path = out / "manifest.json"
    if path.exists():
        old = read(path)
        if old["fingerprint"] != digest(stable):
            raise ValueError(
                "Benchmark inputs, policy, weights or runtime changed; use a new output directory"
            )
        return old
    manifest = {
        **stable,
        "fingerprint": digest(stable),
        "created_at": now(),
        "platform": platform.platform(),
        "snapshot": snapshot.as_posix(),
        "protocol": "No weight training. Two predefined schemas x two logit aggregations; selection and yes thresholds use calibration only. No Qwen calls.",
        "test_scope": "Same previously inspected public test as feedback-v1; not new independent gold. Pretraining overlap with this public corpus is unknown.",
        "missing_label_policy": "Predict all inputs; mask missing teacher labels in metrics; never convert them to no.",
        "windowing": "Up to 512 content subwords plus schema within 1024 encoded tokens; actual processor lengths checked; all source characters retained in nonoverlapping chunks; aggregate logits before softmax.",
    }
    write_json(path, manifest)
    return manifest


def calibrate_decide(predictor, source, out, manifest):
    out = Path(out)
    if (out / "selection.json").exists():
        selection = read(out / "selection.json")
        if selection["manifest_hash"] != manifest["fingerprint"]:
            raise ValueError("Selection belongs to a different benchmark")
        return selection
    rows, y = load_partition(source, "calibration")
    candidates, results = [], {}
    for schema in predictor.schemas:
        print("CALIBRATION", schema, len(rows), flush=True)
        records = score_partition(
            predictor, rows, schema, out, manifest["fingerprint"], "calibration"
        )
        for mode in predictor.policy["aggregations"]:
            name = schema + "-" + mode
            p = probabilities(records, mode)
            results[name] = p
            candidates.append(
                {"name": name, "schema": schema, "aggregation": mode, "metrics": task_metrics(y, p)}
            )
    winner = sorted(
        candidates,
        key=lambda c: (-c["metrics"]["mean_macro_f1"], -c["metrics"]["mean_yes_f1"], c["name"]),
    )[0]
    thresholds = {
        t: [
            select_recall_threshold(y[:, j], results[winner["name"]][:, j, 1], target)
            for target in predictor.policy["recall_targets"]
        ]
        for j, t in enumerate(TASKS)
    }
    selection = {
        "created_at": now(),
        "manifest_hash": manifest["fingerprint"],
        "winner": winner,
        "candidates": candidates,
        "yes_thresholds": thresholds,
        "test_used": False,
    }
    write_json(out / "selection.json", selection)
    return selection


def average_precision(y, p):
    from sklearn.metrics import average_precision_score

    return {
        t: float(average_precision_score(y[y[:, j] >= 0, j] == 1, p[y[:, j] >= 0, j, 1]))
        for j, t in enumerate(TASKS)
    }


def evaluate_decide(predictor, source, out, manifest, selection):
    import numpy as np
    from .sentiment_metrics import clustered_intervals

    source, out = Path(source), Path(out)
    if (out / "evaluation.json").exists():
        old = read(out / "evaluation.json")
        if old["selection_hash"] != digest(selection):
            raise ValueError("Frozen evaluation selection changed")
        return old
    rows, y = load_partition(source, "test")
    winner = selection["winner"]
    print("TEST", winner["name"], len(rows), flush=True)
    records = score_partition(
        predictor, rows, winner["schema"], out, manifest["fingerprint"], "test"
    )
    p = probabilities(records, winner["aggregation"])
    predictions = [
        {
            "input_id": row["input_id"],
            "windows": record["windows"],
            "signals": {
                t: {
                    "label": CLASSES[int(prob[j].argmax())],
                    "probabilities": dict(zip(CLASSES, map(float, prob[j]))),
                }
                for j, t in enumerate(TASKS)
            },
        }
        for row, record, prob in zip(rows, records, p, strict=True)
    ]
    write_jsonl(out / "test-predictions.jsonl", predictions)
    intervals = {}
    for j, task in enumerate(TASKS):
        mask = y[:, j] >= 0
        remap = np.asarray([1, 0, 2])
        ci = clustered_intervals(
            remap[y[mask, j]],
            p[mask, j][:, [1, 0, 2]],
            [r["group_id"] for r, keep in zip(rows, mask) if keep],
            replicates=predictor.policy["bootstrap_replicates"],
            seed=42,
        )
        ci["yes_recall_95"] = ci.pop("negative_recall_95")
        intervals[task] = ci
    baselines = []
    for baseline in read(source / "evaluation.json")["models"]:
        saved = read_jsonl(source / f"test-predictions/{baseline['name']}.jsonl")
        if [r["input_id"] for r in rows] != [r["input_id"] for r in saved]:
            raise ValueError("Baseline uses a different test subset/order")
        bp = np.asarray(
            [[[r["signals"][t]["probabilities"][c] for c in CLASSES] for t in TASKS] for r in saved]
        )
        baselines.append(
            {
                "name": baseline["name"],
                "test": baseline["test"],
                "yes_average_precision": average_precision(y, bp),
                "runtime": baseline["runtime"],
            }
        )
    result = {
        "completed_at": now(),
        "selection_hash": digest(selection),
        "candidate": winner["name"],
        "test": task_metrics(y, p),
        "intervals": intervals,
        "yes_average_precision": average_precision(y, p),
        "yes_thresholds": {
            t: [
                {
                    **yes_metrics(y[:, j], p[:, j, 1], v["threshold"]),
                    "calibration_target_recall": v["target_recall"],
                    "calibration_precision": v["precision"],
                    "calibration_recall": v["recall"],
                }
                for v in selection["yes_thresholds"][t]
            ]
            for j, t in enumerate(TASKS)
        },
        "test_messages": len(rows),
        "test_sessions": len({r["group_id"] for r in rows}),
        "window_audit": {
            "total_windows": sum(r["windows"] for r in records),
            "multiwindow_messages": sum(r["windows"] > 1 for r in records),
            "max_windows_per_message": max(r["windows"] for r in records),
            "max_encoded_tokens": max(r["max_encoded_tokens"] for r in records),
            "truncated_characters": sum(r["truncated_characters"] for r in records),
        },
        "baselines": baselines,
    }
    write_json(out / "evaluation.json", result)
    return result


def measure_cpu(predictor, source, out, selection):
    import numpy as np

    out = Path(out)
    if (out / "runtime.json").exists():
        value = read(out / "runtime.json")
        if value["selection_hash"] != digest(selection):
            raise ValueError("Runtime measurement belongs to another selection")
        return value
    if predictor.device != "cpu":
        raise ValueError("Compare CPU latency on CPU")
    rows, _ = load_partition(source, "test")
    rows = sorted(rows, key=lambda ex: digest([42, "feedback-latency", ex["input_id"]]))[
        : predictor.policy["latency_sample_messages"]
    ]
    texts = [r["context"][0]["text"] for r in student_inputs(rows)]
    winner = selection["winner"]
    for text in texts[:3]:
        predictor.predict([text], winner["schema"])
    expected = {r["input_id"]: r for r in read_jsonl(out / "test-predictions.jsonl")}
    elapsed, diff, mismatches = [], 0.0, 0
    for row, text in zip(rows, texts, strict=True):
        started = time.perf_counter()
        values = predictor.predict([text], winner["schema"])
        p = probabilities(values, winner["aggregation"])[0]
        elapsed.append(time.perf_counter() - started)
        for j, t in enumerate(TASKS):
            before = expected[row["input_id"]]["signals"][t]
            mismatches += CLASSES[int(p[j].argmax())] != before["label"]
            diff = max(
                diff, max(abs(p[j, k] - before["probabilities"][c]) for k, c in enumerate(CLASSES))
            )
    runtime = {
        "selection_hash": digest(selection),
        "device": "cpu",
        "dtype": "float32",
        "threads": 8,
        "messages": len(rows),
        "load_seconds": predictor.load_seconds,
        "p50_ms": float(np.quantile(elapsed, 0.5) * 1000),
        "p95_ms": float(np.quantile(elapsed, 0.95) * 1000),
        "mean_ms": float(np.mean(elapsed) * 1000),
        "full_model_parameters": predictor.parameter_count,
        "cpu_vs_test_label_mismatches": int(mismatches),
        "cpu_vs_test_probability_max_abs_difference": float(diff),
        "schema_encoded_overheads": predictor.overheads,
        "schema_encoded_limits": predictor.limits,
        "scope": "Same 100 seeded full-text inputs as feedback-v1; old baseline timings are historical, not a contemporaneous speed test.",
    }
    write_json(out / "runtime.json", runtime)
    return runtime


def report_decide(source, out, report_path):
    source, out, report_path = Path(source), Path(out), Path(report_path)
    manifest, selection, evaluation, runtime = (
        read(out / f)
        for f in ["manifest.json", "selection.json", "evaluation.json", "runtime.json"]
    )
    public = {
        "manifest": {k: v for k, v in manifest.items() if k != "snapshot"},
        "selection": selection,
        "evaluation": evaluation,
        "runtime": runtime,
    }
    for name in ("native-api-parity", "execution", "qwen-restored"):
        if (out / f"{name}.json").exists():
            public[name.replace("-", "_")] = read(out / f"{name}.json")
    if "native_api_parity" in public and not public["native_api_parity"]["passed"]:
        raise ValueError("Cannot publish benchmark with failed native API parity")
    baselines = {r["name"]: r for r in evaluation["baselines"]}
    a, b = (evaluation["test"]["tasks"][t] for t in TASKS)
    lines = [
        "# GLiNER2.5-Decide: обратная связь об агенте",
        "",
        f"Test завершён: {evaluation['completed_at']}. Модель `{manifest['policy']['model']}`, revision `{manifest['policy']['revision']}`; gliner2 {manifest['packages']['gliner2']}.",
        "",
        f"Результат без дообучения: mean macro-F1 **{evaluation['test']['mean_macro_f1']:.4f}**; BGE-M3 finetuned — **{baselines['bge-m3-finetuned']['test']['mean_macro_f1']:.4f}**, E5 finetuned — **{baselines['e5-finetuned']['test']['mean_macro_f1']:.4f}**. GLiNER распознал {a['confusion_matrix'][1][1]} из {a['per_class']['yes']['support']} положительных сообщений по недовольству и {b['confusion_matrix'][1][1]} из {b['per_class']['yes']['support']} по исправлениям.",
        "",
        "Это оценка конкретного checkpoint с двумя проверенными схемами, без адаптации его весов к нашим меткам. Она не устанавливает потолок качества GLiNER после дообучения. Чужой benchmark на других задачах и метриках не заменяет эту проверку.",
        "",
        "## Протокол",
        "",
        "Два независимых признака: dissatisfaction (явное недовольство работой агента) и correction (исправление ошибки в его предыдущей работе). Классы no / yes / unclear. Учитель — фиксированная разметка локального Qwen из feedback-v1; человеческого эталона нет. На входе та же отдельная реплика пользователя, без предыдущих ответов агента.",
        "",
        "Веса GLiNER не дообучались. До получения новых оценок заданы две схемы описаний классов и два способа объединения окон (max/mean logits). Победитель и пороги выбраны только на calibration. Сравниваем со студентами, обученными на train этого корпуса; режимы обучения различаются. Test уже изучался в предыдущем исследовании; неизвестно, встречался ли публичный корпус при предобучении GLiNER.",
        "",
        f"Test: {evaluation['test_messages']} сообщений, {evaluation['test_sessions']} сессий; валидных меток по задачам: {dict((t, evaluation['test']['tasks'][t]['messages']) for t in TASKS)}. Пропуски маскируются. Никаких новых вызовов Qwen.",
        "",
        "Полный текст сохраняется в непересекающихся окнах: до 512 текстовых subword-токенов, до 1024 токенов вместе со схемой. Размер каждого окна проверяется реальным процессором. Сначала агрегируются logits окон, затем softmax и независимый argmax для каждой задачи. Молчаливая подстановка пустого входа при ошибке библиотеки отключена.",
        "",
        f"Аудит окон: `{evaluation['window_audit']}`.",
        "",
        "## Выбор на calibration",
        "",
        "| Схема / агрегация | Mean macro-F1 | Mean yes F1 |",
        "|---|---:|---:|",
    ]
    for c in selection["candidates"]:
        lines.append(
            f"| {c['name']} | {c['metrics']['mean_macro_f1']:.4f} | {c['metrics']['mean_yes_f1']:.4f} |"
        )
    lines += [
        "",
        f"Выбран **{selection['winner']['name']}** ({selection['created_at']}), до test.",
        "",
        "## Сравнение на одном test",
        "",
        "| Модель | Mean macro-F1 | Недовольство yes P / R / F1 | Исправление yes P / R / F1 | AP yes: недовольство / исправление |",
        "|---|---:|---|---|---|",
    ]
    models = [
        {
            "name": "GLiNER2.5-Decide (без дообучения)",
            "test": evaluation["test"],
            "yes_average_precision": evaluation["yes_average_precision"],
        },
        *evaluation["baselines"],
    ]
    for r in models:
        a, b = [r["test"]["tasks"][t]["per_class"]["yes"] for t in TASKS]
        ap = r["yes_average_precision"]
        lines.append(
            f"| {r['name']} | {r['test']['mean_macro_f1']:.4f} | {a['precision']:.3f} / {a['recall']:.3f} / {a['f1']:.3f} | {b['precision']:.3f} / {b['recall']:.3f} / {b['f1']:.3f} | {ap[TASKS[0]]:.3f} / {ap[TASKS[1]]:.3f} |"
        )
    lines += [
        "",
        "AP — average precision для yes против no/unclear; рассчитывается по ранжированию, не по выбранному порогу. Macro-F1 усредняет три фиксированных класса и две задачи.",
        "",
        "## Матрицы GLiNER",
        "",
        "Строки — Qwen, столбцы — GLiNER. Все числа — количества сообщений.",
        "",
    ]
    for task in TASKS:
        lines += [
            "",
            f"### {task}",
            "",
            "| Qwen / GLiNER | no | yes | unclear |",
            "|---|---:|---:|---:|",
        ]
        for name, row in zip(CLASSES, evaluation["test"]["tasks"][task]["confusion_matrix"]):
            lines.append(f"| {name} | " + " | ".join(map(str, row)) + " |")
    lines += [
        "",
        "## Пороги высокой полноты",
        "",
        "Порог выбран по максимальной calibration precision при recall не ниже заданной. Ниже фактические результаты на test; целевая полнота не гарантируется при переносе. Это бинарный отбор yes, отдельный от трёхклассового argmax. no и unclear считаются не-yes.",
        "",
        "| Признак | Цель calibration recall | Test precision | Test recall | Доля отобранных |",
        "|---|---:|---:|---:|---:|",
    ]
    for t in TASKS:
        for v in evaluation["yes_thresholds"][t]:
            lines.append(
                f"| {t} | {v['calibration_target_recall']:.0%} | {v['precision']:.2%} | {v['recall']:.2%} | {v['selected_fraction']:.2%} |"
            )
    lines += [
        "",
        "## Исполнение",
        "",
        f"CPU, float32, 8 потоков; 100 тех же случайно зафиксированных test-входов: p50 **{runtime['p50_ms']:.1f} мс**, p95 **{runtime['p95_ms']:.1f} мс**. Включены токенизация, обработка всех окон и обе задачи. Веса загружаются один раз. Предыдущие времена E5/BGE исторические, не одновременно измеренный speedup.",
        "",
        f"Полный счёт параметров загруженного модуля: {runtime['full_model_parameters']:,}; в карточке модель называется 340M. Различия CPU и основного test-прогона на 100 сообщениях: {runtime['cpu_vs_test_label_mismatches']} меток, max |Δp|={runtime['cpu_vs_test_probability_max_abs_difference']:.3g}.",
        "",
        "На этой версии Transformers DeBERTa использует eager attention: библиотека автоматически отклонила запрошенный checkpoint-ом SDPA и выбрала поддерживаемую реализацию. Не применялись FlashDeBERTa, квантование, torch.compile или ONNX. Измерена скорость данного Python-стека, не предел производительности архитектуры.",
        "",
        "Модель заявлена как английская; этот эксперимент не устанавливает качество на русскоязычных корпоративных трейсах. Калиброванность вероятностей по умолчанию не предполагается. Независимая проверка качества учителя не проводилась.",
        "",
        "## Источники и воспроизведение",
        "",
        "- [Карточка выбранной модели](https://huggingface.co/fastino/GLiNER2.5-Decide)",
        "- [Официальная библиотека](https://github.com/fastino-ai/GLiNER2)",
        "- Политика: `configs/gliner25-feedback.json`; запуск: `scripts/run_gliner_benchmark.py`.",
        "- Основные метки и веса остаются локально; рядом с отчётом опубликован только агрегированный JSON.",
        "",
    ]
    if "native_api_parity" in public:
        checks = public["native_api_parity"]["checks"]
        lines += [
            "## Проверка адаптера",
            "",
            f"Сравнение с официальным `model.classify_text`: {sum(r['task_decisions'] for r in checks)} решений на calibration и синтетических входах, {sum(r['label_mismatches'] for r in checks)} расхождений. Проверены обе схемы, обработка logits и порядок классов. Это проверка интеграции, не независимая оценка смысловой точности.",
            "",
        ]
    write_text(out / "report.md", "\n".join(lines))
    write_text(report_path, "\n".join(lines))
    write_json(report_path.with_name(report_path.stem + "-metrics.json"), public)
    write_json(
        out / "complete.json",
        {"completed_at": now(), "status": "complete", "report": str(report_path)},
    )
    write_json(out / "progress.json", {"status": "complete", "completed_at": now()})
