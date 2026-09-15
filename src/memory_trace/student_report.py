"""Write a compact research report, standard plots, and a portable selected artifact."""

import importlib.metadata
import os
import platform
import shutil
import sys
from pathlib import Path

from .io import write_json, write_text
from .sentiment import load_student
from .student_study import read_json


def percent(value):
    return "—" if value is None else f"{100*value:.1f}%"


def render_report(base, *, report_path=None, model_output=None):
    base = Path(base)
    evaluation = read_json(base / "evaluation.json")
    design = read_json(base / "data/design.json")
    baseline = read_json(base / "baseline-selection.json")
    winner = next(r for r in evaluation["models"] if r["selected_on_calibration"])
    teacher = evaluation["teacher"]
    output = Path(model_output or "artifacts/sentiment-student-v1")
    source = base / winner["path"]
    if output.exists():
        if load_student(output)["model_version"] != winner["model_version"]:
            raise ValueError("Portable artifact directory already contains another model")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, output)
    exported = load_student(output)
    if exported["kind"] == "finetuned":
        from .sentiment_encoder import encoder_directory_hash

        if encoder_directory_hash(output / "encoder") != exported["encoder"]["finetuned_sha256"]:
            raise ValueError("Exported encoder checksum mismatch")
    write_json(
        output / "routing-policy.json",
        {
            "source": "calibration",
            "mode": "shadow",
            "model_version": winner["model_version"],
            "policies": evaluation["selection"]["routing_policies"],
            "audit_rates": evaluation["selection"]["audit_rates"],
            "note": "Targets are calibration recall levels, not production guarantees. NEGATIVE follows the Qwen content-valence rubric.",
        },
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "libraries": {
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
                "pyarrow",
                "psutil",
                "matplotlib",
            )
        },
        "portability": "Training ran on Windows. Windows and Linux core/linear tests were executed; macOS CI is configured but was not run on this host. CUDA was used only if reported.",
        "verification": (
            read_json(base / "verification.json") if (base / "verification.json").exists() else None
        ),
    }
    write_json(base / "environment.json", environment)
    write_json(output / "environment.json", environment)
    write_json(
        output / "study-provenance.json",
        {
            "model_version": winner["model_version"],
            "teacher_version": evaluation["selection"]["teacher_version"],
            "study_policy_hash": evaluation["selection"]["study_policy_hash"],
            "selected_at": evaluation["selection"]["created_at"],
            "selected_before_test": evaluation["selection"]["selected_before_test"],
            "teacher_weight_file": (
                read_json(base / "teacher-weight-file.json")
                if (base / "teacher-weight-file.json").exists()
                else None
            ),
        },
    )
    lines = [
        (
            "# Синтетическая проверка программы — не результат исследования"
            if design.get("synthetic")
            else "# Студенты Qwen: разметка, обучение и проверка"
        ),
        "",
        f"Завершено: {evaluation['completed_at']}. Режим применения: **shadow**.",
        "",
        f"По calibration выбран **{winner['name']}**. На финальной проверке: совпадение с Qwen **{percent(winner['metrics']['agreement'])}**, macro-F1 **{winner['metrics']['macro_f1']:.3f}**.",
        f"Модель выбрана до финальной проверки; ее имя не менялось по результатам test. Эталон сравнения — фиксированный автоматический учитель, человеческих меток нет.",
        "",
        "## Данные и учитель",
        "",
        f"Из {design['original_messages']} исходных сообщений допущено {design['retained_messages']}. Сессии и компоненты точных/близких дубликатов между splits не пересекаются. Ранее просмотренные сессии и близкие тексты исключены из test. Распределение классов не выравнивалось.",
        "",
        "| Split | Допущено | Сессий | Меток Qwen | NEGATIVE | NEUTRAL | POSITIVE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, data in design["splits"].items():
        labelled = teacher["splits"][split]
        counts = labelled["labels"]
        lines.append(
            f"| {split} | {data['messages']} | {data['sessions']} | {labelled['labelled']} | {counts.get('NEGATIVE',0)} | {counts.get('NEUTRAL',0)} | {counts.get('POSITIVE',0)} |"
        )
    lines += [
        "",
        f"Исключения: {design['exclusion_counts']}. Удаление шаблонных пересечений меняет допустимую популяцию; частоты ниже относятся именно к ней.",
        f"Учитель: `{evaluation['selection']['teacher_version']}`. Использовано {teacher['reused_calls']} идентичных старых запросов. Всего в журнале {teacher['attempts_including_original_cached_and_retries']} попыток с учетом старых и повторных; статусы итоговых вызовов: {teacher['call_status_counts']}.",
        f"Сумма времени сохраненных попыток: {teacher['sum_attempt_seconds_including_reused']/3600:.2f} ч; это включает время переиспользованных ответов и не равно длительности новой серии. Токены: {teacher['tokens']}.",
        "",
        "Исходные метки, их объяснения, ответы и reasoning Qwen не входят в признаки студентов. Обучающие цели — только три класса учителя. Неразмеченные входы не подменяются NEUTRAL; покрытие показывается отдельно.",
        "",
        "## Качество и скорость",
        "",
        "Основные модели обучены на всех доступных train-метках. Интервалы получены bootstrap по сессиям. Accuracy/F1 считаются на известных ответах учителя; отсутствие ответа видно в покрытии и отдельном согласии на всех допустимых входах.",
        "",
        "| Модель | Совпадение | Macro-F1 | Recall NEGATIVE | CPU/device p50, мс | p95, мс | Батч, сообщений/с | Модель, МиБ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evaluation["models"]:
        if row["size"] != "all":
            continue
        m, r = row["metrics"], row["runtime"]
        name = row["name"] + (" **выбрана**" if row["selected_on_calibration"] else "")
        storage = f"{row['artifact_bytes']/2**20:.1f}" if "artifact_bytes" in row else "—"
        lines.append(
            f"| {name} | {percent(m['agreement'])} | {m['macro_f1']:.3f} | {percent(m['per_class']['NEGATIVE']['recall'])} | {r['single_message_p50_seconds']*1000:.2f} | {r['single_message_p95_seconds']*1000:.2f} | {r['batch_messages_per_second']:.1f} | {storage} |"
        )
    lo, hi = winner["intervals"]["agreement_95"]
    lines += [
        "",
        "Размер включает голову, а для E5 — также веса и токенизатор, даже если замороженный энкодер хранится в отдельном кэше. Это размер файлов модели, не пиковая память обучения.",
        f"Для выбранной модели 95% интервал согласия: **{percent(lo)}…{percent(hi)}**; F1: {winner['intervals']['macro_f1_95']}. Из {winner['eligible_test_messages']} входов учитель ответил на {winner['known_teacher_test_messages']}; согласие на всех входах: {percent(winner['joint_agreement_all_eligible'])}.",
        f"Устройство замера: {winner['runtime']['device']}, torch threads={winner['runtime']['torch_threads']}. Измерены все {winner['runtime']['single_message_samples']} тестовых сообщений после прогрева. Обработка текста и классификация включены; загрузка модели отдельно: {winner['runtime']['model_load_seconds']:.2f} с. Наблюдаемый максимум RSS всего процесса: {winner['runtime']['observed_process_rss_max_bytes']/2**20:.0f} МиБ, включая библиотеки.",
        "",
        "Длинный текст энкодер делит на последовательные окна по токенам с повторением query-префикса, затем усредняет нормализованные эмбеддинги с весами по числу токенов. Конец сообщения не отбрасывается. Такой pooling — наша инженерная настройка, отдельно его оптимальность не устанавливалась.",
        "",
        "| Класс выбранной модели | Примеров Qwen | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, metrics in winner["metrics"]["per_class"].items():
        lines.append(
            f"| {label} | {metrics['support']} | {percent(metrics['precision'])} | {percent(metrics['recall'])} | {metrics['f1']:.3f} |"
        )
    lines += [
        "",
        "Редкие классы представлены небольшим числом примеров. Состав test после исключения ранее просмотренных сессий отличается от train/calibration; высокая общая доля совпадений не заменяет проверки редких классов. Это не оценка качества на рабочем потоке или на русском языке.",
        "",
        "## Кривая обучения",
        "",
        "| Модель | Train-сообщений | Calibration macro-F1 | Test macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    cal = {r["name"]: r for r in baseline["candidates"]}
    for row in evaluation["models"]:
        if row["kind"] in {"tfidf", "encoder"}:
            lines.append(
                f"| {row['name']} | {row['training_messages']} | {cal[row['name']]['calibration']['macro_f1']:.3f} | {row['metrics']['macro_f1']:.3f} |"
            )
    lines += [
        "",
        "Подвыборки вложенные, набирались целыми сессиями; фактический размер может превышать целевой. Проверка трех размеров использует одну фиксированную группировку, поэтому не является оценкой вариации по всем возможным обучающим наборам.",
        "",
        "## Полнота и стоимость фильтра",
        "",
        "Порог выбирался на calibration. В таблице показана симуляция направления целой сессии при хотя бы одном срабатывании, с Bernoulli-аудитом остальных сессий. Это офлайн-оценка; реальная ранняя маршрутизация по потоку событий здесь не проверялась.",
        "",
        "| Цель recall на calibration | Аудит | Recall сообщений моделью | Покрытие NEGATIVE с аудитом, ожидание | Доля вызовов Qwen, ожидание | Экономия времени, оценка |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in evaluation["routing"]:
        lines.append(
            f"| {percent(row['policy']['target_calibration_recall'])} | {percent(row['audit_rate'])} | {percent(row['model_negative_message_recall'])} | {percent(row['expected_coverage_with_audit'])} | {percent(row['expected_teacher_call_fraction'])} | {percent(row['expected_cost_saving_fraction'])} |"
        )
    lines += [
        "",
        "Расчет стоимости использует измеренное время вызовов Qwen на тех же входах плюс время студента для всех сообщений. Это эквивалент последовательной вычислительной нагрузки; денежная стоимость, задержка очереди и стоимость обучения сюда не включены. Указанная цель recall не является гарантией на test или рабочем потоке.",
        "",
        "## Оценка доли NEGATIVE",
        "",
        "Сравнены полная разметка Qwen, случайная выборка с заданной вероятностью, каскад с аудитом и случайная выборка при таком же ожидаемом числе вызовов. Использован Horvitz–Thompson total / фиксированный размер популяции; 5 000 симуляций по сессиям. Вероятности сохранены в JSON. Оценки не обрезались до [0,1].",
        "",
        "| Цель recall | Аудит | Дизайн | Вызовов Qwen, ожидание | RMSE доли, п.п. | Покрытие нормального 95% интервала |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in evaluation["prevalence"]:
        for name, stat in row["designs"].items():
            lines.append(
                f"| {percent(row['target_calibration_recall'])} | {percent(row['audit_rate'])} | {name} | {stat['expected_teacher_calls']:.1f} | {stat['rmse']*100:.2f} | {percent(stat['normal_interval_empirical_coverage'])} |"
            )
    first = evaluation["prevalence"][0]
    lines += [
        "",
        f"Доля NEGATIVE среди известных ответов Qwen: {percent(first['teacher_negative_fraction_known'])}. Границы для всех допустимых входов с учетом отсутствующих ответов: {first['full_eligible_negative_fraction_bounds']}. Если учитель не ответил, симуляция явно оценивает подмножество известных ответов; неизвестные классы не считаются отрицательными или нейтральными.",
        "На небольшом числе сессий нормальные интервалы могут недопокрывать истинное значение — их фактическое покрытие показано в таблице. Это диагностика метода выборки; модельную ошибку Qwen она не устраняет.",
    ]
    weak_intervals = [
        row
        for row in evaluation["prevalence"]
        if row["designs"]["cascade_with_audit"]["normal_interval_empirical_coverage"] < 0.9
    ]
    if weak_intervals:
        details = "; ".join(
            f"цель {percent(row['target_calibration_recall'])}, аудит {percent(row['audit_rate'])}: покрытие {percent(row['designs']['cascade_with_audit']['normal_interval_empirical_coverage'])}"
            for row in weak_intervals
        )
        lines += [
            f"Обнаружено слабое покрытие нормальных интервалов каскада: {details}.",
            "При редких неотобранных NEGATIVE аудит может не найти ни одного из них, и оцененная дисперсия оказывается нулевой. Точечная HT-оценка несмещенная по дизайну, но такие интервалы нельзя использовать как надежный контроль частоты. Для применения нужны более крупная независимая проверка и интервалы, пригодные для редких событий.",
        ]
    if any(
        row["designs"]["cascade_with_audit"]["exact_design_standard_error"] == 0
        for row in evaluation["prevalence"]
    ):
        lines.append(
            f"Нулевой RMSE для некоторых порогов относится только к этой конечной тестовой популяции: все ее {winner['metrics']['per_class']['NEGATIVE']['support']} NEGATIVE уже попали в обязательную проверку Qwen. Это не доказательство нулевой ошибки оценки частоты на новых данных. Порог после просмотра test не изменялся."
        )
    lines += [
        "",
        "## Артефакты и воспроизведение",
        "",
        "- `data/design.json`, `study-design.json`: состав данных и протокол до обучения.",
        "- `data/reviewed-addendum.json`: дополнительная проверка девяти fewshot-примеров; состав замороженных splits не изменился.",
        "- `teacher/`: возобновляемые вызовы, метки, причины и ошибки.",
        "- `baseline-selection.json`, `finetune-selection.json` (если применимо), `selection.json`: выбор только по calibration.",
        "- `evaluation.json`: метрики, интервалы, стоимость, вероятности аудита и симуляции.",
        "- `models/`, `test-predictions/`: модели всех кандидатов и финальные предсказания.",
        "- `test-disagreements.jsonl`: расхождения выбранного студента с учителем, тексты и короткие объяснения Qwen; пропуски учителя отмечены отдельно.",
        "- `artifacts/sentiment-student-v1/` от корня проекта: выбранная переносимая модель и shadow-политики.",
        "",
        "```text",
        "python scripts/run_sentiment_study.py baselines",
        "python scripts/run_sentiment_study.py finetune --device cuda",
        "python scripts/run_sentiment_study.py select",
        "python scripts/run_sentiment_study.py evaluate --device cpu",
        "python scripts/run_sentiment_study.py report",
        "python -m memory_trace predict-sentiment runs/sentiment-students-v1/data/test.jsonl --model-dir artifacts/sentiment-student-v1 --out runs/student-predictions.jsonl --device cpu",
        "```",
        "",
        "Веса энкодера зафиксированы по revision; дообученные веса имеют checksum и входят в модель. JSON-голова не использует pickle. Проверки Windows/Linux, независимое сравнение токенизации и эмбеддингов со стандартным E5, а также backward/save/reload на синтетических входах описаны в verification.json. Для macOS настроена CI-проверка, но здесь она не запускалась.",
        "",
        "[Модель E5 и рекомендации автора](https://huggingface.co/intfloat/multilingual-e5-small/blob/614241f622f53c4eeff9890bdc4f31cfecc418b3/README.md) · [Sentence Transformers](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).",
        "",
    ]
    content = "\n".join(lines)
    write_text(base / "report.md", content + "\n![Сравнение моделей и политик](comparison.png)\n")
    report_target = Path(report_path or "docs/experiments/sentiment-students-2026-09-14.md")
    figure_link = Path(os.path.relpath(base / "comparison.png", report_target.parent)).as_posix()
    write_text(report_target, content + f"\n![Сравнение моделей и политик]({figure_link})\n")
    write_text(
        output / "README.md",
        "# Выбранный студент Qwen\n\nКлассы: NEGATIVE / NEUTRAL / POSITIVE. По умолчанию shadow.\n\nИз корня проекта:\n\n```text\npython -m memory_trace predict-sentiment INPUT.jsonl --model-dir artifacts/sentiment-student-v1 --out predictions.jsonl --device cpu\n```\n\nНужен Python 3.11+ и зависимости ML. Для замороженного энкодера сначала загрузите его зафиксированную ревизию или явно передайте --allow-download. Дообученный энкодер включен в эту папку. Модель воспроизводит автоматического учителя, а не независимую человеческую оценку.\n",
    )
    make_plot(base, evaluation, baseline)
    return {"report": str(base / "report.md"), "model": str(output), "winner": winner["name"]}


def make_plot(base, evaluation, baseline):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = {"tfidf": "#2679ad", "encoder": "#d47826"}
    for kind in ("tfidf", "encoder"):
        rows = sorted(
            [r for r in baseline["candidates"] if r["kind"] == kind],
            key=lambda r: r["training_messages"],
        )
        axes[0, 0].plot(
            [r["training_messages"] for r in rows],
            [r["calibration"]["macro_f1"] for r in rows],
            "o-",
            label=kind,
            color=colors[kind],
        )
    for row in evaluation["selection"]["candidates"]:
        if row["kind"] == "finetuned":
            axes[0, 0].scatter(
                [row["training_messages"]],
                [row["calibration"]["macro_f1"]],
                marker="D",
                s=65,
                color="#238b45",
                label="fine-tuned encoder",
                zorder=3,
            )
    axes[0, 0].set(
        title="Learning curve (calibration)",
        xlabel="Labelled training messages",
        ylabel="Macro-F1",
        ylim=(0, 1),
    )
    axes[0, 0].legend()
    winner = next(r for r in evaluation["models"] if r["selected_on_calibration"])
    matrix = np.asarray(winner["metrics"]["confusion_matrix"])
    axes[0, 1].imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        axes[0, 1].text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
            color="white" if value > matrix.max() * 0.5 else "black",
        )
    axes[0, 1].set(
        title=f"Held-out test: {winner['name']}",
        xticks=range(3),
        yticks=range(3),
        xticklabels=["NEG", "NEU", "POS"],
        yticklabels=["NEG", "NEU", "POS"],
        xlabel="Student",
        ylabel="Qwen",
    )
    for audit in sorted({r["audit_rate"] for r in evaluation["routing"]}):
        rows = [r for r in evaluation["routing"] if r["audit_rate"] == audit]
        axes[1, 0].plot(
            [r["expected_teacher_call_fraction"] * 100 for r in rows],
            [r["expected_coverage_with_audit"] * 100 for r in rows],
            "o-",
            label=f"audit {audit:.0%}",
        )
    axes[1, 0].set(
        title="Session routing (offline)",
        xlabel="Expected Qwen calls, %",
        ylabel="Known NEGATIVE coverage, %",
        xlim=(0, 105),
        ylim=(0, 105),
    )
    axes[1, 0].legend()
    labels = {
        "teacher_census": "full Qwen census",
        "random_audit_rate": "random",
        "cascade_with_audit": "cascade + audit",
        "random_equal_expected_calls": "random, equal cost",
    }
    for name, label in labels.items():
        points = [row["designs"][name] for row in evaluation["prevalence"]]
        axes[1, 1].scatter(
            [p["expected_teacher_calls"] for p in points],
            [p["rmse"] * 100 for p in points],
            label=label,
            alpha=0.8,
        )
    axes[1, 1].set(
        title="NEGATIVE prevalence sampling",
        xlabel="Expected Qwen calls",
        ylabel="RMSE, percentage points",
    )
    axes[1, 1].legend(fontsize=8)
    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Student distillation from a fixed Qwen teacher\n"
        + f"Test: {winner['metrics']['messages']} messages; "
        + f"{winner['metrics']['per_class']['NEGATIVE']['support']} NEGATIVE, "
        + f"{winner['metrics']['per_class']['POSITIVE']['support']} POSITIVE",
        fontsize=14,
    )
    fig.savefig(Path(base) / "comparison.png")
    fig.savefig(Path(base) / "comparison.pdf")
    plt.close(fig)
