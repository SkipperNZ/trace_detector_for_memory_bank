"""Agreement with an automatic reference and repeated-call consistency, not human accuracy."""

from collections import Counter, defaultdict

from .dataset import SENTIMENTS
from .io import index_rows
from .metrics import ratio


def render_markdown(report: dict) -> str:
    def percent(value):
        return "—" if value is None else f"{100 * value:.1f}%"

    lines = [
        "# Qwen: проверка LLM-судьи",
        "",
        "Сравнение с автоматическими метками agent-trace-sentiment. "
        "Числа описывают согласие двух разметчиков, а не точность относительно человеческого эталона.",
        "",
        f"Сообщений в пилоте: **{report['messages']}**. "
        f"Успешных первых оценок: **{report['successful_primary_judgments']}**. "
        f"Пропущено или завершилось ошибкой: **{report['missing_or_failed_primary_judgments']}**.",
        "",
        "| Показатель | Результат |",
        "|---|---:|",
        f"| Согласие по тональности на успешных входах | {percent(report['sentiment_agreement_on_successful'])} |",
        f"| Согласие на всех входах, пропуски как несовпадение | {percent(report['sentiment_agreement_all_inputs_failures_as_nonagreement'])} |",
        f"| Macro-F1 относительно исходной разметки | {percent(report['sentiment_macro_f1_on_successful'])} |",
        f"| Совпадение повторов: тональность | {percent(report['sentiment_repeat_agreement'])} |",
        *(
            [
                f"| Совпадение повторов: претензия к агенту | {percent(report['complaint_repeat_agreement'])} |"
            ]
            if report.get("primary_complaint_counts")
            else []
        ),
        "",
        "## По классам исходной тональности",
        "",
        "| Класс | Успешных входов | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, metrics in report["per_class"].items():
        lines.append(
            f"| {label} | {metrics['support_successful']} | {percent(metrics['precision'])} | "
            f"{percent(metrics['recall'])} | {percent(metrics['f1'])} |"
        )
    lines += [
        "",
        "## Матрица: строки — исходные метки, столбцы — Qwen",
        "",
        "| Источник / Qwen | NEGATIVE | NEUTRAL | POSITIVE |",
        "|---|---:|---:|---:|",
    ]
    for label, counts in report["confusion_matrix_rows_source_columns_judge"].items():
        lines.append(
            f"| {label} | {counts['NEGATIVE']} | {counts['NEUTRAL']} | {counts['POSITIVE']} |"
        )
    if report.get("primary_complaint_counts"):
        lines += [
            "",
            "## Явное недовольство агентом",
            "",
            "Эта задача оценивается отдельно от общей тональности.",
            "",
        ]
        for label, count in sorted(report["primary_complaint_counts"].items()):
            lines.append(f"- {label}: {count}")
    latency = report["successful_attempt_latency_seconds"]
    lines += [
        "",
        "## Выполнение",
        "",
        f"Попыток API с учетом retries: {report['api_attempts_including_retries']}. "
        f"Сумма времени попыток: {report['sum_attempt_elapsed_seconds']:.1f} с.",
        "",
        f"Латентность успешных запросов p50/p95, с: {latency['p50']} / {latency['p95']}.",
        "",
        (
            f"Сообщений с полным набором повторов: {report['repeat_comparable_messages']}."
            if report["expected_repeats"] >= 2
            else "На сообщение выполнен один вызов; повторяемость не измерялась."
        ),
        "",
        "Usage, сообщенный сервером: " + str(report["reported_token_usage"]),
        "",
        "## Как читать результат",
        "",
        "Результат зависит от состава и способа отбора выборки; "
        "его доли нельзя автоматически переносить на естественный поток. Повторяемость при temperature=0 "
        "не гарантирует правильность. Разные рубрики могут объяснять часть расхождений.",
        "",
        "[Примеры расхождений](disagreements.jsonl) · [Полный машинный отчет](report.json)",
        "",
    ]
    return "\n".join(lines)


def compare(
    examples: list[dict], calls: list[dict], references: list[dict], *, expected_repeats: int = 2
):
    if not 1 <= expected_repeats <= 10:
        raise ValueError("expected_repeats must be between 1 and 10")
    inputs, refs = index_rows(examples), index_rows(references)
    if inputs.keys() - refs.keys():
        raise ValueError("Every evaluated input needs a source sentiment reference")
    if len({call["judge_version"] for call in calls}) > 1:
        raise ValueError("Compare one judge configuration at a time")
    by_input = defaultdict(list)
    seen = set()
    for call in calls:
        if call["input_id"] not in inputs or (call["input_id"], call["repeat"]) in seen:
            raise ValueError("Unknown input or duplicate repeat in judge calls")
        seen.add((call["input_id"], call["repeat"]))
        by_input[call["input_id"]].append(call)
    matrix = {a: {b: 0 for b in SENTIMENTS} for a in SENTIMENTS}
    disagreements, valid = [], 0
    repeat_valid, repeat_sentiment, repeat_complaint = 0, 0, 0
    complaint_repeat_valid = 0
    complaint_counts, crosswalk = Counter(), defaultdict(Counter)
    usage, unreported_usage = Counter(), 0
    attempts_count, elapsed_seconds = 0, 0.0
    cached_prompt_tokens, successful_latencies = [], []
    for call in calls:
        records = [call] + call.get("previous_runs", [])
        for record in records:
            for attempt in record["attempts"]:
                attempts_count += 1
                elapsed_seconds += attempt["elapsed_seconds"]
                measured = (attempt.get("response") or {}).get("usage")
                if attempt["runtime_status"] == "ok":
                    successful_latencies.append(attempt["elapsed_seconds"])
                if not isinstance(measured, dict):
                    unreported_usage += 1
                    continue
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = measured.get(key)
                    if isinstance(value, (float, int)) and not isinstance(value, bool):
                        usage[key] += value
                cached = (measured.get("prompt_tokens_details") or {}).get("cached_tokens")
                if isinstance(cached, int) and not isinstance(cached, bool):
                    cached_prompt_tokens.append(cached)
    for key, example in inputs.items():
        group = sorted(by_input.get(key, []), key=lambda c: c["repeat"])
        primary = next((call for call in group if call["repeat"] == 0), None)
        if primary is not None and primary["runtime_status"] == "ok":
            answer = primary["answer"]
            source = refs[key]["sentiment_label"]
            predicted = answer["sentiment_label"]
            if source not in SENTIMENTS or predicted not in SENTIMENTS:
                raise ValueError("Invalid sentiment label in comparison")
            matrix[source][predicted] += 1
            valid += 1
            if "label" in answer:
                complaint_counts[answer["label"]] += 1
                crosswalk[source][answer["label"]] += 1
            if source != predicted:
                disagreements.append(
                    {
                        "input_id": key,
                        "source_sentiment": source,
                        "judge_sentiment": predicted,
                        **({"judge_complaint": answer["label"]} if "label" in answer else {}),
                        "reason": answer["reason"],
                        "context": example["context"],
                    }
                )
        if (
            expected_repeats >= 2
            and {c["repeat"] for c in group} == set(range(expected_repeats))
            and all(c["runtime_status"] == "ok" for c in group)
        ):
            repeat_valid += 1
            repeat_sentiment += len({c["answer"]["sentiment_label"] for c in group}) == 1
            if all("label" in c["answer"] for c in group):
                complaint_repeat_valid += 1
                repeat_complaint += len({c["answer"]["label"] for c in group}) == 1
    per_class = {}
    for label in SENTIMENTS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted_count = sum(matrix[other][label] for other in SENTIMENTS)
        per_class[label] = {
            "support_successful": support,
            "precision": ratio(tp, predicted_count),
            "recall": ratio(tp, support),
            "f1": ratio(2 * tp, support + predicted_count),
        }
    correct = sum(matrix[label][label] for label in SENTIMENTS)
    f1 = [row["f1"] for row in per_class.values() if row["f1"] is not None]

    def percentile(values, quantile):
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    report = {
        "reference_type": "dataset_llm",
        "input_splits": dict(Counter(ex.get("split", "unspecified") for ex in examples)),
        "messages": len(examples),
        "successful_primary_judgments": valid,
        "missing_or_failed_primary_judgments": len(examples) - valid,
        "sentiment_agreement_on_successful": ratio(correct, valid),
        "sentiment_agreement_all_inputs_failures_as_nonagreement": ratio(correct, len(examples)),
        "sentiment_macro_f1_on_successful": sum(f1) / len(f1) if f1 else None,
        "confusion_matrix_rows_source_columns_judge": matrix,
        "per_class": per_class,
        "repeat_comparable_messages": repeat_valid,
        "expected_repeats": expected_repeats,
        "sentiment_repeat_agreement": ratio(repeat_sentiment, repeat_valid),
        **(
            {
                "complaint_repeat_agreement": ratio(repeat_complaint, complaint_repeat_valid),
                "primary_complaint_counts": dict(complaint_counts),
                "source_sentiment_to_judge_complaint": {k: dict(v) for k, v in crosswalk.items()},
            }
            if complaint_counts
            else {}
        ),
        "api_attempts_including_retries": attempts_count,
        "sum_attempt_elapsed_seconds": elapsed_seconds,
        "successful_attempt_latency_seconds": {
            "p50": percentile(successful_latencies, 0.5),
            "p95": percentile(successful_latencies, 0.95),
        },
        "reported_cached_prompt_tokens": (
            sum(cached_prompt_tokens) if cached_prompt_tokens else None
        ),
        "reported_token_usage": dict(usage),
        "attempts_without_usage": unreported_usage,
        "limits": [
            "Agreement with automatic sentiment labels is not human-validated accuracy.",
            *(
                ["Sentiment and explicit agent dissatisfaction are different targets."]
                if complaint_counts
                else []
            ),
            "Balanced pilot class frequencies are not natural prevalence.",
            "Repeat agreement does not establish correctness; deterministic requests can agree on errors.",
            "Token totals cover reported fields only; no monetary savings are inferred.",
        ],
    }
    return report, disagreements
