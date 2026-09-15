"""Small synthetic examples for plumbing checks, never for model validation."""

from .prepare import annotation_template


def events():
    rows = []
    conversations = {
        "complaint": [
            ("user", "Сделай пример на Python."),
            ("assistant", "Вот реализация на Java."),
            ("user", "Ты опять проигнорировал мою просьбу использовать Python!"),
        ],
        "correction": [
            ("user", "Построй график."),
            ("assistant", "График с синей линией готов."),
            ("user", "Поменяй цвет линии на зеленый, пожалуйста."),
        ],
        "external": [("user", "The external service is broken again. Can you help?")],
        "ambiguous": [("user", "Ну и что это такое?")],
    }
    for trace_id, messages in conversations.items():
        for sequence, (role, text) in enumerate(messages):
            rows.append(
                {
                    "tenant_id": "synthetic",
                    "trace_id": trace_id,
                    "branch_id": "main",
                    "event_id": str(sequence),
                    "parent_id": str(sequence - 1) if sequence else None,
                    "sequence": sequence,
                    "role": role,
                    "visibility": "user_visible",
                    "text": text,
                    "language": "en" if trace_id == "external" else "ru",
                }
            )
    return rows


def annotations_and_scores(examples):
    annotations = annotation_template(examples)
    scores = []
    for example, annotation in zip(examples, annotations, strict=True):
        label = "no"
        if example["trace_id"] == "complaint" and example["target_event_id"] == "2":
            label = "yes"
        elif example["trace_id"] == "ambiguous":
            label = "unclear"
        annotation.update(
            label=label,
            label_source="synthetic_demo",
            annotator="demo-fixture",
            evidence_event_ids=[example["target_event_id"]] if label == "yes" else [],
        )
        scores.append(
            {
                "input_id": example["input_id"],
                "score": {"yes": 0.9, "no": 0.1, "unclear": None}[label],
                "runtime_status": "ok" if label != "unclear" else "not_evaluated",
                "model_abstention": label == "unclear",
                "model_version": "synthetic-fixture-v1",
            }
        )
    return annotations, scores
