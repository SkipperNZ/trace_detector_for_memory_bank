"""Reference evaluation and design-weighted monitoring, with explicit denominators."""

import math

from .io import digest, index_rows, probability, uniform
from .routing import Policy, route


def validate_labels(
    examples: list[dict],
    annotations: list[dict],
    *,
    synthetic: bool = False,
    label_source: str = "human",
):
    if label_source not in {"human", "llm_judge"}:
        raise ValueError("label_source must be human or llm_judge")
    inputs = index_rows(examples)
    labels = index_rows(annotations)
    if labels.keys() != inputs.keys():
        raise ValueError("Labels must cover exactly the requested inputs, once each")
    for input_id, row in labels.items():
        example = inputs[input_id]
        if row.get("label") not in {"yes", "no", "unclear"}:
            raise ValueError("Every reference label must be yes/no/unclear")
        sources = {label_source, "synthetic_demo"} if synthetic else {label_source}
        if row.get("label_source") not in sources:
            raise ValueError(f"Reference labels must have label_source={label_source}")
        if label_source == "llm_judge" and (
            row.get("runtime_status") != "ok"
            or not isinstance(row.get("judge_version"), str)
            or not row["judge_version"]
        ):
            raise ValueError("Judge labels need a successful status and judge_version")
        if not isinstance(row.get("annotator"), str) or not row["annotator"].strip():
            raise ValueError("Every label needs an annotator")
        if row.get("metric_version") != example["metric_version"]:
            raise ValueError("Annotation metric_version mismatch")
        evidence = row.get("evidence_event_ids")
        allowed = {event["event_id"] for event in example["context"]}
        if not isinstance(evidence, list) or any(
            not isinstance(e, str) or e not in allowed for e in evidence
        ):
            raise ValueError("Evidence must reference events in this input's causal context")
        if row["label"] == "yes" and example["target_event_id"] not in evidence:
            raise ValueError("A yes label must cite the target user event")
    return labels


def checked_decisions(examples: list[dict], decisions: list[dict]):
    expected = {}
    for ex in examples:
        expected.setdefault((ex["tenant_id"], ex["trace_id"]), set()).add(ex["input_id"])
    seen = set()
    for row in decisions:
        key = row["tenant_id"], row["trace_id"]
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unknown decision trace")
        seen.add(key)
        if set(row["input_ids"]) != expected[key] or len(row["input_ids"]) != len(expected[key]):
            raise ValueError("Decision inputs are stale or incomplete")
        for field in ("filter_selected", "audit_selected", "would_route", "routed"):
            if type(row.get(field)) is not bool:
                raise ValueError(f"{field} must be boolean")
        for field in ("model_selected_ids", "safety_selected_ids"):
            if not set(row[field]).issubset(expected[key]):
                raise ValueError("Trigger IDs must belong to their trace")
        probability(row["inclusion_probability"], "inclusion_probability", positive=True)
        # Verify decisions against their recorded policy, including the random audit.
        policy = Policy(**row["policy"])
        if digest(row["policy"]) != row["policy_id"]:
            raise ValueError("Policy checksum mismatch")
        selected = bool(row["model_selected_ids"] or row["safety_selected_ids"])
        audited = not selected and uniform(policy.seed, "audit-v1", list(key)) < policy.audit_rate
        expected_probability = 1.0 if policy.mode == "shadow" or selected else policy.audit_rate
        if (
            row["filter_selected"] != selected
            or row["audit_selected"] != audited
            or row["would_route"] != (selected or audited)
            or row["routed"] != (policy.mode == "shadow" or selected or audited)
            or row["inclusion_probability"] != expected_probability
        ):
            raise ValueError("Decision is inconsistent with the recorded sampling policy")
    if seen != expected.keys():
        raise ValueError("Decisions must cover all traces")
    return decisions


def ratio(numerator: float, denominator: float):
    return numerator / denominator if denominator else None


def recall_lower_bound(hits: int, positives: int, alpha: float = 0.05):
    """One-sided Clopper-Pearson bound; assumes independent positive traces."""
    if not 0 <= hits <= positives or not 0 < alpha < 1:
        raise ValueError("Invalid binomial interval arguments")
    if positives == 0:
        return None
    if hits == 0:
        return 0.0
    if hits == positives:
        return alpha ** (1 / positives)
    # Solve P[Bin(n, p) >= hits] = alpha in log space, without scipy.
    log_coefficients = [
        math.lgamma(positives + 1) - math.lgamma(k + 1) - math.lgamma(positives - k + 1)
        for k in range(hits, positives + 1)
    ]
    low, high = 0.0, 1.0
    for _ in range(60):
        p = (low + high) / 2
        logs = [
            c + k * math.log(p) + (positives - k) * math.log1p(-p)
            for c, k in zip(log_coefficients, range(hits, positives + 1), strict=True)
        ]
        maximum = max(logs)
        tail = math.exp(maximum) * math.fsum(math.exp(x - maximum) for x in logs)
        if tail < alpha:
            low = p
        else:
            high = p
    return (low + high) / 2


def evaluate(
    examples: list[dict],
    predictions: list[dict],
    decisions: list[dict],
    annotations: list[dict],
    *,
    synthetic: bool = False,
    label_source: str = "human",
) -> dict:
    labels = validate_labels(examples, annotations, synthetic=synthetic, label_source=label_source)
    checked_decisions(examples, decisions)
    scores = index_rows(predictions)
    if scores.keys() - labels.keys():
        raise ValueError("Unknown prediction inputs")
    if len({d["policy_id"] for d in decisions}) > 1:
        raise ValueError("Evaluate one frozen policy at a time")
    if decisions:
        recomputed = route(examples, predictions, Policy(**decisions[0]["policy"]))
        actual = {(d["tenant_id"], d["trace_id"]): d for d in decisions}
        for expected in recomputed:
            if actual[(expected["tenant_id"], expected["trace_id"])] != expected:
                raise ValueError("Decisions do not match these predictions and input order")
    positives = {key for key, row in labels.items() if row["label"] == "yes"}
    hits, routed_hits, candidate_hits, positive_traces = 0, 0, 0, 0
    negative_traces, unclear_traces, false_positives = 0, 0, 0
    model_message_hits, filter_coverage = 0, 0
    actual_messages, candidate_messages = 0, 0
    safety_traces = 0
    for decision in decisions:
        ids = set(decision["input_ids"])
        trace_positive = bool(ids & positives)
        trace_unclear = any(labels[i]["label"] == "unclear" for i in ids)
        positive_traces += trace_positive
        negative_traces += not trace_positive and not trace_unclear
        unclear_traces += not trace_positive and trace_unclear
        hits += trace_positive and decision["filter_selected"]
        candidate_hits += trace_positive and decision["would_route"]
        routed_hits += trace_positive and decision["routed"]
        false_positives += not trace_positive and not trace_unclear and decision["filter_selected"]
        model_message_hits += len(positives & set(decision["model_selected_ids"]))
        filter_coverage += len(ids & positives) if decision["filter_selected"] else 0
        actual_messages += len(ids) if decision["routed"] else 0
        candidate_messages += len(ids) if decision["would_route"] else 0
        safety_traces += bool(decision["safety_selected_ids"])
    unclear_messages = sum(row["label"] == "unclear" for row in labels.values())
    group_traces = {}
    for ex in examples:
        group_traces.setdefault((ex["tenant_id"], ex["group_id"]), set()).add(ex["trace_id"])
    clustered = any(len(traces) > 1 for traces in group_traces.values())
    return {
        "purpose": (
            "synthetic pipeline demonstration" if synthetic else f"full {label_source} reference"
        ),
        "reference_label_source": label_source,
        "traces": len(decisions),
        "messages": len(examples),
        "positive_traces": positive_traces,
        "negative_traces": negative_traces,
        "unclear_traces": unclear_traces,
        "unclear_messages": unclear_messages,
        "filter_trace_recall": ratio(hits, positive_traces),
        "filter_trace_recall_lower_95_one_sided": (
            None if clustered else recall_lower_bound(hits, positive_traces)
        ),
        "multiple_traces_per_group": clustered,
        "filter_false_positive_traces": false_positives,
        "model_message_recall": ratio(model_message_hits, len(positives)),
        "filter_message_coverage": ratio(filter_coverage, len(positives)),
        "candidate_routing_trace_recall_with_audit": ratio(candidate_hits, positive_traces),
        "actual_routing_trace_recall": ratio(routed_hits, positive_traces),
        "filter_selected_traces": sum(d["filter_selected"] for d in decisions),
        "safety_selected_traces": safety_traces,
        "audit_selected_traces": sum(d["audit_selected"] for d in decisions),
        "candidate_routed_traces": sum(d["would_route"] for d in decisions),
        "actual_routed_traces": sum(d["routed"] for d in decisions),
        "candidate_routed_messages": candidate_messages,
        "actual_routed_messages": actual_messages,
        "unscored_or_failed_messages": sum(
            scores.get(ex["input_id"], {}).get("runtime_status") != "ok"
            or scores.get(ex["input_id"], {}).get("score") is None
            for ex in examples
        ),
        "judge_end_to_end_recall": None,
        "token_or_money_savings": None,
        "limitations": [
            "With llm_judge labels, all quality metrics measure agreement with the teacher, not human truth.",
            "The binomial bound requires independent representative positive traces and a frozen policy.",
            "Routing coverage is not judge recall; judge quality and token costs are not measured.",
            "Synthetic data and reused development data do not establish production quality.",
        ],
    }


def prevalence(
    examples: list[dict],
    decisions: list[dict],
    annotations: list[dict],
    *,
    label_source: str = "human",
) -> dict:
    checked_decisions(examples, decisions)
    selected_ids = {i for d in decisions if d["routed"] for i in d["input_ids"]}
    selected = [ex for ex in examples if ex["input_id"] in selected_ids]
    labels = validate_labels(selected, annotations, label_source=label_source)
    totals = {"yes": 0.0, "no": 0.0, "unclear": 0.0}
    for decision in decisions:
        if decision["routed"]:
            weight = 1 / decision["inclusion_probability"]
            for input_id in decision["input_ids"]:
                totals[labels[input_id]["label"]] += weight
    return {
        "reference_label_source": label_source,
        "estimated_message_totals_ht": totals,
        "dissatisfaction_among_clear": ratio(totals["yes"], totals["yes"] + totals["no"]),
        "unclear_fraction": ratio(totals["unclear"], sum(totals.values())),
        "reviewed_traces": sum(d["routed"] for d in decisions),
        "reviewed_messages": len(labels),
        "population_messages": len(examples),
        "confidence_interval": None,
        "limitations": [
            "Requires correct logged inclusion probabilities and complete reference labels in selected traces.",
            "With llm_judge labels, frequency refers to that judge's labels, not human truth.",
            "HT totals are design-unbiased; their ratio need not be unbiased in a finite sample.",
            "No design-aware cluster confidence interval is implemented in this MVP.",
        ],
    }
