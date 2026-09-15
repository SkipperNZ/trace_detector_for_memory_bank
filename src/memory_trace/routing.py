"""Trace-level shadow/cascade routing and reproducible Bernoulli auditing."""

from collections import defaultdict
from dataclasses import asdict, dataclass

from .io import digest, index_rows, probability, uniform


@dataclass(frozen=True)
class Policy:
    mode: str = "shadow"
    threshold: float = 0.5
    audit_rate: float = 0.1
    seed: int = 42
    version: str = "routing-v1"

    def __post_init__(self):
        if self.mode not in {"shadow", "cascade"}:
            raise ValueError("mode must be shadow or cascade")
        probability(self.threshold, "threshold")
        probability(self.audit_rate, "audit_rate", positive=True)
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")


def route(examples: list[dict], predictions: list[dict], policy: Policy) -> list[dict]:
    inputs = index_rows(examples)
    scores = index_rows(predictions)
    if scores.keys() - inputs.keys():
        raise ValueError("Predictions contain unknown or stale input_id values")
    traces = defaultdict(list)
    for example in examples:
        traces[(example["tenant_id"], example["trace_id"])].append(example)
    decisions = []
    policy_id = digest(asdict(policy))
    for key, members in sorted(traces.items()):
        model_selected_ids, safety_selected_ids = [], []
        versions = set()
        for example in members:
            input_id = example["input_id"]
            prediction = scores.get(input_id, {})
            score = prediction.get("score")
            if score is not None:
                score = probability(score, "score")
            runtime = prediction.get("runtime_status", "missing")
            if runtime not in {"ok", "error", "not_evaluated", "missing"}:
                raise ValueError(f"Invalid runtime_status: {runtime}")
            if prediction.get("context_status", "complete") != "complete":
                runtime = "not_evaluated"
            abstention = prediction.get("model_abstention", False)
            if type(abstention) is not bool:
                raise ValueError("model_abstention must be a boolean")
            model_version = prediction.get("model_version")
            if prediction and (not isinstance(model_version, str) or not model_version):
                raise ValueError("Every prediction needs a model_version")
            if model_version:
                versions.add(model_version)
            if runtime == "ok" and score is not None and score >= policy.threshold:
                model_selected_ids.append(input_id)
            if (
                example["context_status"] != "complete"
                or runtime != "ok"
                or abstention
                or score is None
                or example["language"] == "unknown"
            ):
                safety_selected_ids.append(input_id)
        selected = bool(model_selected_ids or safety_selected_ids)
        audit = not selected and uniform(policy.seed, "audit-v1", list(key)) < policy.audit_rate
        candidate_probability = 1.0 if selected else policy.audit_rate
        decisions.append(
            {
                "tenant_id": key[0],
                "trace_id": key[1],
                "input_ids": [e["input_id"] for e in members],
                "model_selected_ids": model_selected_ids,
                "safety_selected_ids": safety_selected_ids,
                "filter_selected": selected,
                "audit_selected": audit,
                "would_route": selected or audit,
                "routed": policy.mode == "shadow" or selected or audit,
                "inclusion_probability": 1.0 if policy.mode == "shadow" else candidate_probability,
                "candidate_inclusion_probability": candidate_probability,
                "policy": asdict(policy),
                "policy_id": policy_id,
                "model_versions": sorted(versions),
            }
        )
    return decisions
