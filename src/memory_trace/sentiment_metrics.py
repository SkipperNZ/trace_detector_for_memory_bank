"""Teacher agreement, clustered uncertainty, routing, and probability-sampling experiments."""

from collections import defaultdict
from .dataset import SENTIMENTS
from .io import digest


def classification(y, probabilities, *, classes=SENTIMENTS):
    import numpy as np
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    y, p = np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float)
    if p.shape != (len(y), 3) or not len(y):
        raise ValueError("Expected nonempty aligned three-class probabilities")
    predicted = p.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=[0, 1, 2], zero_division=0
    )
    return {
        "messages": len(y),
        "agreement": float(np.mean(y == predicted)),
        "macro_f1": float(f1.mean()),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist(),
        "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "brier": float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))),
    }


def clustered_intervals(y, probabilities, groups, *, replicates=2000, seed=42):
    import numpy as np

    y, p = np.asarray(y), np.asarray(probabilities)
    unique = sorted(set(groups))
    matrices = []
    groups = np.asarray(groups)
    for group in unique:
        selected = groups == group
        matrices.append(
            np.bincount(3 * y[selected] + p[selected].argmax(axis=1), minlength=9).reshape(3, 3)
        )
    matrices = np.asarray(matrices)
    rng = np.random.default_rng(seed)
    sampled = matrices[rng.integers(0, len(unique), size=(replicates, len(unique)))].sum(axis=1)
    diag = np.diagonal(sampled, axis1=1, axis2=2)
    denominator = sampled.sum(axis=1) + sampled.sum(axis=2)
    f1 = np.divide(
        2 * diag, denominator, out=np.zeros_like(diag, dtype=float), where=denominator > 0
    ).mean(axis=1)
    accuracy = diag.sum(axis=1) / sampled.sum(axis=(1, 2))
    support = sampled[:, 0, :].sum(axis=1)
    recall = np.divide(diag[:, 0], support, out=np.full(replicates, np.nan), where=support > 0)
    return {
        "method": "percentile bootstrap resampling complete sessions",
        "sessions": len(unique),
        "replicates": replicates,
        "seed": seed,
        "agreement_95": np.quantile(accuracy, [0.025, 0.975]).tolist(),
        "macro_f1_95": np.quantile(f1, [0.025, 0.975]).tolist(),
        "negative_recall_95": (
            np.nanquantile(recall, [0.025, 0.975]).tolist() if np.isfinite(recall).any() else None
        ),
    }


def choose_thresholds(y, negative_scores, *, recalls=(0.95, 0.98, 0.99)):
    import numpy as np

    y, scores = np.asarray(y), np.asarray(negative_scores)
    positive = y == 0
    if not positive.any():
        raise ValueError("Calibration requires teacher NEGATIVE messages")
    candidates = sorted(set(scores[positive].tolist()), reverse=True)
    policies = []
    for target in recalls:
        if not 0 < target <= 1:
            raise ValueError("Recall target must be in (0,1]")
        threshold = next(t for t in candidates if np.mean(scores[positive] >= t) >= target)
        selected = scores >= threshold
        policies.append(
            {
                "target_calibration_recall": target,
                "threshold": float(threshold),
                "calibration_recall": float(np.mean(selected[positive])),
                "calibration_routed_fraction": float(selected.mean()),
                "mode": "shadow",
            }
        )
    return policies


def routing_metrics(
    examples,
    y_by_id,
    negative_scores,
    policy,
    teacher_seconds,
    *,
    student_seconds=0.0,
    audit_rate=0.0,
):
    import numpy as np

    if not 0 <= audit_rate <= 1:
        raise ValueError("Audit rate must be in [0,1]")
    selected = np.asarray(negative_scores) >= policy["threshold"]
    sessions = defaultdict(list)
    for i, ex in enumerate(examples):
        sessions[ex["group_id"]].append(i)
    routed = selected.copy()
    for indices in sessions.values():
        routed[indices] = selected[indices].any()
    truth = np.array([y_by_id.get(ex["input_id"], -1) for ex in examples])
    known, negative = truth >= 0, truth == 0
    weights = routed.astype(float) + (~routed) * audit_rate
    cost = np.array([teacher_seconds[ex["input_id"]] for ex in examples])
    positive_sessions = [ids for ids in sessions.values() if negative[ids].any()]
    routed_sessions = int(sum(routed[ids].any() for ids in sessions.values()))
    expected_negative_coverage = float(weights[negative].mean()) if negative.any() else None
    missing = int((~known).sum())
    return {
        "policy": policy,
        "audit_rate": audit_rate,
        "eligible_messages": len(examples),
        "teacher_missing": missing,
        "model_negative_message_recall": (
            float(selected[negative].mean()) if negative.any() else None
        ),
        "model_negative_message_precision": (
            float(negative[selected & known].mean()) if (selected & known).any() else None
        ),
        "session_routed_messages": int(routed.sum()),
        "session_routed_fraction": float(routed.mean()),
        "routed_sessions": routed_sessions,
        "sessions": len(sessions),
        "negative_session_recall": (
            sum(routed[ids].any() for ids in positive_sessions) / len(positive_sessions)
            if positive_sessions
            else None
        ),
        "negative_message_coverage": float(routed[negative].mean()) if negative.any() else None,
        "expected_coverage_with_audit": expected_negative_coverage,
        "expected_teacher_calls": float(weights.sum()),
        "expected_teacher_call_fraction": float(weights.mean()),
        "teacher_all_seconds": float(cost.sum()),
        "student_seconds": student_seconds,
        "expected_cascade_seconds": float(cost @ weights + student_seconds),
        "expected_cost_saving_fraction": (
            float(1 - (cost @ weights + student_seconds) / cost.sum()) if cost.sum() > 0 else None
        ),
        "unknown_teacher_negative_coverage_bounds": (
            [
                float((routed & negative).sum() / (negative.sum() + missing)),
                float(((routed & negative).sum() + missing) / (negative.sum() + missing)),
            ]
            if negative.sum() + missing
            else None
        ),
        "scope": "Offline session OR simulation; measured API wall time is a serial service-cost proxy, not currency or end-to-end concurrent latency.",
    }


def prevalence_experiment(
    examples, y_by_id, negative_scores, policy, *, audit_rate=0.1, replicates=5000, seed=42
):
    """Finite-population simulation: Bernoulli session sampling, HT total / fixed N.

    Unknown teacher answers are not silently deleted. For this diagnostic the reported
    estimand is explicitly the known-answer subpopulation, with full-population bounds.
    """
    import numpy as np

    if not 0 < audit_rate <= 1:
        raise ValueError("Every unselected session needs nonzero audit probability")
    sessions = defaultdict(list)
    for i, ex in enumerate(examples):
        sessions[ex["group_id"]].append(i)
    groups = sorted(sessions)
    truths = np.asarray([y_by_id.get(ex["input_id"], -1) for ex in examples])
    selected = np.asarray(negative_scores) >= policy["threshold"]
    group_negative = np.array([sum(truths[i] == 0 for i in sessions[g]) for g in groups])
    group_n = np.array([sum(truths[i] >= 0 for i in sessions[g]) for g in groups])
    group_calls = np.array([len(sessions[g]) for g in groups])
    candidate = np.array([selected[sessions[g]].any() for g in groups])
    denominator = int(group_n.sum())
    if not denominator:
        raise ValueError("No valid teacher answers for prevalence comparison")
    truth = float(group_negative.sum() / denominator)
    cascade_pi = np.where(candidate, 1.0, audit_rate)
    equal_budget_rate = float(cascade_pi @ group_calls / group_calls.sum())
    designs = {
        "teacher_census": np.ones(len(groups)),
        "random_audit_rate": np.full(len(groups), audit_rate),
        "cascade_with_audit": cascade_pi,
        "random_equal_expected_calls": np.full(len(groups), equal_budget_rate),
    }
    reports = {}
    for k, (name, pi) in enumerate(designs.items()):
        rng = np.random.default_rng(seed + k)
        observed = rng.random((replicates, len(groups))) < pi
        estimates = observed @ (group_negative / pi) / denominator
        exact_variance = float(np.sum((1 - pi) / pi * group_negative**2) / denominator**2)
        # Standard design-based variance estimator; report empirical coverage rather
        # than promising a normal approximation for this small, clustered population.
        variance_hat = observed @ ((1 - pi) / pi**2 * group_negative**2) / denominator**2
        margin = 1.96 * np.sqrt(variance_hat)
        calls = observed @ group_calls
        reports[name] = {
            "inclusion_probabilities": {g: float(p) for g, p in zip(groups, pi)},
            "expected_teacher_calls": float(pi @ group_calls),
            "mean_teacher_calls": float(calls.mean()),
            "mean_estimate": float(estimates.mean()),
            "bias": float(estimates.mean() - truth),
            "rmse": float(np.sqrt(np.mean((estimates - truth) ** 2))),
            "exact_design_standard_error": float(np.sqrt(exact_variance)),
            "estimate_95_sampling_interval": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "normal_interval_empirical_coverage": float(
                np.mean((estimates - margin <= truth) & (truth <= estimates + margin))
            ),
            "zero_sample_fraction": float(np.mean(calls == 0)),
        }
    routed_known = selected & (truths >= 0)
    missing = int((truths < 0).sum())
    return {
        "teacher_known_messages": denominator,
        "eligible_messages": len(examples),
        "teacher_missing": missing,
        "teacher_negative_fraction_known": truth,
        "full_eligible_negative_fraction_bounds": [
            float(group_negative.sum() / len(examples)),
            float((group_negative.sum() + missing) / len(examples)),
        ],
        "naive_selected_message_fraction": (
            float(np.mean(truths[routed_known] == 0)) if routed_known.any() else None
        ),
        "replicates": replicates,
        "seed": seed,
        "designs": reports,
        "estimator": "Horvitz-Thompson NEGATIVE total / fixed known-answer message count; Bernoulli sampling of complete eligible sessions; estimates are not clipped.",
        "limitation": "When teacher errors exist, the estimand is explicitly the known-answer subset and bounds cover the full eligible population. This does not correct teacher misclassification.",
    }


def nested_training_sets(examples, sizes=(500, 2000), seed=42):
    grouped = defaultdict(list)
    for ex in examples:
        if ex["split"] != "train":
            raise ValueError("Learning curves accept train only")
        grouped[ex["group_id"]].append(ex)
    ordered = [
        grouped[g]
        for g in sorted(grouped, key=lambda g: digest([seed, "student-learning-curve-v1", g]))
    ]
    result = {}
    for size in list(sizes) + [len(examples)]:
        selected = []
        for group in ordered:
            selected.extend(sorted(group, key=lambda ex: ex["input_id"]))
            if len(selected) >= size:
                break
        result[str(size) if size < len(examples) else "all"] = selected
    return result
