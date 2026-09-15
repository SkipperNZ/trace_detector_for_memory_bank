import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from memory_trace.demo import annotations_and_scores, events
from memory_trace.embeddings import select_device
from memory_trace.io import digest, read_jsonl, uniform, write_jsonl
from memory_trace.metrics import evaluate, prevalence, recall_lower_bound, validate_labels
from memory_trace.prepare import annotation_template, normalize, prepare
from memory_trace.routing import Policy, route


class CausalTests(unittest.TestCase):
    def test_no_future_sibling_or_internal_visibility(self):
        rows = events()[:3]
        rows.append(
            {
                **rows[1],
                "event_id": "hidden",
                "sequence": 2,
                "parent_id": "1",
                "text": "PRIVATE REASONING",
                "visibility": "internal",
            }
        )
        rows[2] = {**rows[2], "parent_id": "hidden", "sequence": 3}
        rows.append(
            {**rows[1], "event_id": "future", "parent_id": "2", "sequence": 4, "text": "FUTURE"}
        )
        rows.append(
            {
                **rows[1],
                "event_id": "sibling",
                "branch_id": "alternative",
                "sequence": 2,
                "text": "OTHER BRANCH",
            }
        )
        example = prepare(list(reversed(rows)))[-1]
        self.assertEqual([e["event_id"] for e in example["context"]], ["0", "1", "2"])
        self.assertNotIn("PRIVATE", json.dumps(example))
        self.assertNotIn("FUTURE", json.dumps(example))
        self.assertNotIn("OTHER BRANCH", json.dumps(example))

    def test_duplicates_are_delivery_ids_not_text(self):
        rows = events()[:1]
        self.assertEqual(len(normalize(rows + rows)), 1)
        separate = {**rows[0], "event_id": "other", "parent_id": "0", "sequence": 1}
        self.assertEqual(len(normalize(rows + [separate])), 2)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            normalize(rows + [{**rows[0], "text": "changed"}])

    def test_missing_parent_and_late_repair(self):
        rows = events()[:3]
        incomplete = prepare([rows[2]])[0]
        complete = prepare(rows)[-1]
        self.assertEqual(incomplete["context_status"], "missing_parent")
        self.assertNotEqual(incomplete["input_id"], complete["input_id"])
        decision = route([incomplete], [], Policy(mode="cascade"))[0]
        self.assertTrue(decision["routed"])

    def test_parent_cannot_cross_tenant_and_cycles_fail(self):
        rows = events()[:3]
        rows[1]["tenant_id"] = "another"
        self.assertEqual(prepare(rows)[-1]["context_status"], "missing_parent")
        rows = events()[:3]
        rows[0]["parent_id"] = "2"
        with self.assertRaisesRegex(ValueError, "cycle"):
            prepare(rows)

    def test_target_preserved_when_over_budget(self):
        example = prepare(events()[:3], max_chars=1)[-1]
        self.assertEqual(example["context"][-1]["text"], events()[2]["text"])
        self.assertEqual(example["context_status"], "truncated")

    def test_partial_user_not_eligible_and_group_split_stable(self):
        rows = events()
        rows[0]["status"] = "partial"
        for row in rows:
            row["group_id"] = "one-project"
        examples = prepare(rows)
        self.assertEqual(len(examples), 5)
        self.assertEqual(len({ex["split"] for ex in examples}), 1)
        self.assertEqual(examples, prepare(list(reversed(rows))))

    def test_annotation_is_blind(self):
        template = annotation_template(prepare(events()))[0]
        self.assertIsNone(template["label"])
        self.assertNotIn("score", template)
        self.assertNotIn("split", template)


class RoutingMetricsTests(unittest.TestCase):
    def setUp(self):
        self.examples = prepare(events())
        self.labels, self.scores = annotations_and_scores(self.examples)

    def test_shadow_routes_all_but_retains_candidate_cost_counts(self):
        decisions = route(self.examples, self.scores, Policy(audit_rate=0.01))
        report = evaluate(self.examples, self.scores, decisions, self.labels, synthetic=True)
        self.assertEqual(report["actual_routed_traces"], 4)
        self.assertEqual(report["filter_selected_traces"], 2)
        self.assertEqual(report["filter_trace_recall"], 1)
        self.assertAlmostEqual(report["filter_trace_recall_lower_95_one_sided"], 0.05)
        self.assertTrue(all(d["inclusion_probability"] == 1 for d in decisions))
        self.assertIsNone(report["judge_end_to_end_recall"])

    def test_audit_is_stable_when_reordered_and_probability_valid(self):
        policy = Policy(mode="cascade", audit_rate=0.2)
        a = route(self.examples, self.scores, policy)
        b = route(list(reversed(self.examples)), list(reversed(self.scores)), policy)
        self.assertEqual([d["audit_selected"] for d in a], [d["audit_selected"] for d in b])
        for d in a:
            self.assertEqual(d["inclusion_probability"], 1 if d["filter_selected"] else 0.2)
        draws = [uniform(42, "audit-v1", ["tenant", str(i)]) for i in range(10000)]
        self.assertTrue(0.18 < sum(x < 0.2 for x in draws) / len(draws) < 0.22)

    def test_missing_scores_and_abstention_route_for_review(self):
        decisions = route(self.examples, [], Policy(mode="cascade"))
        self.assertTrue(all(d["routed"] for d in decisions))
        self.assertTrue(all(not d["model_selected_ids"] for d in decisions))
        with self.assertRaises(ValueError):
            route(self.examples, [{**self.scores[0], "score": math.nan}], Policy())
        with self.assertRaises(ValueError):
            Policy(audit_rate=0)
        with self.assertRaises(ValueError):
            route(self.examples, [{**self.scores[0], "input_id": "stale"}], Policy())
        incomplete = [{**p, "score": 0.0, "context_status": "token_limit"} for p in self.scores]
        self.assertTrue(
            all(d["routed"] for d in route(self.examples, incomplete, Policy(mode="cascade")))
        )

    def test_filter_recall_excludes_audit_and_no_positive_is_null(self):
        scores = [
            {**p, "score": 0.0, "runtime_status": "ok", "model_abstention": False}
            for p in self.scores
        ]
        decisions = route(self.examples, scores, Policy(mode="cascade", audit_rate=1))
        report = evaluate(self.examples, scores, decisions, self.labels, synthetic=True)
        self.assertEqual(report["filter_trace_recall"], 0)
        self.assertEqual(report["candidate_routing_trace_recall_with_audit"], 1)
        for label in self.labels:
            label["label"] = "no"
        report = evaluate(self.examples, scores, decisions, self.labels, synthetic=True)
        self.assertIsNone(report["filter_trace_recall"])

    def test_reference_validation_and_evidence(self):
        with self.assertRaisesRegex(ValueError, "human"):
            validate_labels(self.examples, self.labels)
        for label in self.labels:
            label["label_source"] = "human"
        validate_labels(self.examples, self.labels)
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_labels(self.examples, self.labels[:-1])
        self.labels[0]["evidence_event_ids"] = ["future-event"]
        with self.assertRaisesRegex(ValueError, "Evidence"):
            validate_labels(self.examples, self.labels)

    def test_ht_weighting_and_incomplete_review_rejected(self):
        decisions = route(self.examples, self.scores, Policy())
        for label in self.labels:
            label["label_source"] = "human"
        result = prevalence(self.examples, decisions, self.labels)
        self.assertEqual(result["estimated_message_totals_ht"], {"yes": 1, "no": 4, "unclear": 1})
        self.assertAlmostEqual(result["dissatisfaction_among_clear"], 0.2)
        with self.assertRaises(ValueError):
            prevalence(self.examples, decisions, self.labels[:-1])

    def test_cascade_weighting_uses_saved_probabilities(self):
        scores = [
            {**p, "score": 0.0, "runtime_status": "ok", "model_abstention": False}
            for p in self.scores
        ]
        # Find an a-priori reproducible seed that samples the positive trace for this fixture.
        seed = next(
            s for s in range(100) if uniform(s, "audit-v1", ["synthetic", "complaint"]) < 0.2
        )
        decisions = route(self.examples, scores, Policy(mode="cascade", audit_rate=0.2, seed=seed))
        selected_ids = {i for d in decisions if d["routed"] for i in d["input_ids"]}
        labels = [
            {**row, "label_source": "human"}
            for row in self.labels
            if row["input_id"] in selected_ids
        ]
        result = prevalence(self.examples, decisions, labels)
        self.assertEqual(result["estimated_message_totals_ht"]["yes"], 5.0)
        changed = copy.deepcopy(decisions)
        changed[0]["inclusion_probability"] = 0.5
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            prevalence(self.examples, changed, labels)

    def test_clustered_interval_is_suppressed_and_scores_must_match(self):
        for example in self.examples:
            example["group_id"] = "same-project"
        decisions = route(self.examples, self.scores, Policy())
        report = evaluate(self.examples, self.scores, decisions, self.labels, synthetic=True)
        self.assertTrue(report["multiple_traces_per_group"])
        self.assertIsNone(report["filter_trace_recall_lower_95_one_sided"])
        changed = [{**p, "score": 0.99} for p in self.scores]
        with self.assertRaisesRegex(ValueError, "do not match"):
            evaluate(self.examples, changed, decisions, self.labels, synthetic=True)

    def test_exact_lower_bound(self):
        self.assertIsNone(recall_lower_bound(0, 0))
        self.assertEqual(recall_lower_bound(0, 10), 0)
        self.assertAlmostEqual(recall_lower_bound(59, 59), 0.05 ** (1 / 59))
        self.assertAlmostEqual(recall_lower_bound(1, 2), 1 - 0.95**0.5)
        self.assertAlmostEqual(recall_lower_bound(2, 3), 0.13535036217158378)


class PortabilityTests(unittest.TestCase):
    def test_devices(self):
        def runtime(cuda=False, mps=False):
            return SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: cuda),
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
            )

        self.assertEqual(select_device("auto", runtime()), "cpu")
        self.assertEqual(select_device("auto", runtime(mps=True)), "mps")
        self.assertEqual(select_device("auto", runtime(cuda=True, mps=True)), "cuda")
        with self.assertRaises(ValueError):
            select_device("cuda", runtime())

    def test_unicode_artifact_path_and_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "путь с пробелами"
            result = subprocess.run(
                [sys.executable, "-m", "memory_trace", "demo", "--out", str(output)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["purpose"], "synthetic pipeline demonstration")
            self.assertEqual(len(read_jsonl(output / "events.jsonl")), 8)
            write_jsonl(output / "кириллица.jsonl", [{"text": "Привет"}])
            self.assertEqual(read_jsonl(output / "кириллица.jsonl"), [{"text": "Привет"}])

    def test_core_does_not_import_heavy_runtime(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import memory_trace.cli, sys; "
                "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_hash_is_canonical_and_tenant_isolated(self):
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))
        rows = events()[:1]
        other = copy.deepcopy(rows)
        other[0]["tenant_id"] = "other"
        self.assertNotEqual(prepare(rows)[0]["input_id"], prepare(other)[0]["input_id"])


if __name__ == "__main__":
    unittest.main()
