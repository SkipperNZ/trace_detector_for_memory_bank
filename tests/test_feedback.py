import copy
import importlib.util
import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from memory_trace.feedback import (
    TASKS,
    feedback_input,
    canonical_key,
    student_inputs,
    validate_feedback,
    PROMPT,
)
from memory_trace.io import read_jsonl
from memory_trace.judge import JudgeConfig, messages, run_judge


def sample(text="Your patch crashes. Fix it.", tenant="fixture", number="1"):
    return feedback_input(
        {"text": text, "tenant_id": tenant, "session_id": number, "message_id": number}
    )


def answer():
    return {
        "dissatisfaction": {"label": "no", "reason": "Calm correction.", "evidence": []},
        "correction": {
            "label": "yes",
            "reason": "Identified faulty prior work.",
            "evidence": ["Your patch crashes."],
        },
        "reason_tags": ["wrong_result"],
    }


class FeedbackTests(unittest.TestCase):
    def test_independent_signals_and_exact_evidence(self):
        self.assertEqual(
            validate_feedback(json.dumps(answer()), sample())["correction"]["label"], "yes"
        )
        bad = answer()
        bad["correction"]["evidence"] = ["The patch is broken"]
        with self.assertRaisesRegex(ValueError, "exact substring"):
            validate_feedback(json.dumps(bad), sample())
        bad["correction"]["evidence"] = []
        with self.assertRaisesRegex(ValueError, "require evidence"):
            validate_feedback(json.dumps(bad), sample())

    def test_tags_require_explicit_positive(self):
        bad = answer()
        bad["correction"]["label"] = "unclear"
        with self.assertRaisesRegex(ValueError, "positive signal"):
            validate_feedback(json.dumps(bad), sample())

    def test_canonical_cache_does_not_cross_tenants_or_contexts(self):
        a, b = sample(), sample(number="2")
        self.assertNotEqual(a["input_id"], b["input_id"])
        self.assertEqual(canonical_key(a), canonical_key(b))
        self.assertNotEqual(canonical_key(a), canonical_key(sample(tenant="other")))
        b["history_available"] = True
        self.assertNotEqual(canonical_key(a), canonical_key(b))
        self.assertEqual(a, feedback_input(a))

    def test_context_must_end_at_target_and_serialization_keeps_previous_work(self):
        ex = sample()
        ex["context_scope"] = "causal_prefix"
        ex["history_available"] = True
        ex["context"].insert(
            0, {"event_id": "previous", "role": "assistant", "text": "VISIBLE PREVIOUS PATCH"}
        )
        ex = feedback_input(ex)
        self.assertIn("VISIBLE PREVIOUS PATCH", student_inputs([ex])[0]["context"][0]["text"])
        ex["context"].append({"event_id": "future", "role": "assistant", "text": "FUTURE"})
        with self.assertRaisesRegex(ValueError, "causal prefix"):
            feedback_input(ex)

    def test_prompt_and_task_cache_are_separate_from_sentiment(self):
        c = JudgeConfig("http://localhost/v1", "fixture", "fixture", task="feedback", retries=0)
        other = JudgeConfig(
            "http://localhost/v1", "fixture", "fixture", task="sentiment", retries=0
        )
        self.assertNotEqual(c.version, other.version)
        self.assertEqual(messages(sample(), "feedback")[0]["content"], PROMPT)
        ex = sample()
        ex["source_label"] = "FORBIDDEN_SOURCE_LABEL"
        self.assertNotIn("FORBIDDEN_SOURCE_LABEL", json.dumps(messages(ex, "feedback")))

    def test_journal_resumes_both_heads_without_new_calls(self):
        calls = []

        def request(payload, config):
            calls.append(payload)
            return {
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer())}}]
            }

        c = JudgeConfig("http://localhost/v1", "fixture", "fixture", task="feedback", retries=0)
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(2):
                r = run_judge([sample()], c, Path(tmp), repeats=1, request_fn=request)
                self.assertEqual(r["successful_calls"], 1)
            label = read_jsonl(Path(tmp) / "labels.jsonl")[0]
            self.assertEqual(set(label["signals"]), set(TASKS))
            self.assertEqual(label["signals"]["dissatisfaction"]["label"], "no")
            self.assertEqual(label["label_source"], "llm_judge")
        self.assertEqual(len(calls), 1)

    def test_restart_estimate_counts_only_pending_and_failed_requests(self):
        import sqlite3
        from memory_trace.feedback_data import remaining_estimate
        from memory_trace.io import write_json

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_json(base / "design.json", {"judge_version": "fixture", "canonical_messages": 5})
            write_json(base / "pilot-gate.json", {"mean_seconds_per_call": 99})
            (base / "teacher").mkdir()
            with closing(sqlite3.connect(base / "teacher/calls.sqlite3")) as db:
                db.execute("CREATE TABLE calls (record TEXT)")
                for i, status in enumerate(("ok", "not_evaluated", "error")):
                    db.execute(
                        "INSERT INTO calls VALUES (?)",
                        (
                            json.dumps(
                                {
                                    "input_id": str(i),
                                    "judge_version": "fixture",
                                    "runtime_status": status,
                                    "attempts": [{"elapsed_seconds": 10}],
                                }
                            ),
                        ),
                    )
                db.commit()
            estimate = remaining_estimate(base)
            self.assertEqual(estimate["remaining_or_retry_calls"], 3)
            self.assertEqual(estimate["mean_seconds_per_call"], 10)
            self.assertEqual(estimate["conservative_remaining_seconds_with_training"], 2440.5)


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "ML extra required")
class FeedbackModelTests(unittest.TestCase):
    def test_binary_and_missing_targets_keep_three_class_probabilities(self):
        import numpy as np
        from memory_trace.feedback_models import fit_feedback_head, task_metrics
        from memory_trace.sentiment import head_logits, softmax

        x = np.asarray([[-2], [-1], [1], [2], [10000]], dtype=float)
        h = fit_feedback_head(x, np.asarray([0, 0, 1, 1, -1]), c=10)
        p = softmax(head_logits(x, h))
        self.assertEqual(p.shape, (5, 3))
        self.assertTrue(np.allclose(p.sum(axis=1), 1))
        self.assertEqual(p[:4].argmax(axis=1).tolist(), [0, 0, 1, 1])
        self.assertEqual(h["observed_classes"], ["no", "yes"])
        both = np.stack([p, p], axis=1)
        y = np.asarray([[0, 0], [0, 0], [1, 1], [1, 1], [-1, -1]])
        self.assertEqual(task_metrics(y, both)["tasks"]["correction"]["missing_teacher_labels"], 1)
        h = fit_feedback_head(x, np.asarray([1, 1, 1, 1, -1]))
        self.assertTrue((softmax(head_logits(x, h)).argmax(axis=1) == 1).all())


if __name__ == "__main__":
    unittest.main()
