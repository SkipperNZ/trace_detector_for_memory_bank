import importlib.util
import unittest

from memory_trace.dataset import convert
from memory_trace.judge import JudgeConfig
from memory_trace.student_data import clean_splits, duplicate_components
from memory_trace.sentiment import targets, make_sentiment_input
from memory_trace.sentiment_encoder import token_windows


def example(i, text, split, session=None):
    rows, _ = convert(
        [
            {
                "id": i,
                "source_dataset": "fixture",
                "session_id": str(session or i),
                "content_text": text,
                "sentiment_label": "NEUTRAL",
            }
        ]
    )
    return {**rows[0], "split": split}


class StudentDataTests(unittest.TestCase):
    def setUp(self):
        self.config = JudgeConfig("http://localhost/v1", "fixture", "fixture", task="sentiment")

    def test_new_messages_need_no_labels_and_keep_tenants_separate(self):
        a = make_sentiment_input("A message", tenant_id="a", session_id="1", message_id="1")
        b = make_sentiment_input("Next message", tenant_id="a", session_id="1", message_id="2")
        other = make_sentiment_input("A message", tenant_id="b", session_id="1", message_id="1")
        self.assertEqual(a["group_id"], b["group_id"])
        self.assertNotEqual(a["input_id"], b["input_id"])
        self.assertNotEqual(a["group_id"], other["group_id"])
        self.assertEqual(a["split"], "inference")
        self.assertNotIn("sentiment_label", a)
        self.assertNotIn("dataset_id", a)
        self.assertEqual(a["context"][0]["text"], "A message")

    def test_reviewed_sessions_and_unicode_duplicates_do_not_leak(self):
        seen = example(1, "Already reviewed", "test", "old")
        rows = [
            seen,
            example(2, "Different text in same session", "test", "old"),
            example(3, "Ａlready   REVIEWED", "test"),
            example(4, "unique held out", "test"),
            example(5, "UNIQUE held out", "train"),
            example(6, "new training example", "train"),
        ]
        kept, excluded, _ = clean_splits(rows, [seen], self.config, near=False)
        self.assertEqual({ex["target_event_id"] for ex in kept}, {"4", "6"})
        self.assertEqual(len(excluded), 4)

    def test_limit_accounts_for_prompt_and_serialization(self):
        config = JudgeConfig(
            "http://localhost/v1", "fixture", "fixture", task="sentiment", max_input_chars=10
        )
        kept, excluded, _ = clean_splits([example(1, "ok", "train")], [], config, near=False)
        self.assertFalse(kept)
        self.assertEqual(excluded[0]["reason"], "judge_input_limit")

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "ML extra required")
    def test_near_template_cross_split_leakage(self):
        body = "Investigate the rendering pipeline and its session lifecycle. " * 20
        rows = [
            example(1, body + "ticket 100", "train"),
            example(2, body + "ticket 101", "test"),
            example(3, "Thank you!", "train"),
        ]
        components, pairs = duplicate_components(rows)
        self.assertEqual(components[0], components[1])
        self.assertTrue(pairs)
        kept, excluded, _ = clean_splits(rows, [], self.config)
        self.assertEqual({ex["target_event_id"] for ex in kept}, {"2", "3"})
        self.assertEqual(excluded[0]["reason"], "duplicate_in_higher_priority_split")

    def test_sentiment_targets_reject_binary_or_mixed_teacher_labels(self):
        rows = [example(1, "good", "train"), example(2, "bad", "train")]
        labels = [
            {
                "input_id": ex["input_id"],
                "task": "sentiment",
                "label_source": "llm_judge",
                "sentiment_label": "POSITIVE",
                "judge_version": "v1",
            }
            for ex in rows
        ]
        kept, y, missing = targets(rows, labels[:1])
        self.assertEqual(y, [2])
        self.assertEqual(missing, [rows[1]["input_id"]])
        labels[1]["judge_version"] = "v2"
        with self.assertRaisesRegex(ValueError, "mix"):
            targets(rows, labels)
        labels[1]["task"] = "sentiment_and_complaint"
        with self.assertRaisesRegex(ValueError, "sentiment labels"):
            targets(rows, labels)

    def test_window_serializer_preserves_every_content_token(self):
        class Tokenizer:
            cls_token_id, sep_token_id = 101, 102

            def __call__(self, text, **kwargs):
                return {
                    "input_ids": [ord(c) for c in text],
                    "offset_mapping": [(i, i + 1) for i in range(len(text))],
                }

            def num_special_tokens_to_add(self, **kwargs):
                return 2

        text = "full text including the critical ending"
        windows, count = token_windows(Tokenizer(), text, max_length=15, prefix="q: ")
        recovered = [token for ids, _ in windows for token in ids[3:-1]]
        self.assertEqual(recovered, [ord(c) for c in " " + text])
        self.assertEqual(sum(w for _, w in windows), count)
        self.assertTrue(all(len(ids) <= 15 for ids, _ in windows))

    def test_short_window_matches_joint_tokenization(self):
        import re

        class Tokenizer:
            cls_token_id, sep_token_id = 101, 102

            def __call__(self, text, **kwargs):
                pieces = list(re.finditer(r" ?\S+| +$", text))
                return {
                    "input_ids": [match.group() for match in pieces],
                    "offset_mapping": [match.span() for match in pieces],
                }

            def num_special_tokens_to_add(self, **kwargs):
                return 2

        tokenizer = Tokenizer()
        for text in ("Please inspect this code.", "", "  code", "!!!"):
            windows, _ = token_windows(tokenizer, text)
            expected = [101] + tokenizer("query: " + text)["input_ids"] + [102]
            self.assertEqual(windows[0][0], expected)
            self.assertEqual(len(windows), 1)

    def test_unprefixed_windows_preserve_first_token_and_empty_messages(self):
        class Tokenizer:
            cls_token_id, sep_token_id = 0, 2

            def __call__(self, text, **kwargs):
                return {
                    "input_ids": [ord(c) for c in text],
                    "offset_mapping": [(i, i + 1) for i in range(len(text))],
                }

            def num_special_tokens_to_add(self, **kwargs):
                return 2

        for text in ("", "a", "  пробелы и длинный текст" * 10):
            windows, count = token_windows(Tokenizer(), text, max_length=8, prefix="")
            self.assertEqual([t for ids, _ in windows for t in ids[1:-1]], [ord(c) for c in text])
            self.assertEqual(count, len(text))
            self.assertTrue(all(len(ids) <= 8 for ids, _ in windows))
            self.assertTrue(all(weight >= 1 for _, weight in windows))


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "ML extra required")
class StudentModelTests(unittest.TestCase):
    def test_study_selection_never_requests_test_targets(self):
        import tempfile
        import numpy as np
        from pathlib import Path
        from unittest.mock import patch
        from memory_trace.io import write_json, write_jsonl, digest
        from memory_trace.student_study import (
            freeze_study,
            fit_baselines,
            freeze_winner,
            split_targets,
        )

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            splits = {}
            labels = []
            for split, count in (("train", 18), ("calibration", 9), ("test", 6)):
                rows = []
                for i in range(count):
                    value = i % 3
                    text = ("broken bad bug", "plain technical task", "excellent wonderful good")[
                        value
                    ]
                    row = example(f"{split}-{i}", text + f" example {i}", split)
                    rows.append(row)
                    labels.append(
                        {
                            "input_id": row["input_id"],
                            "task": "sentiment",
                            "label_source": "llm_judge",
                            "sentiment_label": ("NEGATIVE", "NEUTRAL", "POSITIVE")[value],
                            "judge_version": "fixture-v1",
                        }
                    )
                write_jsonl(base / f"data/{split}.jsonl", rows)
                splits[split] = {"hash": digest(rows)}
                vectors = np.asarray([np.eye(3)[i % 3] for i in range(count)])
                folder = base / f"embeddings/{split}"
                folder.mkdir(parents=True)
                np.save(folder / "vectors.npy", vectors)
                write_json(
                    folder / "manifest.json",
                    {
                        "input_ids": [ex["input_id"] for ex in rows],
                        "inputs_hash": digest(rows),
                        "vectors_hash": digest(vectors.tolist()),
                        "encoder_disk_bytes": 1000000,
                        "encoder": {"model": "fixture", "revision": "a" * 40},
                    },
                )
            write_jsonl(base / "teacher/labels.jsonl", labels)
            write_json(
                base / "teacher/summary.json",
                {"unattempted_calls": 0, "judge_version": "fixture-v1"},
            )
            write_json(base / "data/design.json", {"judge_version": "fixture-v1", "splits": splits})
            write_json(base / "encoder-source.json", {"model": "fixture", "revision": "a" * 40})
            policy = {
                "teacher_version": "fixture-v1",
                "encoder_model": "fixture",
                "encoder_revision": "a" * 40,
                "seed": 42,
                "learning_curve_messages": [6, 12],
                "tfidf_C": [1.0],
                "encoder_C": [1.0],
                "class_weight_balanced": [False],
                "recall_targets": [0.95],
                "audit_rates": [0.1],
            }
            write_json(base / "policy.json", policy)
            freeze_study(base, base / "policy.json")

            def guarded(folder, split, config):
                self.assertNotEqual(split, "test", "Selection touched final test targets")
                return split_targets(folder, split, config)

            with patch("memory_trace.student_study.split_targets", side_effect=guarded):
                baseline = fit_baselines(base)
                self.assertFalse(baseline["finetune_required"])
                selection = freeze_winner(base)
            self.assertTrue(selection["selected_before_test"])
            self.assertGreater(selection["winner"]["calibration"]["macro_f1"], 0.9)
            self.assertEqual(selection["winner"]["kind"], "tfidf")
            encoder_candidate = next(
                r for r in baseline["candidates"] if r["name"] == "encoder-all"
            )
            self.assertGreater(encoder_candidate["artifact_bytes"], 1000000)

    def test_portable_three_class_model_roundtrip_and_leakage_guard(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from memory_trace.sentiment import (
            TfidfFeatures,
            fit_head,
            head_logits,
            provenance,
            save_student,
            load_student,
            predict_student,
        )

        rows = [
            example(i + 1, text, "train")
            for i, text in enumerate(
                [
                    "bad broken bug",
                    "bad broken failure",
                    "do technical work",
                    "do technical task",
                    "great lovely result",
                    "great wonderful result",
                ]
            )
        ]
        labels = [
            {
                "input_id": ex["input_id"],
                "task": "sentiment",
                "label_source": "llm_judge",
                "sentiment_label": ("NEGATIVE", "NEUTRAL", "POSITIVE")[i // 2],
                "judge_version": "v1",
            }
            for i, ex in enumerate(rows)
        ]
        features = TfidfFeatures()
        vectors = features.fit_transform([ex["context"][0]["text"] for ex in rows])
        head = fit_head(vectors, [0, 0, 1, 1, 2, 2])
        with tempfile.TemporaryDirectory() as temp:
            artifact = save_student(
                temp,
                kind="tfidf",
                training=provenance(rows, labels),
                head=head,
                features=features.state(),
            )
            loaded = load_student(temp)
            predicted, _ = predict_student(rows, loaded)
            self.assertEqual(
                [p["sentiment_label"] for p in predicted], [l["sentiment_label"] for l in labels]
            )
            restored = TfidfFeatures(loaded["features"]).transform(
                [ex["context"][0]["text"] for ex in rows]
            )
            np.testing.assert_allclose(
                head_logits(restored, head), head_logits(vectors, head), atol=1e-12
            )
            leaked = [{**rows[0], "split": "test"}]
            with self.assertRaisesRegex(ValueError, "overlaps"):
                predict_student(leaked, loaded)
            path = Path(temp) / "model.json"
            path.write_text(path.read_text().replace('"shadow"', '"cascade"'))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_student(temp)

    def test_threshold_ties_and_probability_sampling(self):
        from memory_trace.sentiment_metrics import choose_thresholds, prevalence_experiment

        policies = choose_thresholds([0, 0, 0, 1], [0.8, 0.8, 0.2, 0.1], recalls=(2 / 3, 1.0))
        self.assertEqual(policies[0]["threshold"], 0.8)
        self.assertEqual(policies[1]["threshold"], 0.2)
        rows = [example(i + 1, "fixture " + str(i), "test") for i in range(20)]
        labels = {ex["input_id"]: 0 if i < 5 else 1 for i, ex in enumerate(rows)}
        report = prevalence_experiment(
            rows, labels, [0.9] * 2 + [0.1] * 18, policies[0], audit_rate=0.25, replicates=10000
        )
        self.assertEqual(report["teacher_negative_fraction_known"], 0.25)
        self.assertGreater(report["naive_selected_message_fraction"], 0.25)
        for design in report["designs"].values():
            self.assertLess(abs(design["bias"]), 0.01)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            prevalence_experiment(rows, labels, [0.1] * 20, policies[0], audit_rate=0)
