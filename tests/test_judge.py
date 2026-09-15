import json
import tempfile
import threading
import unittest
import urllib.error
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from memory_trace.dataset import SENTIMENTS, convert, pilot_sample
from memory_trace.io import read_jsonl
from memory_trace.judge import (
    JudgeConfig,
    call_judge,
    messages,
    read_config,
    run_judge,
    validate_answer,
)
from memory_trace.judge_report import compare, render_markdown
from memory_trace.metrics import validate_labels


def dataset_rows(count=6):
    return [
        {
            "id": i,
            "source_dataset": "fixture/source",
            "session_id": f"s{i // 2}",
            "content_text": f"Message {i}",
            "sentiment_label": SENTIMENTS[i % 3],
            "sentiment_reason": "FORBIDDEN SOURCE REASON",
            "nTurns": 999,
            "input_tokens_total": 123456789,
        }
        for i in range(count)
    ]


def response(payload, config):
    target = json.loads(payload["messages"][-1]["content"])["target_event_id"]
    answer = {
        "sentiment_label": "NEUTRAL",
        "label": "no",
        "language": "en",
        "context_sufficient": True,
        "evidence_event_ids": [target],
        "reason": "An instruction.",
    }
    return {
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


class DatasetTests(unittest.TestCase):
    def test_import_is_message_only_blind_and_grouped(self):
        examples, references = convert(dataset_rows())
        for ex in examples:
            serialized = json.dumps(messages(ex))
            self.assertNotIn("FORBIDDEN SOURCE REASON", serialized)
            self.assertNotIn("123456789", serialized)
            self.assertNotIn("sentiment_reason", json.dumps(ex))
            self.assertNotIn("sentiment_label", json.dumps(ex))
            self.assertFalse(ex["history_available"])
            self.assertEqual(len(ex["context"]), 1)
        self.assertEqual(examples[0]["split"], examples[1]["split"])
        self.assertEqual(references[0]["label_source"], "dataset_llm")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            convert(dataset_rows() + dataset_rows())

    def test_pilot_stratification_and_probabilities(self):
        examples, refs = convert(dataset_rows(12))
        for example in examples:
            example["split"] = "calibration"
        selected, design = pilot_sample(examples, refs, 2, 42)
        self.assertEqual(len(selected), 6)
        self.assertTrue(all(d["inclusion_probability"] == 0.5 for d in design))
        reversed_result, _ = pilot_sample(list(reversed(examples)), list(reversed(refs)), 2, 42)
        self.assertEqual(selected, reversed_result)
        examples[0]["split"] = "test"
        selected, _ = pilot_sample(examples, refs, 100, 42)
        self.assertNotIn(examples[0], selected)


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.examples, self.refs = convert(dataset_rows())
        self.config = JudgeConfig(
            "http://localhost:1234/v1", "test-model", "test-deployment", retries=0, concurrency=2
        )

    def test_response_validation(self):
        ex = self.examples[0]
        raw = response({"messages": messages(ex)}, self.config)["choices"][0]["message"]["content"]
        answer = validate_answer(raw, ex)
        self.assertEqual(answer["label"], "no")
        answer["evidence_event_ids"] = ["future"]
        with self.assertRaisesRegex(ValueError, "Evidence"):
            validate_answer(json.dumps(answer), ex)
        answer["evidence_event_ids"] = []
        answer["label"] = "yes"
        with self.assertRaisesRegex(ValueError, "cite"):
            validate_answer(json.dumps(answer), ex)
        answer["label"] = "no"
        answer["context_sufficient"] = False
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            validate_answer(json.dumps(answer), ex)

    def test_sentiment_only_pipeline_is_blind_and_exports_three_classes(self):
        config = replace(self.config, task="sentiment")
        self.assertNotEqual(config.version, self.config.version)
        observed = []

        def transport(payload, config):
            observed.append(payload)
            target = json.loads(payload["messages"][-1]["content"])["target_event_id"]
            label = SENTIMENTS[int(target) % 3]
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps({"label": label, "reason": "Fixture sentiment."})
                        },
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            result = run_judge(self.examples, config, out, repeats=1, request_fn=transport)
            self.assertEqual(result["labeled_messages"], 6)
            run_judge(self.examples, config, out, repeats=1, request_fn=transport)
            self.assertEqual(len(observed), 6)
            for payload in observed:
                serialized = json.dumps(payload)
                self.assertNotIn("FORBIDDEN SOURCE REASON", serialized)
                self.assertNotIn("123456789", serialized)
                self.assertNotIn("yes|no|unclear", serialized)
            labels = read_jsonl(out / "labels.jsonl")
            self.assertEqual({row["sentiment_label"] for row in labels}, set(SENTIMENTS))
            self.assertTrue(
                all("label" not in row and row["repeat_consistent"] is None for row in labels)
            )
            report, disagreements = compare(
                self.examples, read_jsonl(out / "calls.jsonl"), self.refs, expected_repeats=1
            )
            self.assertEqual(report["sentiment_agreement_on_successful"], 1)
            self.assertEqual(disagreements, [])
            self.assertNotIn("primary_complaint_counts", report)
            self.assertIsNone(report["sentiment_repeat_agreement"])
            self.assertNotIn("Явное недовольство", render_markdown(report))
            with self.assertRaisesRegex(ValueError, "another"):
                run_judge(self.examples, self.config, out, repeats=1, request_fn=transport)

    def test_sentiment_validation_rejects_complaint_and_extra_fields(self):
        for answer in (
            {"label": "yes", "reason": "x"},
            {"label": "NEUTRAL", "reason": "x", "complaint": "no"},
            {"label": "NEUTRAL", "reason": ""},
        ):
            with self.assertRaises(ValueError):
                validate_answer(json.dumps(answer), self.examples[0], "sentiment")

    def test_custom_prompt_is_versioned_sent_and_exported(self):
        config = replace(
            self.config,
            task="sentiment",
            system_prompt="Classify sentiment. Return label and reason.",
            prompt_version="fixture-v2",
        )
        self.assertNotEqual(
            config.version,
            replace(config, system_prompt=config.system_prompt + " Use three classes.").version,
        )
        observed = []

        def transport(payload, config):
            observed.append(payload)
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"label":"NEUTRAL","reason":"An instruction."}'},
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            run_judge(self.examples[:1], config, out, repeats=1, request_fn=transport)
            self.assertEqual(observed[0]["messages"][0]["content"], config.system_prompt)
            manifest = json.loads((out / "run.json").read_text())
            self.assertEqual(manifest["system_prompt"], config.system_prompt)
            self.assertEqual(manifest["prompt_version"], "fixture-v2")
            self.assertNotIn("prompt_source", manifest)
            self.assertEqual(read_jsonl(out / "labels.jsonl")[0]["metric_version"], "fixture-v2")
        with self.assertRaisesRegex(ValueError, "requires"):
            replace(config, prompt_version=None)

    def test_sentiment_repeat_disagreement_stays_unlabeled(self):
        config = replace(self.config, task="sentiment", concurrency=1)
        chosen = iter(["NEGATIVE", "NEUTRAL"])

        def transport(payload, config):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"label": next(chosen), "reason": "Ambiguous message."}
                            )
                        },
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            result = run_judge(self.examples[:1], config, out, request_fn=transport)
            self.assertEqual(result["labeled_messages"], 0)
            self.assertEqual(read_jsonl(out / "labels.jsonl"), [])
            self.assertEqual(
                read_jsonl(out / "unresolved.jsonl")[0]["reason"], "sentiment_disagreement"
            )
            report, disagreements = compare(
                self.examples[:1], read_jsonl(out / "calls.jsonl"), self.refs
            )
            self.assertEqual(report["sentiment_repeat_agreement"], 0)
            self.assertEqual(report["sentiment_agreement_on_successful"], 1)
            self.assertNotIn("complaint_repeat_agreement", report)

    def test_sentiment_accepts_only_a_complete_json_fence(self):
        raw = json.dumps({"label": "NEUTRAL", "reason": "An instruction."})
        answer = validate_answer("```json\n" + raw + "\n```", self.examples[0], "sentiment")
        self.assertEqual(answer["sentiment_label"], "NEUTRAL")
        for text in (
            "Here is the answer: " + raw,
            "```json\n" + raw + "\n```\nExtra text",
            "```json\n" + raw + "\n```\n```json\n" + raw + "\n```",
        ):
            with self.assertRaises(ValueError):
                validate_answer(text, self.examples[0], "sentiment")

    def test_resume_uses_checkpoint_and_stores_llm_provenance(self):
        count, lock = [0], threading.Lock()

        def transport(payload, config):
            with lock:
                count[0] += 1
            return response(payload, config)

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            report = run_judge(self.examples, self.config, out, request_fn=transport)
            self.assertEqual(count[0], 12)
            self.assertEqual(report["labeled_messages"], 6)
            run_judge(self.examples, self.config, out, request_fn=transport)
            self.assertEqual(count[0], 12)
            labels = read_jsonl(out / "labels.jsonl")
            exs = read_jsonl(out / "labeled_examples.jsonl")
            validate_labels(exs, labels, label_source="llm_judge")
            self.assertTrue(all(ex["language"] == "en" for ex in exs))
            with self.assertRaisesRegex(ValueError, "human"):
                validate_labels(exs, labels)
            with self.assertRaisesRegex(ValueError, "another"):
                run_judge(self.examples[:1], self.config, out, request_fn=transport)
            calls = read_jsonl(out / "calls.jsonl")
            comparison, disagreements = compare(self.examples, calls, self.refs)
            self.assertAlmostEqual(comparison["sentiment_agreement_on_successful"], 1 / 3)
            self.assertEqual(comparison["sentiment_repeat_agreement"], 1)
            self.assertEqual(comparison["reported_token_usage"]["total_tokens"], 1440)
            self.assertEqual(len(disagreements), 4)
            partial, _ = compare(self.examples, calls, self.refs, expected_repeats=3)
            self.assertEqual(partial["repeat_comparable_messages"], 0)

    def test_canary_stops_on_authentication_failure(self):
        def denied(payload, config):
            raise urllib.error.HTTPError("http://localhost", 401, "Unauthorized", {}, None)

        with tempfile.TemporaryDirectory() as temp:
            report = run_judge(self.examples, self.config, Path(temp), request_fn=denied)
            self.assertEqual(report["completed_calls"], 1)
            self.assertEqual(report["failed_calls"], 1)
            self.assertEqual(report["unattempted_calls"], 11)
            self.assertEqual(report["labeled_messages"], 0)

    def test_transport_timeout_override_preserves_request_version_and_is_recorded(self):
        from memory_trace.judge import effective_http_timeout

        version = self.config.version
        with patch.dict("os.environ", {"MB_JUDGE_HTTP_TIMEOUT_SECONDS": "300"}):
            self.assertEqual(effective_http_timeout(self.config), 300)
            self.assertEqual(self.config.version, version)
            record = call_judge(self.examples[0], 0, self.config, response)
            self.assertEqual(record["effective_http_timeout_seconds"], 300)
            self.assertEqual(record["runtime_status"], "ok")
        for value in ("0", "nan", "inf", "-1"):
            with patch.dict("os.environ", {"MB_JUDGE_HTTP_TIMEOUT_SECONDS": value}):
                with self.assertRaises(ValueError):
                    effective_http_timeout(self.config)

    def test_mid_run_outage_stops_new_requests_and_resume_reuses_successes(self):
        counter = [0]

        def outage(payload, config):
            counter[0] += 1
            if counter[0] > 2:
                raise urllib.error.URLError("server went offline")
            return response(payload, config)

        config = replace(self.config, concurrency=1, retries=0)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            report = run_judge(self.examples, config, out, repeats=1, request_fn=outage)
            self.assertTrue(report["stopped_due_to_transport"])
            self.assertEqual(counter[0], 3)
            self.assertEqual(report["successful_calls"], 2)
            self.assertEqual(report["unattempted_calls"], 3)
            resumed = [0]

            def recovered(payload, config):
                resumed[0] += 1
                return response(payload, config)

            report = run_judge(
                self.examples, config, out, repeats=1, retry_errors=True, request_fn=recovered
            )
            self.assertFalse(report["stopped_due_to_transport"])
            self.assertEqual(report["successful_calls"], 6)
            self.assertEqual(resumed[0], 4)

    def test_failed_attempts_count_and_retry_is_saved(self):
        number = [0]

        def sometimes_bad(payload, config):
            number[0] += 1
            if number[0] == 1:
                return {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}
            return response(payload, config)

        config = JudgeConfig("http://localhost/v1", "test", "test", retries=1)
        with patch("memory_trace.judge.time.sleep"):
            result = call_judge(self.examples[0], 0, config, sometimes_bad)
        self.assertEqual(result["runtime_status"], "ok")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["attempts"][0]["runtime_status"], "error")

    def test_persistent_format_error_does_not_block_other_recorded_retries(self):
        config = replace(self.config, concurrency=1, retries=0)
        first = [True]

        def initial(payload, config):
            result = response(payload, config)
            if first[0]:
                first[0] = False
            else:
                result["choices"][0]["message"]["content"] = "malformed"
            return result

        stubborn = [True]

        def retry(payload, config):
            result = response(payload, config)
            if stubborn[0]:
                stubborn[0] = False
                result["choices"][0]["message"]["content"] = "malformed"
            return result

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            initial_result = run_judge(self.examples, config, output, repeats=1, request_fn=initial)
            self.assertEqual(initial_result["failed_calls"], len(self.examples) - 1)
            retried = run_judge(
                self.examples, config, output, repeats=1, request_fn=retry, retry_errors=True
            )
            self.assertEqual(retried["unattempted_calls"], 0)
            self.assertEqual(retried["failed_calls"], 1)
            self.assertEqual(retried["successful_calls"], len(self.examples) - 1)

    def test_input_limit_never_silently_truncates_or_calls_model(self):
        config = JudgeConfig("http://localhost/v1", "test", "test", max_input_chars=10)

        def forbidden(payload, config):
            raise AssertionError("Should not send an over-limit input")

        result = call_judge(self.examples[0], 0, config, forbidden)
        self.assertEqual(result["runtime_status"], "not_evaluated")
        self.assertEqual(result["error_type"], "input_limit")

    def test_config_secret_file_relative_to_config(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "judge.json"
            path.write_text(
                json.dumps(
                    {
                        "base_url": "http://localhost/v1",
                        "model": "test",
                        "deployment_id": "snapshot",
                        "api_key_file": "secret.txt",
                    }
                )
            )
            config = read_config(path)
            # macOS /var symlink and Windows 8.3 temp aliases resolve to the same file.
            self.assertEqual(Path(config.api_key_file), (Path(temp) / "secret.txt").resolve())

    def test_prompt_file_is_relative_and_fingerprint_uses_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "prompt.txt").write_text("Classify tone.", encoding="utf-8")
            path = root / "judge.json"
            config = {
                "task": "sentiment",
                "base_url": "http://localhost/v1",
                "model": "test",
                "deployment_id": "snapshot",
                "prompt_file": "prompt.txt",
                "prompt_version": "fixture-file-v1",
            }
            path.write_text(json.dumps(config), encoding="utf-8")
            first = read_config(path)
            self.assertEqual(first.system_prompt, "Classify tone.")
            (root / "prompt.txt").write_text("Classify tone carefully.", encoding="utf-8")
            self.assertNotEqual(first.version, read_config(path).version)
            config["system_prompt"] = "Conflicting inline prompt"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "either"):
                read_config(path)

    def test_no_credential_in_config_hash_or_manifest(self):
        with patch.dict("os.environ", {"MB_JUDGE_API_KEY": "credential-one"}):
            first = self.config.version
        with patch.dict("os.environ", {"MB_JUDGE_API_KEY": "credential-two"}):
            self.assertEqual(first, self.config.version)


if __name__ == "__main__":
    unittest.main()
