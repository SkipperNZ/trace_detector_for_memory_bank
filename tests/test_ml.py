"""Explicit integration test using an already cached checkpoint, never downloads weights."""

import os
import tempfile
import unittest
from pathlib import Path

from memory_trace.demo import annotations_and_scores, events
from memory_trace.embeddings import FrozenEncoder, predict, train
from memory_trace.prepare import prepare


@unittest.skipUnless(os.environ.get("MB_TRACE_ML_TEST") == "1", "opt-in cached model test")
class CachedEncoderTest(unittest.TestCase):
    def test_frozen_encoder_head_roundtrip_and_token_limit(self):
        examples = prepare(events())
        labels, _ = annotations_and_scores(examples)
        for ex in examples:
            ex["split"] = "train"
        for label in labels:
            label["label_source"] = "human"  # Hand-authored fixture, not research gold.
        encoder = FrozenEncoder(
            "sentence-transformers/all-MiniLM-L6-v2",
            "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
            prefix="",
            device=os.environ.get("MB_TRACE_ML_DEVICE", "cpu"),
            local_files_only=True,
        )
        with tempfile.TemporaryDirectory() as temp:
            head = train(examples, labels, encoder, Path(temp) / "head.json")
            self.assertEqual(head["training_messages"], 5)
            self.assertEqual(head["excluded_messages"], 1)
            predictions = predict(examples, head, encoder)
            self.assertTrue(all(0 <= p["score"] <= 1 for p in predictions))
            long_example = prepare([{**events()[0], "text": "word " * 2000}])[0]
            long_example["split"] = "train"
            result = predict([long_example], head, encoder)[0]
            self.assertIsNone(result["score"])
            self.assertEqual(result["context_status"], "token_limit")
            examples[0]["split"] = "test"
            with self.assertRaisesRegex(ValueError, "overlaps"):
                predict(examples, head, encoder)
