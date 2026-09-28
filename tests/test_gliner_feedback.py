import importlib.util
import unittest

from memory_trace.gliner_feedback import split_complete


class CompleteTextTests(unittest.TestCase):
    def test_long_unicode_and_whitespace_are_retained_in_order(self):
        text = "  начало\n" + "оченьДлинноеСлово" * 20 + "\n\t END 😊  "
        chunks = split_complete(text, lambda s: len(s) + 10, 27)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(s for s, _ in chunks), text)
        self.assertTrue(all(n <= 27 for _, n in chunks))

    def test_no_room_for_text_fails_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "no room"):
            split_complete("x", lambda s: 100 + len(s), 100)

    def test_short_input_and_empty_input(self):
        for text in ("", "A small sentence."):
            self.assertEqual(split_complete(text, len, 100), [(text, len(text))])


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "ML extra required")
class DecideMetricsTests(unittest.TestCase):
    def test_aggregate_logits_before_softmax_preserves_two_tasks(self):
        import numpy as np
        from memory_trace.gliner_feedback import aggregate_logits, probabilities

        logits = [[[4, 1, 0], [2, 0, 1]], [[0, 3, 2], [0, 4, 2]]]
        record = {"logits": {m: aggregate_logits(logits, m).tolist() for m in ("max", "mean")}}
        p = probabilities([record], "max")
        self.assertEqual(p.shape, (1, 2, 3))
        self.assertEqual(p.argmax(axis=2).tolist(), [[0, 1]])
        self.assertTrue(np.allclose(p.sum(axis=2), 1))
        with self.assertRaises(ValueError):
            aggregate_logits([[[float("nan")] * 3] * 2], "max")

    def test_recall_threshold_masks_missing_teacher_and_handles_ties(self):
        import numpy as np
        from memory_trace.gliner_feedback import select_recall_threshold, yes_metrics

        y = np.array([1, 1, 0, 2, -1])
        scores = np.array([0.9, 0.5, 0.5, 0.1, 0.95])
        selected = select_recall_threshold(y, scores, 1.0)
        self.assertEqual(selected["threshold"], 0.5)
        self.assertEqual(selected["recall"], 1)
        self.assertAlmostEqual(selected["precision"], 2 / 3)
        self.assertEqual(selected["messages"], 4)
        self.assertEqual(yes_metrics(y, scores, 0.8)["tp"], 1)


if __name__ == "__main__":
    unittest.main()
