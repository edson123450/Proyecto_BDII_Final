import unittest

import numpy as np

from benchmarks.parte2_audio import (assert_no_split_leakage, precision_at,
                                     probe_counts, recall_at, select_stratified_test)
from benchmarks.identificacion_audio import crop_audio, identification_score, noise_at_snr


class AudioBenchmarkContractTests(unittest.TestCase):
    def records(self):
        rows = []
        for genre_number, genre in enumerate(("Rock", "Folk", "Jazz")):
            for i in range(20):
                rows.append({"id": 100 * genre_number + i, "genre": genre,
                             "split": "training" if i < 10 else "test"})
        return rows

    def test_test_selection_is_stratified_deterministic_and_disjoint(self):
        records = self.records()
        queries = select_stratified_test(records, 4, seed=71)
        self.assertEqual(queries, select_stratified_test(records, 4, seed=71))
        self.assertEqual(len(queries), 12)
        self.assertTrue(all(row["split"] == "test" for row in queries))
        for genre in ("Rock", "Folk", "Jazz"):
            self.assertEqual(sum(row["genre"] == genre for row in queries), 4)
        with self.assertRaises(ValueError):
            select_stratified_test(records, 11)

    def test_codebook_and_gallery_cannot_include_test(self):
        records = self.records()
        gallery = [row for row in records if row["split"] == "training"]
        queries = select_stratified_test(records, 4)
        ids = [row["id"] for row in gallery]
        assert_no_split_leakage(records, gallery, queries, ids, full=True)
        with self.assertRaises(ValueError):
            assert_no_split_leakage(records, gallery, queries, ids + [queries[0]["id"]])
        with self.assertRaises(ValueError):
            assert_no_split_leakage(records, gallery + [queries[0]], queries, ids)
        with self.assertRaises(ValueError):
            assert_no_split_leakage(records, gallery, queries, ids[:2], full=True)

    def test_precision_counts_missing_results_as_misses(self):
        labels = {1: "Rock", 2: "Rock", 3: "Folk"}
        self.assertEqual(precision_at([1, 2, 3], "Rock", labels, 5), .4)
        self.assertEqual(precision_at([], "Rock", labels, 10), 0)

    def test_recall_uses_exact_ids_not_labels_or_scores(self):
        self.assertEqual(recall_at([9, 2, 3], [1, 2, 3], 3), 2 / 3)
        self.assertEqual(recall_at([1, 1, 1], [1, 2, 3], 3), 1 / 3)
        self.assertEqual(recall_at([1, 2, 3], [3, 1, 2], 3), 1)
        with self.assertRaises(ValueError):
            recall_at([1], [])

    def test_probe_schedule_includes_all_without_duplicates(self):
        self.assertEqual(probe_counts(80), [1, 2, 5, 10, 20, 50, 80])
        self.assertEqual(probe_counts(8), [1, 2, 5, 8])
        self.assertEqual(probe_counts(1), [1])

    def test_identification_noise_has_measured_20db_snr(self):
        audio = np.sin(np.arange(16000, dtype=np.float64) * .07).astype(np.float32)
        noisy, measured = noise_at_snr(audio, 20, 42)
        self.assertAlmostEqual(measured, 20, places=5)
        np.testing.assert_array_equal(noisy, noise_at_snr(audio, 20, 42)[0])
        self.assertFalse(np.array_equal(audio, noisy))

    def test_identification_crop_and_failures_are_not_concealed(self):
        audio = np.arange(3000, dtype=np.float32)
        np.testing.assert_array_equal(crop_audio(audio, 100, 7.2, 5), audio[720:1220])
        with self.assertRaises(ValueError):
            crop_audio(audio, 100, 29, 5)
        self.assertEqual(identification_score([{"id": 2}, {"id": 3}], 7)["hit_at_5"], 0)
        self.assertEqual(identification_score([{"id": 2}, {"id": 3}], 3)["hit_at_1"], 0)
        self.assertEqual(identification_score([{"id": 2}, {"id": 3}], 3)["hit_at_5"], 1)


if __name__ == "__main__":
    unittest.main()
