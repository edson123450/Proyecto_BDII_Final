import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import numpy as np

from benchmarks.parte2_audio import (AudioBenchmark, assert_no_split_leakage, compare_features, precision_at,
                                     probe_counts, read_timings, recall_at, select_stratified_test)
from benchmarks.identificacion_audio import crop_audio, identification_score, make_plan, noise_at_snr, relative_audio_path


class AudioBenchmarkContractTests(unittest.TestCase):
    def feature_sample(self):
        return {"mfcc_descriptors": np.ones((4, 20), np.float32),
                "sequence": np.ones((1, 13), np.float32),
                "embedding": np.ones(512, np.float32),
                "segment_embeddings": np.ones((2, 512), np.float32),
                "segment_starts": np.array([0, 5], np.float32), "duration": 15.0}

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
        gallery[0]["artist_id"] = "overlapping_artist"
        queries[0]["artist_id"] = "overlapping_artist"
        with self.assertRaises(ValueError):
            assert_no_split_leakage(records, gallery, queries, ids, full=True)

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

    def test_timing_trace_uses_last_valid_record_per_track(self):
        records = [
            {"id": 7, "status": "ok", "mfcc_ms": 12, "embedding_ms": 180, "run_id": "pilot"},
            {"id": 7, "status": "ok", "mfcc_ms": 10, "embedding_ms": 140, "run_id": "full"},
            {"id": 7, "status": "error", "run_id": "failed"},
            {"id": 7, "status": "ok", "feature_version": "obsolete", "run_id": "wrong_version"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in records) + '\n{"id":', encoding="utf-8")
            trace = read_timings(path)
        self.assertEqual(trace[7]["run_id"], "full")
        self.assertEqual(trace[7]["embedding_ms"], 140)

    def test_controlled_extraction_must_match_cached_features(self):
        cached = self.feature_sample()
        self.assertTrue(all(error == 0 for error in compare_features(cached, cached).values()))
        changed = {**cached, "embedding": cached["embedding"] + .01}
        with self.assertRaises(AssertionError):
            compare_features(changed, cached)
        with self.assertRaises(AssertionError):
            compare_features({**cached, "sequence": np.ones((2, 13))}, cached)

    def test_controlled_extraction_excludes_warmup_and_resumes_checkpoints(self):
        cached = self.feature_sample()
        observed = {**cached, "timings": {"decode_ms": 2.0, "mfcc_ms": 3.0,
                                          "embedding_ms": 11.0, "extraction_ms": 14.0, "total_ms": 16.0}}
        extractor = Mock()
        extractor.extract.return_value = observed
        with tempfile.TemporaryDirectory() as directory:
            bench = AudioBenchmark.__new__(AudioBenchmark)
            bench.scale, bench.fingerprint = "full", "unit-test"
            bench.cache, bench.checkpoints = Path(directory) / "private/audio", Path(directory) / "checkpoints"
            bench.checkpoints.mkdir()
            bench.queries = [{"id": i, "genre": "Rock", "path": f"/private/source/{i}.mp3"} for i in (7, 8)]
            bench.timings = {7: {"id": 7, "run_id": "historical", "path": "/private/not-public"}}
            bench.codebook = Mock()
            bench.codebook.counts.return_value = {"aw000": 4}
            bench.log = Mock()
            bench.verify_sources = Mock()
            bench.export = Mock()
            with patch("benchmarks.parte2_audio.AudioExtractor", return_value=extractor) as factory, \
                    patch("benchmarks.parte2_audio.load_features", return_value=cached):
                bench.measure_extraction()
                bench.measure_extraction()
            factory.assert_called_once_with(cache_dir=bench.cache.parent, device="cpu", batch_size=2, threads=2)
            self.assertEqual(extractor.extract.call_count, 3)  # warmup + two measured queries
            self.assertEqual(bench.export.call_count, 2)
            files = sorted(bench.checkpoints.glob("extraction_*.json"))
            self.assertEqual(len(files), 2)
            record = json.loads(files[0].read_text())
            self.assertEqual(record["timings"]["embedding_ms"], 11.0)
            self.assertEqual(record["historical_trace"], {"id": 7, "run_id": "historical"})
            self.assertNotIn("/private/", files[0].read_text())

    def test_identification_noise_has_measured_20db_snr(self):
        audio = np.sin(np.arange(16000, dtype=np.float64) * .07).astype(np.float32)
        noisy, measured = noise_at_snr(audio, 20, 42)
        self.assertAlmostEqual(measured, 20, places=5)
        np.testing.assert_array_equal(noisy, noise_at_snr(audio, 20, 42)[0])
        self.assertFalse(np.array_equal(audio, noisy))

    def test_public_audio_path_is_relative_to_configured_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fma_small"
            self.assertEqual(relative_audio_path(root / "139/139933.mp3", root), "139/139933.mp3")
            with self.assertRaises(ValueError):
                relative_audio_path(Path(directory) / "outside.mp3", root)

    def test_identification_plan_is_not_resampled_after_corpus_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [{**row, "path": str(root / f"{row['id']:06d}.mp3")} for row in self.records()]
            plan = make_plan(records, audio_root=root)
            selected = {query["track_id"] for query in plan}
            self.assertEqual(make_plan(records, selected, root), plan)
            with self.assertRaises(ValueError):
                make_plan(records, {row["id"] for row in records} - {plan[0]["track_id"]}, root)

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
