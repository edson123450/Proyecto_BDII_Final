"""Pruebas acústicas: cotas, exactitud de filtros, corpus y extracción reproducible."""
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from backend.audio.dtw import best_subsequence, dtw_distance, lb_keogh, search_tracks
from backend.audio.features import AudioCodebook, AudioExtractor, FEATURE_VERSION, segment_audio
from backend.audio.prepare import _save_cache, cache_path, train_codebook, valid_cache


def reference_dtw(query, candidate, radius):
    matrix = np.full((len(query) + 1, len(candidate) + 1), math.inf)
    matrix[0, 0] = 0
    for i in range(1, len(query) + 1):
        for j in range(1, len(candidate) + 1):
            if abs(i - j) <= radius:
                matrix[i, j] = float(np.sum((query[i - 1] - candidate[j - 1]) ** 2)) + min(
                    matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1])
    return matrix[-1, -1]


class AudioDTWTests(unittest.TestCase):
    def test_dtw_matches_independent_full_matrix_and_lower_bound(self):
        rng = np.random.default_rng(162)
        for length in (1, 4, 9, 25):
            for dimension in (1, 3, 13):
                query, candidate = rng.normal(size=(2, length, dimension))
                for radius in (0, 1, 3, length):
                    with self.subTest(length=length, dimension=dimension, radius=radius):
                        expected = reference_dtw(query, candidate, radius)
                        self.assertAlmostEqual(dtw_distance(query, candidate, radius), expected, places=9)
                        self.assertLessEqual(lb_keogh(query, candidate, radius), expected + 1e-9)

    def test_radius_zero_is_squared_euclidean_and_cutoff_is_safe(self):
        query = np.arange(15).reshape(5, 3).astype(float)
        candidate = query + 2
        cost = float(np.square(query - candidate).sum())
        self.assertEqual(dtw_distance(query, candidate, 0), cost)
        self.assertEqual(lb_keogh(query, candidate, 0), cost)
        self.assertEqual(dtw_distance(query, candidate, 0, cutoff=cost), cost)
        self.assertTrue(math.isinf(dtw_distance(query, candidate, 0, cutoff=cost / 10)))

    def test_all_subsequence_starts_match_exhaustive_dtw(self):
        rng = np.random.default_rng(812)
        query, track = rng.normal(size=(8, 3)), rng.normal(size=(29, 3))
        track[13:21] = query + .01
        expected = min((reference_dtw(query, track[start:start + len(query)], 2), start)
                       for start in range(len(track) - len(query) + 1))
        for filtered in (False, True):
            result = best_subsequence(query, track, radius=2, use_lb=filtered)
            self.assertAlmostEqual(result['cost'], expected[0])
            self.assertEqual(result['start_frame'], expected[1])
            self.assertEqual(result['windows'], 22)

    def test_filter_and_refine_has_identical_topk_even_with_bad_ann_seeds(self):
        rng = np.random.default_rng(74)
        query = rng.normal(size=(10, 3))
        tracks = [(number, rng.normal(size=(35, 3)) * 4) for number in range(30)]
        tracks[17][1][12:22] = query
        exhaustive, _ = search_tracks(query, iter(tracks), k=5, use_lb=False)
        filtered, stats = search_tracks(query, iter(tracks), k=5, seed_tracks=[tracks[0], tracks[9]])
        self.assertEqual(filtered, exhaustive)
        self.assertEqual(filtered[0]['track_id'], 17)
        self.assertEqual(filtered[0]['cost'], 0)
        self.assertEqual(stats['tracks'], 30)
        self.assertGreater(stats['pruned'], 0)

    def test_ties_are_refined_and_ordered_by_track_id(self):
        query = np.zeros((5, 2))
        tracks = [(9, query.copy()), (3, query.copy()), (1, query.copy())]
        result, stats = search_tracks(query, tracks, k=2)
        self.assertEqual([row['track_id'] for row in result], [1, 3])
        self.assertEqual(stats['pruned'], 0)

    def test_invalid_shapes_and_short_tracks(self):
        with self.assertRaises(ValueError):
            dtw_distance(np.zeros((3, 2)), np.zeros((4, 2)))
        with self.assertRaises(ValueError):
            lb_keogh(np.full((3, 2), np.nan), np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            dtw_distance(np.zeros((3, 2)), np.zeros((3, 2)), -1)
        result, _ = search_tracks(np.zeros((6, 2)), [(1, np.zeros((5, 2)))])
        self.assertEqual(result, [])


class FakeEmbedder:
    def embed(self, segments):
        result = np.ones((len(segments), 512), dtype=np.float32)
        result[:, 0] = [np.mean(np.square(segment)) + 1 for segment in segments]
        return result / np.linalg.norm(result, axis=1)[:, None]


class AudioFeatureTests(unittest.TestCase):
    def test_segments_have_five_overlapping_windows_without_copying_files(self):
        segments, starts = segment_audio(np.zeros(30 * 32000, np.float32), 32000)
        self.assertEqual(len(segments), 5)
        self.assertEqual(starts.tolist(), [0, 5, 10, 15, 20])
        self.assertTrue(all(len(segment) == 320000 for segment in segments))
        short, starts = segment_audio(np.zeros(5 * 32000, np.float32), 32000)
        self.assertEqual(len(short), 1)
        self.assertEqual(len(short[0]), 160000)

    def test_same_frontend_for_uploaded_clip_and_waveform(self):
        import soundfile as sf
        waveform = (.2 * np.sin(2 * np.pi * 440 * np.arange(5 * 16000) / 16000)).astype(np.float32)
        extractor = AudioExtractor(model=FakeEmbedder())
        direct = extractor.extract_waveform(waveform, 16000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'query.wav'
            sf.write(path, waveform, 16000, subtype='FLOAT')
            uploaded = extractor.extract(path)
        self.assertEqual(direct['mfcc_descriptors'].shape[1], 20)
        self.assertEqual(direct['sequence'].shape, (25, 13))
        self.assertEqual(direct['embedding'].shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(direct['embedding'])), 1, places=5)
        np.testing.assert_allclose(uploaded['mfcc_descriptors'], direct['mfcc_descriptors'])
        self.assertTrue(np.isfinite(uploaded['sequence']).all())

    def test_codebook_cache_roundtrip_counts_and_query_normalization(self):
        rng = np.random.default_rng(21)
        descriptors = rng.normal(size=(300, 20)).astype(np.float32)
        codebook = AudioCodebook.fit(descriptors, words=8, training_ids=[1, 3], seed=8)
        counts = codebook.counts(descriptors)
        self.assertEqual(sum(counts.values()), len(descriptors))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'codebook.npz'
            codebook.save(path)
            reopened = AudioCodebook.load(path)
        self.assertEqual(reopened.training_ids, (1, 3))
        self.assertEqual(reopened.counts(descriptors), counts)
        sequence = descriptors[:13, :13]
        np.testing.assert_allclose(reopened.normalize_sequence(sequence), (sequence - codebook.mean[:13]) / codebook.std[:13])

    def test_codebook_training_excludes_test_and_cache_detects_changed_source(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows = []
            for track_id, split, center in ((1, 'training', 0), (2, 'test', 1000)):
                source = output / f'{track_id}.mp3'
                source.write_bytes(b'source fingerprint')
                row = dict(id=track_id, split=split, path=str(source))
                rows.append(row)
                descriptors = (rng.normal(size=(300, 20)) + center).astype(np.float32)
                result = dict(mfcc_descriptors=descriptors, sequence=descriptors[:20, :13],
                              embedding=np.ones(512, np.float32), segment_embeddings=np.ones((1, 512), np.float32),
                              segment_starts=np.zeros(1), duration=4., sample_rate=16000)
                _save_cache(cache_path(output, track_id), result, source)
            book = train_codebook(rows, output, words=4, sample_size=100)
            self.assertEqual(book.training_ids, (1,))
            self.assertLess(abs(book.mean).max(), 1)
            source = Path(rows[0]['path'])
            self.assertTrue(valid_cache(cache_path(output, 1), source))
            source.write_bytes(b'changed source')
            self.assertFalse(valid_cache(cache_path(output, 1), source))


if __name__ == '__main__':
    unittest.main()
