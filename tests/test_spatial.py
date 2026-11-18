import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from backend.minidb.io import DiskCounter
from backend.minidb.spatial import RTree, PointStore, canonical_point, outward


class SpatialTests(unittest.TestCase):
    def expected(self, points, center, k=None, radius=None):
        center = np.asarray(center, np.float32).astype(np.float64)
        distances = np.sqrt(np.sum((points.astype(np.float64) - center) ** 2, axis=1))
        order = sorted(range(len(points)), key=lambda i: (distances[i], (i // 7 + 1, i % 7)))
        if radius is not None:
            order = [i for i in order if distances[i] <= radius]
        return [((i // 7 + 1, i % 7), float(distances[i])) for i in (order if k is None else order[:k])]

    def assert_answers(self, actual, expected):
        self.assertEqual([rid for rid, _ in actual], [rid for rid, _ in expected])
        np.testing.assert_allclose([d for _, d in actual], [d for _, d in expected], atol=1e-12, rtol=1e-12)

    def test_random_2d_split_reopen_and_exact_queries(self):
        points = np.random.default_rng(42).uniform(-10, 10, (1200, 2)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rtree.bin"
            counter = DiskCounter()
            with RTree(path, page_size=512, counter=counter, create=True) as tree:
                for i, point in enumerate(points):
                    tree.insert(point, (i // 7 + 1, i % 7))
                self.assertGreater(tree.height, 2)
                self.assertEqual(tree.size, len(points))
            with RTree(path, page_size=512, counter=counter) as tree:
                for center in ([0, 0], [3.17, -4.2], [100, 100]):
                    for k in (1, 5, 50, 2000):
                        counter.reset()
                        self.assert_answers(tree.knn(center, k), self.expected(points, center, k=k))
                        self.assertEqual(counter.disk_writes, 0)
                        self.assertGreater(counter.disk_reads, 0)
                    self.assert_answers(tree.radius_search(center, 3), self.expected(points, center, radius=3))
                ids = tree.range_search([-2, -3], [2, 3])
                expected = [(i // 7 + 1, i % 7) for i, p in enumerate(points) if -2 <= p[0] <= 2 and -3 <= p[1] <= 3]
                self.assertEqual(ids, sorted(expected))
            self.assertEqual(path.stat().st_size % 512, 0)

    def test_128_dimensions_capacity_three(self):
        points = np.random.default_rng(7).random((100, 128), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "high.bin"
            with RTree(path, dimensions=128, create=True) as tree:
                self.assertEqual(tree.capacity, 3)
                for i, point in enumerate(points):
                    tree.insert(point, (i // 7 + 1, i % 7))
                self.assertGreater(tree.height, 2)
            with RTree(path, dimensions=128) as tree:
                for center in (points[17], np.zeros(128), np.ones(128)):
                    self.assert_answers(tree.knn(center, 20), self.expected(points, center, k=20))
                    self.assert_answers(tree.radius_search(center, 5), self.expected(points, center, radius=5))

    def test_duplicate_points_ties_signed_rids_and_zero_radius(self):
        with tempfile.TemporaryDirectory() as directory:
            with RTree(Path(directory) / "ties", page_size=256, create=True) as tree:
                pairs = [([0.1, 0.2], (10 - i, -i)) for i in range(20)]
                for point, rid in pairs:
                    tree.insert(point, rid)
                expected = sorted(rid for _, rid in pairs)
                self.assertEqual([rid for rid, d in tree.knn([.1, .2], 5)], expected[:5])
                self.assertTrue(all(distance == 0 for _, distance in tree.radius_search([.1, .2], 0)))
                self.assertEqual(tree.range_search([.1, .2], [.1, .2]), expected)

    def test_point_store_baseline_persistence(self):
        points = np.random.default_rng(13).random((400, 8), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points"
            counter = DiskCounter()
            with PointStore(path, dimensions=8, counter=counter, create=True) as store:
                for i, point in enumerate(points):
                    store.insert(point, (i // 7 + 1, i % 7))
            with PointStore(path, dimensions=8, counter=counter) as store:
                counter.reset()
                self.assert_answers(store.knn(points[15], 20), self.expected(points, points[15], k=20))
                self.assertEqual(counter.disk_reads, store.file.page_count - 1)
                self.assert_answers(store.radius_search(points[15], 1), self.expected(points, points[15], radius=1))

    def test_rounding_bounds_and_invalid_input(self):
        v = np.array([.1, -.1])
        self.assertTrue(np.all(outward(v, 2, True).astype(np.float64) <= v))
        self.assertTrue(np.all(outward(v, 2, False).astype(np.float64) >= v))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty"
            with RTree(path, create=True) as tree:
                self.assertEqual(tree.knn([0, 0], 5), [])
                self.assertEqual(tree.radius_search([0, 0], 5), [])
                for bad in ([1], [np.nan, 2], [np.inf, 1], [1e100, 1]):
                    with self.assertRaises(ValueError):
                        tree.insert(bad, (1, 0))
                with self.assertRaises(ValueError):
                    tree.knn([0, 0], 0)
                with self.assertRaises(ValueError):
                    tree.radius_search([0, 0], -1)
            with self.assertRaises(ValueError):
                RTree(path, dimensions=4)
            with self.assertRaises(FileExistsError):
                RTree(path, create=True)


if __name__ == "__main__":
    unittest.main()
