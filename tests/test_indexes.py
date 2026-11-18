"""Verificación de estructuras, persistencia y transferencias de los índices."""
import random
import tempfile
import unittest
from pathlib import Path

from backend.minidb.indexes import BPlusTree, ExtendibleHash, PAGE_HEADER
from backend.minidb.io import DiskCounter


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bplus_multilevel_duplicates_and_reopen(self):
        path = self.root / "tree.idx"
        entries = [(i % 73, (i // 10, i % 10)) for i in range(1700)]
        random.Random(19).shuffle(entries)
        counter = DiskCounter()
        tree = BPlusTree(path, page_size=128, counter=counter, create=True)
        for key, rid in entries:
            tree.insert(key, rid)
        expected = sorted(entries)
        self.assertEqual(list(tree.iter_entries()), expected)
        self.assertGreaterEqual(tree.stats()["height"], 4)
        self.assertEqual(tree.stats()["size"], len(entries))
        for key in range(73):
            self.assertEqual(tree.search(key), [rid for k, rid in expected if k == key])
        node = tree._read_node(tree._first_leaf)
        previous, count = -1, 0
        while True:
            self.assertEqual(node.prev, previous)
            self.assertEqual(node.entries, sorted(node.entries))
            count += len(node.entries)
            previous = node.pid
            if node.next == -1:
                break
            node = tree._read_node(node.next)
        self.assertEqual(count, len(entries))
        tree.close()
        self.assertEqual(path.stat().st_size % 128, 0)
        with BPlusTree(path, page_size=128, counter=counter) as reopened:
            self.assertEqual(list(reopened.iter_entries()), expected)
            self.assertEqual(reopened.range_search(20, 22), [rid for key, rid in expected if 20 <= key <= 22])
            self.assertEqual(reopened.range_search(20, 22, False, False), [rid for key, rid in expected if 20 < key < 22])
            self.assertEqual(reopened.range_search(high=1), [rid for key, rid in expected if key <= 1])
            self.assertEqual(reopened.range_search(low=72), [rid for key, rid in expected if key >= 72])
            self.assertEqual(reopened.range_search(50, 20), [])
            self.assertEqual(reopened.search(100), [])

    def test_bplus_deletion_empty_leaves_and_new_inserts(self):
        path = self.root / "delete.idx"
        entries = [(i % 17, (i, 0)) for i in range(600)]
        with BPlusTree(path, page_size=128, create=True) as tree:
            for entry in entries:
                tree.insert(*entry)
            deleted = entries[::2]
            for entry in deleted:
                self.assertTrue(tree.delete(*entry))
                self.assertFalse(tree.delete(*entry))
            self.assertEqual(list(tree.iter_entries()), sorted(entries[1::2]))
            for entry in entries[1::2]:
                self.assertTrue(tree.delete(*entry))
            self.assertEqual(tree.range_search(), [])
            self.assertEqual(tree.stats()["size"], 0)
            for entry in reversed(entries):
                tree.insert(*entry)
            self.assertEqual(list(tree.iter_entries()), sorted(entries))
        with BPlusTree(path, page_size=128) as tree:
            self.assertEqual(list(tree.iter_entries()), sorted(entries))

    def test_point_queries_read_height_not_entire_tree(self):
        counter = DiskCounter()
        with BPlusTree(self.root / "io.idx", page_size=256, counter=counter, create=True) as tree:
            for i in range(3000):
                tree.insert(i, (i, 0))
            counter.reset()
            self.assertEqual(tree.search(1234), [(1234, 0)])
            reads = counter.snapshot()["disk_reads"]
            self.assertLessEqual(reads, tree.stats()["height"] + 1)
            self.assertGreaterEqual(reads, tree.stats()["height"])
            self.assertEqual(counter.snapshot()["disk_writes"], 0)
            self.assertEqual(counter.snapshot()["by_role"]["index"]["disk_reads"], reads)

    def test_extendible_hash_splits_paged_directory_and_reopen(self):
        path = self.root / "hash.idx"
        entries = [(i % 607, (i, 0)) for i in range(2000)]
        random.Random(87).shuffle(entries)
        expected = {}
        counter = DiskCounter()
        with ExtendibleHash(path, page_size=128, counter=counter, create=True) as index:
            for key, rid in entries:
                index.insert(key, rid)
                expected.setdefault(key, []).append(rid)
            self.assertGreater(index.stats()["global_depth"], 1)
            self.assertGreater(index.stats()["directory_pages"], 1)
            self.assertEqual(index.stats()["size"], len(entries))
            for key, rids in expected.items():
                self.assertEqual(sorted(index.search(key)), sorted(rids))
            self.assertEqual(index.search(-1), [])
        with ExtendibleHash(path, page_size=128, counter=counter) as index:
            for key, rids in expected.items():
                self.assertEqual(sorted(index.search(key)), sorted(rids))
            counter.reset()
            self.assertEqual(len(index.search(302)), len(expected[302]))
            self.assertEqual(counter.snapshot()["disk_reads"], 2)
            self.assertEqual(counter.snapshot()["disk_writes"], 0)
        self.assertEqual(path.stat().st_size % 128, 0)

    def test_hash_identical_hash_overflow_and_subsequent_split(self):
        path = self.root / "overflow.idx"
        counter = DiskCounter()
        with ExtendibleHash(path, page_size=128, counter=counter, create=True) as index:
            for i in range(800):
                index.insert(42, (i, 0))
            self.assertEqual(index.stats()["global_depth"], 0)
            self.assertEqual(sorted(index.search(42)), [(i, 0) for i in range(800)])
            counter.reset()
            index.insert(42, (800, 0))
            self.assertLessEqual(counter.snapshot()["disk_reads"], 3)
            for key in range(50):
                if key != 42:
                    index.insert(key, (1000 + key, 0))
            self.assertGreater(index.stats()["global_depth"], 0)
            self.assertEqual(sorted(index.search(42)), [(i, 0) for i in range(801)])
            for key in range(50):
                if key != 42:
                    self.assertEqual(index.search(key), [(1000 + key, 0)])
            for i in range(801):
                self.assertTrue(index.delete(42, (i, 0)))
            self.assertEqual(index.search(42), [])
            self.assertFalse(index.delete(42, (0, 0)))
            index.insert(42, (9000, 0))
        with ExtendibleHash(path, page_size=128) as index:
            self.assertEqual(index.search(42), [(9000, 0)])
            self.assertEqual(index.size, 50)

    def test_hash_collision_keeps_different_keys(self):
        with ExtendibleHash(self.root / "collision.idx", page_size=128, create=True) as index:
            index._hash = lambda key: 19
            for i in range(100):
                index.insert(i % 9, (i, 0))
            for key in range(9):
                self.assertEqual(index.search(key), [(i, 0) for i in range(100) if i % 9 == key])
            self.assertEqual(index.global_depth, 0)

    def test_character_float_keys_and_duplicate_pairs(self):
        for implementation in (BPlusTree, ExtendibleHash):
            for key_type, width, values in (
                ("CHAR", 18, ["Ana", "á", "", "z", "a\0b", "á", "z"]),
                ("FLOAT", 8, [3.5, -7.1, 0.0, -0.0, 2.25, 3.5, 1e100]),
            ):
                with self.subTest(index=implementation.__name__, key_type=key_type):
                    path = self.root / (implementation.__name__ + key_type)
                    entries = [(key, (i, 0)) for i, key in enumerate(values)]
                    with implementation(path, key_type=key_type, key_size=width, page_size=256, create=True) as index:
                        for entry in entries:
                            index.insert(*entry)
                        index.insert(values[0], (0, 0))
                        for value in values:
                            expected = [rid for key, rid in entries if key == value]
                            if value == values[0]:
                                expected.append((0, 0))
                            self.assertEqual(sorted(index.search(value)), sorted(expected))
                        self.assertTrue(index.delete(values[0], (0, 0)))
                    with implementation(path, key_type=key_type, key_size=width, page_size=256) as index:
                        for value in values:
                            self.assertEqual(sorted(index.search(value)), sorted(rid for key, rid in entries if key == value))
                        if key_type == "FLOAT":
                            with self.assertRaises(ValueError):
                                index.insert(float("nan"), (0, 0))
                        else:
                            with self.assertRaises(ValueError):
                                index.insert("a" * 19, (0, 0))

    def test_signed_overflow_rids(self):
        for implementation in (BPlusTree, ExtendibleHash):
            with implementation(self.root / implementation.__name__, page_size=128, create=True) as index:
                rids = [(-7, i) for i in range(50)] + [(7, 4), (-(1 << 31), 0)]
                for rid in rids:
                    index.insert(100, rid)
                self.assertEqual(sorted(index.search(100)), sorted(rids))
                self.assertTrue(index.delete(100, (-7, 4)))
                rids.remove((-7, 4))
                self.assertEqual(sorted(index.search(100)), sorted(rids))

    def test_persisted_metadata_rejects_wrong_configuration(self):
        path = self.root / "metadata.idx"
        with BPlusTree(path, create=True) as index:
            index.insert(10, (1, 2))
        with self.assertRaises(ValueError):
            ExtendibleHash(path)
        with self.assertRaises(ValueError):
            BPlusTree(path, key_type="FLOAT")
        with BPlusTree(path) as index:
            raw = index.file.read(index.root_page_id)
            fields = PAGE_HEADER.unpack_from(raw)
            self.assertEqual(fields[1], index.root_page_id)
            self.assertEqual(fields[2], 1)
            self.assertEqual(fields[3], PAGE_HEADER.size + index.entry_size)
            self.assertEqual(fields[4:6], (-1, -1))


if __name__ == "__main__":
    unittest.main()
