"""Pruebas de persistencia, transferencias físicas, reutilización y overflow."""
import random
import tempfile
import unittest
from pathlib import Path

from backend.minidb.io import DiskCounter, DiskFile
from backend.minidb.storage import HEADER_SIZE, PAGE_HEADER, HeapFile, Schema, SequentialFile


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        self.schema = Schema([{"name": "id", "type": "INT", "primary": True},
                              {"name": "name", "type": "CHAR(12)"},
                              {"name": "score", "type": "FLOAT"}])

    def tearDown(self):
        self.temporary.cleanup()

    def row(self, number):
        return {"id": number, "name": f"fila{number}", "score": number / 2}

    def test_disk_counter_includes_roles_and_exact_pages(self):
        counter = DiskCounter()
        with DiskFile(self.path / "disk", 128, counter) as disk:
            self.assertEqual(disk.allocate(), 0)
            self.assertEqual(disk.allocate(b"a" * 128), 1)
            self.assertEqual(disk.read(1), b"a" * 128)
            disk.write(0, b"b" * 128)
            self.assertEqual(disk.page_count, 2)
            with self.assertRaises(ValueError):
                disk.write(0, b"x")
            with self.assertRaises(EOFError):
                disk.read(9)
        with DiskFile(self.path / "index", 128, counter, "index") as disk:
            disk.allocate()
            disk.read(0)
        self.assertEqual(counter.disk_reads, 2)
        self.assertEqual(counter.disk_writes, 4)
        snapshot = counter.snapshot()
        self.assertEqual(snapshot["total"], 6)
        self.assertEqual(snapshot["by_role"]["index"], {"disk_reads": 1, "disk_writes": 1})
        counter.reset()
        self.assertEqual(counter.snapshot()["total"], 0)

    def test_schema_unicode_bounds_and_types(self):
        row = {"id": -5, "name": "Perú", "score": 3.75}
        self.assertEqual(self.schema.decode(self.schema.encode(row)), row)
        for invalid in ({"id": 1, "name": "x" * 13, "score": 1.0},
                        {"id": 1.2, "name": "x", "score": 1.0},
                        {"id": 1, "name": "x", "score": float("nan")},
                        {"id": 1, "name": "x\x00", "score": 1.0}):
            with self.assertRaises(ValueError):
                self.schema.encode(invalid)

    def test_heap_free_list_persistence_and_scan_io(self):
        path = self.path / "heap"
        counter = DiskCounter()
        heap = HeapFile(path, self.schema, counter, page_size=512, create=True)
        capacity = heap._store.capacity
        rids = [heap.insert(self.row(i)) for i in range(capacity * 3)]
        pages = heap.page_count
        self.assertEqual(heap._store.free_head, -1)
        for index in (0, capacity + 3, capacity * 2 + 2):
            self.assertTrue(heap.delete(rids[index]))
            self.assertFalse(heap.delete(rids[index]))
        heap.close()
        heap = HeapFile(path, self.schema, counter, page_size=512)
        self.assertEqual(heap.record_count, capacity * 3 - 3)
        inserted = {heap.insert(self.row(1000 + i)) for i in range(3)}
        self.assertEqual(inserted, {rids[i] for i in (0, capacity + 3, capacity * 2 + 2)})
        self.assertEqual(heap.page_count, pages)
        counter.reset()
        self.assertEqual(len(list(heap.scan())), capacity * 3)
        self.assertEqual(counter.disk_reads, pages - 1)
        self.assertEqual(counter.disk_writes, 0)
        counter.reset()
        self.assertEqual(len(list(heap.scan_equal("id", 1001))), 1)
        self.assertEqual(counter.disk_reads, pages - 1)
        page = heap.file.read(1)
        self.assertEqual(PAGE_HEADER.unpack_from(page)[:2], (1, capacity))
        self.assertGreater(PAGE_HEADER.unpack_from(page)[2], HEADER_SIZE)
        heap.close()

    def test_heap_does_not_cache_pages_between_gets(self):
        counter = DiskCounter()
        with HeapFile(self.path / "heap", self.schema, counter) as heap:
            rid = heap.insert(self.row(2))
            counter.reset()
            self.assertEqual(heap.get(rid), self.row(2))
            self.assertEqual(heap.get(rid), self.row(2))
            self.assertEqual(counter.disk_reads, 2)

    def test_sequential_random_overflow_delete_reopen_reorganize(self):
        path = self.path / "sequential"
        keys = list(range(600))
        random.Random(222).shuffle(keys)
        seq = SequentialFile(path, self.schema, page_size=512, create=True)
        rids = {key: seq.insert(self.row(key)) for key in keys}
        self.assertGreater(seq.overflow_count, 0)
        self.assertTrue(any(rid[0] < 0 for rid in rids.values()))
        for key in keys:
            self.assertEqual(seq.get(rids[key]), self.row(key))
        for key in range(0, 600, 3):
            self.assertTrue(seq.delete(rids[key]))
            self.assertIsNone(seq.get(rids[key]))
        seq.close()
        seq = SequentialFile(path, self.schema, page_size=512)
        self.assertEqual(seq.record_count, 400)
        self.assertEqual([row["id"] for _, row in seq.range_search(50, 100)],
                         [key for key in range(50, 101) if key % 3])
        for key in (1, 55, 599):
            self.assertEqual(seq.search(key)[0][1], self.row(key))
        self.assertEqual(seq.search(3), [])
        seq.reorganize(.75)
        self.assertEqual(seq.overflow_count, 0)
        self.assertEqual(seq._overflow.file.page_count, 1)
        self.assertEqual([row["id"] for _, row in seq.range_search(-1, 900)],
                         [key for key in range(600) if key % 3])
        for page_id in range(1, seq.file.page_count - 1):
            count = PAGE_HEADER.unpack_from(seq.file.read(page_id))[1]
            self.assertEqual(count, int(seq._main.capacity * .75))
        for key in range(0, 600, 3):
            seq.insert(self.row(key))
        self.assertEqual([row["id"] for _, row in seq.range_search(0, 599)], list(range(600)))
        seq.close()

    def test_sequential_empty_main_page_keeps_binary_search_fence(self):
        with SequentialFile(self.path / "seq", self.schema, page_size=512) as seq:
            seq.bulk_load([self.row(i) for i in range(90)])
            selected = [(rid, row) for rid, row in seq.scan() if rid[0] == 2]
            for rid, _ in selected:
                seq.delete(rid)
            remaining = sorted(row["id"] for _, row in seq.scan())
            self.assertEqual([row["id"] for _, row in seq.range_search(-1, 99)], remaining)
            for _, row in selected:
                seq.insert(row)
            self.assertEqual([row["id"] for _, row in seq.range_search(-1, 99)], list(range(90)))

    def test_sequential_duplicates_and_overflow_slot_reuse(self):
        with SequentialFile(self.path / "seq", self.schema, page_size=512) as seq:
            rids = [seq.insert(self.row(7)) for _ in range(100)]
            self.assertEqual(len(seq.search(7)), 100)
            pages = seq.page_count
            for rid in rids[::2]:
                self.assertTrue(seq.delete(rid))
            self.assertEqual(len(seq.search(7)), 50)
            for _ in range(50):
                seq.insert(self.row(7))
            self.assertEqual(seq.page_count, pages)
            self.assertEqual(len(seq.search(7)), 100)
            seq.reorganize()
            self.assertEqual(len(seq.search(7)), 100)
            self.assertEqual(seq.overflow_count, 0)

    def test_sequential_char_key_and_empty_reorganization(self):
        schema = Schema([{"name": "code", "type": "CHAR(8)", "primary": True}, {"name": "n", "type": "INT"}])
        with SequentialFile(self.path / "chars", schema, page_size=512) as seq:
            seq.reorganize()
            keys = [f"a{i:03d}" for i in range(100)]
            for key in reversed(keys):
                seq.insert({"code": key, "n": int(key[1:])})
            self.assertEqual([row["code"] for _, row in seq.range_search("a010", "a020")], keys[10:21])
            seq.reorganize()
            self.assertEqual(seq.search("a099")[0][1]["n"], 99)

    def test_wrong_schema_and_storage_format_rejected(self):
        path = self.path / "seq"
        with SequentialFile(path, self.schema):
            pass
        with self.assertRaises(ValueError):
            HeapFile(path, self.schema)
        with self.assertRaises(FileExistsError):
            SequentialFile(path, self.schema, create=True)


if __name__ == "__main__":
    unittest.main()
