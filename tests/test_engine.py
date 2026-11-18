"""Integración SQL, planes, almacenamiento y persistencia de índices."""
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.minidb.engine import Database
from backend.minidb.indexes import BPlusTree


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        self.db = Database(self.path, page_size=512)

    def tearDown(self):
        self.db.close()
        self.temporary.cleanup()

    def create(self, organization='HEAP'):
        self.db.execute(f'CREATE TABLE items (id INT PRIMARY KEY, name CHAR(12), score FLOAT) USING {organization};')

    def fill(self, count=30):
        for value in range(count):
            self.db.execute(f"INSERT INTO items VALUES ({value}, 'item{value}', {value + .5});")

    def values(self, sql):
        return [row['id'] for row in self.db.execute(sql, page_size=1000)['rows']]

    def test_sql_scan_projection_order_limit_pagination_and_io(self):
        self.create()
        self.fill()
        result = self.db.execute('SELECT id, name FROM items WHERE id >= 4 AND id < 24 ORDER BY id DESC LIMIT 9;', page=2, page_size=4)
        self.assertEqual([row['id'] for row in result['rows']], [19, 18, 17, 16])
        self.assertEqual(set(result['rows'][0]), {'id', 'name'})
        self.assertEqual(result['total'], 9)
        self.assertEqual(result['plan']['operation'], 'SeqScan')
        self.assertEqual(result['metrics']['disk_writes'], 0)
        self.assertEqual(result['metrics']['disk_reads'], self.db.tables['items'].page_count - 1)
        self.assertEqual(self.db.execute('SELECT * FROM items LIMIT 0')['total'], 0)
        for page, size in ((0, 25), (1, 0), (1.5, 2), (True, 2)):
            with self.assertRaises(ValueError):
                self.db.execute('SELECT * FROM items', page=page, page_size=size)

    def test_index_selection_ranges_delete_and_reinsert(self):
        self.create()
        self.fill(80)
        self.db.execute('CREATE INDEX id_tree ON items (id) USING BTREE')
        self.db.execute('CREATE INDEX id_hash ON items (id) USING HASH')
        equal = self.db.execute('SELECT * FROM items WHERE id = 17')
        self.assertEqual(equal['plan']['index'], 'id_hash')
        self.assertEqual(equal['rows'][0]['id'], 17)
        self.assertGreater(equal['metrics']['by_role']['index']['disk_reads'], 0)
        ranged = self.db.execute('SELECT * FROM items WHERE id > 10 AND id <= 15 ORDER BY id')
        self.assertEqual(ranged['plan']['operation'], 'IndexRangeScan')
        self.assertEqual(ranged['plan']['index'], 'id_tree')
        self.assertEqual([row['id'] for row in ranged['rows']], [11, 12, 13, 14, 15])
        deleted = self.db.execute('DELETE FROM items WHERE id >= 10 AND id <= 15')
        self.assertEqual(deleted['affected'], 6)
        self.assertEqual(self.values('SELECT * FROM items WHERE id = 12'), [])
        self.db.execute("INSERT INTO items VALUES (12, 'again', 1.0)")
        self.assertEqual(self.values('SELECT * FROM items WHERE id = 12'), [12])
        self.assertEqual(len(self.db.indexes['id_tree'].search(12)), 1)
        self.assertEqual(len(self.db.indexes['id_hash'].search(12)), 1)

    def test_sequential_overflow_signed_rids_reorganize_and_reopen(self):
        self.create('SEQUENTIAL')
        self.db.execute('CREATE INDEX id_tree ON items (id) USING BTREE')
        self.db.execute('CREATE INDEX id_hash ON items (id) USING HASH')
        values = list(range(100))
        random.Random(912).shuffle(values)
        for value in values:
            self.db.insert('items', dict(id=value, name=str(value), score=float(value)))
        self.assertGreater(self.db.tables['items'].overflow_count, 0)
        overflow_rid, overflow_row = next((rid, row) for rid, row in self.db.tables['items'].scan() if rid[0] < 0)
        self.assertIn(overflow_rid, self.db.indexes['id_tree'].search(overflow_row['id']))
        self.assertIn(overflow_rid, self.db.indexes['id_hash'].search(overflow_row['id']))
        with self.assertRaises(ValueError):
            self.db.insert('items', overflow_row)
        self.db.execute('DELETE FROM items WHERE id = 45')
        self.db.reorganize('items')
        self.assertEqual(self.db.tables['items'].overflow_count, 0)
        self.db.close()
        self.db = Database(self.path, page_size=512)
        self.assertEqual(self.values('SELECT * FROM items WHERE id >= 40 AND id < 50 ORDER BY id'), [40, 41, 42, 43, 44, 46, 47, 48, 49])
        self.assertEqual(self.values('SELECT * FROM items WHERE id = 45'), [])
        self.db.execute("INSERT INTO items VALUES (45, 'restored', 45)")
        self.assertEqual(self.values('SELECT * FROM items WHERE id = 45'), [45])
        with self.assertRaises(ValueError):
            self.db.execute("INSERT INTO items VALUES (45, 'duplicate', 5)")

    def test_fractional_integer_predicates_keep_index_semantics(self):
        self.create()
        self.fill(10)
        queries = [
            'id = 3.0', 'id = 3.5', 'id > 3.5', 'id >= 3.5', 'id < 3.5', 'id <= 3.5',
            'id > 3 AND id < 4', 'id >= 3 AND id <= 3', 'id > 4 AND id <= 4',
            'id > -999999999999999999999', 'id < 999999999999999999999',
            'id > 999999999999999999999', 'id < -999999999999999999999',
            'id = 999999999999999999999',
        ]
        baseline = {q: self.values(f'SELECT * FROM items WHERE {q} ORDER BY id') for q in queries}
        self.db.execute('CREATE INDEX id_tree ON items (id) USING BTREE')
        for query, expected in baseline.items():
            with self.subTest(query=query):
                self.assertEqual(self.values(f'SELECT * FROM items WHERE {query} ORDER BY id'), expected)
        self.db.execute('CREATE INDEX id_hash ON items (id) USING HASH')
        self.assertEqual(self.values('SELECT * FROM items WHERE id = 3.0'), [3])
        self.assertEqual(self.values('SELECT * FROM items WHERE id = 3.5'), [])

    def test_float_integer_precision_and_char_bounds(self):
        self.create()
        self.db.execute("INSERT INTO items VALUES (1, 'a', 9007199254740992)")
        self.db.execute("INSERT INTO items VALUES (2, 'z', 9007199254740994)")
        self.db.execute('CREATE INDEX score_tree ON items (score) USING BTREE')
        self.db.execute('CREATE INDEX name_tree ON items (name) USING BTREE')
        self.assertEqual(self.values('SELECT * FROM items WHERE score < 9007199254740993'), [1])
        self.assertEqual(self.values('SELECT * FROM items WHERE score > 9007199254740993'), [2])
        self.assertEqual(self.values('SELECT * FROM items WHERE score = 9007199254740993'), [])
        self.assertEqual(self.values("SELECT * FROM items WHERE name < 'zzzzzzzzzzzzzzzzzz' ORDER BY id"), [1, 2])
        self.assertEqual(self.values("SELECT * FROM items WHERE name = 'zzzzzzzzzzzzzzzzzz'"), [])

    def test_or_predicates_do_not_incorrectly_select_one_index(self):
        self.create()
        self.fill(20)
        self.db.execute('CREATE INDEX id_tree ON items (id) USING BTREE')
        self.assertEqual(self.values('SELECT * FROM items WHERE id = 2 OR id = 15 ORDER BY id'), [2, 15])
        self.assertEqual(self.values('SELECT * FROM items WHERE (id = 2 OR id = 15) AND score > 10 ORDER BY id'), [15])
        self.assertEqual(self.values('SELECT * FROM items WHERE id >= 1 AND (id = 2 OR id = 15) ORDER BY id'), [2, 15])

    def test_invalid_types_names_and_duplicate_primary_do_not_change_rows(self):
        self.create()
        self.fill(1)
        bad = ["INSERT INTO items VALUES (0, 'duplicate', 2)",
               "INSERT INTO items VALUES (2.5, 'bad', 2)",
               "INSERT INTO items VALUES (2, 'much-too-long-for-column', 2)",
               "INSERT INTO items VALUES (2, 'bad', 'text')",
               "SELECT * FROM items WHERE id = 'text'",
               'SELECT * FROM items WHERE score = 1e999',
               'SELECT missing FROM items', 'SELECT * FROM items ORDER BY missing']
        for sql in bad:
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                self.db.execute(sql)
        self.assertEqual(self.db.tables['items'].record_count, 1)
        for size in ('1.5', "'12'", '0'):
            with self.assertRaises(ValueError):
                self.db.execute(f'CREATE TABLE bad (name CHAR({size}))')
        with self.assertRaises(ValueError):
            self.db.create_table('../outside', [{'name': 'id', 'type': 'INT'}])

    def test_failed_create_rolls_back_catalog_and_new_files(self):
        with patch.object(self.db, '_save_catalog', side_effect=OSError('fallo simulado')):
            with self.assertRaises(OSError):
                self.create('SEQUENTIAL')
        self.assertNotIn('items', self.db.catalog)
        self.assertNotIn('items', self.db.tables)
        self.assertFalse((self.path / 'items.tbl').exists())
        self.assertFalse((self.path / 'items.tbl.overflow').exists())
        self.create()
        self.fill(3)
        with patch.object(BPlusTree, 'insert', side_effect=ValueError('fallo simulado')):
            with self.assertRaises(ValueError):
                self.db.create_index('items', 'broken', 'id', 'BTREE')
        self.assertNotIn('broken', self.db.indexes)
        self.assertEqual(self.db.catalog['items']['indexes'], [])
        self.assertFalse((self.path / 'broken.idx').exists())
        self.assertFalse((self.path / 'broken.building').exists())
        self.db.create_index('items', 'broken', 'id', 'BTREE')
        self.assertEqual(self.values('SELECT * FROM items WHERE id = 1'), [1])

    def test_catalog_spans_pages_and_reopens(self):
        for number in range(12):
            self.db.execute(f'CREATE TABLE table_{number} (id INT PRIMARY KEY, label CHAR(16))')
        self.assertGreater((self.path / 'catalog.bin').stat().st_size, 512)
        self.db.close()
        self.db = Database(self.path, page_size=512)
        self.assertEqual(len(self.db.describe()), 12)
        self.db.execute("INSERT INTO table_11 VALUES (1, 'persisted')")
        self.assertEqual(self.db.execute('SELECT * FROM table_11')['rows'][0]['label'], 'persisted')


if __name__ == '__main__':
    unittest.main()
