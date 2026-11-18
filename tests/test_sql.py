"""Gramática SQL, literales y precedencia de condiciones."""
import unittest

from backend.minidb.sql import conjuncts, matches, parse_sql


class SQLTests(unittest.TestCase):
    def test_create_types_and_index(self):
        statement = parse_sql('CREATE TABLE Employees (id INT PRIMARY KEY, name CHAR(30), salary FLOAT) USING SEQUENTIAL;')
        self.assertEqual(statement['table'], 'employees')
        self.assertEqual(statement['organization'], 'SEQUENTIAL')
        self.assertEqual(statement['columns'][0], {'name': 'id', 'type': 'INT', 'primary': True})
        self.assertEqual(statement['columns'][1]['size'], 30)
        self.assertEqual(parse_sql('CREATE INDEX by_id ON Employees (id) USING HASH')['kind'], 'HASH')

    def test_escaped_text_signed_numbers_and_comments(self):
        statement = parse_sql("-- encabezado\nINSERT INTO t VALUES (-2, 'O''Brien; --texto', +1.25e2); -- final")
        self.assertEqual(statement['values'], [-2, "O'Brien; --texto", 125.0])

    def test_and_precedes_or_and_parentheses_override(self):
        condition = parse_sql('SELECT * FROM t WHERE a = 1 OR a = 2 AND b > 5')['where']
        self.assertTrue(matches({'a': 1, 'b': 0}, condition))
        self.assertFalse(matches({'a': 2, 'b': 0}, condition))
        grouped = parse_sql('SELECT * FROM t WHERE (a = 1 OR a = 2) AND b > 5')['where']
        self.assertFalse(matches({'a': 1, 'b': 0}, grouped))
        self.assertEqual(conjuncts(condition), [])
        self.assertEqual(conjuncts(grouped), [('>', 'b', 5)])

    def test_order_projection_limit_and_delete(self):
        query = parse_sql('SELECT id, label FROM t ORDER BY id DESC LIMIT 12;')
        self.assertEqual(query['columns'], ['id', 'label'])
        self.assertEqual(query['order'], ('id', True))
        self.assertEqual(query['limit'], 12)
        self.assertEqual(parse_sql('SELECT * FROM t ORDER BY id ASC')['order'], ('id', False))
        self.assertEqual(parse_sql('DELETE FROM t WHERE id != 3')['where'], ('!=', 'id', 3))

    def test_malformed_and_multiple_statements_are_rejected(self):
        for sql in ("'SELECT' * FROM t", "SELECT * FROM t WHERE id '=' 1",
                    'SELECT * FROM t ORDER BY id DESC ASC', 'SELECT * FROM t LIMIT -1',
                    "SELECT * FROM t LIMIT '2'", 'SELECT * FROM t LIMIT 2.5',
                    'SELECT * FROM t; DELETE FROM t;', 'INSERT INTO t VALUES (1,)',
                    'SELECT * FROM t WHERE (id = 3', 'SELECT * FROM t WHERE id ~~ 3',
                    'DROP TABLE t', 'SELECT * FROM t WHERE id = NULL'):
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                parse_sql(sql)


if __name__ == '__main__':
    unittest.main()
