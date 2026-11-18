import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from backend.minidb.io import DiskCounter
from backend.minidb.text import InvertedIndex, TextNormalizer, merge_postings


class TextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'text'

    def tearDown(self):
        self.temp.cleanup()

    def test_normalization_boolean_and_reopen(self):
        docs = [(1, 'The cats are running. Música clásica'), (2, 'Cats sing jazz'),
                (3, 'Rock and jazz music'), (4, 'La música clásica es bella'), (5, '')]
        index = InvertedIndex(self.path, create=True)
        index.build(docs)
        self.assertEqual(index.boolean('cats AND jazz'), [2])
        self.assertEqual(index.boolean('cats OR rock'), [1, 2, 3])
        self.assertEqual(index.boolean('(cats OR rock) AND-NOT jazz'), [1])
        self.assertEqual(index.boolean('cats AND NOT jazz'), [1])
        self.assertEqual(index.boolean('cats jazz'), [2])
        self.assertEqual(index.boolean('música AND clásica'), [1, 4])
        self.assertEqual(index.boolean('ghost AND cats'), [])
        self.assertEqual(index.gettext(1), docs[0][1])
        self.assertEqual(list(index.iter_docs()), docs)
        self.assertIsNone(index.gettext(999))
        expected = index.query('cats music', k=5)
        self.assertTrue(expected)
        index.close()
        reopened = InvertedIndex(self.path)
        self.assertEqual(reopened.query('cats music', k=5), expected)
        self.assertEqual(reopened.boolean('cats'), [1, 2])
        self.assertEqual(list(reopened.iter_docs()), docs)
        for path in self.path.iterdir():
            self.assertEqual(path.stat().st_size % 4096, 0)
        reopened.close()

    def test_cosine_matches_direct_formula_and_acoustic_counts(self):
        docs = [(30, Counter(aw001=3, aw002=1)), (10, Counter(aw001=1, aw003=5)),
                (20, Counter(aw002=2, aw003=1)), (40, Counter(aw004=7))]
        query = Counter(aw001=2, aw002=1)
        index = InvertedIndex(self.path, create=True)
        index.build_tokens(docs)
        df = Counter(term for _, counts in docs for term in counts)
        def weights(counts):
            return {t: (1 + math.log10(tf)) * math.log10(len(docs)/df[t]) for t, tf in counts.items() if t in df}
        def cosine(counts):
            q, d = weights(query), weights(counts)
            numerator = sum(w*d.get(t, 0) for t, w in q.items())
            return numerator / math.sqrt(sum(w*w for w in q.values()) * sum(w*w for w in d.values()))
        expected = sorted([(docid, cosine(c)) for docid, c in docs if set(c) & set(query)], key=lambda row: (-row[1], row[0]))
        actual = index.query_tokens(query, k=10)
        self.assertEqual([x[0] for x in actual], [x[0] for x in expected])
        for a,b in zip(actual,expected):
            self.assertAlmostEqual(a[1], b[1], places=14)
        self.assertEqual(index.get_postings('aw001'), [(10, 1), (30, 3)])
        self.assertEqual(index.query_tokens(['missing']), [])
        index.close()

    def test_many_pages_binary_vocabulary_and_sorted_postings(self):
        counter = DiskCounter()
        index = InvertedIndex(self.path, counter=counter, page_size=512, create=True)
        index.build_tokens((i, [f'term{i:05d}', 'common']) for i in reversed(range(2000)))
        counter.reset()
        self.assertEqual(index.get_postings('term00777'), [(777, 1)])
        self.assertLessEqual(counter.disk_reads, 14)
        self.assertEqual(index.get_postings('common'), [(i, 1) for i in range(2000)])
        self.assertEqual(index.query_tokens(['common']), [])  # idf=0
        self.assertEqual(index.query_tokens(['term00777']), [(777, 1.0)])
        index.close()
        reopened = InvertedIndex(self.path, page_size=512)
        self.assertEqual(reopened.get_postings('term01999'), [(1999, 1)])
        reopened.close()

    def test_merge_semantics_and_zero_norm_documents(self):
        self.assertEqual(merge_postings([1, 3, 5], [2, 3, 6], 'AND'), [3])
        self.assertEqual(merge_postings([1, 3, 5], [2, 3, 6], 'OR'), [1, 2, 3, 5, 6])
        self.assertEqual(merge_postings([1, 3, 5], [2, 3, 6], 'AND-NOT'), [1, 5])
        index = InvertedIndex(self.path, create=True)
        index.build_tokens([(1, []), (2, ['unique']), (3, ['unique'])])
        self.assertEqual(index.query_tokens(['unique']), [(2, 1.0), (3, 1.0)])
        for query in ['AND cats', 'cats OR', '(cats', 'cats )']:
            with self.assertRaises(ValueError):
                index.boolean(query)
        index.close()

    def test_reject_invalid_document_and_frequency(self):
        index = InvertedIndex(self.path, create=True)
        with self.assertRaises(ValueError):
            index.build_tokens([(1, ['rock']), (1, ['jazz'])])
        with self.assertRaises(ValueError):
            index.build_tokens([(1, {'rock': 0})])
        with self.assertRaises(ValueError):
            index.build_tokens([(-1, ['rock'])])
        index.close()


if __name__ == '__main__':
    unittest.main()
