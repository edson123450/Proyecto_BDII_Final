"""Contratos que evitan resultados inventados, sesgo de muestras y reanudación corrupta."""

import tempfile
import unittest
from pathlib import Path

from benchmarks.parte1 import (FULL_NS, PAGE_SIZES, SELECTIVITIES, atomic_json,
                               dataset_row, experiment_config, insertion_order,
                               read_json, summarize_queries, summary_statistics)


class BenchmarkContractTests(unittest.TestCase):
    def test_full_configuration_matches_statement(self):
        cfg = experiment_config("full", 42)
        self.assertEqual(cfg["insert_ns"], [1000, 10000, 50000, 100000, 250000, 500000])
        self.assertEqual(cfg["query_n"], 100000)
        self.assertEqual(cfg["point_queries"], 1000)
        self.assertEqual(cfg["page_sizes"], [1024, 2048, 4096, 8192])
        self.assertEqual(cfg["selectivities"], [.001, .01, .05, .1, .25])
        self.assertGreaterEqual(cfg["range_queries"], 2)

    def test_dataset_reproduces_without_duplicates(self):
        order = insertion_order(2000, 42)
        self.assertEqual(order, insertion_order(2000, 42))
        self.assertNotEqual(order, insertion_order(2000, 43))
        self.assertEqual(set(order), set(range(1, 2001)))
        for key in (1, 100000, 500000):
            row = dataset_row(key)
            self.assertEqual(row, dataset_row(key))
            self.assertEqual(row["id"], key)
            self.assertLessEqual(len(row["nombre"].encode()), 30)
            self.assertLessEqual(len(row["dept"].encode()), 20)

    def test_statistics_use_observations_and_sample_deviation(self):
        result = summary_statistics([1, 2, 3])
        self.assertEqual(result["mean"], 2)
        self.assertEqual(result["stdev"], 1)
        with self.assertRaises(ValueError):
            summary_statistics([])

    def test_groups_do_not_mix_methods_or_selectivities(self):
        records = []
        for method, sel, read in [("heap", .01, 10), ("heap", .01, 20), ("btree", .01, 2), ("btree", .1, 8)]:
            records.append({"method": method, "selectivity": sel, "disk_reads": read,
                            "disk_writes": 0, "index_reads": 0, "data_reads": read,
                            "latency_ms": read / 10, "result_count": 1})
        summarized = summarize_queries(records, ("method", "selectivity"))
        self.assertEqual(len(summarized), 3)
        heap = next(r for r in summarized if r["method"] == "heap")
        self.assertEqual(heap["queries"], 2)
        self.assertEqual(heap["disk_reads_mean"], 15)

    def test_checkpoint_round_trip_keeps_real_precision_and_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "checkpoint.json"
            content = [{"latency_ms": .123456789, "method": "Árbol B+", "disk_reads": 17}]
            atomic_json(path, content)
            self.assertEqual(read_json(path), content)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
