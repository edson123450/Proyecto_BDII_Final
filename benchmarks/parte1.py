#!/usr/bin/env python3
"""Experimentos P1 reales, deterministas y reanudables sobre archivos binarios.

Uso (desde la raíz):
    .venv/bin/python benchmarks/parte1.py --scale smoke
    .venv/bin/python benchmarks/parte1.py --scale full

E1 mide una serie acumulada desde cero: copiar el último estado confirmado
permite reanudar sin volver a medir un prefijo. Las copias no se cronometran ni
se contabilizan como I/O del motor. E2/E3 incluyen materialización de las
tuplas del heap para los índices. E4 aísla el índice para observar su estructura.
No se altera el contador, no se extrapolan resultados y no se vacía la caché
del sistema operativo: I/O significa transferencias realizadas por el gestor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FULL_NS = (1_000, 10_000, 50_000, 100_000, 250_000, 500_000)
SELECTIVITIES = (0.001, 0.01, 0.05, 0.1, 0.25)
PAGE_SIZES = (1_024, 2_048, 4_096, 8_192)
INSERT_METHODS = ("heap", "sequential_no_reorg", "sequential_reorg", "btree", "hash")
POINT_METHODS = ("heap", "sequential", "btree", "hash")
RANGE_METHODS = ("heap", "sequential", "btree")
LABELS = {
    "heap": "Heap / Full scan", "sequential": "Sequential ordenado",
    "sequential_no_reorg": "Sequential sin reorganizar",
    "sequential_reorg": "Sequential con reorganización",
    "btree": "Árbol B+", "hash": "Hash extensible",
}
COLORS = {"heap": "#4477AA", "sequential": "#228833",
          "sequential_no_reorg": "#66CCEE", "sequential_reorg": "#228833",
          "btree": "#CC6677", "hash": "#AA3377"}
COLUMNS = [
    {"name": "id", "type": "INT", "primary": True},
    {"name": "nombre", "type": "CHAR", "size": 30},
    {"name": "dept", "type": "CHAR", "size": 20},
    {"name": "salario", "type": "FLOAT"},
]
DEPARTMENTS = ("Ventas", "Ingenieria", "Finanzas", "Operaciones", "Marketing", "RRHH", "Legal", "Soporte")


def dataset_row(key: int) -> dict[str, Any]:
    """Una misma clave siempre produce exactamente la misma tupla."""
    return {"id": key, "nombre": f"Empleado {key:06d}",
            "dept": DEPARTMENTS[key % len(DEPARTMENTS)],
            "salario": round(1500 + ((key * 7919) % 1_850_000) / 100, 2)}


def insertion_order(n: int, seed: int) -> list[int]:
    keys = list(range(1, n + 1))
    random.Random(seed).shuffle(keys)
    return keys


def experiment_config(scale: str, seed: int, range_queries: int | None = None) -> dict[str, Any]:
    if scale not in {"smoke", "full"}:
        raise ValueError("scale debe ser smoke o full")
    return {"format_version": 1, "scale": scale, "seed": seed,
            "insert_ns": list(FULL_NS if scale == "full" else (100, 500, 2000)),
            "query_n": 100_000 if scale == "full" else 2_000,
            "point_queries": 1_000 if scale == "full" else 30,
            "range_queries": range_queries if range_queries is not None else (20 if scale == "full" else 3),
            "sensitivity_n": 100_000 if scale == "full" else 2_000,
            "page_size": 4096, "page_sizes": list(PAGE_SIZES),
            "selectivities": list(SELECTIVITIES), "schema": COLUMNS,
            "fill_factor": 0.75, "query_checkpoint_size": 50}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def source_fingerprint(config: dict[str, Any]) -> tuple[str, dict[str, str]]:
    # Sólo módulos del gestor físico: cambios en API/frontend no invalidan I/O.
    physical = ("storage.py", "indexes.py", "io.py", "pager.py", "pages.py")
    sources = [Path(__file__).resolve()]
    sources += [ROOT / "backend" / "minidb" / name for name in physical
                if (ROOT / "backend" / "minidb" / name).exists()]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    encoded = json.dumps({"config": config, "sources": hashes}, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:20], hashes


def summary_statistics(values: Iterable[float]) -> dict[str, float | int]:
    items = list(values)
    if not items:
        raise ValueError("No se puede resumir una muestra vacía")
    return {"count": len(items), "mean": statistics.fmean(items),
            "stdev": statistics.stdev(items) if len(items) > 1 else 0.0,
            "min": min(items), "max": max(items)}


def summarize_queries(rows: list[dict[str, Any]], groups: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(tuple(row[key] for key in groups), []).append(row)
    summaries = []
    for key, sample in buckets.items():
        item = dict(zip(groups, key))
        item["queries"] = len(sample)
        for metric in ("disk_reads", "disk_writes", "index_reads", "data_reads", "latency_ms", "result_count"):
            stats = summary_statistics(row[metric] for row in sample)
            for stat in ("mean", "stdev", "min", "max"):
                item[f"{metric}_{stat}"] = stats[stat]
        summaries.append(item)
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


class Benchmark:
    def __init__(self, config: dict[str, Any], output: Path, work: Path):
        self.config, self.output = config, output.resolve()
        self.fingerprint, self.source_hashes = source_fingerprint(config)
        self.checkpoints = self.output / "checkpoints" / self.fingerprint
        self.work = work.resolve() / self.fingerprint
        self.output.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        storage = importlib.import_module("backend.minidb.storage")
        indexes = importlib.import_module("backend.minidb.indexes")
        self.Heap, self.Sequential, self.Schema = storage.HeapFile, storage.SequentialFile, storage.Schema
        self.Counter = getattr(storage, "DiskCounter", None)
        if self.Counter is None:
            self.Counter = importlib.import_module("backend.minidb.io").DiskCounter
        self.BTree, self.Hash = indexes.BPlusTree, indexes.ExtendibleHash
        self.schema = self.Schema(COLUMNS)
        self.started = utc_now()

    def log(self, message: str) -> None:
        print(f"[{utc_now()}] {message}", flush=True)

    def assert_sources_unchanged(self) -> None:
        current, _ = source_fingerprint(self.config)
        if current != self.fingerprint:
            raise RuntimeError("El código físico cambió durante la medición. Reinicie para obtener resultados de una sola versión.")

    def checkpoint(self, name: str, rows: list[dict[str, Any]]) -> None:
        self.assert_sources_unchanged()
        atomic_json(self.checkpoints / f"{name}.json", rows)

    def cached(self, name: str) -> list[dict[str, Any]] | None:
        path = self.checkpoints / f"{name}.json"
        return read_json(path) if path.exists() else None

    def rows(self, prefix: str) -> list[dict[str, Any]]:
        return [row for path in sorted(self.checkpoints.glob(prefix + "*.json")) for row in read_json(path)]

    def new_store(self, method: str, directory: Path, counter: Any, create: bool = False,
                  page_size: int | None = None):
        size = page_size or self.config["page_size"]
        directory.mkdir(parents=True, exist_ok=True)
        common = {"counter": counter, "page_size": size, "create": create}
        if method == "heap":
            return self.Heap(directory / "heap.bin", self.schema, **common)
        if method.startswith("sequential"):
            return self.Sequential(directory / "sequential.bin", self.schema, key_name="id", **common)
        cls = self.BTree if method == "btree" else self.Hash
        return cls(directory / f"{method}.bin", key_type="INT", **common)

    @staticmethod
    def io(counter: Any) -> dict[str, int]:
        snap = counter.snapshot()
        roles = snap.get("by_role", {})
        return {"disk_reads": int(snap["disk_reads"]), "disk_writes": int(snap["disk_writes"]),
                "index_reads": int(roles.get("index", {}).get("disk_reads", 0)),
                "data_reads": int(roles.get("data", {}).get("disk_reads", 0))}

    def experiment1(self) -> None:
        keys = insertion_order(max(self.config["insert_ns"]), self.config["seed"])
        for method in INSERT_METHODS:
            if all(self.cached(f"e1_{method}_{n:07d}") is not None for n in self.config["insert_ns"]):
                self.log(f"E1 {method}: checkpoints completos")
                continue
            state = self.work / f"e1_{method}_state"
            backup = self.work / f"e1_{method}_backup"
            stage = self.work / f"e1_{method}_stage"
            # Recuperar una interrupción entre los dos renombrados atómicos.
            if not state.exists() and backup.exists():
                backup.replace(state)
            if backup.exists():
                shutil.rmtree(backup)
            history = read_json(state / "state.json")["history"] if (state / "state.json").exists() else []
            for previous in history:
                self.checkpoint(f"e1_{method}_{previous['n']:07d}", [previous])
            previous_n = history[-1]["n"] if history else 0
            for n in self.config["insert_ns"]:
                if n <= previous_n:
                    continue
                self.log(f"E1 {method}: {previous_n:,} → {n:,} registros")
                if stage.exists():
                    shutil.rmtree(stage)
                if state.exists():
                    shutil.copytree(state, stage)
                else:
                    stage.mkdir(parents=True)
                counter = self.Counter()
                start = time.perf_counter_ns()
                engine = self.new_store(method, stage, counter, create=not history)
                # Incluir creación de páginas, pero excluir reapertura de un checkpoint.
                if history:
                    counter.reset()
                    start = time.perf_counter_ns()
                for offset in range(previous_n, n):
                    key = keys[offset]
                    if method in {"btree", "hash"}:
                        engine.insert(key, (offset // 50 + 1, offset % 50))
                    else:
                        engine.insert(dataset_row(key))
                inserted = time.perf_counter_ns()
                reorg_ms = 0.0
                if method == "sequential_reorg":
                    engine.reorganize(self.config["fill_factor"])
                    reorg_ms = (time.perf_counter_ns() - inserted) / 1e6
                engine.close()
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                counts = self.io(counter)
                prior = history[-1] if history else {}
                row = {"experiment": 1, "method": method, "n": n,
                       "page_size": self.config["page_size"],
                       "elapsed_ms": prior.get("elapsed_ms", 0) + elapsed_ms,
                       "disk_reads": prior.get("disk_reads", 0) + counts["disk_reads"],
                       "disk_writes": prior.get("disk_writes", 0) + counts["disk_writes"],
                       "reorganization_ms": prior.get("reorganization_ms", 0) + reorg_ms,
                       "segment_elapsed_ms": elapsed_ms, "segment_disk_reads": counts["disk_reads"],
                       "segment_disk_writes": counts["disk_writes"],
                       "stored_bytes": sum(p.stat().st_size for p in stage.rglob("*") if p.is_file() and p.name != "state.json"),
                       "scope": "key_rid_index_only" if method in {"btree", "hash"} else "full_record_storage",
                       "checkpoint_at": utc_now()}
                row["total_io"] = row["disk_reads"] + row["disk_writes"]
                history.append(row)
                atomic_json(stage / "state.json", {"history": history})
                self.assert_sources_unchanged()
                if state.exists():
                    state.replace(backup)
                stage.replace(state)
                if backup.exists():
                    shutil.rmtree(backup)
                self.checkpoint(f"e1_{method}_{n:07d}", [row])
                previous_n = n
                self.export()
            if state.exists():
                shutil.rmtree(state)

    def fixture(self) -> Path:
        target = self.work / "query_fixture"
        ready = target / "ready.json"
        if ready.exists() and read_json(ready).get("fingerprint") == self.fingerprint:
            return target
        stage = self.work / "query_fixture_stage"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        self.log(f"Preparando tablas e índices para {self.config['query_n']:,} registros")
        counter = self.Counter()
        keys = insertion_order(self.config["query_n"], self.config["seed"])
        with ExitStack() as stack:
            heap = self.new_store("heap", stage, counter, True)
            bt = self.new_store("btree", stage, counter, True)
            hs = self.new_store("hash", stage, counter, True)
            for engine in (heap, bt, hs):
                stack.callback(engine.close)
            for key in keys:
                rid = heap.insert(dataset_row(key))
                bt.insert(key, rid)
                hs.insert(key, rid)
        seq = self.new_store("sequential", stage, counter, True)
        try:
            seq.bulk_load(dataset_row(key) for key in keys)
        finally:
            seq.close()
        atomic_json(stage / "ready.json", {"fingerprint": self.fingerprint, "n": len(keys), "created_at": utc_now()})
        if target.exists():
            shutil.rmtree(target)
        stage.replace(target)
        return target

    def open_query(self, method: str, fixture: Path, stack: ExitStack):
        counter = self.Counter()
        engine = self.new_store(method, fixture, counter)
        stack.callback(engine.close)
        heap = self.new_store("heap", fixture, counter) if method in {"btree", "hash"} else None
        if heap is not None:
            stack.callback(heap.close)
        return counter, engine, heap

    @staticmethod
    def materialize_heap(heap: Any, rids: Iterable[Any]) -> list[dict[str, Any]]:
        return [heap.get(rid) for rid in rids]

    def point_query(self, method: str, engine: Any, heap: Any, key: int) -> list[dict[str, Any]]:
        if method == "heap":
            return [row for _, row in engine.scan_equal("id", key)]
        if method == "sequential":
            return [row for _, row in engine.search(key)]
        return self.materialize_heap(heap, engine.search(key))

    def range_query(self, method: str, engine: Any, heap: Any, low: int, high: int) -> list[dict[str, Any]]:
        if method == "heap":
            return [row for _, row in engine.scan() if low <= row["id"] <= high]
        if method == "sequential":
            return [row for _, row in engine.range_search(low, high)]
        return self.materialize_heap(heap, engine.range_search(low, high))

    def measure(self, counter: Any, action: Callable[[], list[Any]]) -> tuple[list[Any], dict[str, Any]]:
        counter.reset()
        start = time.perf_counter_ns()
        records = action()
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        return records, {**self.io(counter), "latency_ms": elapsed_ms, "result_count": len(records)}

    def experiment2(self) -> None:
        chunk = self.config["query_checkpoint_size"]
        if all(self.cached(f"e2_{method}_{start:05d}") is not None
               for method in POINT_METHODS for start in range(0, self.config["point_queries"], chunk)):
            self.log("E2: checkpoints completos")
            return
        fixture = self.fixture()
        rng = random.Random(self.config["seed"] + 2)
        queries = [rng.randint(1, self.config["query_n"]) for _ in range(self.config["point_queries"])]
        for method in POINT_METHODS:
            with ExitStack() as stack:
                counter, engine, heap = self.open_query(method, fixture, stack)
                for start in range(0, len(queries), chunk):
                    name = f"e2_{method}_{start:05d}"
                    if self.cached(name) is not None:
                        continue
                    self.log(f"E2 {method}: consultas {start + 1}–{min(start + chunk, len(queries))}")
                    rows = []
                    for i in range(start, min(start + chunk, len(queries))):
                        key = queries[i]
                        records, metrics = self.measure(counter, lambda: self.point_query(method, engine, heap, key))
                        if len(records) != 1 or records[0] != dataset_row(key):
                            raise AssertionError(f"E2 {method}: resultado incorrecto para id={key}: {records[:3]}")
                        rows.append({"experiment": 2, "method": method, "n": self.config["query_n"],
                                     "query_id": i, "key": key, "page_size": self.config["page_size"], **metrics})
                    self.checkpoint(name, rows)
                    self.export()

    def experiment3(self) -> None:
        if all(self.cached(f"e3_{method}_{selectivity:.3f}") is not None
               for method in RANGE_METHODS for selectivity in self.config["selectivities"]):
            self.log("E3: checkpoints completos")
            return
        fixture = self.fixture()
        rng = random.Random(self.config["seed"] + 3)
        queries = []
        n = self.config["query_n"]
        for selectivity in self.config["selectivities"]:
            width = max(1, round(n * selectivity))
            for i in range(self.config["range_queries"]):
                low = rng.randint(1, n - width + 1)
                queries.append((selectivity, i, low, low + width - 1))
        for method in RANGE_METHODS:
            with ExitStack() as stack:
                counter, engine, heap = self.open_query(method, fixture, stack)
                for selectivity in self.config["selectivities"]:
                    name = f"e3_{method}_{selectivity:.3f}"
                    if self.cached(name) is not None:
                        continue
                    self.log(f"E3 {method}: selectividad {selectivity:.1%}")
                    rows = []
                    for sel, i, low, high in queries:
                        if sel != selectivity:
                            continue
                        records, metrics = self.measure(counter, lambda: self.range_query(method, engine, heap, low, high))
                        if len(records) != high - low + 1 or {row["id"] for row in records} != set(range(low, high + 1)):
                            raise AssertionError(f"E3 {method}: rango incorrecto {low}..{high}")
                        rows.append({"experiment": 3, "method": method, "n": n, "query_id": i,
                                     "selectivity": selectivity, "actual_selectivity": (high - low + 1) / n,
                                     "low": low, "high": high, "page_size": self.config["page_size"], **metrics})
                    self.checkpoint(name, rows)
                    self.export()

    def experiment4(self) -> None:
        n = self.config["sensitivity_n"]
        keys = insertion_order(n, self.config["seed"])
        point_rng = random.Random(self.config["seed"] + 4)
        point_queries = [point_rng.randint(1, n) for _ in range(self.config["point_queries"])]
        range_rng = random.Random(self.config["seed"] + 5)
        width = max(1, round(n * .01))
        ranges = [(start, start + width - 1) for start in
                  (range_rng.randint(1, n - width + 1) for _ in range(self.config["range_queries"]))]
        for page_size in self.config["page_sizes"]:
            name = f"e4_summary_{page_size:05d}"
            if self.cached(name) is not None:
                continue
            self.log(f"E4 B+: páginas {page_size} bytes, N={n:,}")
            directory = self.work / f"e4_{page_size}"
            if directory.exists():
                shutil.rmtree(directory)
            counter = self.Counter()
            started = time.perf_counter_ns()
            tree = self.new_store("btree", directory, counter, True, page_size)
            try:
                for offset, key in enumerate(keys):
                    tree.insert(key, (offset // 50 + 1, offset % 50))
                insertion_ms = (time.perf_counter_ns() - started) / 1e6
                insertion_io = self.io(counter)
                stats = tree.stats()
                rows = []
                for i, key in enumerate(point_queries):
                    rids, metrics = self.measure(counter, lambda: tree.search(key))
                    if len(rids) != 1:
                        raise AssertionError(f"E4 B={page_size}: id {key} no encontrado")
                    rows.append({"experiment": 4, "method": "btree", "query_type": "equality",
                                 "n": n, "page_size": page_size, "query_id": i, **metrics})
                for i, (low, high) in enumerate(ranges):
                    rids, metrics = self.measure(counter, lambda: tree.range_search(low, high))
                    if len(rids) != width or len(set(rids)) != width:
                        raise AssertionError(f"E4 B={page_size}: rango incorrecto")
                    rows.append({"experiment": 4, "method": "btree", "query_type": "range_1pct",
                                 "n": n, "page_size": page_size, "query_id": i, **metrics})
                eq = [row for row in rows if row["query_type"] == "equality"]
                rg = [row for row in rows if row["query_type"] == "range_1pct"]
                summary = {"experiment": 4, "n": n, "page_size": page_size,
                           "height": stats["height"], "fanout": stats["fanout"], "pages": stats["pages"],
                           "insert_elapsed_ms": insertion_ms, "insert_disk_reads": insertion_io["disk_reads"],
                           "insert_disk_writes": insertion_io["disk_writes"],
                           "insert_total_io": insertion_io["disk_reads"] + insertion_io["disk_writes"],
                           "point_reads_mean": statistics.fmean(row["disk_reads"] for row in eq),
                           "point_reads_stdev": statistics.stdev(row["disk_reads"] for row in eq) if len(eq) > 1 else 0.0,
                           "point_latency_ms_mean": statistics.fmean(row["latency_ms"] for row in eq),
                           "range_reads_mean": statistics.fmean(row["disk_reads"] for row in rg),
                           "range_latency_ms_mean": statistics.fmean(row["latency_ms"] for row in rg),
                           "scope": "key_rid_index_only", "checkpoint_at": utc_now()}
                self.checkpoint(f"e4_queries_{page_size:05d}", rows)
                self.checkpoint(name, [summary])
            finally:
                tree.close()
            shutil.rmtree(directory)
            self.export()

    def export(self, *, plots: bool = False) -> dict[str, Any]:
        self.assert_sources_unchanged()
        e1, e2, e3 = self.rows("e1_"), self.rows("e2_"), self.rows("e3_")
        e4, e4_raw = self.rows("e4_summary_"), self.rows("e4_queries_")
        e2_summary = summarize_queries(e2, ("method", "n", "page_size"))
        e3_summary = summarize_queries(e3, ("method", "n", "page_size", "selectivity"))
        expected = {"experiment1": len(self.config["insert_ns"]) * len(INSERT_METHODS),
                    "experiment2": len(POINT_METHODS) * self.config["point_queries"],
                    "experiment3": len(RANGE_METHODS) * len(SELECTIVITIES) * self.config["range_queries"],
                    "experiment4": len(PAGE_SIZES)}
        actual = dict(zip(expected, map(len, (e1, e2, e3, e4))))
        complete = {key: actual[key] == count for key, count in expected.items()}
        methodology = {
            "dataset": "Registros sintéticos deterministas, claves únicas 1..N, orden barajado con semilla fija; esquema id INT, nombre CHAR(30), dept CHAR(20), salario FLOAT.",
            "experiment1": "Series acumuladas desde cero sobre el mismo orden de 500000 claves (2000 en smoke). Cada punto N es el prefijo de la serie. Sequential con reorganización ejecuta reorganize(0.75) al alcanzar cada N; su coste está incluido. Heap y Sequential almacenan tuplas completas; B+ y Hash miden sólo pares clave/RID. Los RIDs de E1 son referencias sintéticas de formato válido; E2/E3 usan RIDs reales del Heap.",
            "experiment2": "1000 igualdades aleatorias con reemplazo y todas presentes (30 en smoke). Misma secuencia para cada método. Full scan agota las páginas. Sequential usa área principal ordenada preparada con bulk_load. Índices incluyen búsqueda y lectura de la tupla en Heap.",
            "experiment3": "Rangos inclusivos de anchura exactamente round(N*selectividad), mismos límites por método. Todas las tuplas se materializan; B+ incluye accesos al Heap aleatoriamente organizado, sin agrupación ni deduplicación de páginas por el benchmark.",
            "experiment4": "100000 claves (2000 en smoke), idéntico orden aleatorio por tamaño de página. Se mide sólo el índice; fanout es la capacidad máxima de hijos, altura y páginas se obtienen de la estructura real. Incluye igualdad y rangos del 1%.",
            "io": "Contador de transferencias de bloques del gestor: cada read/write físico solicitado por la implementación, no fallos de caché del dispositivo. No se vacía la caché del SO. Creación de archivos incluida en inserción; reapertura de checkpoints y fixture de consultas excluidas. Diagnóstico stats() excluido de consultas.",
            "timing": "Reloj monotónico perf_counter_ns, milisegundos. Validación de resultados, exportación, copia de checkpoints y gráficos fuera de los intervalos medidos. Desviación estándar muestral (ddof=1). Una serie de inserción; consultas repetidas para distribuciones de latencia.",
            "checkpoint": "Checkpoints JSON atómicos por versión de código y configuración. E1 preserva estado binario confirmado y copia antes del siguiente segmento; no se extrapolan observaciones.",
        }
        manifest = {"started_at": self.started, "updated_at": utc_now(), "fingerprint": self.fingerprint,
                    "config": self.config, "source_hashes": self.source_hashes,
                    "environment": {"python": sys.version, "platform": platform.platform(),
                                    "processor": platform.processor(), "machine": platform.machine()},
                    "methodology": methodology, "expected_observations": expected,
                    "actual_observations": actual, "experiments_complete": complete,
                    "full_requirement_satisfied": self.config["scale"] == "full" and all(complete.values())}
        tables = {"experiment1_insertion": e1, "experiment2_point_raw": e2,
                  "experiment2_point_summary": e2_summary, "experiment3_range_raw": e3,
                  "experiment3_range_summary": e3_summary, "experiment4_page_size": e4,
                  "experiment4_queries_raw": e4_raw}
        for name, rows in tables.items():
            if rows:
                write_csv(self.output / f"{name}.csv", rows)
        atomic_json(self.output / "manifest.json", manifest)
        atomic_json(self.output / "results.json", {"manifest": manifest, **tables})
        if plots:
            make_plots(self.output, tables, self.config["scale"])
        return manifest


def make_plots(output: Path, tables: dict[str, list[dict[str, Any]]], scale: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2, "figure.dpi": 150})
    suffix = " · PRUEBA SMOKE" if scale == "smoke" else ""

    def finish(fig, name):
        fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    e1 = tables["experiment1_insertion"]
    if e1:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
        for method in INSERT_METHODS:
            rows = sorted((r for r in e1 if r["method"] == method), key=lambda r: r["n"])
            if not rows:
                continue
            for ax, metric, divisor in ((axes[0], "elapsed_ms", 1000), (axes[1], "disk_writes", 1)):
                ax.plot([r["n"] for r in rows], [r[metric] / divisor for r in rows], marker="o",
                        label=LABELS[method], color=COLORS[method])
                ax.set(xscale="log", yscale="log", xlabel="Registros insertados (N)")
        axes[0].set_ylabel("Tiempo acumulado (s, escala log)")
        axes[1].set_ylabel("Bloques escritos acumulados (escala log)")
        axes[1].legend(fontsize=8, loc="best")
        fig.suptitle("E1 · Inserción masiva" + suffix)
        fig.supxlabel("B+ y Hash: sólo clave/RID. Sequential con reorganización: mantenimiento incluido.", fontsize=9)
        finish(fig, "experiment1_insertion")

    e2 = tables["experiment2_point_summary"]
    if e2:
        order = [next((r for r in e2 if r["method"] == method), None) for method in POINT_METHODS]
        rows = [r for r in order if r]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
        for ax, metric, ylabel in ((axes[0], "disk_reads", "Lecturas por consulta (escala log)"),
                                   (axes[1], "latency_ms", "Latencia (ms, escala log)")):
            means = [r[f"{metric}_mean"] for r in rows]
            errors = [r[f"{metric}_stdev"] for r in rows]
            ax.bar(range(len(rows)), means, color=[COLORS[r["method"]] for r in rows])
            lower = [min(err, mean * .95) for err, mean in zip(errors, means)]
            ax.errorbar(range(len(rows)), means, yerr=[lower, errors], fmt="none", ecolor="#333333", capsize=4)
            ax.set(yscale="log", ylabel=ylabel, xticks=range(len(rows)),
                   xticklabels=[LABELS[r["method"]].replace(" / ", "\n") for r in rows])
            ax.tick_params(axis="x", labelsize=8)
        fig.suptitle("E2 · Igualdad: media y desviación estándar" + suffix)
        fig.supxlabel("Incluye recuperación de tuplas. Las barras inferiores se recortan a valores positivos por la escala log.", fontsize=9)
        finish(fig, "experiment2_point")

    e3 = tables["experiment3_range_summary"]
    if e3:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
        for method in RANGE_METHODS:
            rows = sorted((r for r in e3 if r["method"] == method), key=lambda r: r["selectivity"])
            if not rows:
                continue
            for ax, metric in ((axes[0], "disk_reads_mean"), (axes[1], "latency_ms_mean")):
                ax.plot([r["selectivity"] * 100 for r in rows], [r[metric] for r in rows],
                        marker="o", label=LABELS[method], color=COLORS[method])
                ax.set(xscale="log", yscale="log", xlabel="Selectividad (%)")
                ax.set_xticks([s * 100 for s in SELECTIVITIES], [f"{s * 100:g}" for s in SELECTIVITIES])
        axes[0].set_ylabel("Lecturas medias por consulta (escala log)")
        axes[1].set_ylabel("Latencia media (ms, escala log)")
        axes[0].legend(fontsize=9)
        fig.suptitle("E3 · Rangos de selectividad variable" + suffix)
        fig.supxlabel("B+ incluye materialización sobre Heap con inserción aleatoria; cada RID recuperado tiene su coste real.", fontsize=9)
        finish(fig, "experiment3_range")

    e4 = tables["experiment4_page_size"]
    if e4:
        rows = sorted(e4, key=lambda r: r["page_size"])
        xs = [r["page_size"] for r in rows]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for ax, key, ylabel in ((axes[0, 0], "fanout", "Capacidad máxima de hijos (fan-out)"),
                                (axes[0, 1], "height", "Altura real (niveles)"),
                                (axes[1, 1], "insert_total_io", "I/O total de construcción")):
            ax.plot(xs, [r[key] for r in rows], marker="o", color=COLORS["btree"])
            ax.set_ylabel(ylabel)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        axes[1, 0].plot(xs, [r["point_reads_mean"] for r in rows], marker="o", label="Igualdad")
        axes[1, 0].plot(xs, [r["range_reads_mean"] for r in rows], marker="s", label="Rango 1 %")
        axes[1, 0].set_ylabel("Lecturas medias, sólo índice")
        axes[1, 0].legend()
        max_height = max(r["height"] for r in rows)
        axes[0, 1].set_yticks(range(1, max_height + 1))
        axes[0, 1].set_ylim(.5, max_height + .5)
        for ax in axes.flat:
            ax.set_xticks(xs, [str(x) for x in xs])
            ax.set_xlabel("Tamaño de página (bytes)")
        fig.suptitle("E4 · Sensibilidad del árbol B+ al tamaño de página" + suffix)
        finish(fig, "experiment4_page_size")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--experiments", nargs="+", choices=("1", "2", "3", "4"), default=["1", "2", "3", "4"])
    parser.add_argument("--range-queries", type=int, help="Repeticiones por selectividad; por defecto 20 full / 3 smoke")
    parser.add_argument("--output", type=Path, help="Por defecto benchmarks/results/{scale}")
    parser.add_argument("--work", type=Path, default=ROOT / "benchmarks" / "work")
    parser.add_argument("--plots-only", action="store_true", help="Regenerar figuras desde results.json sin ejecutar el motor")
    parser.add_argument("--keep-work", action="store_true", help="Conservar fixture binario después de completar todos los experimentos")
    args = parser.parse_args(argv)
    output = args.output or ROOT / "benchmarks" / "results" / args.scale
    if args.plots_only:
        stored = read_json(output / "results.json")
        make_plots(output, stored, stored["manifest"]["config"]["scale"])
        return 0
    if args.range_queries is not None and args.range_queries < 2:
        parser.error("--range-queries debe ser al menos 2")
    config = experiment_config(args.scale, args.seed, args.range_queries)
    bench = Benchmark(config, output, args.work)
    bench.log(f"Iniciando {args.scale}; versión {bench.fingerprint}; salida {output}")
    try:
        for number in args.experiments:
            getattr(bench, f"experiment{number}")()
    finally:
        manifest = bench.export(plots=True)
    if all(manifest["experiments_complete"].values()) and not args.keep_work:
        shutil.rmtree(bench.work)
    bench.log(f"Finalizado. Requisito completo P1: {manifest['full_requirement_satisfied']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
