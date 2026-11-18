#!/usr/bin/env python3
"""E5 real: R-Tree vs escaneo paginado, escala 2D y dimensiones 2..128.

.venv/bin/python benchmarks/parte2_espacial.py --scale smoke
.venv/bin/python benchmarks/parte2_espacial.py --scale full
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.minidb.io import DiskCounter
from backend.minidb.spatial import RTree, PointStore
from benchmarks.parte1 import atomic_json, read_json, utc_now, write_csv


def configuration(scale):
    return {"scale": scale, "seed": 20261118, "page_size": 4096,
            "spatial_ns": [10000, 50000, 100000] if scale == "full" else [300, 1000],
            "dimensions": [2, 4, 8, 16, 32, 64, 128] if scale == "full" else [2, 8, 128],
            "dimensionality_n": 2000 if scale == "full" else 100,
            "query_count": 5 if scale == "full" else 2, "ks": [5, 20, 50],
            "radii_km": [.25, 1.0, 3.0], "dimension_k": 20}


def points(n, dimensions, seed, spatial=False):
    rng = np.random.default_rng(seed)
    for start in range(0, n, 4096):
        block = rng.random((min(4096, n - start), dimensions), dtype=np.float32)
        if spatial:
            block *= np.asarray([50.0, 40.0], np.float32)
        for i, point in enumerate(block, start):
            yield point, (i // 100 + 1, i % 100)


def summaries(rows):
    groups = {}
    fields = ("series", "n", "dimensions", "query_type", "parameter", "method")
    for row in rows:
        groups.setdefault(tuple(row[f] for f in fields), []).append(row)
    result = []
    for key, sample in groups.items():
        item = dict(zip(fields, key))
        item["queries"] = len(sample)
        for field in ("disk_reads", "disk_writes", "index_reads", "data_reads", "latency_ms", "result_count", "fraction_pages_read"):
            values = [row[field] for row in sample]
            item[f"{field}_mean"] = statistics.fmean(values)
            item[f"{field}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result.append(item)
    return result


class SpatialBenchmark:
    def __init__(self, config, output):
        self.config, self.output = config, output
        sources = [Path(__file__), ROOT / "backend/minidb/spatial.py", ROOT / "backend/minidb/io.py"]
        self.source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        self.fingerprint = hashlib.sha256(json.dumps([config, self.source_hashes], sort_keys=True).encode()).hexdigest()[:20]
        self.checkpoints = output / "checkpoints" / self.fingerprint
        self.work = ROOT / "benchmarks/work" / f"spatial_{self.fingerprint}"
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        self.started_at = utc_now()

    def verify_sources(self):
        for name, expected in self.source_hashes.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Las fuentes espaciales cambiaron durante la medición; reinicie el benchmark")

    def log(self, text):
        print(f"[{utc_now()}] {text}", flush=True)

    def run_unit(self, series, n, dimensions):
        name = f"{series}_n{n:07d}_d{dimensions:03d}"
        checkpoint = self.checkpoints / f"{name}.json"
        if checkpoint.exists():
            self.log(f"{name}: checkpoint completo")
            return
        folder = self.work / name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        seed = self.config["seed"] + (0 if series == "spatial_n" else dimensions)
        counters, engines, builds = {}, {}, []
        for method, cls in (("rtree", RTree), ("sequential", PointStore)):
            self.log(f"Construcción {method}, {series}: N={n:,}, D={dimensions}")
            counter = counters[method] = DiskCounter()
            start = time.perf_counter_ns()
            engine = cls(folder / f"{method}.bin", dimensions=dimensions, page_size=self.config["page_size"], counter=counter, create=True)
            for point, rid in points(n, dimensions, seed, series == "spatial_n"):
                engine.insert(point, rid)
            engine.close()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            stats = counter.snapshot()
            engine = engines[method] = cls(folder / f"{method}.bin", dimensions=dimensions, page_size=self.config["page_size"], counter=counter)
            structure = engine.stats()
            builds.append({"series": series, "n": n, "dimensions": dimensions, "method": method,
                           "elapsed_ms": elapsed, "disk_reads": stats["disk_reads"], "disk_writes": stats["disk_writes"],
                           "pages": structure["pages"], "capacity": structure["capacity"], "height": structure.get("height", 1),
                           "stored_bytes": structure["pages"] * self.config["page_size"]})
        rng = np.random.default_rng(self.config["seed"] + dimensions + 19)
        centers = rng.random((self.config["query_count"], dimensions), dtype=np.float32)
        if series == "spatial_n":
            centers *= np.asarray([50, 40], np.float32)
            queries = [("knn", k) for k in self.config["ks"]] + [("radius", r) for r in self.config["radii_km"]]
        else:
            queries = [("knn", self.config["dimension_k"])]
        rows = []
        try:
            # Archivos recién construidos + una consulta de calentamiento por método.
            for engine in engines.values():
                engine.knn(centers[0], self.config["dimension_k"])
            for query_type, parameter in queries:
                for query_id, center in enumerate(centers):
                    answers = {}
                    for method in (("sequential", "rtree") if query_id % 2 == 0 else ("rtree", "sequential")):
                        counter, engine = counters[method], engines[method]
                        counter.reset()
                        start = time.perf_counter_ns()
                        actual = engine.knn(center, parameter) if query_type == "knn" else engine.radius_search(center, parameter)
                        elapsed = (time.perf_counter_ns() - start) / 1e6
                        snapshot = counter.snapshot()
                        answers[method] = actual
                        role = snapshot["by_role"]
                        rows.append({"series": series, "n": n, "dimensions": dimensions, "query_type": query_type,
                                     "parameter": parameter, "method": method, "query_id": query_id,
                                     "disk_reads": snapshot["disk_reads"], "disk_writes": snapshot["disk_writes"],
                                     "index_reads": role["index"]["disk_reads"], "data_reads": role["data"]["disk_reads"],
                                     "latency_ms": elapsed, "result_count": len(actual),
                                     "fraction_pages_read": snapshot["disk_reads"] / max(1, engine.stats()["pages"] - 1)})
                    left, right = answers["sequential"], answers["rtree"]
                    if [rid for rid, _ in left] != [rid for rid, _ in right] or not np.allclose([d for _, d in left], [d for _, d in right], rtol=1e-12, atol=1e-12):
                        raise AssertionError(f"R-Tree no coincide con escaneo exacto: {name}, {query_type}, {parameter}, {query_id}")
            self.verify_sources()
            atomic_json(checkpoint, {"builds": builds, "queries": rows, "all_results_equal": True, "completed_at": utc_now()})
        finally:
            for engine in engines.values():
                engine.close()
        shutil.rmtree(folder)
        self.export()

    def export(self, plots=False):
        self.verify_sources()
        units = [read_json(path) for path in sorted(self.checkpoints.glob("*.json"))]
        rows = [row for unit in units for row in unit["queries"]]
        builds = [row for unit in units for row in unit["builds"]]
        summary = summaries(rows)
        expected = len(self.config["spatial_ns"]) + len(self.config["dimensions"])
        metadata = {"config": self.config, "fingerprint": self.fingerprint, "source_hashes": self.source_hashes,
                    "started_at": self.started_at, "updated_at": utc_now(), "units_complete": len(units), "units_expected": expected,
                    "full_requirement_satisfied": self.config["scale"] == "full" and len(units) == expected,
                    "all_results_equal_to_paginated_exact_scan": all(unit["all_results_equal"] for unit in units),
                    "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
                    "methodology": {
                        "spatial_dataset": "Puntos sintéticos uniformes en región local de 50x40 km, coordenadas float32, semilla fija. Los N comparten el mismo prefijo. Escala comparable a un área metropolitana; no son ubicaciones reales de negocios.",
                        "dimension_dataset": "Vectores sintéticos uniformes [0,1]^D; N fijo=2000 full (100 smoke), dimensiones 2..128, k=20. Se aísla D del crecimiento de N. El menor fanout por entrada más ancha y el solapamiento contribuyen conjuntamente a la degradación.",
                        "memory": "Construcción genera bloques de 4096 puntos. No se conserva el dataset en RAM para consultar. PointStore lee páginas completas una por vez, calcula distancias exactas y mantiene Top-K en max-heap. RTree consulta nodos en disco, sin caché persistente.",
                        "queries": "Cinco centros independientes por configuración full (dos smoke), idénticos entre métodos; orden de métodos alternado. Coordenadas y centros float32, distancias Euclídeas calculadas float64. Resultados completos comparados fuera de la medición, con desempate por RID.",
                        "timing": "Latencias calientes: datos recién escritos al SO y una consulta de calentamiento por método. No se vacía caché SO. perf_counter_ns; media y desviación muestral. La ejecución full se realiza con la extracción de audio en pausa, coordinada antes de medir.",
                        "io": "Transferencias paginadas solicitadas al gestor, desglosadas índice/datos. Consultas RTree incluyen coordenadas en hojas y devuelven RID/distancia; no incluyen materialización de otras columnas del Heap. El baseline contiene exactamente puntos/RIDs.",
                        "pages": "Páginas físicas 4096B, cabecera32B, entradaRTree 8D+8 bytes (MBRfloat32+RID/child). Capacity D128=3; no se utiliza una librería de índice espacial.",
                        "correctness": "Cada respuesta RTree debe coincidir con baseline paginado antes de escribir el checkpoint. No se ajustan ni extrapolan métricas para producir un cruce esperado."}}
        result = {"manifest": metadata, "builds": builds, "queries": rows, "summary": summary}
        for name, table in (("experiment5_builds", builds), ("experiment5_queries_raw", rows), ("experiment5_summary", summary)):
            write_csv(self.output / f"{name}.csv", table)
        atomic_json(self.output / "manifest.json", metadata)
        atomic_json(self.output / "results.json", result)
        if plots:
            make_plots(self.output, result)
        return metadata


def make_plots(output, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2})
    rows, config = result["summary"], result["manifest"]["config"]
    suffix = " · SMOKE" if config["scale"] == "smoke" else ""
    colors = ["#4477AA", "#228833", "#CC6677"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for row_idx, (kind, params) in enumerate((("knn", config["ks"]), ("radius", config["radii_km"]))):
        for method in ("rtree", "sequential"):
            for color, param in zip(colors, params):
                sample = sorted((r for r in rows if r["series"] == "spatial_n" and r["query_type"] == kind and r["parameter"] == param and r["method"] == method), key=lambda r: r["n"])
                if not sample:
                    continue
                label = f"{'R-Tree' if method == 'rtree' else 'Scan'} · {'k=' if kind == 'knn' else 'r='}{param}{' km' if kind == 'radius' else ''}"
                for ax, metric in ((axes[row_idx, 0], "disk_reads_mean"), (axes[row_idx, 1], "latency_ms_mean")):
                    ax.plot([r["n"] for r in sample], [r[metric] for r in sample], marker="o", linestyle="-" if method == "rtree" else "--", color=color, label=label)
                    ax.set(xscale="log", yscale="log", xlabel="Número de puntos (N)")
        axes[row_idx, 0].set_ylabel("Bloques leídos por consulta")
        axes[row_idx, 1].set_ylabel("Latencia media (ms)")
        axes[row_idx, 1].legend(fontsize=7, ncol=2)
        axes[row_idx, 0].set_title("k vecinos más cercanos" if kind == "knn" else "Consulta por radio")
    fig.suptitle("E5 · Escalabilidad espacial 2D" + suffix)
    fig.supxlabel("Páginas 4 KB · puntos sintéticos · mismas consultas por método · caché SO caliente", fontsize=9)
    fig.savefig(output / "experiment5_scalability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for method, color in (("rtree", colors[0]), ("sequential", colors[1])):
        sample = sorted((r for r in rows if r["series"] == "dimensionality" and r["method"] == method), key=lambda r: r["dimensions"])
        if not sample:
            continue
        xs = [r["dimensions"] for r in sample]
        label = "R-Tree" if method == "rtree" else "Scan paginado"
        for ax, field in ((axes[0, 0], "disk_reads_mean"), (axes[0, 1], "latency_ms_mean"), (axes[1, 0], "fraction_pages_read_mean")):
            ax.plot(xs, [r[field] for r in sample], marker="o", color=color, label=label)
        builds = sorted((r for r in result["builds"] if r["series"] == "dimensionality" and r["method"] == method), key=lambda r: r["dimensions"])
        axes[1, 1].plot([r["dimensions"] for r in builds], [r["pages"] - 1 for r in builds], marker="o", color=color, label=label)
    for ax, label in ((axes[0, 0], "Bloques leídos por consulta"), (axes[0, 1], "Latencia media (ms)"),
                      (axes[1, 0], "Fracción de páginas visitadas"), (axes[1, 1], "Páginas de nodos / puntos")):
        ax.set(xlabel="Dimensión D", ylabel=label, xscale="log")
        ax.set_xticks(config["dimensions"], [str(d) for d in config["dimensions"]])
        ax.legend(fontsize=8)
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_ylim(0, 1.05)
    fig.suptitle(f"E5 · Dimensionalidad: N={config['dimensionality_n']:,}, k=20" + suffix)
    fig.supxlabel("Mayor D aumenta solapamiento y reduce entradas por página; métricas observadas sin extrapolación.", fontsize=9)
    fig.savefig(output / "experiment5_dimensionality.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or ROOT / "benchmarks/results/parte2_espacial" / args.scale
    if args.plots_only:
        make_plots(output, read_json(output / "results.json"))
        return 0
    bench = SpatialBenchmark(configuration(args.scale), output)
    try:
        for n in bench.config["spatial_ns"]:
            bench.run_unit("spatial_n", n, 2)
        for dimensions in bench.config["dimensions"]:
            bench.run_unit("dimensionality", bench.config["dimensionality_n"], dimensions)
    finally:
        metadata = bench.export(plots=True)
    if metadata["units_complete"] == metadata["units_expected"]:
        shutil.rmtree(bench.work)
    bench.log(f"Terminado: requisito completo E5={metadata['full_requirement_satisfied']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
