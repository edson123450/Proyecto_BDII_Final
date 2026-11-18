#!/usr/bin/env python3
"""E7/E8 de audio sobre FMA: evaluación sin fuga de test hacia la galería.

    .venv/bin/python benchmarks/parte2_audio.py --scale smoke
    .venv/bin/python benchmarks/parte2_audio.py --scale full

Full requiere extracción completa y codebook de producción. Smoke usa las
cachés reales del piloto; nunca fabrica tiempos, embeddings ni relevancia.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import statistics
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.audio.features import AudioCodebook, FEATURE_VERSION
from backend.audio.prepare import cache_path, load_features, valid_cache
from backend.minidb.io import DiskCounter
from backend.minidb.text import InvertedIndex
from backend.minidb.vector import VectorStore, IVFIndex
from benchmarks.parte1 import atomic_json, read_json, utc_now, write_csv


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_stratified_test(records, per_genre=4, seed=20261118, required_genres=None):
    """Muestreo reproducible: cada género aporta el mismo número de consultas."""
    groups = defaultdict(list)
    for row in records:
        if row["split"] == "test":
            groups[row["genre"]].append(row)
    genres = sorted(required_genres if required_genres is not None else groups)
    if not genres:
        raise ValueError("No hay consultas test disponibles")
    rng, selected = random.Random(seed), []
    for genre in genres:
        available = sorted(groups.get(genre, []), key=lambda row: int(row["id"]))
        if len(available) < per_genre:
            raise ValueError(f"Se necesitan {per_genre} consultas test de {genre}; disponibles={len(available)}")
        selected.extend(rng.sample(available, per_genre))
    return selected


def assert_no_split_leakage(metadata, gallery, queries, codebook_ids, full=False):
    by_id = {int(row["id"]): row for row in metadata}
    if len(by_id) != len(metadata):
        raise ValueError("IDs duplicados en los metadatos")
    gallery_ids, query_ids = {int(r["id"]) for r in gallery}, {int(r["id"]) for r in queries}
    training_ids = {int(r["id"]) for r in metadata if r["split"] == "training"}
    if not gallery_ids or not query_ids or gallery_ids & query_ids:
        raise ValueError("Galería y consultas deben ser no vacías y disjuntas")
    if not gallery_ids <= training_ids or any(by_id[i]["split"] != "test" for i in query_ids):
        raise ValueError("La galería debe ser training y las consultas deben ser test")
    vocabulary_training = {int(i) for i in codebook_ids}
    if not vocabulary_training or not vocabulary_training <= training_ids:
        raise ValueError("Codebook contiene IDs ajenos a training o carece de procedencia")
    if full and vocabulary_training != gallery_ids:
        raise ValueError("El codebook full debe declarar todos los IDs training válidos de la galería")


def precision_at(retrieved_ids, query_genre, labels, k):
    if k < 1:
        raise ValueError("k debe ser positivo")
    # Si un método devuelve menos de k resultados, faltantes cuentan como no relevantes.
    return sum(labels[int(i)] == query_genre for i in retrieved_ids[:k]) / k


def recall_at(retrieved_ids, exact_ids, k=10):
    truth = set(exact_ids[:k])
    if not truth:
        raise ValueError("La verdad exacta no puede ser vacía")
    return len(set(retrieved_ids[:k]) & truth) / len(truth)


def probe_counts(clusters):
    if clusters < 1:
        raise ValueError("IVF requiere al menos un cluster")
    return sorted({min(p, clusters) for p in (1, 2, 5, 10, 20, 50)} | {clusters})


def read_metadata(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["id"] = int(row["id"])
    return rows


def read_timings(path):
    result = {}
    if not Path(path).exists():
        return result
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Una última línea incompleta puede pertenecer a extracción activa.
                continue
            if row.get("status") == "ok" and row.get("feature_version", FEATURE_VERSION) == FEATURE_VERSION:
                result[int(row["id"])] = row
    return result


def describe(values):
    values = list(values)
    return {"mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def summarize_e7(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["genre"])].append(row)
        groups[(row["method"], "ALL_MACRO")].append(row)
    output = []
    metrics = ("precision_at_5", "precision_at_10", "latency_ms", "disk_reads", "disk_writes", "index_reads", "data_reads", "extraction_ms", "quantization_ms", "decode_ms")
    for (method, genre), sample in groups.items():
        row = {"method": method, "genre": genre, "queries": len(sample)}
        for metric in metrics:
            values = [item[metric] for item in sample if item.get(metric) is not None]
            row[f"{metric}_observations"] = len(values)
            if values:
                row.update({f"{metric}_{key}": value for key, value in describe(values).items()})
        output.append(row)
    return output


def summarize_e8(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["probes"], row["clusters"])].append(row)
    output = []
    for (probes, clusters), sample in sorted(groups.items()):
        row = {"probes": probes, "clusters": clusters, "all_clusters": probes == clusters, "queries": len(sample)}
        for metric in ("recall_at_10", "latency_ms", "disk_reads", "disk_writes", "index_reads", "data_reads"):
            row.update({f"{metric}_{key}": value for key, value in describe(item[metric] for item in sample).items()})
        output.append(row)
    return output


class AudioBenchmark:
    def __init__(self, scale, cache, output, per_genre=None, codebook_path=None):
        self.scale, self.cache, self.output = scale, cache.resolve(), output.resolve()
        self.per_genre = per_genre if per_genre is not None else (4 if scale == "full" else 1)
        if self.per_genre < 1:
            raise ValueError("Se requiere al menos una consulta por género")
        if scale == "full" and self.per_genre not in (4, 8):
            raise ValueError("Full utiliza 4 u 8 consultas por género: 32 o 64 consultas en total")
        self.metadata = read_metadata(self.cache / "metadata.csv")
        self.codebook_path = codebook_path or self.cache / ("codebook.npz" if scale == "full" else "codebook.pilot.npz")
        if not self.codebook_path.exists():
            raise FileNotFoundError(f"Vocabulario aún no disponible: {self.codebook_path}")
        if scale == "full" and "pilot" in self.codebook_path.name:
            raise ValueError("Full no admite vocabulario del piloto")
        self.codebook = AudioCodebook.load(self.codebook_path)
        self.valid = [row for row in self.metadata if valid_cache(cache_path(self.cache, row["id"]), Path(row["path"]))]
        self.gallery = [row for row in self.valid if row["split"] == "training"]
        if scale == "smoke":
            self.gallery = self.gallery[:256]
        required = sorted({row["genre"] for row in self.metadata}) if scale == "full" else None
        self.queries = select_stratified_test(self.valid, self.per_genre, required_genres=required)
        assert_no_split_leakage(self.metadata, self.gallery, self.queries, self.codebook.training_ids, full=scale == "full")
        if len(self.gallery) < 10:
            raise ValueError("Se necesitan al menos diez pistas training para Precision@10")
        self.labels = {row["id"]: row["genre"] for row in self.metadata}
        self.timings = read_timings(self.cache / "manifest.jsonl")
        self.preparation_report = None
        if scale == "full":
            report_path = self.cache / "preparation_report.json"
            if not report_path.exists():
                raise ValueError("Full espera preparation_report.json: extracción y BoAW completos")
            self.preparation_report = read_json(report_path)
            if len(self.metadata) != 8000 or self.preparation_report.get("selected") != 8000 or len(self.valid) < 5000:
                raise ValueError("Full requiere las 8000 pistas FMA Small procesadas y al menos 5000 cachés válidas")
            if self.preparation_report.get("valid") != len(self.valid):
                raise ValueError("El informe de preparación y las cachés válidas no coinciden")
            missing_timing = [row["id"] for row in self.queries if not {"mfcc_ms", "embedding_ms"} <= self.timings.get(row["id"], {}).keys()]
            if missing_timing:
                raise ValueError(f"Faltan tiempos de extracción reales para consultas: {missing_timing}")
        self.config = {"scale": scale, "seed": 20261118, "queries_per_genre": self.per_genre,
                       "feature_version": FEATURE_VERSION, "page_size": 4096, "metric": "COSINE", "dimension": 512,
                       "gallery_ids": [row["id"] for row in self.gallery], "query_ids": [row["id"] for row in self.queries],
                       "codebook_sha256": file_sha256(self.codebook_path), "clusters": math.isqrt(len(self.gallery)),
                       "metadata_sha256": file_sha256(self.cache / "metadata.csv")}
        source_names = ("benchmarks/parte2_audio.py", "backend/audio/features.py", "backend/audio/model.py", "backend/audio/prepare.py",
                        "backend/minidb/vector.py", "backend/minidb/text.py", "backend/minidb/io.py")
        self.source_hashes = {name: file_sha256(ROOT / name) for name in source_names}
        cache_signature = [(row["id"], cache_path(self.cache, row["id"]).stat().st_size,
                            cache_path(self.cache, row["id"]).stat().st_mtime_ns) for row in self.gallery + self.queries]
        self.config["cache_signature"] = hashlib.sha256(json.dumps(cache_signature).encode()).hexdigest()
        self.fingerprint = hashlib.sha256(json.dumps([self.config, self.source_hashes], sort_keys=True).encode()).hexdigest()[:20]
        self.checkpoints = self.output / "checkpoints" / self.fingerprint
        self.work = ROOT / "benchmarks/work" / f"audio_{self.fingerprint}"
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        self.started_at = utc_now()

    def log(self, message):
        print(f"[{utc_now()}] {message}", flush=True)

    def verify_sources(self):
        for name, digest in self.source_hashes.items():
            if file_sha256(ROOT / name) != digest:
                raise RuntimeError("Fuentes de audio/índices cambiaron durante la medición; reinicie")

    def token_documents(self):
        counts_path = self.cache / "boaw.jsonl"
        provenance = self.cache / "boaw_metadata.json"
        gallery_ids = set(self.config["gallery_ids"])
        if self.scale == "full" and counts_path.exists() and provenance.exists():
            description = read_json(provenance)
            if description.get("codebook_sha256") == self.config["codebook_sha256"] and description.get("feature_version") == FEATURE_VERSION:
                found = set()
                with counts_path.open(encoding="utf-8") as stream:
                    for line in stream:
                        document = json.loads(line)
                        docid = int(document["id"])
                        if docid in gallery_ids:
                            if docid in found:
                                raise ValueError("Documento duplicado en boaw.jsonl")
                            found.add(docid)
                            yield docid, Counter(document["counts"])
                if found != gallery_ids:
                    raise ValueError("boaw.jsonl no contiene toda la galería training")
                return
        # Fallback fiel: cuantizar los descriptores reales con el codebook elegido.
        for row in self.gallery:
            features = load_features(cache_path(self.cache, row["id"]))
            yield row["id"], self.codebook.counts(features["mfcc_descriptors"])

    def build(self):
        checkpoint = self.checkpoints / "build.json"
        ready = self.work / "ready.json"
        if checkpoint.exists() and ready.exists() and read_json(ready).get("fingerprint") == self.fingerprint:
            return
        self.log(f"Construyendo galería training={len(self.gallery)}, consultas test={len(self.queries)}")
        for name in ("vectors.bin", "boaw", "ivf"):
            path = self.work / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        builds = []
        counter = DiskCounter()
        started = time.perf_counter_ns()
        store = VectorStore(self.work / "vectors.bin", dimension=512, counter=counter, create=True)
        for row in self.gallery:
            features = load_features(cache_path(self.cache, row["id"]))
            store.append((row["id"], 0), features["embedding"])
        store.close()
        builds.append({"method": "deep_exact", "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
                       **self.metrics(counter), "stored_bytes": (self.work / "vectors.bin").stat().st_size})
        counter.reset()
        started = time.perf_counter_ns()
        inverted = InvertedIndex(self.work / "boaw", counter=counter, create=True)
        inverted.build_tokens(self.token_documents())
        inverted.close()
        builds.append({"method": "boaw_tfidf", "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
                       **self.metrics(counter), "stored_bytes": sum(p.stat().st_size for p in (self.work / "boaw").iterdir() if p.is_file())})
        counter.reset()
        store = VectorStore(self.work / "vectors.bin", counter=counter)
        index = IVFIndex(self.work / "ivf", counter=counter)
        counter.reset()
        started = time.perf_counter_ns()
        try:
            stats = index.build(store, clusters=self.config["clusters"], seed=self.config["seed"])
            builds.append({"method": "deep_ivf", "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
                           **self.metrics(counter), "stored_bytes": stats["index_bytes"], "clusters": stats["clusters"]})
        finally:
            index.close()
            store.close()
        self.verify_sources()
        atomic_json(checkpoint, builds)
        atomic_json(ready, {"fingerprint": self.fingerprint})

    @staticmethod
    def metrics(counter):
        snapshot = counter.snapshot()
        return {"disk_reads": snapshot["disk_reads"], "disk_writes": snapshot["disk_writes"],
                "index_reads": snapshot["by_role"]["index"]["disk_reads"],
                "data_reads": snapshot["by_role"]["data"]["disk_reads"]}

    def measure(self, counter, action):
        counter.reset()
        started = time.perf_counter_ns()
        answer = action()
        elapsed = (time.perf_counter_ns() - started) / 1e6
        if counter.snapshot()["disk_writes"]:
            raise AssertionError("Una consulta de benchmark no debe modificar archivos")
        return answer, {"latency_ms": elapsed, **self.metrics(counter)}

    def experiment7(self):
        if all((self.checkpoints / f"e7_{row['id']:06d}.json").exists() for row in self.queries):
            self.log("E7: checkpoints completos")
            return
        self.build()
        counters = {method: DiskCounter() for method in ("boaw_tfidf", "deep_exact")}
        inverted = InvertedIndex(self.work / "boaw", counter=counters["boaw_tfidf"])
        store = VectorStore(self.work / "vectors.bin", counter=counters["deep_exact"])
        first = load_features(cache_path(self.cache, self.queries[0]["id"]))
        inverted.query_tokens(self.codebook.counts(first["mfcc_descriptors"]), 10)
        store.knn(first["embedding"], 10)
        gallery_ids = set(self.config["gallery_ids"])
        try:
            for query_number, query in enumerate(self.queries):
                checkpoint = self.checkpoints / f"e7_{query['id']:06d}.json"
                if checkpoint.exists():
                    continue
                self.log(f"E7 {query_number + 1}/{len(self.queries)}: {query['id']} ({query['genre']})")
                features = load_features(cache_path(self.cache, query["id"]))
                started = time.perf_counter_ns()
                counts = self.codebook.counts(features["mfcc_descriptors"])
                quantization_ms = (time.perf_counter_ns() - started) / 1e6
                timing = self.timings.get(query["id"], {})
                rows, exact = [], None
                methods = ("boaw_tfidf", "deep_exact") if query_number % 2 == 0 else ("deep_exact", "boaw_tfidf")
                for method in methods:
                    if method == "boaw_tfidf":
                        result, measured = self.measure(counters[method], lambda: inverted.query_tokens(counts, 10))
                        ids = [docid for docid, _ in result]
                        extraction_ms = timing.get("mfcc_ms")
                    else:
                        result, measured = self.measure(counters[method], lambda: store.knn(features["embedding"], 10))
                        ids = [rid[0] for rid, _ in result]
                        exact = [{"id": rid[0], "distance": distance} for rid, distance in result]
                        extraction_ms = timing.get("embedding_ms")
                    if not set(ids) <= gallery_ids or query["id"] in ids or len(ids) != len(set(ids)):
                        raise AssertionError("Fuga de consulta o IDs incorrectos en recuperación semántica")
                    rows.append({"query_id": query["id"], "genre": query["genre"], "method": method,
                                 "gallery_count": len(self.gallery), "result_count": len(ids),
                                 "precision_at_5": precision_at(ids, query["genre"], self.labels, 5),
                                 "precision_at_10": precision_at(ids, query["genre"], self.labels, 10),
                                 **measured, "extraction_ms": extraction_ms,
                                 "quantization_ms": quantization_ms if method == "boaw_tfidf" else 0.0,
                                 "decode_ms": timing.get("decode_ms"), "extraction_source": "manifest.jsonl" if timing else "unavailable",
                                 "extraction_run_id": timing.get("run_id", "pilot_trace" if timing else ""),
                                 "extraction_device": timing.get("device", "cpu (pilot configuration)" if timing else ""),
                                 "retrieved_ids": ids})
                self.verify_sources()
                atomic_json(checkpoint, {"rows": rows, "exact": exact, "extraction_trace": timing})
                self.export()
        finally:
            inverted.close()
            store.close()

    def experiment8(self):
        probes = probe_counts(self.config["clusters"])
        if all((self.checkpoints / f"e8_{query['id']:06d}.json").exists() for query in self.queries):
            self.log("E8: checkpoints completos")
            return
        self.build()
        counter = DiskCounter()
        index = IVFIndex(self.work / "ivf", counter=counter)
        first = load_features(cache_path(self.cache, self.queries[0]["id"]))
        index.knn(first["embedding"], 10, probes=min(5, self.config["clusters"]))
        try:
            for position, query in enumerate(self.queries):
                checkpoint = self.checkpoints / f"e8_{query['id']:06d}.json"
                if checkpoint.exists():
                    continue
                truth_path = self.checkpoints / f"e7_{query['id']:06d}.json"
                if not truth_path.exists():
                    raise ValueError("E8 requiere primero la verdad exacta de E7")
                truth = read_json(truth_path)["exact"]
                expected_ids = [row["id"] for row in truth]
                embedding = load_features(cache_path(self.cache, query["id"]))["embedding"]
                rows = []
                previous_recall = 0.0
                for count in probes:
                    result, measured = self.measure(counter, lambda: index.knn(embedding, 10, probes=count))
                    ids = [rid[0] for rid, _ in result]
                    recall = recall_at(ids, expected_ids)
                    if recall < previous_recall or not set(ids) <= set(self.config["gallery_ids"]):
                        raise AssertionError("Recall IVF no monótono o resultados ajenos a la galería")
                    previous_recall = recall
                    if count == index.clusters:
                        if ids != expected_ids or not np.allclose([d for _, d in result], [row["distance"] for row in truth], atol=1e-12, rtol=1e-12):
                            raise AssertionError("IVF con todas las listas no coincide con búsqueda exacta")
                    rows.append({"query_id": query["id"], "genre": query["genre"], "probes": count,
                                 "clusters": index.clusters, "all_clusters": count == index.clusters,
                                 "gallery_count": len(self.gallery), "recall_at_10": recall,
                                 "result_count": len(ids), **measured, "retrieved_ids": ids,
                                 "exact_ids": expected_ids})
                self.verify_sources()
                atomic_json(checkpoint, rows)
                self.log(f"E8 {position + 1}/{len(self.queries)}: IVF all-clusters coincide con exacto")
                self.export()
        finally:
            index.close()

    def export(self, plots=False):
        self.verify_sources()
        e7_units = [read_json(path) for path in sorted(self.checkpoints.glob("e7_*.json"))]
        e7 = [row for unit in e7_units for row in unit["rows"]]
        e8 = [row for path in sorted(self.checkpoints.glob("e8_*.json")) for row in read_json(path)]
        build_path = self.checkpoints / "build.json"
        builds = read_json(build_path) if build_path.exists() else []
        expected7 = 2 * len(self.queries)
        expected8 = len(self.queries) * len(probe_counts(self.config["clusters"]))
        e7_complete, e8_complete = len(e7) == expected7, len(e8) == expected8
        raw_splits, valid_splits = Counter(row["split"] for row in self.metadata), Counter(row["split"] for row in self.valid)
        valid_ids = {row["id"] for row in self.valid}
        excluded = [{"id": row["id"], "split": row["split"], "genre": row["genre"]}
                    for row in self.metadata if row["id"] not in valid_ids]
        train_artists = {row.get("artist_id") for row in self.gallery} - {None, "", "0"}
        query_artists = {row.get("artist_id") for row in self.queries} - {None, "", "0"}
        manifest = {"config": self.config, "fingerprint": self.fingerprint, "source_hashes": self.source_hashes,
                    "started_at": self.started_at, "updated_at": utc_now(), "metadata_splits": dict(raw_splits),
                    "valid_splits": dict(valid_splits), "excluded_invalid_or_missing": excluded,
                    "query_genres": dict(Counter(row["genre"] for row in self.queries)),
                    "shared_artist_ids_gallery_queries": sorted(train_artists & query_artists),
                    "experiments_complete": {"experiment7": e7_complete, "experiment8": e8_complete},
                    "observations": {"experiment7": len(e7), "experiment8": len(e8)},
                    "full_requirement_satisfied": self.scale == "full" and e7_complete and e8_complete,
                    "all_probes_exact": e8_complete and all(row["recall_at_10"] == 1 for row in e8 if row["all_clusters"]),
                    "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "numeric_threads": 2},
                    "methodology": {
                        "splits": "Galería formada únicamente por training; consultas estratificadas únicamente de test; validation no se usa para ajustar ni consultar. Codebook declara solo IDs training. DF, IDF y normas BoAW se recalculan sobre la galería; centroides IVF se aprenden únicamente de sus vectores.",
                        "semantic_relevance": "Relevancia aproximada por coincidencia de género musical FMA, evaluada con Precision@5 y Precision@10. Esta métrica mide recuperación semántica por género; no demuestra identificación de una grabación ni atribución de derechos de autor.",
                        "query_sampling": "Muestreo determinista equilibrado por género, cuatro consultas por cada uno de los ocho géneros en full por defecto. Promedio global equivale a macro-promedio por género. Los vecinos faltantes cuentan como no relevantes.",
                        "representations": "BoAW: MFCC20 sin c0 + normalización aprendida en training + codebook K-Means + histograma TFIDF + coseno en índice invertido propio. Deep: embeddings PANNs CNN6 preentrenado en AudioSet, 512D, promedio normalizado de segmentos de 10s; búsqueda exacta de coseno paginada.",
                        "extraction": "Consultas semánticas de pista completa (hasta 30 s), idéntico audio en ambos métodos. Tiempos reales por pista del último registro exitoso manifest.jsonl: MFCC y embedding de sus ventanas se muestran separados. Cuantización BoAW se mide nuevamente sobre descriptores reales. Decodificación se informa aparte. Lectura NPZ e inferencia se excluyen de latencia de consulta; no se presenta una suma de fases como medición end-to-end. Las consultas recortadas pertenecen a la evaluación independiente de identificación.",
                        "ivf": "IVF propio paginado entrenado solo en galería; C=floor(sqrt(N)). Probes 1,2,5,10,20,50 y C; valores mayores que C se deduplican en smoke. Recall@10 se calcula contra VectorStore exacto con idénticos vectores normalizados. Con C probes se exige igualdad de IDs y distancias.",
                        "timing": "Consultas con caché de SO caliente y una consulta de calentamiento por índice. Reloj perf_counter_ns; estadísticas muestrales. Una consulta por pista y configuración; variación refleja consultas distintas. Fases de construcción y extracción se registran separadamente.",
                        "io": "Transferencias reales de páginas de 4096B solicitadas al gestor, con desglose índice/datos. Vector512D ocupa una entrada por página en este diseño. Los resultados son IDs/similitudes; no se materializan títulos ni otros metadatos durante la medición.",
                        "scope": "Smoke usa únicamente pistas reales disponibles del piloto y no satisface los requisitos masivos. Full exige preparación completa de las 8000 pistas originales; archivos no decodificables se excluyen explícitamente, sin sustituirlos por muestras sintéticas.",
                        "checkpoints": "Resultados atómicos por consulta y fingerprint de código, vocabulario, galería y cachés. No se extrapolan ni rellenan resultados faltantes."}}
        result = {"manifest": manifest, "builds": builds, "experiment7_queries": e7, "experiment7_summary": summarize_e7(e7),
                  "experiment8_queries": e8, "experiment8_summary": summarize_e8(e8)}
        for name, rows in result.items():
            if name != "manifest":
                write_csv(self.output / f"{name}.csv", rows)
        atomic_json(self.output / "manifest.json", manifest)
        atomic_json(self.output / "results.json", result)
        if plots:
            make_plots(self.output, result)
        return manifest


def make_plots(output, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2})
    suffix = " · SMOKE" if result["manifest"]["config"]["scale"] == "smoke" else ""
    colors = {"boaw_tfidf": "#4477AA", "deep_exact": "#CC6677"}
    labels = {"boaw_tfidf": "BoAW + TF-IDF", "deep_exact": "PANNs + exacto"}
    e7 = result["experiment7_summary"]
    overall = {row["method"]: row for row in e7 if row["genre"] == "ALL_MACRO"}
    if overall:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
        methods = [m for m in colors if m in overall]
        xs = np.arange(len(methods))
        for offset, metric, label in ((-.18, "precision_at_5", "P@5"), (.18, "precision_at_10", "P@10")):
            axes[0].bar(xs + offset, [overall[m][f"{metric}_mean"] for m in methods], .35, label=label)
        axes[0].set(ylim=(0, 1), ylabel="Precisión por coincidencia de género", xticks=xs, xticklabels=[labels[m] for m in methods])
        axes[0].legend()
        for ax, metric, label in ((axes[1], "latency_ms", "Latencia media de búsqueda (ms)"), (axes[2], "disk_reads", "Bloques leídos por búsqueda")):
            ax.bar(xs, [overall[m][f"{metric}_mean"] for m in methods], color=[colors[m] for m in methods])
            ax.set(yscale="log", ylabel=label, xticks=xs, xticklabels=[labels[m] for m in methods])
        for ax in axes:
            ax.tick_params(axis="x", labelsize=8)
        fig.suptitle("E7 · Representación clásica y embedding profundo" + suffix)
        fig.supxlabel("Consultas test estratificadas · galería training · inferencia y cuantización fuera del tiempo de búsqueda", fontsize=8)
        fig.savefig(output / "experiment7_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        genres = sorted({r["genre"] for r in e7 if r["genre"] != "ALL_MACRO"})
        fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
        for offset, method in zip((-.18, .18), methods):
            data = {r["genre"]: r for r in e7 if r["method"] == method}
            ax.bar(np.arange(len(genres)) + offset, [data[g]["precision_at_10_mean"] for g in genres], .35, color=colors[method], label=labels[method])
        ax.set(xticks=range(len(genres)), xticklabels=genres, ylim=(0, 1), ylabel="Precision@10", title="E7 · Recuperación semántica por género" + suffix)
        ax.legend()
        fig.savefig(output / "experiment7_genres.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        for i, method in enumerate(methods):
            row = overall[method]
            extraction = row.get("extraction_ms_mean", 0)
            quantization = row.get("quantization_ms_mean", 0)
            ax.bar(i, extraction, color="#4477AA", label="Extracción observada" if i == 0 else None)
            if quantization:
                ax.bar(i, quantization, bottom=extraction, color="#DDCC77", label="Cuantización" if i == 0 else None)
        ax.set(xticks=xs, xticklabels=["MFCC + cuantización" if m == "boaw_tfidf" else "Embedding PANNs" for m in methods], ylabel="Tiempo medio por consulta (ms)", title="E7 · Coste de representación, separado de búsqueda" + suffix)
        ax.legend()
        fig.supxlabel("Extracción: registros reales de preparación por pista. Cuantización: medida nuevamente sobre MFCC almacenados.", fontsize=8)
        fig.savefig(output / "experiment7_extraction.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    e8 = result["experiment8_summary"]
    if e8:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for ax, xfield, xlabel in ((axes[0], "latency_ms_mean", "Latencia media (ms)"), (axes[1], "disk_reads_mean", "Bloques leídos por consulta")):
            xs = [row[xfield] for row in e8]
            ys = [row["recall_at_10_mean"] * 100 for row in e8]
            ax.plot(xs, ys, marker="o", color="#228833")
            for x, y, row in zip(xs, ys, e8):
                ax.annotate(f"p={row['probes']}", (x, y), xytext=(5, -12 if y > 95 else 6), textcoords="offset points", fontsize=8)
            ax.set(xlabel=xlabel, ylabel="Recall@10 (%)", ylim=(0, 105))
        fig.suptitle("E8 · Calidad y coste del IVF paginado" + suffix)
        fig.supxlabel("Verdad base: búsqueda exacta de los mismos vectores · todas las listas deben recuperar Recall@10 = 100 %", fontsize=9)
        fig.savefig(output / "experiment8_tradeoff.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/cache/audio")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--codebook", type=Path)
    parser.add_argument("--queries-per-genre", type=int)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or ROOT / "benchmarks/results/parte2_audio" / args.scale
    if args.plots_only:
        make_plots(output, read_json(output / "results.json"))
        return 0
    with threadpool_limits(limits=2):
        bench = AudioBenchmark(args.scale, args.cache, output, args.queries_per_genre, args.codebook)
        try:
            bench.experiment7()
            bench.experiment8()
        finally:
            manifest = bench.export(plots=True)
        if all(manifest["experiments_complete"].values()) and not args.keep_work:
            shutil.rmtree(bench.work)
        bench.log(f"Finalizado: requisito completo E7/E8={manifest['full_requirement_satisfied']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
