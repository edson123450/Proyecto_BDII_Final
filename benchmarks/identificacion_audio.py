#!/usr/bin/env python3
"""Validación de identificación sobre originales incluidos en el corpus.

Esta evaluación de fragmentos es independiente de E7 (género, test→training)
y de E8 (Recall IVF frente al exacto). No exige ni fabrica aciertos.

    .venv/bin/python benchmarks/identificacion_audio.py --plan-only
    .venv/bin/python benchmarks/identificacion_audio.py --limit-queries 1
    .venv/bin/python benchmarks/identificacion_audio.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.audio.features import load_audio
from backend.config import DATA_DIR, AUDIO_CACHE
from benchmarks.parte1 import atomic_json, read_json, utc_now, write_csv
from benchmarks.parte2_audio import read_metadata, select_stratified_test, file_sha256


VARIANTS = (
    {"name": "crop_5s", "start_seconds": 7.2, "duration_seconds": 5.0, "snr_db": None},
    {"name": "crop_10s", "start_seconds": 2.0, "duration_seconds": 10.0, "snr_db": None},
    {"name": "noise_20db_10s", "start_seconds": 7.2, "duration_seconds": 10.0, "snr_db": 20.0},
)


def crop_audio(waveform, sample_rate, start_seconds, duration_seconds):
    if sample_rate <= 0 or start_seconds < 0 or duration_seconds <= 0:
        raise ValueError("Parámetros de recorte inválidos")
    first, count = round(start_seconds * sample_rate), round(duration_seconds * sample_rate)
    if len(waveform) < first + count:
        raise ValueError("La pista no alcanza la duración requerida por el recorte predefinido")
    return np.ascontiguousarray(waveform[first:first + count], dtype=np.float32)


def noise_at_snr(waveform, snr_db=20.0, seed=42):
    """AWGN reproducible con potencia medida: no recortar amplitud tras añadir ruido."""
    signal = np.asarray(waveform, dtype=np.float64)
    if not signal.size or not np.isfinite(signal).all() or not math.isfinite(snr_db):
        raise ValueError("La señal y SNR deben ser finitos")
    signal_power = float(np.mean(signal * signal))
    if signal_power <= 0:
        raise ValueError("No se define SNR finito sobre señal silenciosa")
    noise = np.random.default_rng(seed).standard_normal(signal.shape)
    noise *= math.sqrt(signal_power / (10 ** (snr_db / 10)) / float(np.mean(noise * noise)))
    noisy = (signal + noise).astype(np.float32)
    measured_noise = noisy.astype(np.float64) - signal
    measured_snr = 10 * math.log10(signal_power / float(np.mean(measured_noise * measured_noise)))
    return noisy, measured_snr


def identification_score(retrieved, track_id):
    ids = [int(item["id"]) for item in retrieved]
    if len(ids) != len(set(ids)):
        raise ValueError("Los resultados agrupados por pista contienen duplicados")
    rank = ids.index(track_id) + 1 if track_id in ids else None
    return {"hit_at_1": int(rank == 1), "hit_at_5": int(rank is not None and rank <= 5),
            "rank_within_top5": rank, "retrieved_ids": ids}


def make_plan(metadata, allowed_ids=None):
    candidates = [row for row in metadata if allowed_ids is None or row["id"] in allowed_ids]
    genres = sorted({row["genre"] for row in metadata})
    tracks = select_stratified_test(candidates, 1, seed=20261118, required_genres=genres)
    return [{"track_id": row["id"], "genre": row["genre"], "split": row["split"], "path": row["path"],
             "variant": variant["name"], "start_seconds": variant["start_seconds"],
             "duration_seconds": variant["duration_seconds"], "snr_db": variant["snr_db"], "noise_seed": 42}
            for row in tracks for variant in VARIANTS]


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["method"])].append(row)
    output = []
    for (variant, method), sample in groups.items():
        row = {"variant": variant, "method": method, "queries": len(sample),
               "hits_at_1": sum(r["hit_at_1"] for r in sample), "hits_at_5": sum(r["hit_at_5"] for r in sample)}
        for metric in ("hit_at_1", "hit_at_5", "execution_ms", "disk_reads", "disk_writes", "index_reads", "data_reads"):
            values = [item[metric] for item in sample]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(row)
    return output


class IdentificationBenchmark:
    def __init__(self, database_path, output, probes=5):
        from backend.minidb.multimodal import MultimodalDatabase
        from backend.audio.service import AudioService
        self.output, self.database_path = output.resolve(), database_path.resolve()
        self.db = MultimodalDatabase(self.database_path)
        self.service = AudioService(self.db)
        if not self.service.ready():
            self.db.close()
            raise ValueError("La base de datos aún no contiene la colección acústica completa")
        tracks = {row["id"]: row for _, row in self.db.tables["pistas"].scan()}
        name = self.db.catalog["pistas"]["multimodal"]["codebook"]
        if len(tracks) < 5000 or "pilot" in name:
            self.db.close()
            raise ValueError("La validación final requiere corpus completo y codebook de producción; no piloto")
        report = read_json(AUDIO_CACHE / "preparation_report.json")
        if report.get("selected") != 8000 or report.get("valid") != len(tracks):
            self.db.close()
            raise ValueError("La cantidad de pistas paginadas no coincide con la preparación completa FMA Small")
        self.plan = make_plan(read_metadata(AUDIO_CACHE / "metadata.csv"), set(tracks))
        if len(self.plan) != 24:
            self.db.close()
            raise ValueError("Se requieren ocho géneros y tres variantes: 24 consultas")
        self.config = {"probes": probes, "k": 5, "corpus_tracks": len(tracks),
                       "corpus_segments": self.db.tables["segmentos"].record_count,
                       "codebook_sha256": file_sha256(AUDIO_CACHE / name),
                       "catalog_sha256": file_sha256(self.database_path / "catalog.bin"),
                       "plan": self.plan}
        names = ("benchmarks/identificacion_audio.py", "backend/audio/service.py", "backend/audio/features.py", "backend/audio/model.py",
                 "backend/audio/dtw.py", "backend/minidb/vector.py", "backend/minidb/text.py", "backend/minidb/multimodal.py")
        self.source_hashes = {name: file_sha256(ROOT / name) for name in names}
        self.fingerprint = hashlib.sha256(json.dumps([self.config, self.source_hashes], sort_keys=True).encode()).hexdigest()[:20]
        self.checkpoints = self.output / "checkpoints" / self.fingerprint
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.started_at = utc_now()

    def verify_sources(self):
        for name, sha in self.source_hashes.items():
            if file_sha256(ROOT / name) != sha:
                raise RuntimeError("La implementación de identificación cambió durante la evaluación")

    def run(self, limit=None):
        if all((self.checkpoints / f"{query['track_id']:06d}_{query['variant']}.json").exists() for query in self.plan):
            return self.export(plots=True)
        # Cargar pesos y compilar DTW fuera de las consultas cronometradas.
        self.service._load()
        selected = self.plan if limit is None else self.plan[:limit]
        for number, query in enumerate(selected, 1):
            path = self.checkpoints / f"{query['track_id']:06d}_{query['variant']}.json"
            if path.exists():
                continue
            print(f"[{utc_now()}] Identificación {number}/{len(selected)}: {query['track_id']} {query['variant']}", flush=True)
            started = time.perf_counter_ns()
            original, sample_rate = load_audio(query["path"], max_seconds=30)
            decode_ms = (time.perf_counter_ns() - started) / 1e6
            clip = crop_audio(original, sample_rate, query["start_seconds"], query["duration_seconds"])
            actual_snr = None
            if query["snr_db"] is not None:
                clip, actual_snr = noise_at_snr(clip, query["snr_db"], seed=query["noise_seed"])
            started = time.perf_counter_ns()
            response = self.service.search(clip, sample_rate, k=5, probes=self.config["probes"], include_dtw=True)
            service_ms = (time.perf_counter_ns() - started) / 1e6
            rows = []
            for method in response["methods"]:
                counts = method["metrics"]
                if counts["disk_writes"]:
                    raise AssertionError("La identificación no debe modificar páginas")
                score = identification_score(method["rows"], query["track_id"])
                rows.append({"track_id": query["track_id"], "genre": query["genre"], "variant": query["variant"],
                             "method": method["name"], **score, "execution_ms": counts["execution_ms"],
                             "disk_reads": counts["disk_reads"], "disk_writes": counts["disk_writes"],
                             "index_reads": counts["by_role"]["index"]["disk_reads"],
                             "data_reads": counts["by_role"]["data"]["disk_reads"],
                             "actual_snr_db": actual_snr, "start_seconds": query["start_seconds"],
                             "duration_seconds": query["duration_seconds"]})
            if {row["method"] for row in rows} != {"BoAW", "IVF", "Exacto", "DTW + LB_Keogh"}:
                raise AssertionError("No se evaluaron los cuatro métodos de identificación")
            self.verify_sources()
            atomic_json(path, {"query": query, "rows": rows, "decode_ms": decode_ms,
                               "service_total_ms": service_ms, "response": response, "completed_at": utc_now()})
            self.export()
        return self.export(plots=True)

    def export(self, plots=False):
        units = [read_json(path) for path in sorted(self.checkpoints.glob("*.json"))]
        rows = [row for unit in units for row in unit["rows"]]
        extractions = [{"track_id": unit["query"]["track_id"], "variant": unit["query"]["variant"],
                        "decode_ms": unit["decode_ms"], "service_total_ms": unit["service_total_ms"],
                        **unit["response"]["extraction_ms"]} for unit in units]
        result = {"manifest": {"config": self.config, "fingerprint": self.fingerprint, "source_hashes": self.source_hashes,
                              "started_at": self.started_at, "updated_at": utc_now(), "queries_completed": len(units),
                              "queries_expected": 24, "method_observations": len(rows), "full_requirement_satisfied": len(units) == 24 and len(rows) == 96,
                              "methodology": {
                                  "task": "Identificación de grabaciones cuyo original está incluido deliberadamente en el corpus completo. Ocho pistas test, una por género; esto no mide generalización a originales ausentes y es independiente de E7/E8.",
                                  "variants": "Recorte5s desde7.2s, recorte10s desde2.0s y recorte10s desde7.2s con ruido gaussiano aditivo a SNR20dB. Semilla42. Inicios desalineados respecto a ventanas de10s con stride5s. Selección fijada antes de observar aciertos.",
                                  "corpus": "Corpus FMA Small completo preparado: BoAW sobre pistas, IVF y exacto sobre segmentos persistidos y agrupados por pista. El original permanece en cada método. Se informa Top1/Top5 y también los fallos.",
                                  "dtw": "AudioService ejecuta DTW con LB_Keogh sobre todas las pistas; ANN solo siembra el heap. Universo exacto: ventanas MFCC13 de igual duración, inicios cada200ms, banda10% y normalización del codebooktraining. La garantía no abarca todas las alineaciones continuas posibles.",
                                  "timing": "Extracción nueva del mismo fragmento para los cuatro métodos, tiempos de extracción separados. Latencia de cada método incluye recuperación de filas/metadata. service_total_ms mide extracción y las cuatro búsquedas completas, excluyendo decodificación/carga inicial del modelo. Pesos y compilaciónDTW se calientan antes. No se reutilizan tiempos de pistas30s para fragmentos5/10s.",
                                  "metrics": "Hit@1 y Hit@5 por identidad, sin imponer éxito. I/O paginado del motor; métricas exactas de los métodos retornadas por AudioService. Checkpoints conservan respuesta completa auditable."}},
                  "queries": rows, "summary": summarize(rows), "extraction": extractions}
        self.output.mkdir(parents=True, exist_ok=True)
        atomic_json(self.output / "results.json", result)
        atomic_json(self.output / "manifest.json", result["manifest"])
        for name in ("queries", "summary", "extraction"):
            write_csv(self.output / f"identification_{name}.csv", result[name])
        if plots and rows:
            make_plots(self.output, result)
        return result["manifest"]

    def close(self):
        self.service.close()
        self.db.close()


def make_plots(output, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2})
    methods = ["BoAW", "IVF", "Exacto", "DTW + LB_Keogh"]
    variants = [variant["name"] for variant in VARIANTS]
    labels = {"crop_5s": "Recorte 5 s", "crop_10s": "Recorte 10 s", "noise_20db_10s": "10 s + ruido 20 dB"}
    colors = ["#4477AA", "#228833", "#CC6677", "#AA3377"]
    records = {(r["variant"], r["method"]): r for r in result["summary"]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for method_number, (method, color) in enumerate(zip(methods, colors)):
        for ax, metric in ((axes[0], "hit_at_1_mean"), (axes[1], "hit_at_5_mean")):
            values = [records.get((variant, method), {}).get(metric, np.nan) * 100 for variant in variants]
            ax.bar(np.arange(3) + (method_number - 1.5) * .19, values, width=.18, color=color, label=method)
            ax.set(xticks=range(3), xticklabels=[labels[v] for v in variants], ylim=(0, 105), ylabel="Consultas identificadas (%)")
    axes[0].set_title("Primera respuesta (Hit@1)")
    axes[1].set_title("Primeras cinco respuestas (Hit@5)")
    axes[1].legend(fontsize=8, loc="lower right")
    state = "" if result["manifest"]["full_requirement_satisfied"] else " · PARCIAL"
    fig.suptitle("Identificación de fragmentos con originales incluidos" + state)
    fig.supxlabel("Ocho pistas, una por género · corpus completo · selección e inicios fijados previamente · aciertos y fallos observados", fontsize=8)
    fig.savefig(output / "identification_accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for method, color in zip(methods, colors):
        for ax, metric, ylabel in ((axes[0], "execution_ms_mean", "Latencia media de búsqueda (ms)"),
                                   (axes[1], "disk_reads_mean", "Bloques leídos por búsqueda")):
            ax.plot(range(3), [records.get((v, method), {}).get(metric, np.nan) for v in variants], marker="o", label=method, color=color)
            ax.set(xticks=range(3), xticklabels=[labels[v] for v in variants], yscale="log", ylabel=ylabel)
    axes[0].legend(fontsize=8)
    fig.suptitle("Identificación · Coste medido de cada método" + state)
    fig.savefig(output / "identification_cost.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/parte2_audio/identificacion/full")
    parser.add_argument("--probes", type=int, default=5)
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args(argv)
    if args.probes < 1 or args.limit_queries is not None and args.limit_queries < 1:
        parser.error("probes y limit-queries deben ser positivos")
    if args.plan_only:
        plan = make_plan(read_metadata(AUDIO_CACHE / "metadata.csv"))
        atomic_json(args.output / "planned_queries.json", plan)
        print(json.dumps({"planned_tracks": len({r['track_id'] for r in plan}), "planned_queries": len(plan), "path": str(args.output / 'planned_queries.json')}))
        return 0
    if args.plots_only:
        make_plots(args.output, read_json(args.output / "results.json"))
        return 0
    with threadpool_limits(limits=2):
        bench = IdentificationBenchmark(args.database, args.output, args.probes)
        try:
            manifest = bench.run(args.limit_queries)
        finally:
            bench.close()
    print(json.dumps({"queries_completed": manifest["queries_completed"], "full_requirement_satisfied": manifest["full_requirement_satisfied"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
