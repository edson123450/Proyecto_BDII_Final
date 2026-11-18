"""Preparación reanudable de FMA Small; NPZ es una caché intermedia de extracción.

Los índices consultables finales deben construirse con los archivos paginados
propios de minidb. No se copian MP3 ni se serializa una base de datos con NumPy.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np

from .features import AudioCodebook, AudioExtractor, FEATURE_VERSION


METADATA_FIELDS = ('id', 'genre', 'split', 'artist_id', 'title', 'artist', 'license', 'path')


def fma_metadata(audio_dir, metadata_dir):
    """Esta función lee únicamente los registros Small y conserva las divisiones oficiales."""
    audio_dir, metadata_dir = Path(audio_dir), Path(metadata_dir)
    records = []
    with open(metadata_dir / 'tracks.csv', encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle)
        columns = list(zip(next(reader), next(reader)))
        next(reader)  # Fila con el nombre del índice: track_id.
        positions = {pair: number for number, pair in enumerate(columns)}
        def cell(row, category, name):
            position = positions.get((category, name))
            return row[position] if position is not None else ''
        for row in reader:
            if cell(row, 'set', 'subset') != 'small':
                continue
            track_id = int(row[0])
            name = f'{track_id:06d}'
            records.append(dict(id=track_id, genre=cell(row, 'track', 'genre_top'),
                                split=cell(row, 'set', 'split'), artist_id=cell(row, 'artist', 'id'),
                                title=cell(row, 'track', 'title'), artist=cell(row, 'artist', 'name'),
                                license=cell(row, 'track', 'license'),
                                path=str(audio_dir / name[:3] / (name + '.mp3'))))
    records.sort(key=lambda row: row['id'])
    return records


def cache_path(output_dir, track_id):
    return Path(output_dir) / f'{int(track_id):06d}.npz'


def valid_cache(path, source):
    if not path.exists() or not source.exists():
        return False
    try:
        stat = source.stat()
        with np.load(path, allow_pickle=False) as data:
            return (str(data['feature_version']) == FEATURE_VERSION
                    and int(data['source_size']) == stat.st_size
                    and int(data['source_mtime_ns']) == stat.st_mtime_ns
                    and data['embedding'].shape == (512,)
                    and data['sequence'].ndim == 2 and data['sequence'].shape[1] == 13)
    except (ValueError, KeyError, OSError, EOFError):
        return False


def _save_cache(path, result, source):
    temporary = path.with_suffix('.temporary')
    stat = source.stat()
    with open(temporary, 'wb') as handle:
        np.savez_compressed(handle, mfcc_descriptors=result['mfcc_descriptors'],
                            sequence=result['sequence'], embedding=result['embedding'],
                            segment_embeddings=result['segment_embeddings'],
                            segment_starts=result['segment_starts'], duration=result['duration'],
                            sample_rate=result['sample_rate'], feature_version=FEATURE_VERSION,
                            source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns)
    os.replace(temporary, path)


def load_features(path):
    """Esta función carga una caché validada sin permitir objetos pickle."""
    with np.load(path, allow_pickle=False) as data:
        if str(data['feature_version']) != FEATURE_VERSION:
            raise ValueError('Versión incompatible de características')
        return {name: data[name] for name in ('mfcc_descriptors', 'sequence', 'embedding',
                                             'segment_embeddings', 'segment_starts', 'duration')}


def train_codebook(records, output_dir, words=256, sample_size=128000, seed=2026):
    """Esta función muestrea solo training; nunca usa validation ni test para ajustar."""
    training = [row for row in records if row['split'] == 'training'
                and valid_cache(cache_path(output_dir, row['id']), Path(row['path']))]
    if not training:
        raise ValueError('No hay pistas training extraídas para construir BoAW')
    rng = np.random.default_rng(seed)
    per_track = max(1, int(np.ceil(sample_size / len(training))))
    samples = []
    for row in training:
        with np.load(cache_path(output_dir, row['id']), allow_pickle=False) as data:
            descriptors = data['mfcc_descriptors']
        count = min(per_track, len(descriptors))
        samples.append(descriptors[rng.choice(len(descriptors), count, replace=False)])
    values = np.concatenate(samples)
    if len(values) > sample_size:
        values = values[rng.choice(len(values), sample_size, replace=False)]
    return AudioCodebook.fit(values, words=words, training_ids=[row['id'] for row in training], seed=seed)


def build_boaw_cache(records, output_dir, codebook, path):
    """Esta función produce conteos para InvertedIndex.build_tokens, sin stemming."""
    temporary = Path(str(path) + '.temporary')
    documents = 0
    with open(temporary, 'w', encoding='utf-8') as handle:
        for row in records:
            target = cache_path(output_dir, row['id'])
            if not valid_cache(target, Path(row['path'])):
                continue
            with np.load(target, allow_pickle=False) as data:
                counts = codebook.counts(data['mfcc_descriptors'])
            handle.write(json.dumps(dict(id=row['id'], counts=counts), separators=(',', ':')) + '\n')
            documents += 1
    os.replace(temporary, path)
    return documents


def write_boaw_metadata(codebook_path, codebook, document_count, path):
    """Esta función vincula los conteos BoAW al vocabulario y a sus pistas training."""
    digest = hashlib.sha256()
    with open(codebook_path, 'rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    training_bytes = json.dumps(list(codebook.training_ids), separators=(',', ':')).encode('utf-8')
    metadata = dict(codebook_sha256=digest.hexdigest(), feature_version=FEATURE_VERSION,
                    document_count=int(document_count), training_count=len(codebook.training_ids),
                    training_ids_digest=hashlib.sha256(training_bytes).hexdigest(),
                    training_ids_digest_encoding='UTF-8 compact JSON array of ordered integer IDs',
                    idf_scope='computed by each InvertedIndex on its own gallery')
    Path(path).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def prepare(audio_dir, metadata_dir, output_dir='data/cache/audio', limit=100,
            resume=True, device='cpu', batch_size=2, build_codebook=True, threads=2):
    """Esta función mide un piloto o extrae el corpus completo registrando cada fallo."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = fma_metadata(audio_dir, metadata_dir)
    selected = records if limit is None else records[:limit]
    with open(output_dir / 'metadata.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    start = perf_counter()
    run_id = datetime.now(timezone.utc).isoformat(timespec='seconds')
    extractor = None
    successful, failed, resumed, timing_records = [], [], 0, []
    with open(output_dir / 'manifest.jsonl', 'a', encoding='utf-8') as manifest:
        for number, row in enumerate(selected, 1):
            source, target = Path(row['path']), cache_path(output_dir, row['id'])
            if resume and valid_cache(target, source):
                successful.append(row)
                resumed += 1
                continue
            if extractor is None:
                extractor = AudioExtractor(output_dir.parent, device=device, batch_size=batch_size, threads=threads)
            try:
                result = extractor.extract(source)
                _save_cache(target, result, source)
                successful.append(row)
                timing_records.append(result['timings'])
                entry = dict(id=row['id'], status='ok', run_id=run_id, device=str(extractor.model.device),
                             batch_size=batch_size, threads=threads, feature_version=FEATURE_VERSION,
                             **result['timings'], duration=result['duration'],
                             frames=len(result['sequence']), segments=len(result['segment_embeddings']))
            except Exception as exc:
                entry = dict(id=row['id'], status='error', run_id=run_id, error=f'{type(exc).__name__}: {exc}')
                failed.append(entry)
            manifest.write(json.dumps(entry, ensure_ascii=False) + '\n')
            manifest.flush()
            if number % 10 == 0 or number == len(selected):
                print(json.dumps(dict(processed=number, total=len(selected), valid=len(successful),
                                      errors=len(failed), elapsed_seconds=round(perf_counter() - start, 2))), flush=True)
    extraction_seconds = perf_counter() - start
    codebook_start = perf_counter()
    suffix = '.pilot' if limit is not None and limit < len(records) else ''
    codebook_path = output_dir / f'codebook{suffix}.npz'
    boaw_path = output_dir / f'boaw{suffix}.jsonl'
    boaw_documents = 0
    if build_codebook and successful:
        codebook = train_codebook(successful, output_dir)
        codebook.save(codebook_path)
        boaw_documents = build_boaw_cache(successful, output_dir, codebook, boaw_path)
        write_boaw_metadata(codebook_path, codebook, boaw_documents, output_dir / f'boaw_metadata{suffix}.json')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        peak *= 1024
    report = dict(feature_version=FEATURE_VERSION, run_id=run_id, batch_size=batch_size, threads=threads, selected=len(selected), valid=len(successful),
                  failed=failed, resumed=resumed, newly_extracted=len(timing_records),
                  extraction_seconds=extraction_seconds, preparation_seconds=perf_counter() - start,
                  codebook_seconds=perf_counter() - codebook_start if build_codebook else 0,
                  codebook=str(codebook_path) if build_codebook else None,
                  boaw=str(boaw_path) if build_codebook else None, boaw_documents=boaw_documents,
                  peak_rss_bytes=peak, device=str(extractor.model.device) if extractor else device,
                  mean_timings_ms={key: float(np.mean([entry[key] for entry in timing_records]))
                                   for key in timing_records[0]} if timing_records else {},
                  valid_ids=[row['id'] for row in successful],
                  cache_bytes=sum(cache_path(output_dir, row['id']).stat().st_size for row in successful))
    report_name = ('pilot_report.json' if suffix else 'preparation_report.json') if build_codebook else f'extraction_report{suffix}.json'
    report_path = output_dir / report_name
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key not in ('valid_ids', 'failed')}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir', default=os.environ.get('FMA_AUDIO_DIR'))
    parser.add_argument('--metadata-dir', default=os.environ.get('FMA_METADATA_DIR'))
    parser.add_argument('--output-dir', default='data/cache/audio')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--extract-only', action='store_true')
    args = parser.parse_args()
    if not args.audio_dir or not args.metadata_dir:
        parser.error('Configurar FMA_AUDIO_DIR y FMA_METADATA_DIR o proporcionar las rutas')
    if args.limit < 1:
        parser.error('--limit debe ser positivo')
    report = prepare(args.audio_dir, args.metadata_dir, args.output_dir,
                     limit=None if args.all else args.limit, resume=not args.no_resume,
                     device=args.device, batch_size=args.batch_size, build_codebook=not args.extract_only, threads=args.threads)
    if args.all and report['valid'] < 5000:
        raise SystemExit('No se alcanzaron las 5000 pistas válidas requeridas')


if __name__ == '__main__':
    main()
