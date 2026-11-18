"""Carga reproducible de colecciones e índices sobre páginas binarias propias."""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import struct
from pathlib import Path

import numpy as np

from .config import DATA_DIR, AUDIO_CACHE, METADATA_DIR
from .minidb.multimodal import MultimodalDatabase
from .minidb.blob import BlobStore
from .minidb.vector import VectorStore, IVFIndex
from .minidb.text import InvertedIndex


def column(name, kind='INT', size=None, primary=False):
    return dict(name=name, type=kind, **({'size': size} if size else {}), **({'primary': True} if primary else {}))


def clipped(text, maximum):
    return str(text).encode('utf-8')[:maximum].decode('utf-8', errors='ignore')


def text_documents(metadata_dir, limit=20000):
    """Esta función reúne texto original de títulos, biografías y descripciones de FMA."""
    with (Path(metadata_dir) / 'tracks.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream)
        header = list(zip(next(reader), next(reader)))
        next(reader)
        fields = [('track','title'), ('artist','name'), ('album','title'), ('track','genre_top'),
                  ('artist','bio'), ('album','information'), ('track','information'),
                  ('track','tags'), ('artist','tags'), ('album','tags')]
        positions = [header.index(f) for f in fields if f in header]
        title_position = header.index(('track', 'title'))
        for number, row in enumerate(reader):
            if number >= limit:
                return
            body = html.unescape(re.sub(r'<[^>]+>', ' ', ' '.join(row[p] for p in positions if row[p])))
            yield int(row[0]), row[title_position], re.sub(r'\s+', ' ', body).strip()


def prepare_space(db, count=100000):
    if 'locales' in db.catalog:
        return
    db.create_table('locales', [column('id', primary=True), column('nombre','CHAR',64), column('lon','FLOAT'), column('lat','FLOAT')])
    rng = np.random.default_rng(42)
    file = db.tables['locales']
    for identifier in range(1, count+1):
        lon, lat = (-77.04, -12.05) + rng.uniform(-.1, .1, 2)
        file.insert(dict(id=identifier, nombre=f'Local simulado {identifier}', lon=float(lon), lat=float(lat)))
    db.create_index('locales','idx_local_id','id','BTREE')
    db.register('locales', virtual_columns={'ubicacion':'POINT'}, source='Coordenadas sintéticas de Lima; semilla 42')
    db.create_multimodal_index('locales','idx_geo','ubicacion','RTREE')
    print(f'locales: {count:,} puntos e índice R-Tree.', flush=True)


def prepare_text(db, metadata_dir, count=20000):
    if 'documentos' in db.catalog:
        return
    db.create_table('documentos',[column('id',primary=True),column('titulo','CHAR',512),column('cuerpo_ptr')])
    with_blob = BlobStore(db.directory/'documentos.blob',db.counter)
    try:
        for docid, title, body in text_documents(metadata_dir, count):
            db.tables['documentos'].insert(dict(id=docid,titulo=clipped(title,512),cuerpo_ptr=with_blob.append(body.encode('utf-8'))))
    finally:
        with_blob.close()
    db.create_index('documentos','idx_doc_id','id','BTREE')
    db.register('documentos',virtual_columns={'cuerpo':'TEXT'},blobs='documentos.blob',source='Metadatos FMA, no audio adicional')
    db.create_multimodal_index('documentos','idx_texto','cuerpo','INVERTED')
    print(f'documentos: {db.tables["documentos"].record_count:,} textos e índice invertido.',flush=True)


def prepare_audio(db, cache_dir, limit=None, pilot=False):
    from .audio.features import AudioCodebook
    from .audio.prepare import load_features, cache_path
    cache_dir = Path(cache_dir)
    codebook_path = cache_dir / ('codebook.pilot.npz' if pilot else 'codebook.npz')
    if not codebook_path.exists():
        raise ValueError(f'Completar antes la extracción y el vocabulario acústico: {codebook_path.name}')
    if 'pistas' in db.catalog:
        return
    with (cache_dir/'metadata.csv').open(encoding='utf-8',newline='') as stream:
        records = [r for r in csv.DictReader(stream) if cache_path(cache_dir,r['id']).exists()]
    records = records[:limit] if limit is not None else records
    if not pilot and len(records) < 5000:
        raise ValueError('La colección final requiere al menos 5000 pistas extraídas.')
    codebook = AudioCodebook.load(codebook_path)
    db.create_table('pistas',[column('id',primary=True),column('titulo','CHAR',512),column('artista','CHAR',256),
                            column('genero','CHAR',48),column('particion','CHAR',16),column('duracion','FLOAT'),
                            column('embedding_ptr'),column('secuencia_ptr'),column('licencia','CHAR',256)])
    db.create_table('segmentos',[column('id',primary=True),column('pista_id'),column('inicio','FLOAT'),column('embedding_ptr')])
    blobs = BlobStore(db.directory/'pistas.blob',db.counter)
    segment_blobs = BlobStore(db.directory/'segmentos.blob',db.counter)
    vectors = VectorStore(db.directory/'pistas.vec',counter=db.counter,create=True)
    segment_vectors = VectorStore(db.directory/'segmentos.vec',counter=db.counter,create=True)
    counts, segment_id = [], 0
    try:
        for number, record in enumerate(records,1):
            identifier = int(record['id'])
            features = load_features(cache_path(cache_dir,identifier))
            sequence = codebook.normalize_sequence(features['sequence']).astype('<f4')
            seq_bytes = struct.pack('<II',*sequence.shape)+sequence.tobytes()
            embedding = features['embedding'].astype('<f4')
            rid = db.tables['pistas'].insert(dict(id=identifier,titulo=clipped(record['title'],512),artista=clipped(record['artist'],256),
                    genero=record['genre'],particion=record['split'],duracion=float(features['duration']),
                    embedding_ptr=blobs.append(embedding.tobytes()),secuencia_ptr=blobs.append(seq_bytes),licencia=clipped(record['license'],256)))
            vectors.append(rid,embedding)
            counts.append((identifier,codebook.counts(features['mfcc_descriptors'])))
            for start, vector in zip(features['segment_starts'],features['segment_embeddings']):
                segment_id += 1
                srid = db.tables['segmentos'].insert(dict(id=segment_id,pista_id=identifier,inicio=float(start),
                                            embedding_ptr=segment_blobs.append(vector.astype('<f4').tobytes())))
                segment_vectors.append(srid,vector)
            if number%1000 == 0:
                print(f'Paginando audio: {number}/{len(records)}',flush=True)
    finally:
        blobs.close();segment_blobs.close();vectors.close();segment_vectors.close()
    for table in ('pistas','segmentos'):
        db.create_index(table,f'idx_{table}_id','id','BTREE')
        db.register(table,virtual_columns={'embedding':'VECTOR(512)'},blobs=f'{table}.blob',vectors=f'{table}.vec',
                    codebook=str(codebook_path.name),source='FMA Small / PANNs Cnn6 AudioSet')
        db.create_multimodal_index(table,f'idx_{table}_ivf','embedding','IVF')
    db.create_index('pistas','idx_pistas_genero','genero','HASH')
    db.create_index('segmentos','idx_segmentos_pista','pista_id','HASH')
    boaw = InvertedIndex(db.directory/'pistas_boaw',counter=db.counter,create=True)
    boaw.build_tokens(counts)
    boaw.close()
    print(f'pistas: {len(records):,}; segmentos: {segment_id:,}; BoAW e IVF construidos.',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=DATA_DIR)
    parser.add_argument('--cache-dir',type=Path,default=AUDIO_CACHE)
    parser.add_argument('--metadata-dir',type=Path,default=METADATA_DIR)
    parser.add_argument('--without-audio',action='store_true')
    parser.add_argument('--audio-only',action='store_true')
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--audio-limit',type=int)
    parser.add_argument('--spatial-count',type=int,default=100000)
    parser.add_argument('--text-count',type=int,default=20000)
    args=parser.parse_args()
    db=MultimodalDatabase(args.directory)
    try:
        if not args.audio_only:
            prepare_space(db,args.spatial_count)
            prepare_text(db,args.metadata_dir,args.text_count)
        if not args.without_audio:
            prepare_audio(db,args.cache_dir,args.audio_limit,args.pilot)
    finally:
        db.close()


if __name__=='__main__':
    main()
