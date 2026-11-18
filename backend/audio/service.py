"""Consultas acústicas conectadas con los índices y registros físicos propios."""
from __future__ import annotations

import struct
from time import perf_counter
from pathlib import Path

import numpy as np

from ..config import AUDIO_CACHE, AUDIO_DIR
from ..minidb.text import InvertedIndex
from .features import AudioCodebook, AudioExtractor
from .dtw import search_tracks, warmup


class AudioService:
    def __init__(self, database):
        self.db = database
        self.extractor = None
        self.codebook = None
        self.boaw = None

    def ready(self):
        return 'pistas' in self.db.tables and (self.db.directory/'pistas_boaw/vocab.dic').exists()

    def _load(self):
        if not self.ready():
            raise ValueError('Completar la preparación de las pistas antes de consultar audio.')
        if self.codebook is None:
            name = self.db.catalog['pistas']['multimodal']['codebook']
            self.codebook = AudioCodebook.load(AUDIO_CACHE/name)
            self.boaw = InvertedIndex(self.db.directory/'pistas_boaw',counter=self.db.counter)
        if self.extractor is None:
            self.extractor = AudioExtractor(AUDIO_CACHE.parent,device='cpu',batch_size=2)
        warmup()

    @staticmethod
    def path(identifier):
        if not isinstance(identifier,int) or identifier<1:
            raise ValueError('ID de pista inválido.')
        name=f'{identifier:06d}'
        return AUDIO_DIR/name[:3]/(name+'.mp3')

    def info(self):
        if not self.ready():
            return dict(ready=False,count=0,segments=0,genres=[],dimension=512,clusters=0)
        genres = sorted({row['genero'] for _,row in self.db.tables['pistas'].scan()})
        index,_ = self.db._index('segmentos','IVF')
        return dict(ready=True,count=self.db.tables['pistas'].record_count,
                    segments=self.db.tables['segmentos'].record_count,genres=genres,dimension=512,clusters=index.clusters)

    @staticmethod
    def public_row(row):
        return dict(id=row['id'],title=row['titulo'],artist=row['artista'],genre=row['genero'],
                    duration=row['duracion'],license=row['licencia'],audio_url=f'/api/audio/track/{row["id"]}')

    def catalog(self,limit=100):
        if not self.ready():
            return []
        rows=[]
        for _,row in self.db.tables['pistas'].scan():
            rows.append(self.public_row(row))
            if len(rows)>=limit:
                break
        return rows

    def _track(self,identifier):
        item=self.db.get_by_id('pistas',identifier)
        if not item:
            raise ValueError('La pista no pertenece a esta colección.')
        return item[1]

    def _sequence(self,row):
        raw=self.db.resources['pistas']['blobs'].read(row['secuencia_ptr'])
        length,dimensions=struct.unpack_from('<II',raw)
        return np.frombuffer(raw,offset=8,dtype='<f4').reshape(length,dimensions)

    def _method(self,name,started,rows,plan,**details):
        return dict(name=name,rows=rows,plan=plan,metrics={**self.db.counter.snapshot(),
                    'execution_ms':(perf_counter()-started)*1000},**details)

    def search(self,waveform,sample_rate,k=5,probes=5,include_dtw=True):
        """Esta función compara las representaciones sin usar ANN para excluir pistas de DTW."""
        self._load()
        features=self.extractor.extract_waveform(waveform,sample_rate)
        started=perf_counter()
        counts=self.codebook.counts(features['mfcc_descriptors'])
        features['timings']['quantization_ms']=(perf_counter()-started)*1000
        methods=[]
        self.db.counter.reset();started=perf_counter()
        found=self.boaw.query_tokens(counts,k=k)
        rows=[dict(**self.public_row(self._track(identifier)),score=float(score)) for identifier,score in found]
        methods.append(self._method('BoAW',started,rows,dict(operation='InvertedScan',index='pistas_boaw',reason='MFCC → palabras acústicas → TF-IDF/coseno')))
        seeds=[]
        store=self.db.resources['segmentos']['vectors']
        index,spec=self.db._index('segmentos','IVF')
        for name,approximate in [('IVF',True),('Exacto',False)]:
            self.db.counter.reset();started=perf_counter()
            # Hasta seis ventanas por pista: recuperar suficientes RID antes de agrupar.
            found=index.knn(features['embedding'],k=min(k*6,store.count),probes=probes) if approximate else store.knn(features['embedding'],k=min(k*6,store.count))
            rows=[];seen=set()
            for rid,distance in found:
                segment=self.db.tables['segmentos'].get(rid)
                identifier=segment['pista_id']
                if identifier in seen:
                    continue
                seen.add(identifier)
                rows.append(dict(**self.public_row(self._track(identifier)),distance=float(distance),start_seconds=segment['inicio']))
                if len(rows)==k:
                    break
            if approximate:
                seeds=[r['id'] for r in rows]
            methods.append(self._method(name,started,rows,dict(operation='IVFScan' if approximate else 'VectorScan',
                        index=spec['name'] if approximate else None,reason=f'Coseno de segmentos de 10 s; probes={min(probes,index.clusters)}' if approximate else 'kNN lineal exacto de los mismos segmentos, con max-heap')))
        if include_dtw:
            self.db.counter.reset();started=perf_counter()
            normalized=self.codebook.normalize_sequence(features['sequence'])
            seed_tracks=((identifier,self._sequence(self._track(identifier))) for identifier in seeds)
            def tracks():
                for _,row in self.db.tables['pistas'].scan():
                    if row['id'] not in seeds:
                        yield row['id'],self._sequence(row)
            found,stats=search_tracks(normalized,tracks(),k=k,seed_tracks=seed_tracks)
            rows=[dict(**self.public_row(self._track(item['track_id'])),distance=item['cost'],start_seconds=item['start_seconds']) for item in found]
            methods.append(self._method('DTW + LB_Keogh',started,rows,
                    dict(operation='FilterAndRefine',index=None,reason='Todas las pistas; ANN solo ordena las primeras evaluaciones'),details=stats))
        return dict(query={'duration':features['duration']},extraction_ms=features['timings'],methods=methods,
                    corpus_count=self.db.tables['pistas'].record_count,
                    dtw_scope='DTW exacto entre ventanas de igual duración, pasos de 200 ms y banda del 10 % (mínimo 2 pasos). La cota no omite vecinos dentro de ese universo.')

    def close(self):
        if self.boaw is not None:
            self.boaw.close()
