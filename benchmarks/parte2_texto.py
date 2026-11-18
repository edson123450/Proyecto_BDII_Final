#!/usr/bin/env python3
"""E6: índice invertido propio, escaneo paginado y GIN PostgreSQL reales."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import html
import heapq
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.config import METADATA_DIR
from backend.minidb.io import DiskCounter, DiskFile
from backend.minidb.text import InvertedIndex, TextNormalizer

PAGE = 4096
_HEADER = struct.Struct('<4sIHHiiI')
_DOC = struct.Struct('<QdII')
_TOKEN = struct.Struct('<HId')


def read_fma(path, limit):
    with Path(path).open(encoding='utf8', newline='') as stream:
        reader = csv.reader(stream)
        groups, names = next(reader), next(reader)
        next(reader)
        header = list(zip(groups, names))
        fields = [('track', 'title'), ('artist', 'name'), ('album', 'title'), ('track', 'genre_top'),
                  ('artist', 'bio'), ('album', 'information'), ('track', 'information'),
                  ('track', 'tags'), ('artist', 'tags'), ('album', 'tags')]
        positions = [header.index(field) for field in fields if field in header]
        for index, row in enumerate(reader):
            if index >= limit:
                return
            body = ' '.join(row[pos] for pos in positions if row[pos])
            body = html.unescape(re.sub(r'<[^>]+>', ' ', body))
            body = re.sub(r'\s+', ' ', body).strip()
            yield int(row[0]), body


class ScanCorpus:
    """Documentos preprocesados de longitud variable empaquetados en páginas.

    Cada documento conserva original, frecuencias y pesos. La consulta recorre
    páginas del corpus, sin usar listas de posteo ni un corpus en RAM.
    """
    def __init__(self, path, counter, create=False):
        if create:
            Path(path).write_bytes(b'')
        self.file = DiskFile(path, PAGE, counter, 'data')
        self.capacity = PAGE - _HEADER.size

    def build(self, docs, counts, df):
        buffer = bytearray()
        pid = 0
        for (docid, body), frequencies in zip(docs, counts):
            weights = {term: (1 + math.log10(tf)) * math.log10(len(docs)/df[term]) for term, tf in frequencies.items()}
            norm = math.sqrt(sum(w*w for w in weights.values()))
            body_bytes = body.encode('utf8')
            record = bytearray(_DOC.pack(docid, norm, len(body_bytes), len(frequencies)))
            record.extend(body_bytes)
            for term, tf in sorted(frequencies.items()):
                raw = term.encode('utf8')
                record.extend(_TOKEN.pack(len(raw), tf, weights[term]))
                record.extend(raw)
            buffer.extend(struct.pack('<I', len(record)))
            buffer.extend(record)
            while len(buffer) >= self.capacity:
                self._write_page(pid, buffer[:self.capacity])
                del buffer[:self.capacity]
                pid += 1
        if buffer:
            self._write_page(pid, buffer)

    def _write_page(self, pid, payload):
        page = bytearray(PAGE)
        _HEADER.pack_into(page, 0, b'SCN1', pid, 0, _HEADER.size + len(payload), -1, pid-1, 0)
        page[_HEADER.size:_HEADER.size + len(payload)] = payload
        self.file.write(pid, page)

    def scan(self):
        carry = bytearray()
        for pid in range(self.file.page_count):
            page = self.file.read(pid)
            magic, actual, _, free, *_ = _HEADER.unpack_from(page)
            if magic != b'SCN1' or actual != pid:
                raise ValueError('Corpus de escaneo corrupto')
            carry.extend(page[_HEADER.size:free])
            consumed = 0
            while len(carry)-consumed >= 4:
                size = struct.unpack_from('<I', carry, consumed)[0]
                if len(carry)-consumed < size+4:
                    break
                offset = consumed+4
                docid, norm, text_len, terms = _DOC.unpack_from(carry, offset)
                offset += _DOC.size + text_len
                weights = {}
                for _ in range(terms):
                    length, tf, weight = _TOKEN.unpack_from(carry, offset)
                    offset += _TOKEN.size
                    term = bytes(carry[offset:offset+length]).decode('utf8')
                    offset += length
                    weights[term] = weight
                yield docid, norm, weights
                consumed += size+4
            if consumed:
                del carry[:consumed]
        if carry:
            raise ValueError('Documento truncado en corpus')

    def query(self, terms, operation, qweights=None, k=10):
        result, top = [], []
        qnorm = math.sqrt(sum(w*w for w in (qweights or {}).values()))
        for docid, norm, weights in self.scan():
            present = [term in weights for term in terms]
            if operation == 'rank':
                if not norm or not qnorm:
                    continue
                score = sum(qweights[t]*weights.get(t, 0) for t in terms)/(norm*qnorm)
                if score <= 0:
                    continue
                item = score, -docid
                if len(top) < k:
                    heapq.heappush(top, item)
                elif item > top[0]:
                    heapq.heapreplace(top, item)
            elif (all(present) if operation == 'AND' else any(present) if operation == 'OR' else present[0] and not any(present[1:])):
                result.append((docid, 1.0))
        return [(-negative, score) for score, negative in sorted(top, reverse=True)] if operation == 'rank' else sorted(result)

    def close(self):
        self.file.close()


def pg_connect(dsn):
    import psycopg
    conn = psycopg.connect(dsn, autocommit=True)
    database = 'sonoradb_benchmark'
    if conn.execute('SELECT 1 FROM pg_database WHERE datname=%s', (database,)).fetchone() is None:
        conn.execute('CREATE DATABASE sonoradb_benchmark')
    params = conn.info.get_parameters()
    conn.close()
    params['dbname'] = database
    return psycopg.connect(**params, autocommit=True)


def pg_build(conn, docs, counts, df, table):
    from psycopg.types.json import Jsonb
    conn.execute(f'DROP TABLE IF EXISTS {table}')
    conn.execute(f'CREATE TABLE {table} (id bigint PRIMARY KEY, token_text text, weights jsonb, norm double precision)')
    with conn.cursor().copy(f'COPY {table}(id,token_text,weights,norm) FROM STDIN') as copy:
        for (docid, _), frequencies in zip(docs, counts):
            weights = {term: (1 + math.log10(tf)) * math.log10(len(docs)/df[term]) for term, tf in frequencies.items()}
            norm = math.sqrt(sum(w*w for w in weights.values()))
            copy.write_row((docid, ' '.join(frequencies), Jsonb(weights), norm))
    start = time.perf_counter()
    conn.execute(f"CREATE INDEX {table}_gin ON {table} USING gin(to_tsvector('simple',token_text))")
    elapsed = (time.perf_counter()-start)*1000
    conn.execute(f'ANALYZE {table}')
    size = conn.execute('SELECT pg_relation_size(%s)', (f'{table}_gin',)).fetchone()[0]
    return elapsed, size


def pg_query(terms, operation, qweights, table):
    quoted = ["'" + term.replace("'", "''") + "'" for term in terms]
    if operation in ('OR', 'rank'):
        tsquery = ' | '.join(quoted)
    elif operation == 'AND':
        tsquery = ' & '.join(quoted)
    else:
        tsquery = quoted[0] + ''.join(' & !' + t for t in quoted[1:])
    predicate = "to_tsvector('simple',token_text) @@ to_tsquery('simple',%s)"
    if operation != 'rank':
        return f'SELECT id,1.0 FROM {table} WHERE {predicate} ORDER BY id', (tsquery,)
    qnorm = math.sqrt(sum(w*w for w in qweights.values()))
    sql = (f'SELECT id,(SELECT SUM(COALESCE((weights->>term)::double precision,0)*weight::double precision) '
           f'FROM jsonb_each_text(%s::jsonb) AS q(term,weight))/(norm*%s) AS score '
           f'FROM {table} WHERE {predicate} AND norm>0 ORDER BY score DESC,id LIMIT 10')
    return sql, (json.dumps(qweights), qnorm, tsquery)


def validate(actual, expected, operation):
    if operation != 'rank':
        if actual != expected:
            raise AssertionError('Resultados booleanos diferentes')
    else:
        if len(actual) != len(expected):
            raise AssertionError('Cantidad de resultados de ranking diferente')
        # Diferencias de redondeo pueden permutar documentos empatados.
        if any(not math.isclose(a[1], b[1], rel_tol=1e-10, abs_tol=1e-10) for a,b in zip(actual,expected)):
            raise AssertionError('CosineScore diferente')
        expected_ids = {i for i,s in expected if s > expected[-1][1]+1e-10}
        if not expected_ids.issubset({i for i,_ in actual}):
            raise AssertionError('Top-K omitió resultados sin empate')


def export(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plots(output, construction, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for method in ('inverted','postgres_gin'):
        series = [r for r in construction if r['method']==method]
        axes[0].plot([r['n'] for r in series],[r['build_ms']/1000 for r in series],'-o',label=method)
        axes[1].plot([r['n'] for r in series],[r['index_bytes']/1048576 for r in series],'-o',label=method)
    for ax in axes:
        ax.set_xlabel('Documentos');ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Construcción (s)');axes[1].set_ylabel('Diccionario + postings / GIN (MiB)')
    fig.suptitle('E6 · Construcción de índices de texto');fig.tight_layout();fig.savefig(output/'e6_construction.png',dpi=170);plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(11,7))
    for ax,operation in zip(axes.flat,('AND','OR','AND-NOT','rank')):
        for method in ('inverted','full_scan','postgres_gin'):
            series=[r for r in summary if r['method']==method and r['operation']==operation]
            ax.plot([r['terms'] for r in series],[r['latency_ms_mean'] for r in series],'-o',label=method)
        ax.set_title(operation);ax.set_xticks([1,3,5]);ax.set_xlabel('Términos');ax.set_ylabel('Latencia (ms, log)');ax.set_yscale('log');ax.grid(alpha=.2);ax.legend()
    fig.suptitle(f"E6 · Consultas sobre {summary[0]['n']:,} documentos FMA");fig.tight_layout();fig.savefig(output/'e6_queries.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(11,11))
    for row,operation in enumerate(('AND','OR','AND-NOT','rank')):
        for method in ('inverted','full_scan'):
            series=[r for r in summary if r['method']==method and r['operation']==operation]
            axes[row,0].plot([r['terms'] for r in series],[r['disk_reads_mean'] for r in series],'-o',label=method)
        series=[r for r in summary if r['method']=='postgres_gin' and r['operation']==operation]
        for field,label in [('pg_shared_hit_blocks_mean','shared_hit'),('pg_shared_read_blocks_mean','shared_read')]:
            axes[row,1].plot([r['terms'] for r in series],[r[field] for r in series],'-o',label=label)
        axes[row,0].set_yscale('log');axes[row,0].set_ylabel('Lecturas solicitadas · 4 KB')
        axes[row,1].set_ylabel('Buffers PostgreSQL · 8 KB')
        for ax in axes[row]:
            ax.set_title(operation);ax.set_xticks([1,3,5]);ax.set_xlabel('Términos');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('E6 · I/O del gestor y buffers PostgreSQL (contadores distintos)');fig.tight_layout();fig.savefig(output/'e6_io.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--metadata',type=Path,default=METADATA_DIR/'tracks.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/results/parte2_texto')
    parser.add_argument('--workdir',type=Path,default=ROOT/'data/runtime/e6')
    parser.add_argument('--sizes',type=int,nargs='+',default=[1000,5000,10000,20000])
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--dsn',default=os.environ.get('SONORADB_PG_DSN','host=localhost port=5432 user=postgres dbname=postgres'))
    args=parser.parse_args()
    metadata=args.metadata
    if not metadata.exists():
        parser.error('Configurar --metadata tracks.csv o FMA_METADATA_DIR')
    args.output.mkdir(parents=True,exist_ok=True);args.workdir.mkdir(parents=True,exist_ok=True)
    docs=list(read_fma(metadata,max(args.sizes)))
    if len(docs)<max(args.sizes):
        raise ValueError('No hay suficientes documentos FMA')
    print(f'Normalizando {len(docs)} documentos FMA...',flush=True)
    normalizer=TextNormalizer('auto')
    start=time.perf_counter()
    counts=[Counter(normalizer(body)) for _,body in docs]
    preprocessing_ms=(time.perf_counter()-start)*1000
    conn=pg_connect(args.dsn)
    conn.execute('SET enable_seqscan=off')
    construction=[]
    for n in args.sizes:
        subset=docs[:n];frequencies=counts[:n]
        df=Counter(term for count in frequencies for term in count)
        counter=DiskCounter()
        start=time.perf_counter()
        index=InvertedIndex(args.workdir/f'index_{n}',counter=counter,create=True)
        index.build_tokens((doc[0],count) for doc,count in zip(subset,frequencies))
        build_ms=(time.perf_counter()-start)*1000
        size=sum((index.path/name).stat().st_size for name in ('vocab.dic','postings.inv'))
        construction.append(dict(n=n,method='inverted',build_ms=build_ms,index_bytes=size,norm_bytes=(index.path/'norms.bin').stat().st_size,
                                 disk_reads=counter.disk_reads,disk_writes=counter.disk_writes,terms=index.stats()['terms']))
        index.close()
        table=f'e6_documents_{n}'
        pg_ms,pg_size=pg_build(conn,subset,frequencies,df,table)
        construction.append(dict(n=n,method='postgres_gin',build_ms=pg_ms,index_bytes=pg_size,terms=len(df)))
        export(args.output/'e6_construction.csv',construction)
        print(f'N={n}: propio {build_ms/1000:.3f}s/{size}bytes; GIN {pg_ms/1000:.3f}s/{pg_size}bytes',flush=True)
    n=max(args.sizes);df=Counter(term for count in counts for term in count)
    # Términos reales con frecuencia moderada y coocurrencia: AND no es trivialmente vacío.
    candidates={term for term,freq in df.items() if max(5,n//1000)<=freq<=n//3 and re.fullmatch('[a-z]+',term) and normalizer(term)==[term]}
    selected=None
    for count in counts:
        eligible=sorted(set(count)&candidates,key=lambda t:(-df[t],t))
        if len(eligible)>=5:
            selected=sorted(eligible[:5]);break
    if selected is None:
        raise ValueError('Corpus insuficiente para consultas de cinco términos')
    counter=DiskCounter();index=InvertedIndex(args.workdir/f'index_{n}',counter=counter)
    corpus_counter=DiskCounter();corpus=ScanCorpus(args.workdir/'full_scan.bin',corpus_counter,create=True)
    corpus.build(docs,counts,df)
    raw=[]
    for length in (1,3,5):
        terms=selected[:length];qweights={t:math.log10(n/df[t]) for t in terms}
        for operation in ('AND','OR','AND-NOT','rank'):
            expression=(' '+operation+' ').join(terms)
            expected=index.query_tokens(terms) if operation=='rank' else index.query(expression,mode='boolean')
            sql,params=pg_query(terms,operation,qweights,f'e6_documents_{n}')
            pg_actual=[(int(i),float(s)) for i,s in conn.execute(sql,params).fetchall()]
            validate(pg_actual,expected,operation)
            for repeat in range(args.repeats):
                for method in ('inverted','full_scan','postgres_gin'):
                    current=counter if method=='inverted' else corpus_counter
                    current.reset();start=time.perf_counter()
                    extra={}
                    if method=='inverted':
                        actual=index.query_tokens(terms) if operation=='rank' else index.query(expression,mode='boolean')
                    elif method=='full_scan':
                        actual=corpus.query(terms,operation,qweights)
                    else:
                        actual=[(int(i),float(score)) for i,score in conn.execute(sql,params).fetchall()]
                    elapsed=(time.perf_counter()-start)*1000
                    if method=='postgres_gin':
                        plan=conn.execute('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) '+sql,params).fetchone()[0][0]
                        extra=dict(pg_shared_hit_blocks=plan['Plan'].get('Shared Hit Blocks',0),pg_shared_read_blocks=plan['Plan'].get('Shared Read Blocks',0),
                                   pg_execution_ms=plan['Execution Time'],pg_planning_ms=plan['Planning Time'],pg_block_size=8192,
                                   plan_json=json.dumps(plan,separators=(',',':')))
                    validate(actual,expected,operation)
                    raw.append(dict(n=n,method=method,operation=operation,terms=length,query=' '.join(terms),repeat=repeat,
                                    latency_ms=elapsed,disk_reads=current.disk_reads if method!='postgres_gin' else '',
                                    disk_writes=current.disk_writes if method!='postgres_gin' else '',results=len(actual),**extra))
                export(args.output/'e6_queries_raw.csv',raw)
            print(f'Consulta {operation}/{length}: {len(expected)} resultados, {args.repeats} repeticiones',flush=True)
    summary=[]
    for operation in ('AND','OR','AND-NOT','rank'):
        for length in (1,3,5):
            for method in ('inverted','full_scan','postgres_gin'):
                records=[r for r in raw if r['method']==method and r['operation']==operation and r['terms']==length]
                row=dict(n=n,method=method,operation=operation,terms=length,repeats=len(records),results=records[0]['results'])
                for field in ('latency_ms','disk_reads','pg_shared_hit_blocks','pg_shared_read_blocks','pg_execution_ms'):
                    values=[r[field] for r in records if isinstance(r.get(field),(float,int))]
                    if values:
                        row[field+'_mean']=statistics.mean(values);row[field+'_stdev']=statistics.stdev(values) if len(values)>1 else 0
                summary.append(row)
    export(args.output/'e6_queries_summary.csv',summary)
    manifest=dict(complete=True,n=n,sizes=args.sizes,repeats=args.repeats,queries=selected,observations=len(raw),preprocessing_ms=preprocessing_ms,
                  source='FMA metadata tracks.csv; títulos, artista, álbum, género, biografía, información y tags; HTML eliminado',
                  normalization='NFKD, casefold, stopwords ES/EN; Snowball español seguido de inglés, idéntico por token al indexar y consultar',
                  construction='Preprocesamiento común excluido. Propio construye vocab.dic+postings.inv+norms.bin con build_tokens; tamaño principal dic+inv, normas aparte. GIN se construye sobre to_tsvector(simple, token_text) de los mismos tokens ya normalizados; carga de tabla excluida.',
                  full_scan='Corpus binario paginado 4096 bytes, original UTF8 + frecuencias + pesos TF-IDF; cada consulta recorre todas las páginas y decodifica cada vector documental. Sin postings ni corpus RAM. Preprocesamiento/normas fuera del tiempo de consulta, igual que ambos índices.',
                  cosine='(1+log10(tf))*log10(N/df), normas L2 documento y consulta, top10. PostgreSQL usa los mismos pesos JSONB y norma; GIN genera candidatos. No se usa ts_rank.',
                  timing='Propio/fullscan: perf_counter tiempo cliente. PostgreSQL latency_ms mide SELECT+fetch/roundtrip. EXPLAIN ANALYZE BUFFERS se ejecuta después, fuera del intervalo, sobre la misma consulta; pg_execution_ms corresponde a esa ejecución instrumentada. Validación fuera del cronómetro.',
                  io='Propio y scan: DiskCounter, páginas4096, cada transferencia solicitada. PostgreSQL: BUFFERS shared_read y shared_hit separados, bloques8192; hits no son lecturas de dispositivo. No se vacía caché de SO/PG. enable_seqscan=off para estudiar GIN explícitamente.',
                  single_term='Para una palabra, AND, OR y AND-NOT se reducen al mismo término, sin operando derecho.',
                  pg_version=conn.execute('SELECT version()').fetchone()[0],full_requirement_satisfied=n>=20000)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    plots(args.output,construction,summary)
    index.close();corpus.close();conn.close()
    print('E6 completo: '+str(args.output),flush=True)


if __name__=='__main__':
    main()
