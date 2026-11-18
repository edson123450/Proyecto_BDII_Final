"""Planificador multimodal sobre las tablas, RID y páginas del motor relacional."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from time import perf_counter

import numpy as np

from .engine import Database
from .blob import BlobStore
from .spatial import RTree
from .text import InvertedIndex
from .vector import VectorStore, IVFIndex
from .sql import matches


def project(point):
    """Esta función convierte lon/lat a kilómetros en un plano local de Lima."""
    lon, lat = point
    if not all(math.isfinite(v) for v in (lon, lat)) or not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError('Coordenadas geográficas fuera de rango.')
    return ((lon + 77.04) * 111.32 * math.cos(math.radians(-12.05)), (lat + 12.05) * 111.32)


def flat_and(condition):
    if condition is None:
        return []
    if condition[0] == 'OR':
        raise ValueError('Combinar predicados multimodales con AND; usar OR dentro de MATCH.')
    return flat_and(condition[1]) + flat_and(condition[2]) if condition[0] == 'AND' else [condition]


class MultimodalDatabase(Database):
    def __init__(self, directory, page_size=4096):
        super().__init__(directory, page_size)
        self.probes = 5
        self.mm = {}
        self.resources = {}
        for table, meta in self.catalog.items():
            if 'multimodal' in meta:
                self._open_multimodal(table)

    def _open_multimodal(self, table):
        meta = self.catalog[table]['multimodal']
        opened = {}
        if 'blobs' in meta:
            opened['blobs'] = BlobStore(self.directory / meta['blobs'], self.counter, self.page_size)
        if 'vectors' in meta:
            opened['vectors'] = VectorStore(self.directory / meta['vectors'], counter=self.counter)
        self.resources[table] = opened
        for spec in meta.get('indexes', []):
            self.mm[spec['name']] = self._mm_instance(spec)

    def _mm_instance(self, spec):
        path = self.directory / spec['path']
        if spec['kind'] == 'RTREE':
            return RTree(path, counter=self.counter)
        if spec['kind'] == 'INVERTED':
            return InvertedIndex(path, counter=self.counter)
        return IVFIndex(path, counter=self.counter)

    def register(self, table, **metadata):
        self.catalog[table]['multimodal'] = dict(indexes=[], **metadata)
        self._save_catalog()
        self._open_multimodal(table)

    def _index(self, table, kind):
        specs = self.catalog[table].get('multimodal', {}).get('indexes', [])
        spec = next((s for s in reversed(specs) if s['kind'] == kind), None)
        return (self.mm[spec['name']], spec) if spec else (None, None)

    def create_multimodal_index(self, table, name, column, kind, options=None):
        self._require_table(table)
        self._validate_name(name)
        if name in self.indexes or name in self.mm:
            raise ValueError(f'El índice {name} ya existe.')
        meta = self.catalog[table].get('multimodal', {})
        expected = {'RTREE': 'ubicacion', 'INVERTED': 'cuerpo', 'IVF': 'embedding'}
        if column != expected.get(kind) or column not in meta.get('virtual_columns', {}):
            raise ValueError('Columna multimodal no compatible con el índice solicitado.')
        options = options or {}
        if kind == 'INVERTED' and options.get('scoring', 'TFIDF_COSINE').upper() != 'TFIDF_COSINE':
            raise ValueError('El índice invertido implementa TFIDF_COSINE.')
        path = self.directory / name
        spec = dict(name=name, column=column, kind=kind, path=name, options=options)
        if kind == 'RTREE':
            index = RTree(path, counter=self.counter, create=True)
            for rid, row in self.tables[table].scan():
                index.insert(project((row['lon'], row['lat'])), rid)
        elif kind == 'INVERTED':
            index = InvertedIndex(path, counter=self.counter, create=True)
            blobs = self.resources[table]['blobs']
            index.build_tokens((row['id'], index.tokenize(blobs.read(row['cuerpo_ptr']).decode('utf-8')))
                        for _, row in self.tables[table].scan())
        elif kind == 'IVF':
            store = self.resources[table]['vectors']
            if options.get('metric', store.metric).upper() != store.metric:
                raise ValueError(f'Los vectores de esta colección usan {store.metric}.')
            index = IVFIndex(path, counter=self.counter)
            index.build(store, clusters=options.get('lists'))
        else:
            raise ValueError('Índice multimodal desconocido.')
        self.mm[name] = index
        meta['indexes'].append(spec)
        self._save_catalog()

    def get_by_id(self, table, identifier):
        candidates, _ = self._candidates(table, ('=', 'id', int(identifier)))
        return next(((rid, row) for rid, row in candidates if row['id'] == int(identifier)), None)

    def value(self, table, row, column):
        if column in row:
            return row[column]
        if column == 'ubicacion':
            return [row['lon'], row['lat']]
        if column == 'cuerpo':
            return self.resources[table]['blobs'].read(row['cuerpo_ptr']).decode('utf-8')
        if column == 'embedding':
            return np.frombuffer(self.resources[table]['blobs'].read(row['embedding_ptr']), dtype='<f4').tolist()
        raise ValueError(f'La columna {column} no existe.')

    def _finish(self, rows, plan, parse_ms, started, page, page_size, message=None):
        total = len(rows)
        return dict(message=message or f'{total} registros encontrados.', rows=rows[(page-1)*page_size:page*page_size],
                    total=total, affected=0, page=page, page_size=page_size, plan=plan,
                    metrics={**self.counter.snapshot(), 'parse_ms': parse_ms, 'execution_ms': (perf_counter()-started)*1000})

    def execute(self, sql, page=1, page_size=25):
        from .sql_multimodal import parse_multimodal_sql
        start = perf_counter()
        ast = parse_multimodal_sql(sql)
        parse_ms = (perf_counter()-start)*1000
        table, action = ast.get('table'), ast['action']
        special_projection = action == 'select' and any(isinstance(c, dict) for c in ast['columns'])
        special_index = action == 'create_index' and ast['kind'] in ('RTREE', 'INVERTED', 'IVF')
        is_mm = table in self.catalog and 'multimodal' in self.catalog[table]
        if action != 'set' and not special_index and not is_mm and not special_projection:
            return super().execute(sql, page, page_size)
        with self.lock:
            self.counter.reset()
            started = perf_counter()
            if action == 'set':
                self.probes = ast['value']
                return self._finish([], dict(operation='SetParameter', index=None, reason=f'ivf.probes = {self.probes}'),
                                    parse_ms, started, page, page_size, f'ivf.probes = {self.probes}')
            if special_index:
                self.create_multimodal_index(table, ast['name'], ast['column'], ast['kind'], ast.get('options'))
                return self._finish([], dict(operation='CreateIndex', index=ast['name'], reason=ast['kind']),
                                    parse_ms, started, page, page_size, 'Índice multimodal construido en disco.')
            if action == 'create_index':
                return super().execute(sql, page, page_size)
            if action != 'select':
                raise ValueError('Las colecciones multimedia se cargan con el preparador para mantener sus representaciones sincronizadas.')
            self._require_table(table)
            if not is_mm:
                raise ValueError('La tabla no tiene columnas multimodales.')
            return self._select_multimodal(ast, parse_ms, started, page, page_size)

    def _select_multimodal(self, ast, parse_ms, started, page, page_size):
        table = ast['table']
        columns = ast['columns']
        meta = self.catalog[table]['multimodal']
        virtuals = meta['virtual_columns']
        physical = {c['name'] for c in self.catalog[table]['columns']}
        predicates = flat_and(ast['where'])
        special_ops = {'MATCH', 'WITHIN_RADIUS', 'WITHIN_WINDOW', 'KNN'}
        special = [p for p in predicates if p[0] in special_ops]
        regular = [p for p in predicates if p[0] not in special_ops]
        for predicate in regular:
            self._validate_condition(table, predicate)
        for op, column, value in special:
            if column not in virtuals:
                raise ValueError(f'La columna {column} no existe.')
        distance = next((c for c in columns if isinstance(c, dict) and c['kind'] == 'distance'), None)
        score = next((c for c in columns if isinstance(c, dict) and c['kind'] == 'cosine_score'), None)
        if len(special) > 1 or (special and distance):
            raise ValueError('Usar una búsqueda multimodal por consulta, combinada con filtros relacionales mediante AND.')
        if score and not any(p[0] == 'MATCH' for p in special):
            raise ValueError('COSINE_SCORE requiere una condición MATCH.')
        condition = None
        for predicate in regular:
            condition = predicate if condition is None else ('AND', condition, predicate)
        allowed, filter_plan = None, None
        if condition is not None:
            candidates, filter_plan = self._candidates(table, condition)
            allowed = {rid for rid, row in candidates if matches(row, condition)}
        ranked = None
        plan = dict(operation='SeqScan', index=None, reason='Lectura de la colección')
        if distance:
            if distance['column'] != 'embedding' or 'vectors' not in self.resources[table]:
                raise ValueError('Esta colección no tiene vectores.')
            store = self.resources[table]['vectors']
            if distance['metric'] != store.metric:
                raise ValueError(f'Esta representación usa {store.metric}; elegir el operador correspondiente.')
            if ast['order'] != (distance['alias'], False):
                raise ValueError('Ordenar la distancia ASC para buscar los vecinos más cercanos.')
            if ast['limit'] is None:
                raise ValueError('La búsqueda vectorial requiere LIMIT k.')
            index, spec = self._index(table, 'IVF')
            k = ast['limit']
            ranked = index.knn(distance['vector'], k, self.probes, allowed_rids=allowed) if index else store.knn(distance['vector'], k, allowed_rids=allowed)
            plan = dict(operation='IVFScan' if index else 'VectorScan', index=spec['name'] if spec else None,
                        reason=f'Coseno; probes={min(self.probes, index.clusters)}' if index else 'kNN exacto con max-heap')
        elif special:
            op, column, value = special[0]
            if op == 'MATCH':
                if column != 'cuerpo':
                    raise ValueError('MATCH requiere una columna de texto.')
                index, spec = self._index(table, 'INVERTED')
                if index is None:
                    raise ValueError('Crear primero un índice INVERTED.')
                boolean = bool(re.search(r'\b(?:AND|OR|NOT)\b', value))
                if boolean:
                    ids = set(index.boolean(value))
                    terms = ' '.join(t for t in re.split(r'\b(?:AND|OR|NOT)\b|[()]', value) if t.strip())
                    scores = dict(index.query(terms, k=index.document_count)) if score else {}
                    found = [(i, scores.get(i, 0.0)) for i in ids]
                else:
                    found = index.query(value, k=index.document_count)
                ranked = []
                for docid, similarity in found:
                    item = self.get_by_id(table, docid)
                    if item and (allowed is None or item[0] in allowed):
                        ranked.append((item[0], similarity))
                ranked.sort(key=lambda item: (-item[1], item[0]))
                plan = dict(operation='InvertedScan', index=spec['name'], reason='Mezcla booleana y TF-IDF/coseno' if boolean else 'TF-IDF/coseno')
            else:
                if column != 'ubicacion':
                    raise ValueError('La búsqueda espacial requiere ubicacion.')
                index, spec = self._index(table, 'RTREE')
                if index is None:
                    raise ValueError('Crear primero un índice RTREE.')
                if op == 'WITHIN_RADIUS':
                    center, radius = value
                    ranked = index.radius_search(project(center), radius)
                elif op == 'KNN':
                    center, k = value
                    ranked = index.knn(project(center), k if allowed is None else self.tables[table].record_count)
                    ranked = [item for item in ranked if allowed is None or item[0] in allowed][:k]
                else:
                    lower, upper = value
                    ranked = [(rid, 0.) for rid in index.range_search(project(lower), project(upper))]
                plan = dict(operation='RTreeScan', index=spec['name'], reason=op + '; proyección local en kilómetros')
        if filter_plan:
            plan['strategy'] = 'Pre-filtering mediante ' + filter_plan['operation']
            plan['filter'] = filter_plan
        if ranked is None:
            selected = ((rid, row, None) for rid, row in self.tables[table].scan() if allowed is None or rid in allowed)
        else:
            selected = ((rid, self.tables[table].get(rid), metric) for rid, metric in ranked if allowed is None or rid in allowed)
        if columns == ['*']:
            columns = [c for c in physical if not c.endswith('_ptr')] + [c for c in virtuals if c != 'embedding']
        if any(isinstance(c, str) and c not in physical and c not in virtuals for c in columns):
            raise ValueError('La proyección contiene columnas inexistentes.')
        rows = []
        for rid, row, metric in selected:
            result = {}
            for column in columns:
                if isinstance(column, dict):
                    result[column['alias']] = metric
                else:
                    result[column] = self.value(table, row, column)
            rows.append(result)
        if ast['order']:
            key, reverse = ast['order']
            if key not in {c if isinstance(c, str) else c['alias'] for c in columns}:
                raise ValueError('ORDER BY debe referirse a una columna seleccionada.')
            rows.sort(key=lambda r: r[key], reverse=reverse)
        if ast['limit'] is not None:
            rows = rows[:ast['limit']]
        return self._finish(rows, plan, parse_ms, started, page, page_size)

    def describe(self):
        descriptions = super().describe()
        for item in descriptions:
            meta = item.get('multimodal', {})
            item['indexes'] = item['indexes'] + meta.get('indexes', [])
            item['columns'] = item['columns'] + [dict(name=k, type=v) for k, v in meta.get('virtual_columns', {}).items()]
        return descriptions

    def close(self):
        for index in getattr(self, 'mm', {}).values():
            index.close()
        for resources in getattr(self, 'resources', {}).values():
            for resource in resources.values():
                resource.close()
        super().close()
