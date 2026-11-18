"""Motor relacional: catálogo, ejecución SQL y selección de índices."""

import json
import math
import re
import struct
import threading
from pathlib import Path
from time import perf_counter

from .io import DiskCounter, DiskFile
from .storage import Schema, HeapFile, SequentialFile
from .indexes import BPlusTree, ExtendibleHash
from .sql import parse_sql, matches, conjuncts


CATALOG_HEADER = struct.Struct('<II I q q I')
IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')
INT_MIN, INT_MAX = -(1 << 63), (1 << 63) - 1


class Database:
    def __init__(self, directory, page_size=4096):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.page_size = page_size
        self.counter = DiskCounter()
        self.lock = threading.RLock()
        self.tables = {}
        self.indexes = {}
        self.catalog_path = self.directory / 'catalog.bin'
        self.catalog = self._load_catalog()
        for name in self.catalog:
            self._open_table(name)

    def _load_catalog(self):
        if not self.catalog_path.exists():
            return {}
        with DiskFile(self.catalog_path, self.page_size, self.counter, role='data') as file:
            chunks = []
            for page_id in range(file.page_count):
                page = file.read(page_id)
                actual, count, free, nxt, prev, length = CATALOG_HEADER.unpack_from(page)
                expected_next = page_id + 1 if page_id + 1 < file.page_count else -1
                if (actual != page_id or count != 1 or prev != page_id - 1 or nxt != expected_next
                        or length > self.page_size - CATALOG_HEADER.size
                        or free != CATALOG_HEADER.size + length):
                    raise ValueError('Catálogo inválido.')
                chunks.append(bytes(page[CATALOG_HEADER.size:CATALOG_HEADER.size + length]))
        return json.loads(b''.join(chunks).decode('utf-8'))

    def _save_catalog(self):
        content = json.dumps(self.catalog, ensure_ascii=False).encode('utf-8')
        capacity = self.page_size - CATALOG_HEADER.size
        target = self.catalog_path.with_suffix('.tmp')
        if target.exists():
            target.unlink()
        with DiskFile(target, self.page_size, self.counter, role='data') as file:
            for offset in range(0, len(content), capacity):
                page_id = offset // capacity
                chunk = content[offset:offset + capacity]
                page = bytearray(self.page_size)
                next_page = page_id + 1 if offset + capacity < len(content) else -1
                CATALOG_HEADER.pack_into(page, 0, page_id, 1, CATALOG_HEADER.size + len(chunk), next_page, page_id - 1, len(chunk))
                page[CATALOG_HEADER.size:CATALOG_HEADER.size + len(chunk)] = chunk
                file.allocate(page)
        target.replace(self.catalog_path)

    def _open_table(self, name):
        meta = self.catalog[name]
        schema = Schema(meta['columns'])
        cls = HeapFile if meta['organization'] == 'HEAP' else SequentialFile
        self.tables[name] = cls(self.directory / f'{name}.tbl', schema, counter=self.counter, page_size=self.page_size)
        for index in meta['indexes']:
            self.indexes[index['name']] = self._index_instance(name, index)

    def _index_instance(self, table, index, create=False, path=None):
        column = next(c for c in self.catalog[table]['columns'] if c['name'] == index['column'])
        cls = BPlusTree if index['kind'] == 'BTREE' else ExtendibleHash
        return cls(path or self.directory / f"{index['name']}.idx", key_type=column['type'], key_size=column.get('size', 8), page_size=self.page_size, counter=self.counter, create=create)

    @staticmethod
    def _validate_name(name):
        if not isinstance(name, str) or not IDENTIFIER.fullmatch(name):
            raise ValueError('Nombre inválido: usar letras, números y guion bajo.')

    def create_table(self, name, columns, organization='HEAP'):
        self._validate_name(name)
        if name in self.catalog:
            raise ValueError(f'La tabla {name} ya existe.')
        if organization not in ('HEAP', 'SEQUENTIAL'):
            raise ValueError('La organización debe ser HEAP o SEQUENTIAL.')
        if len({c['name'] for c in columns}) != len(columns):
            raise ValueError('No se admiten nombres de columnas repetidos.')
        if sum(bool(c.get('primary')) for c in columns) > 1:
            raise ValueError('Este motor admite una clave primaria simple.')
        for column in columns:
            self._validate_name(column['name'])
            if str(column['type']).upper() == 'CHAR':
                size = column.get('size')
                if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                    raise ValueError('CHAR requiere un tamaño entero positivo.')
        schema = Schema(columns)
        cls = HeapFile if organization == 'HEAP' else SequentialFile
        path = self.directory / f'{name}.tbl'
        file = cls(path, schema, counter=self.counter, page_size=self.page_size, create=True)
        try:
            self.tables[name] = file
            self.catalog[name] = dict(columns=schema.columns, organization=organization, indexes=[])
            self._save_catalog()
        except Exception:
            self.tables.pop(name, None)
            self.catalog.pop(name, None)
            file.close()
            path.unlink(missing_ok=True)
            Path(str(path) + '.overflow').unlink(missing_ok=True)
            raise

    def create_index(self, table, name, column, kind):
        self._require_table(table)
        self._validate_name(name)
        if name in self.indexes:
            raise ValueError(f'El índice {name} ya existe.')
        if kind not in ('BTREE', 'HASH'):
            raise ValueError('El índice debe ser BTREE o HASH.')
        if column not in {c['name'] for c in self.catalog[table]['columns']}:
            raise ValueError(f'La columna {column} no existe.')
        spec = dict(name=name, column=column, kind=kind)
        index = self._build_index(table, spec)
        try:
            self.indexes[name] = index
            self.catalog[table]['indexes'].append(spec)
            self._save_catalog()
        except Exception:
            index.close()
            self.indexes.pop(name, None)
            self.catalog[table]['indexes'].remove(spec)
            index.path.unlink(missing_ok=True)
            raise

    def _build_index(self, table, spec):
        """Esta función publica un índice únicamente después de construirlo completo."""
        path = self.directory / f"{spec['name']}.idx"
        temporary = path.with_suffix('.building')
        index = None
        try:
            index = self._index_instance(table, spec, create=True, path=temporary)
            for rid, row in self.tables[table].scan():
                index.insert(row[spec['column']], rid)
            index.close()
            temporary.replace(path)
            return self._index_instance(table, spec)
        except Exception:
            if index is not None:
                index.close()
            temporary.unlink(missing_ok=True)
            raise

    def _require_table(self, table):
        if table not in self.tables:
            raise ValueError(f'La tabla {table} no existe.')

    def insert(self, table, row, check_primary=True):
        self._require_table(table)
        file = self.tables[table]
        file.schema.encode(row)
        primary = next((c['name'] for c in self.catalog[table]['columns'] if c.get('primary')), None)
        if check_primary and primary:
            found, _ = self._candidates(table, ('=', primary, row[primary]))
            if any(matches(r, ('=', primary, row[primary])) for _, r in found):
                raise ValueError(f'Ya existe la clave primaria {row[primary]}.')
        for spec in self.catalog[table]['indexes']:
            self.indexes[spec['name']].codec.normalize(row[spec['column']])
        rid = file.insert(row)
        updated = []
        try:
            for spec in self.catalog[table]['indexes']:
                self.indexes[spec['name']].insert(row[spec['column']], rid)
                updated.append(spec)
        except Exception:
            for spec in reversed(updated):
                self.indexes[spec['name']].delete(row[spec['column']], rid)
            file.delete(rid)
            raise
        return rid

    def _validate_condition(self, table, condition):
        if condition is None:
            return
        if condition[0] in ('AND', 'OR'):
            self._validate_condition(table, condition[1])
            self._validate_condition(table, condition[2])
            return
        col = next((c for c in self.catalog[table]['columns'] if c['name'] == condition[1]), None)
        if col is None:
            raise ValueError(f'La columna {condition[1]} no existe.')
        numeric = col['type'] in ('INT', 'FLOAT')
        if numeric != isinstance(condition[2], (int, float)):
            raise ValueError(f'Tipo de comparación incompatible para {col["name"]}.')
        if isinstance(condition[2], float) and not math.isfinite(condition[2]):
            raise ValueError('La comparación requiere un número finito.')

    @staticmethod
    def _empty_candidates(reason):
        return iter(()), dict(operation='EmptyResult', index=None, reason=reason, cost_model='0')

    @staticmethod
    def _index_plan(index, spec, operation, candidates):
        """Esta función expone el modelo analítico; el contador mide las lecturas reales."""
        if spec['kind'] == 'BTREE':
            model = 'h + L + R: altura del árbol, hojas recorridas y accesos a registros'
            parameters = dict(height=index.stats()['height'], candidate_records=len(candidates))
        else:
            model = 'D + B + R: directorio, buckets recorridos y accesos a registros'
            parameters = dict(candidate_records=len(candidates))
        return dict(operation=operation, index=spec['name'], reason=f"Índice en {spec['column']}",
                    cost_model=model, cost_parameters=parameters)

    def _candidates(self, table, condition):
        """Esta función selecciona igualdad, rango o escaneo según los índices disponibles."""
        file = self.tables[table]
        conditions = conjuncts(condition)
        columns = {column['name']: column for column in self.catalog[table]['columns']}
        for op, col, value in conditions:
            if op != '=':
                continue
            column = columns[col]
            if column['type'] == 'INT':
                if value < INT_MIN or value > INT_MAX or int(value) != value:
                    return self._empty_candidates('La igualdad no puede coincidir con un INT de 64 bits')
                key = int(value)
            elif column['type'] == 'FLOAT':
                try:
                    key = float(value)
                except OverflowError:
                    return self._empty_candidates('La igualdad excede el dominio FLOAT')
                if key != value:
                    return self._empty_candidates('La igualdad no tiene representación exacta en FLOAT')
            else:
                if len(value.encode('utf-8')) > column['size']:
                    return self._empty_candidates('El texto excede el ancho de la columna')
                key = value
            specs = sorted(self.catalog[table]['indexes'], key=lambda spec: spec['kind'] != 'HASH')
            for spec in specs:
                if spec['column'] == col:
                    index = self.indexes[spec['name']]
                    rids = index.search(key)
                    return self._rows_at(file, rids), self._index_plan(index, spec, 'IndexScan', rids)
        for spec in self.catalog[table]['indexes']:
            if spec['kind'] != 'BTREE':
                continue
            bounds = [(op, value) for op, col, value in conditions
                      if col == spec['column'] and op in ('<', '<=', '>', '>=')]
            if not bounds:
                continue
            column = columns[spec['column']]
            # La conversión a FLOAT puede redondear enteros grandes; el escaneo
            # conserva la comparación exacta entre enteros y flotantes de Python.
            if column['type'] == 'FLOAT' and any(isinstance(v, int) and abs(v) > 2**53 for _, v in bounds):
                continue
            if column['type'] == 'CHAR' and any(len(v.encode('utf-8')) > column['size'] for _, v in bounds):
                continue
            lows = [(value, op == '>=') for op, value in bounds if op in ('>', '>=')]
            highs = [(value, op == '<=') for op, value in bounds if op in ('<', '<=')]
            low, include_low = max(lows, key=lambda x: (x[0], not x[1])) if lows else (None, True)
            high, include_high = min(highs, key=lambda x: (x[0], x[1])) if highs else (None, True)
            if column['type'] == 'INT':
                low = INT_MIN if low is None else math.ceil(low) if include_low else math.floor(low) + 1
                high = INT_MAX if high is None else math.floor(high) if include_high else math.ceil(high) - 1
                if low > INT_MAX or high < INT_MIN:
                    return self._empty_candidates('El rango no intersecta el dominio INT')
                low, high = max(INT_MIN, low), min(INT_MAX, high)
                include_low = include_high = True
            if low is not None and high is not None and (low > high or (low == high and not (include_low and include_high))):
                return self._empty_candidates('Los límites del rango son incompatibles')
            index = self.indexes[spec['name']]
            rids = index.range_search(low, high, include_low, include_high)
            return self._rows_at(file, rids), self._index_plan(index, spec, 'IndexRangeScan', rids)
        # Sigue siendo un Full Table Scan: se lee cada página exactamente una vez,
        # pero solo se decodifican registros que cumplen esta igualdad necesaria.
        equal = next(((col, value) for op, col, value in conditions if op == '='), None)
        candidates = file.scan_equal(*equal) if equal is not None else file.scan()
        return candidates, dict(operation='SeqScan', index=None,
                                reason='Sin índice aplicable; recorrido de páginas de datos',
                                cost_model='P: páginas de datos de la tabla')

    @staticmethod
    def _rows_at(file, rids):
        for rid in rids:
            row = file.get(rid)
            if row is not None:
                yield rid, row

    def execute(self, sql, page=1, page_size=25):
        with self.lock:
            if (isinstance(page, bool) or not isinstance(page, int) or page < 1
                    or isinstance(page_size, bool) or not isinstance(page_size, int) or page_size < 1):
                raise ValueError('La página y su tamaño deben ser enteros positivos.')
            t0 = perf_counter()
            ast = parse_sql(sql)
            parse_ms = (perf_counter() - t0) * 1000
            self.counter.reset()
            start = perf_counter()
            action = ast['action']
            table = ast['table']
            rows, total, affected = [], 0, 0
            plan = dict(operation=action, index=None)
            if action == 'create_table':
                self.create_table(table, ast['columns'], ast['organization'])
                message = f'Tabla {table} creada.'
            elif action == 'create_index':
                self.create_index(table, ast['name'], ast['column'], ast['kind'])
                message = f'Índice {ast["name"]} construido en disco.'
            elif action == 'insert':
                self._require_table(table)
                columns = self.catalog[table]['columns']
                if len(columns) != len(ast['values']):
                    raise ValueError('La cantidad de valores no coincide con las columnas.')
                rid = self.insert(table, dict(zip([c['name'] for c in columns], ast['values'])))
                affected = 1
                message = f'Registro insertado en RID {rid}.'
            elif action in ('select', 'delete'):
                self._require_table(table)
                self._validate_condition(table, ast['where'])
                candidates, plan = self._candidates(table, ast['where'])
                columns = [c['name'] for c in self.catalog[table]['columns']]
                if action == 'select' and ast['columns'] != ['*']:
                    if not set(ast['columns']).issubset(columns):
                        raise ValueError('La proyección contiene columnas inexistentes.')
                    columns = ast['columns']
                matched = [(rid, row) for rid, row in candidates if matches(row, ast['where'])]
                if action == 'delete':
                    for rid, row in matched:
                        for spec in self.catalog[table]['indexes']:
                            self.indexes[spec['name']].delete(row[spec['column']], rid)
                        self.tables[table].delete(rid)
                    affected = len(matched)
                    message = f'{affected} registros eliminados.'
                else:
                    if ast['order']:
                        col, reverse = ast['order']
                        if col not in [c['name'] for c in self.catalog[table]['columns']]:
                            raise ValueError(f'La columna {col} no existe.')
                        matched.sort(key=lambda item: item[1][col], reverse=reverse)
                    if ast['limit'] is not None:
                        matched = matched[:ast['limit']]
                    total = len(matched)
                    offset = (max(1, page) - 1) * page_size
                    rows = [{c: row[c] for c in columns} for _, row in matched[offset:offset + page_size]]
                    message = f'{total} registros encontrados.'
            elapsed = (perf_counter() - start) * 1000
            return dict(message=message, rows=rows, total=total, affected=affected, page=page, page_size=page_size, plan=plan, metrics={**self.counter.snapshot(), 'parse_ms': parse_ms, 'execution_ms': elapsed})

    def reorganize(self, table):
        with self.lock:
            self._require_table(table)
            if self.catalog[table]['organization'] != 'SEQUENTIAL':
                raise ValueError('Solo las tablas SEQUENTIAL requieren reorganización.')
            self.counter.reset()
            start = perf_counter()
            self.tables[table].reorganize()
            for spec in self.catalog[table]['indexes']:
                self.indexes[spec['name']].close()
                self.indexes[spec['name']] = self._build_index(table, spec)
            return dict(message=f'Tabla {table} reorganizada; índices actualizados con los nuevos RID.', metrics={**self.counter.snapshot(), 'execution_ms': (perf_counter() - start)*1000})

    def describe(self):
        return [dict(name=name, **meta, record_count=self.tables[name].record_count) for name, meta in self.catalog.items()]

    def close(self):
        for index in self.indexes.values():
            index.close()
        for table in self.tables.values():
            table.close()
