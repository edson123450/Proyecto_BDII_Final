"""Heap y secuencial con registros fijos, bitmap y RID estables.

El área principal secuencial conserva rangos de clave ordenados por página.
Los slots no se desplazan: el orden dentro de cada página se obtiene por clave.
El overflow tiene una cadena ordenada de RID y enlaces auxiliares de salto.
Estos enlaces evitan inserciones cuadráticas antes de la reorganización.
"""
from __future__ import annotations

import hashlib
import heapq
import math
import os
import re
import struct
from pathlib import Path

from .io import DiskCounter, DiskFile

RID = tuple[int, int]
PAGE_HEADER = struct.Struct("<qIIqq")
HEADER_SIZE = PAGE_HEADER.size
_META = struct.Struct("<8sIIIIQq16s")
_MAGIC = b"MINIFIX1"
_UINT64 = struct.Struct("<Q")
_NULL = 0
_LEVELS = 12
_NODE_EXTRA = 8 + _LEVELS * 8


class Schema:
    """Esquema de registros fijos: INT/FLOAT de 8 bytes y CHAR en UTF-8."""

    def __init__(self, columns):
        if not columns:
            raise ValueError("El esquema requiere al menos una columna")
        self.columns = []
        self.names = []
        self.offsets = {}
        self._readers = {}
        formats = []
        offset = 0
        for original in columns:
            column = dict(original)
            name = column["name"]
            if not isinstance(name, str) or not name or name in self.names:
                raise ValueError("Nombre de columna inválido o repetido")
            type_name = str(column["type"]).upper().strip()
            char_match = re.fullmatch(r"CHAR\s*\(\s*(\d+)\s*\)", type_name)
            if char_match:
                column["size"] = int(char_match.group(1))
                type_name = "CHAR"
            column["type"] = type_name
            if type_name == "INT":
                fmt, size = "q", 8
            elif type_name == "FLOAT":
                fmt, size = "d", 8
            elif type_name == "CHAR":
                size = int(column.get("size", 0))
                if size <= 0:
                    raise ValueError("CHAR requiere un tamaño positivo")
                fmt = f"{size}s"
                column["size"] = size
            else:
                raise ValueError(f"Tipo no soportado: {type_name}")
            formats.append(fmt)
            self.offsets[name] = offset
            self._readers[name] = struct.Struct("<" + fmt)
            offset += size
            self.names.append(name)
            self.columns.append(column)
        primary = [c["name"] for c in self.columns if c.get("primary", False)]
        if len(primary) > 1:
            raise ValueError("Solo se admite una clave primaria")
        self.key_name = primary[0] if primary else self.names[0]
        self.record_size = offset
        self._struct = struct.Struct("<" + "".join(formats))
        description = repr([(c["name"], c["type"], c.get("size"), bool(c.get("primary"))) for c in self.columns])
        self.digest = hashlib.blake2b(description.encode(), digest_size=16).digest()
        self._char_positions = [i for i, c in enumerate(self.columns) if c["type"] == "CHAR"]

    def encode(self, row):
        if set(row) != set(self.names):
            raise ValueError("Las columnas del registro no coinciden con el esquema")
        values = []
        for column in self.columns:
            value = row[column["name"]]
            if column["type"] == "CHAR":
                if not isinstance(value, str) or "\x00" in value:
                    raise ValueError("CHAR requiere texto sin caracteres nulos")
                value = value.encode("utf-8")
                if len(value) > column["size"]:
                    raise ValueError(f"{column['name']} supera {column['size']} bytes")
            elif column["type"] == "INT":
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(f"{column['name']} requiere un entero")
            elif not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{column['name']} requiere un número finito")
            values.append(value)
        try:
            return self._struct.pack(*values)
        except (struct.error, OverflowError) as exc:
            raise ValueError(str(exc)) from exc

    def decode(self, data, offset=0):
        values = list(self._struct.unpack_from(data, offset))
        for position in self._char_positions:
            values[position] = values[position].rstrip(b"\x00").decode("utf-8")
        return dict(zip(self.names, values))

    def read_field(self, data, name, offset=0):
        value = self._readers[name].unpack_from(data, offset + self.offsets[name])[0]
        return value.rstrip(b"\x00").decode("utf-8") if isinstance(value, bytes) else value


class _RecordFile:
    """Esta clase administra slots y una lista libre persistente en O(1) de I/O."""

    def __init__(self, path, schema, counter, page_size, create, record_size=None, user_size=0, prefix_size=0):
        self.path = Path(path)
        if create and self.path.exists() and self.path.stat().st_size:
            raise FileExistsError(f"El archivo ya existe: {path}")
        self.schema = schema
        self.page_size = page_size
        self.record_size = record_size or schema.record_size
        self.prefix_size = prefix_size
        self.capacity = (page_size - HEADER_SIZE - prefix_size) * 8 // (self.record_size * 8 + 1)
        while HEADER_SIZE + prefix_size + (self.capacity + 7) // 8 + self.capacity * self.record_size > page_size:
            self.capacity -= 1
        self.bitmap_size = (self.capacity + 7) // 8
        self.data_offset = HEADER_SIZE + self.bitmap_size + prefix_size
        if self.capacity < 1 or HEADER_SIZE + _META.size + user_size > page_size:
            raise ValueError("El registro o los metadatos no caben en una página")
        self.file = DiskFile(path, page_size, counter, "data")
        self.user_size = user_size
        self.format_digest = hashlib.blake2b(schema.digest + struct.pack("<III", self.record_size, prefix_size, user_size), digest_size=16).digest()
        if self.file.page_count == 0:
            self.record_count, self.free_head = 0, -1
            self.user = bytearray(user_size)
            self._write_meta()
        else:
            meta = self.file.read(0)
            magic, version, stored_page, stored_record, capacity, self.record_count, self.free_head, digest = _META.unpack_from(meta, HEADER_SIZE)
            if (magic, version, stored_page, stored_record, capacity, digest) != (_MAGIC, 1, page_size, self.record_size, self.capacity, self.format_digest):
                self.file.close()
                raise ValueError("Formato, tamaño de página o esquema incompatible")
            self.user = bytearray(meta[HEADER_SIZE + _META.size:HEADER_SIZE + _META.size + user_size])

    def _write_meta(self):
        meta = bytearray(self.page_size)
        end = HEADER_SIZE + _META.size + self.user_size
        PAGE_HEADER.pack_into(meta, 0, 0, min(self.record_count, 0xffffffff), end, self.free_head, -1)
        _META.pack_into(meta, HEADER_SIZE, _MAGIC, 1, self.page_size, self.record_size, self.capacity,
                        self.record_count, self.free_head, self.format_digest)
        meta[HEADER_SIZE + _META.size:end] = self.user
        self.file.write(0, meta)

    def _empty_page(self, page_id):
        page = bytearray(self.page_size)
        PAGE_HEADER.pack_into(page, 0, page_id, 0, self.data_offset, -1, -1)
        return page

    def _occupied(self, page, slot):
        return bool(page[HEADER_SIZE + slot // 8] & (1 << (slot % 8)))

    def _offset(self, slot):
        return self.data_offset + slot * self.record_size

    def _first_free(self, page):
        for byte_index in range(self.bitmap_size):
            value = page[HEADER_SIZE + byte_index]
            if value != 255:
                bits = (~value) & 255
                slot = byte_index * 8 + (bits & -bits).bit_length() - 1
                if slot < self.capacity:
                    return slot
        return self.capacity

    def _page(self, page_id):
        page = self.file.read(page_id)
        if PAGE_HEADER.unpack_from(page)[0] != page_id:
            raise ValueError("Cabecera de página inválida")
        return page

    def insert_bytes(self, raw, preferred_page=None):
        if len(raw) != self.record_size:
            raise ValueError("Longitud de registro inválida")
        # preferred_page permite insertar en el bloque principal que corresponde a la clave.
        page_id = self.free_head if preferred_page is None else preferred_page
        if page_id == -1:
            page_id = self.file.page_count
            page = self._empty_page(page_id)
            count, next_id, prev_id = 0, -1, -1
        else:
            page = self._page(page_id)
            _, count, _, next_id, prev_id = PAGE_HEADER.unpack_from(page)
            if count == self.capacity:
                raise ValueError("La página está llena")
        slot = self._first_free(page)
        start = self._offset(slot)
        page[start:start + self.record_size] = raw
        page[HEADER_SIZE + slot // 8] |= 1 << (slot % 8)
        count += 1
        if count == self.capacity:
            if prev_id >= 1:
                previous = self._page(prev_id)
                fields = list(PAGE_HEADER.unpack_from(previous))
                fields[3] = next_id
                PAGE_HEADER.pack_into(previous, 0, *fields)
                self.file.write(prev_id, previous)
            elif self.free_head == page_id:
                self.free_head = next_id
            if next_id >= 1:
                following = self._page(next_id)
                fields = list(PAGE_HEADER.unpack_from(following))
                fields[4] = prev_id
                PAGE_HEADER.pack_into(following, 0, *fields)
                self.file.write(next_id, following)
            next_id = prev_id = -1
        elif page_id == self.file.page_count:
            next_id = self.free_head
            if next_id >= 1:
                following = self._page(next_id)
                fields = list(PAGE_HEADER.unpack_from(following))
                fields[4] = page_id
                PAGE_HEADER.pack_into(following, 0, *fields)
                self.file.write(next_id, following)
            self.free_head = page_id
        free_slot = self._first_free(page)
        PAGE_HEADER.pack_into(page, 0, page_id, count, self._offset(free_slot), next_id, prev_id)
        self.file.write(page_id, page)
        self.record_count += 1
        self._write_meta()
        return page_id, slot

    def read_slot(self, rid):
        page_id, slot = rid
        if page_id < 1 or page_id >= self.file.page_count or slot < 0 or slot >= self.capacity:
            return None
        page = self._page(page_id)
        if not self._occupied(page, slot):
            return None
        start = self._offset(slot)
        return bytes(page[start:start + self.record_size])

    def replace_bytes(self, rid, raw):
        page_id, slot = rid
        page = self._page(page_id)
        if not self._occupied(page, slot) or len(raw) != self.record_size:
            raise ValueError("RID o longitud inválidos")
        start = self._offset(slot)
        page[start:start + self.record_size] = raw
        self.file.write(page_id, page)

    def delete(self, rid):
        page_id, slot = rid
        if page_id < 1 or page_id >= self.file.page_count or slot < 0 or slot >= self.capacity:
            return False
        page = self._page(page_id)
        if not self._occupied(page, slot):
            return False
        _, count, _, next_id, prev_id = PAGE_HEADER.unpack_from(page)
        if count == self.capacity:
            next_id, prev_id = self.free_head, -1
            if next_id >= 1:
                following = self._page(next_id)
                fields = list(PAGE_HEADER.unpack_from(following))
                fields[4] = page_id
                PAGE_HEADER.pack_into(following, 0, *fields)
                self.file.write(next_id, following)
            self.free_head = page_id
        page[HEADER_SIZE + slot // 8] &= ~(1 << (slot % 8))
        PAGE_HEADER.pack_into(page, 0, page_id, count - 1, self._offset(self._first_free(page)), next_id, prev_id)
        self.file.write(page_id, page)
        self.record_count -= 1
        self._write_meta()
        return True

    def iter_page(self, page_id, page=None):
        page = self._page(page_id) if page is None else page
        if not PAGE_HEADER.unpack_from(page)[1]:
            return
        for byte_index in range(self.bitmap_size):
            occupied = page[HEADER_SIZE + byte_index]
            while occupied:
                bit = occupied & -occupied
                slot = byte_index * 8 + bit.bit_length() - 1
                yield (page_id, slot), page, self._offset(slot)
                occupied ^= bit

    def scan_raw(self):
        for page_id in range(1, self.file.page_count):
            yield from self.iter_page(page_id)

    def close(self):
        self.file.close()


class HeapFile:
    """Archivo heap: inserción/lista libre O(1), exploración de una lectura por página."""

    def __init__(self, path, schema, counter=None, page_size=4096, create=False):
        self.schema = schema if isinstance(schema, Schema) else Schema(schema)
        self.counter = counter if counter is not None else DiskCounter()
        self._store = _RecordFile(path, self.schema, self.counter, page_size, create)
        self.path, self.page_size = Path(path), page_size
        self.file = self._store.file

    @property
    def record_count(self):
        return self._store.record_count

    @property
    def page_count(self):
        return self.file.page_count

    def insert(self, row):
        return self._store.insert_bytes(self.schema.encode(row))

    def get(self, rid):
        raw = self._store.read_slot(rid)
        return None if raw is None else self.schema.decode(raw)

    def scan(self):
        for rid, page, offset in self._store.scan_raw():
            yield rid, self.schema.decode(page, offset)

    def scan_equal(self, key, value):
        reader = self.schema.read_field
        for rid, page, offset in self._store.scan_raw():
            if reader(page, key, offset) == value:
                yield rid, self.schema.decode(page, offset)

    def delete(self, rid):
        return self._store.delete(rid)

    def close(self):
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class SequentialFile:
    """Principal ordenado por rangos de página y overflow con cadena ordenada.

    RID principal=(pid,slot), overflow=(-pid,slot). El nivel cero enlaza todos
    los registros de overflow; doce niveles, con probabilidad 1/4 de promoción,
    aceleran la localización sin guardar un índice de claves en memoria.
    Solo reorganize/bulk_load cambia RID existentes; el motor reconstruye índices.
    """

    def __init__(self, path, schema, key_name=None, counter=None, page_size=4096, create=False):
        self.schema = schema if isinstance(schema, Schema) else Schema(schema)
        self.key_name = key_name or self.schema.key_name
        if self.key_name not in self.schema.names:
            raise ValueError("La clave secuencial no existe")
        self.path, self.page_size = Path(path), page_size
        self.counter = counter if counter is not None else DiskCounter()
        self._key_reader = self.schema._readers[self.key_name]
        self._main = _RecordFile(path, self.schema, self.counter, page_size, create,
                                 prefix_size=self._key_reader.size, user_size=16)
        key_digest = hashlib.blake2b(self.key_name.encode(), digest_size=16).digest()
        if self._main.user == bytes(16):
            self._main.user[:] = key_digest
            self._main._write_meta()
        elif self._main.user != key_digest:
            self._main.close()
            raise ValueError("La clave secuencial no coincide con el archivo")
        self.overflow_path = Path(str(path) + ".overflow")
        self._overflow = _RecordFile(self.overflow_path, self.schema, self.counter, page_size, create,
                                     record_size=self.schema.record_size + _NODE_EXTRA,
                                     user_size=8 + _LEVELS * 8)
        self.file = self._main.file

    @property
    def record_count(self):
        return self._main.record_count + self._overflow.record_count

    @property
    def page_count(self):
        return self._main.file.page_count + self._overflow.file.page_count

    @property
    def overflow_count(self):
        return self._overflow.record_count

    @staticmethod
    def _token(rid):
        return (rid[0] << 16) | rid[1]

    @staticmethod
    def _rid(token):
        return token >> 16, token & 65535

    def _node(self, token):
        node = self._overflow.read_slot(self._rid(token))
        if node is None:
            raise ValueError("Enlace de overflow apunta a un slot libre")
        return node

    def _next(self, node, level):
        return _UINT64.unpack_from(node, self.schema.record_size + 8 + level * 8)[0]

    def _head(self, level):
        return _UINT64.unpack_from(self._overflow.user, 8 + level * 8)[0]

    def _key(self, raw, offset=0):
        return self.schema.read_field(raw, self.key_name, offset)

    def _find_overflow(self, key, include_equal=False):
        # Esta función desciende por enlaces de salto y devuelve predecesores.
        update = [0] * _LEVELS
        current, node = 0, None
        for level in range(_LEVELS - 1, -1, -1):
            following = self._head(level) if current == 0 else self._next(node, level)
            while following:
                candidate = self._node(following)
                candidate_key = self._key(candidate)
                if candidate_key > key or (candidate_key == key and not include_equal):
                    break
                current, node = following, candidate
                following = self._next(node, level)
            update[level] = current
        following = self._head(0) if current == 0 else self._next(node, 0)
        return update, following

    def _fence(self, page):
        offset = HEADER_SIZE + self._main.bitmap_size
        value = self._key_reader.unpack_from(page, offset)[0]
        return value.rstrip(b"\x00").decode("utf-8") if isinstance(value, bytes) else value

    def _set_fence(self, page_id, raw):
        page = self._main._page(page_id)
        key_offset = self.schema.offsets[self.key_name]
        offset = HEADER_SIZE + self._main.bitmap_size
        page[offset:offset + self._key_reader.size] = raw[key_offset:key_offset + self._key_reader.size]
        self.file.write(page_id, page)

    def _locate_main(self, key):
        if self.file.page_count == 1:
            return None, None
        low, high = 1, self.file.page_count - 1
        while low < high:
            middle = (low + high) // 2
            page = self._main._page(middle)
            if self._fence(page) < key:
                low = middle + 1
            else:
                high = middle
        return low, self._main._page(low)

    def insert(self, row):
        raw = self.schema.encode(row)
        key = row[self.key_name]
        page_id, page = self._locate_main(key)
        if page_id is None:
            rid = self._main.insert_bytes(raw)
            self._set_fence(rid[0], raw)
            return rid
        count = PAGE_HEADER.unpack_from(page)[1]
        fence = self._fence(page)
        if count < self._main.capacity:
            rid = self._main.insert_bytes(raw, preferred_page=page_id)
            if key > fence:
                self._set_fence(page_id, raw)
            return rid
        if page_id == self.file.page_count - 1 and key > fence:
            rid = self._main.insert_bytes(raw, preferred_page=-1)
            self._set_fence(rid[0], raw)
            return rid
        update, following = self._find_overflow(key, include_equal=True)
        serial = _UINT64.unpack_from(self._overflow.user)[0] + 1
        # SplitMix64 proporciona alturas reproducibles incluso después de reabrir.
        mixed = (serial + 0x9e3779b97f4a7c15) & 0xffffffffffffffff
        mixed = ((mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9) & 0xffffffffffffffff
        mixed = ((mixed ^ (mixed >> 27)) * 0x94d049bb133111eb) & 0xffffffffffffffff
        mixed ^= mixed >> 31
        height = 1
        while height < _LEVELS and (mixed & 3) == 0:
            height += 1
            mixed >>= 2
        node = bytearray(raw + bytes(_NODE_EXTRA))
        node[self.schema.record_size] = height
        predecessors = {}
        for level in range(height):
            predecessor = update[level]
            if predecessor:
                if predecessor not in predecessors:
                    predecessors[predecessor] = bytearray(self._node(predecessor))
                following = self._next(predecessors[predecessor], level)
            else:
                following = self._head(level)
            _UINT64.pack_into(node, self.schema.record_size + 8 + level * 8, following)
        rid = self._overflow.insert_bytes(node)
        token = self._token(rid)
        for level in range(height):
            predecessor = update[level]
            if predecessor:
                _UINT64.pack_into(predecessors[predecessor], self.schema.record_size + 8 + level * 8, token)
            else:
                _UINT64.pack_into(self._overflow.user, 8 + level * 8, token)
        for predecessor, predecessor_node in predecessors.items():
            self._overflow.replace_bytes(self._rid(predecessor), predecessor_node)
        _UINT64.pack_into(self._overflow.user, 0, serial)
        self._overflow._write_meta()
        return -rid[0], rid[1]

    def get(self, rid):
        page_id, slot = rid
        raw = self._main.read_slot(rid) if page_id > 0 else self._overflow.read_slot((-page_id, slot))
        return None if raw is None else self.schema.decode(raw)

    def scan(self):
        # Esta función hace un barrido físico de P páginas; SQL no promete orden.
        for rid, page, offset in self._main.scan_raw():
            yield rid, self.schema.decode(page, offset)
        for rid, page, offset in self._overflow.scan_raw():
            yield (-rid[0], rid[1]), self.schema.decode(page, offset)

    def scan_equal(self, key, value):
        for store, sign in ((self._main, 1), (self._overflow, -1)):
            for rid, page, offset in store.scan_raw():
                if self.schema.read_field(page, key, offset) == value:
                    yield (sign * rid[0], rid[1]), self.schema.decode(page, offset)

    def _main_range(self, lo, hi):
        page_id, page = self._locate_main(lo)
        if page_id is None:
            return
        while page_id < self.file.page_count:
            records = [(self._key(buffer, offset), rid, self.schema.decode(buffer, offset))
                       for rid, buffer, offset in self._main.iter_page(page_id, page)]
            records.sort(key=lambda item: item[0])
            for key, rid, row in records:
                if key > hi:
                    return
                if key >= lo:
                    yield rid, row
            if self._fence(page) > hi:
                return
            page_id += 1
            if page_id < self.file.page_count:
                page = self._main._page(page_id)

    def _overflow_range(self, lo, hi):
        _, token = self._find_overflow(lo)
        while token:
            node = self._node(token)
            if self._key(node) > hi:
                return
            page_id, slot = self._rid(token)
            yield (-page_id, slot), self.schema.decode(node)
            token = self._next(node, 0)

    def search(self, key):
        return list(self.range_search(key, key))

    def range_search(self, lo, hi):
        if lo > hi:
            return
        yield from heapq.merge(self._main_range(lo, hi), self._overflow_range(lo, hi),
                               key=lambda item: item[1][self.key_name])

    def delete(self, rid):
        page_id, slot = rid
        if page_id > 0:
            return self._main.delete(rid)
        internal_rid = (-page_id, slot)
        node = self._overflow.read_slot(internal_rid)
        if node is None:
            return False
        token, key = self._token(internal_rid), self._key(node)
        update, _ = self._find_overflow(key)
        # Las claves duplicadas pueden tener varios predecesores con la misma clave.
        changed = {}
        for level in range(node[self.schema.record_size]):
            predecessor = update[level]
            following = self._head(level) if not predecessor else self._next(self._node(predecessor), level)
            while following and following != token:
                predecessor = following
                following = self._next(self._node(predecessor), level)
            if following != token:
                raise ValueError("Cadena de overflow inconsistente")
            replacement = self._next(node, level)
            if predecessor:
                if predecessor not in changed:
                    changed[predecessor] = bytearray(self._node(predecessor))
                _UINT64.pack_into(changed[predecessor], self.schema.record_size + 8 + level * 8, replacement)
            else:
                _UINT64.pack_into(self._overflow.user, 8 + level * 8, replacement)
        for predecessor, previous in changed.items():
            self._overflow.replace_bytes(self._rid(predecessor), previous)
        return self._overflow.delete(internal_rid)

    def bulk_load(self, rows, fill_factor=.75):
        """Esta función reemplaza el contenido por páginas ordenadas al 70–80 %."""
        if not .7 <= fill_factor <= .8:
            raise ValueError("fill_factor debe estar entre 0.70 y 0.80")
        # Los bytes fijos reducen la memoria y evitan conservar diccionarios completos.
        records = [self.schema.encode(row) for row in rows]
        records.sort(key=self._key)
        self._rewrite(records, fill_factor)

    def _rewrite(self, records, fill_factor):
        temporary = Path(str(self.path) + ".reorganizing")
        temporary_overflow = Path(str(temporary) + ".overflow")
        for path in (temporary, temporary_overflow):
            if path.exists():
                path.unlink()
        fresh = SequentialFile(temporary, self.schema, self.key_name, self.counter, self.page_size, create=True)
        target = max(1, int(fresh._main.capacity * fill_factor))
        previous_free = -1
        for start in range(0, len(records), target):
            chunk = records[start:start + target]
            page_id = fresh.file.page_count
            page = fresh._main._empty_page(page_id)
            for slot, raw in enumerate(chunk):
                page[HEADER_SIZE + slot // 8] |= 1 << (slot % 8)
                offset = fresh._main._offset(slot)
                page[offset:offset + len(raw)] = raw
            offset = HEADER_SIZE + fresh._main.bitmap_size
            key_offset = self.schema.offsets[self.key_name]
            page[offset:offset + self._key_reader.size] = chunk[-1][key_offset:key_offset + self._key_reader.size]
            has_space = len(chunk) < fresh._main.capacity
            PAGE_HEADER.pack_into(page, 0, page_id, len(chunk), fresh._main._offset(len(chunk)), -1, previous_free if has_space else -1)
            if has_space and previous_free >= 1:
                previous = fresh._main._page(previous_free)
                fields = list(PAGE_HEADER.unpack_from(previous))
                fields[3] = page_id
                PAGE_HEADER.pack_into(previous, 0, *fields)
                fresh.file.write(previous_free, previous)
            elif has_space:
                fresh._main.free_head = page_id
            fresh.file.write(page_id, page)
            fresh._main.record_count += len(chunk)
            if has_space:
                previous_free = page_id
        fresh._main._write_meta()
        fresh.close()
        self.close()
        os.replace(temporary, self.path)
        os.replace(temporary_overflow, self.overflow_path)
        self.__init__(self.path, self.schema, self.key_name, self.counter, self.page_size)

    def reorganize(self, fill_factor=.75):
        if not .7 <= fill_factor <= .8:
            raise ValueError("fill_factor debe estar entre 0.70 y 0.80")
        records = []
        for store in (self._main, self._overflow):
            records.extend(bytes(page[offset:offset + self.schema.record_size])
                           for _, page, offset in store.scan_raw())
        records.sort(key=self._key)
        self._rewrite(records, fill_factor)

    def close(self):
        self._main.close()
        self._overflow.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
