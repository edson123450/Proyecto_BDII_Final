"""Índices persistentes: árbol B+ y hashing extensible sobre páginas fijas."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
from typing import Iterator

from .io import DiskCounter, DiskFile

RID = tuple[int, int]
# Identificador, cantidad, offset libre, siguiente, anterior, tipo, auxiliar, hash.
PAGE_HEADER = struct.Struct("<4sIHHiiIIQ")
_META = struct.Struct("<8sIIIIQIIII")
_POINTER = struct.Struct("<I")
_RID = struct.Struct("<iI")
_MAGIC = b"IDX1"
_META_PAGE, _LEAF, _INTERNAL, _DIRECTORY, _BUCKET = range(1, 6)
_BPLUS_MAGIC, _HASH_MAGIC = b"BPLUS001", b"EXHASH01"
_KEY_TYPES = {"INT": 1, "FLOAT": 2, "CHAR": 3}


class _KeyCodec:
    def __init__(self, key_type: str, key_size: int):
        self.key_type = key_type.upper()
        if self.key_type not in _KEY_TYPES:
            raise ValueError("Tipo de clave no soportado: " + key_type)
        self.key_size = int(key_size) if self.key_type == "CHAR" else 8
        if not 1 <= self.key_size <= 65500:
            raise ValueError("key_size debe estar entre 1 y 65500")
        self.width = self.key_size + 2 if self.key_type == "CHAR" else 8
        self.record = struct.Struct("<qiI" if self.key_type == "INT" else "<diI") if self.key_type != "CHAR" else None

    def normalize(self, key):
        if self.key_type == "INT":
            if isinstance(key, bool) or not isinstance(key, int):
                raise ValueError("La clave INT debe ser un entero")
            if not -(1 << 63) <= key < (1 << 63):
                raise ValueError("Clave INT fuera de rango de 64 bits")
            return key
        if self.key_type == "FLOAT":
            if isinstance(key, bool) or not isinstance(key, (int, float)):
                raise ValueError("La clave FLOAT debe ser numérica")
            value = float(key)
            if not math.isfinite(value):
                raise ValueError("La clave FLOAT debe ser finita")
            return 0.0 if value == 0 else value
        if not isinstance(key, str):
            raise ValueError("La clave CHAR debe ser texto")
        if len(key.encode("utf-8")) > self.key_size:
            raise ValueError("La clave CHAR excede su ancho en bytes UTF-8")
        return key

    def encode(self, key) -> bytes:
        if self.key_type == "INT":
            return struct.pack("<q", key)
        if self.key_type == "FLOAT":
            return struct.pack("<d", key)
        raw = key.encode("utf-8")
        return struct.pack("<H", len(raw)) + raw.ljust(self.key_size, b"\0")

    def read_entry(self, data, offset):
        if self.record is not None:
            key, page, slot = self.record.unpack_from(data, offset)
            return key, (page, slot)
        length = struct.unpack_from("<H", data, offset)[0]
        if length > self.key_size:
            raise ValueError("Longitud CHAR corrupta en índice")
        key = bytes(data[offset + 2:offset + 2 + length]).decode("utf-8")
        return key, _RID.unpack_from(data, offset + self.width)

    def write_entry(self, data, offset, entry):
        key, rid = entry
        if self.record is not None:
            self.record.pack_into(data, offset, key, *rid)
        else:
            data[offset:offset + self.width] = self.encode(key)
            _RID.pack_into(data, offset + self.width, *rid)


def _normalize_rid(rid) -> RID:
    if (len(rid) != 2 or any(isinstance(x, bool) or not isinstance(x, int) for x in rid)
            or not -(1 << 31) <= rid[0] < (1 << 31) or not 0 <= rid[1] <= 0xFFFFFFFF):
        raise ValueError("RID requiere página signed32 y posición unsigned32")
    return rid[0], rid[1]


def _header(data, pid, kind=None):
    magic, actual, count, free, nxt, prev, actual_kind, aux, token = PAGE_HEADER.unpack_from(data)
    if magic != _MAGIC or actual != pid or (kind is not None and actual_kind != kind):
        raise ValueError(f"Cabecera de página de índice inválida: {pid}")
    if not PAGE_HEADER.size <= free <= len(data):
        raise ValueError(f"Offset libre de página inválido: {pid}")
    return count, free, nxt, prev, actual_kind, aux, token


def _page(page_size, pid, count, free, nxt=-1, prev=-1, kind=_LEAF, aux=0, token=0):
    data = bytearray(page_size)
    PAGE_HEADER.pack_into(data, 0, _MAGIC, pid, count, free, nxt, prev, kind, aux, token)
    return data


class _Index:
    INDEX_MAGIC = b""

    def __init__(self, path, key_type="INT", key_size=8, page_size=4096, counter=None, create=False):
        if not 128 <= page_size <= 65535:
            raise ValueError("page_size debe estar entre 128 y 65535")
        self.path = Path(path)
        self.page_size = page_size
        self.codec = _KeyCodec(key_type, key_size)
        self.entry_size = self.codec.width + _RID.size
        self.capacity = (page_size - PAGE_HEADER.size) // self.entry_size
        self.internal_capacity = (page_size - PAGE_HEADER.size - 4) // (self.entry_size + 4)
        if self.capacity < 3 or self.internal_capacity < 3:
            raise ValueError("La página debe admitir al menos tres claves")
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(b"")
        self.counter = counter if counter is not None else DiskCounter()
        self.file = DiskFile(self.path, page_size=page_size, counter=self.counter, role="index")
        self._closed = False
        if self.file.page_count == 0:
            self._size = 0
            self._root = 0
            self._height = 0
            self._directory_pages = 0
            self._bucket_count = 0
            self._first_leaf = 0
            self._initialize()
        else:
            try:
                self._load_meta()
            except Exception:
                self.file.close()
                raise

    def _load_meta(self):
        data = self.file.read(0)
        _header(data, 0, _META_PAGE)
        magic, key_code, key_size, page_size, root, size, height, dir_pages, buckets, first_leaf = _META.unpack_from(data, PAGE_HEADER.size)
        if magic != self.INDEX_MAGIC:
            raise ValueError("El archivo no corresponde al tipo de índice solicitado")
        if key_code != _KEY_TYPES[self.codec.key_type] or key_size != self.codec.key_size or page_size != self.page_size:
            raise ValueError("El tipo, ancho o tamaño de página no coincide con el índice persistido")
        self._root, self._size, self._height = root, size, height
        self._directory_pages, self._bucket_count, self._first_leaf = dir_pages, buckets, first_leaf

    def _save_meta(self):
        data = _page(self.page_size, 0, 1, PAGE_HEADER.size + _META.size, kind=_META_PAGE)
        _META.pack_into(data, PAGE_HEADER.size, self.INDEX_MAGIC, _KEY_TYPES[self.codec.key_type], self.codec.key_size, self.page_size,
                        self._root, self._size, self._height, self._directory_pages, self._bucket_count, self._first_leaf)
        self.file.write(0, data)

    @property
    def size(self):
        return self._size

    @property
    def height(self):
        return self._height

    @property
    def root_page_id(self):
        return self._root

    def close(self):
        if not self._closed:
            self.file.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@dataclass
class _Node:
    pid: int
    leaf: bool
    entries: list
    children: list
    next: int = -1
    prev: int = -1


class BPlusTree(_Index):
    """Separadores (clave, RID); hojas dobles y división recursiva al 50 %."""
    INDEX_MAGIC = _BPLUS_MAGIC

    def _initialize(self):
        self.file.allocate()
        self._root = self.file.allocate()
        self._first_leaf = self._root
        self._height = 1
        self._write_node(_Node(self._root, True, [], []))
        self._save_meta()

    def _read_node(self, pid):
        data = self.file.read(pid)
        count, free, nxt, prev, kind, _, _ = _header(data, pid)
        if kind not in (_LEAF, _INTERNAL):
            raise ValueError("Página ajena al árbol B+")
        leaf = kind == _LEAF
        stride = self.entry_size if leaf else self.entry_size + 4
        offset = PAGE_HEADER.size + (0 if leaf else 4)
        if free != offset + count * stride or count > (self.capacity if leaf else self.internal_capacity):
            raise ValueError("Cantidad de claves inválida en árbol B+")
        entries, children = [], []
        if not leaf:
            children.append(_POINTER.unpack_from(data, PAGE_HEADER.size)[0])
        for _ in range(count):
            entries.append(self.codec.read_entry(data, offset))
            if not leaf:
                children.append(_POINTER.unpack_from(data, offset + self.entry_size)[0])
            offset += stride
        return _Node(pid, leaf, entries, children, nxt, prev)

    def _write_node(self, node):
        stride = self.entry_size if node.leaf else self.entry_size + 4
        offset = PAGE_HEADER.size + (0 if node.leaf else 4)
        free = offset + len(node.entries) * stride
        if free > self.page_size:
            raise ValueError("El nodo desborda su página")
        data = _page(self.page_size, node.pid, len(node.entries), free, node.next, node.prev, _LEAF if node.leaf else _INTERNAL)
        if not node.leaf:
            _POINTER.pack_into(data, PAGE_HEADER.size, node.children[0])
        for index, entry in enumerate(node.entries):
            self.codec.write_entry(data, offset, entry)
            if not node.leaf:
                _POINTER.pack_into(data, offset + self.entry_size, node.children[index + 1])
            offset += stride
        self.file.write(node.pid, data)

    def _descend(self, entry, left_bias=False):
        path = []
        node = self._read_node(self._root)
        while not node.leaf:
            position = bisect_left(node.entries, entry) if left_bias else bisect_right(node.entries, entry)
            path.append((node.pid, position))
            node = self._read_node(node.children[position])
        return node, path

    def insert(self, key, rid: RID):
        entry = (self.codec.normalize(key), _normalize_rid(rid))
        node, path = self._descend(entry)
        node.entries.insert(bisect_right(node.entries, entry), entry)
        if len(node.entries) <= self.capacity:
            self._write_node(node)
        else:
            midpoint = len(node.entries) // 2
            right = _Node(self.file.allocate(), True, node.entries[midpoint:], [], node.next, node.pid)
            node.entries = node.entries[:midpoint]
            node.next = right.pid
            if right.next != -1:
                following = self._read_node(right.next)
                following.prev = right.pid
                self._write_node(following)
            self._write_node(node)
            self._write_node(right)
            self._propagate_split(node.pid, right.entries[0], right.pid, path)
        self._size += 1
        self._save_meta()

    def _propagate_split(self, left_pid, separator, right_pid, path):
        while path:
            parent_pid, position = path.pop()
            parent = self._read_node(parent_pid)
            parent.entries.insert(position, separator)
            parent.children.insert(position + 1, right_pid)
            if len(parent.entries) <= self.internal_capacity:
                self._write_node(parent)
                return
            midpoint = len(parent.entries) // 2
            separator = parent.entries[midpoint]
            right = _Node(self.file.allocate(), False, parent.entries[midpoint + 1:], parent.children[midpoint + 1:])
            parent.entries = parent.entries[:midpoint]
            parent.children = parent.children[:midpoint + 1]
            self._write_node(parent)
            self._write_node(right)
            left_pid, right_pid = parent.pid, right.pid
        root = _Node(self.file.allocate(), False, [separator], [left_pid, right_pid])
        self._write_node(root)
        self._root = root.pid
        self._height += 1

    def iter_entries(self, low=None, high=None, include_low=True, include_high=True) -> Iterator[tuple[object, RID]]:
        low = self.codec.normalize(low) if low is not None else None
        high = self.codec.normalize(high) if high is not None else None
        if low is not None and high is not None and low > high:
            return
        node = self._read_node(self._first_leaf) if low is None else self._descend((low, (-(1 << 31), -1)), left_bias=True)[0]
        while True:
            for key, rid in node.entries:
                if low is not None and (key < low or (key == low and not include_low)):
                    continue
                if high is not None and (key > high or (key == high and not include_high)):
                    return
                yield key, rid
            if node.next == -1:
                return
            node = self._read_node(node.next)

    def search(self, key) -> list[RID]:
        key = self.codec.normalize(key)
        return [rid for _, rid in self.iter_entries(key, key)]

    def range_search(self, low=None, high=None, include_low=True, include_high=True) -> list[RID]:
        return [rid for _, rid in self.iter_entries(low, high, include_low, include_high)]

    def delete(self, key, rid: RID) -> bool:
        entry = self.codec.normalize(key), _normalize_rid(rid)
        node, _ = self._descend(entry, left_bias=True)
        while True:
            position = bisect_left(node.entries, entry)
            if position < len(node.entries) and node.entries[position] == entry:
                node.entries.pop(position)
                self._write_node(node)
                self._size -= 1
                self._save_meta()
                return True
            if (node.entries and node.entries[-1] > entry) or node.next == -1:
                return False
            node = self._read_node(node.next)

    def stats(self):
        return {"type": "BTREE", "size": self._size, "height": self._height,
                "fanout": self.internal_capacity + 1, "leaf_capacity": self.capacity,
                "pages": self.file.page_count, "root_page_id": self._root, "page_size": self.page_size}


class ExtendibleHash(_Index):
    """Directorio paginado; buckets con profundidad local y overflow de duplicados."""
    INDEX_MAGIC = _HASH_MAGIC

    @property
    def directory_capacity(self):
        return (self.page_size - PAGE_HEADER.size) // 4

    @property
    def global_depth(self):
        return self._height

    def _initialize(self):
        self.file.allocate()
        self._root = self.file.allocate()
        bucket = self.file.allocate()
        self._height = 0
        self._directory_pages = 1
        self._bucket_count = 1
        self._write_directory_page(self._root, [bucket])
        self._write_bucket(bucket, [], 0)
        self._save_meta()

    def _hash(self, key):
        return int.from_bytes(hashlib.blake2b(self.codec.encode(key), digest_size=8).digest(), "little")

    def _write_directory_page(self, pid, pointers):
        data = _page(self.page_size, pid, len(pointers), PAGE_HEADER.size + 4 * len(pointers), kind=_DIRECTORY)
        for index, value in enumerate(pointers):
            _POINTER.pack_into(data, PAGE_HEADER.size + 4 * index, value)
        self.file.write(pid, data)

    def _directory_pointer(self, position):
        page_offset, slot = divmod(position, self.directory_capacity)
        pid = self._root + page_offset
        data = self.file.read(pid)
        count, free, *_ = _header(data, pid, _DIRECTORY)
        if slot >= count or free != PAGE_HEADER.size + count * 4:
            raise ValueError("Entrada de directorio hash inválida")
        return _POINTER.unpack_from(data, PAGE_HEADER.size + slot * 4)[0]

    def _double_directory(self):
        old_size = 1 << self._height
        new_size = old_size * 2
        new_pages = (new_size + self.directory_capacity - 1) // self.directory_capacity
        old_root = self._root
        if new_pages == self._directory_pages:
            data = self.file.read(old_root)
            _header(data, old_root, _DIRECTORY)
            pointers = [_POINTER.unpack_from(data, PAGE_HEADER.size + 4 * i)[0] for i in range(old_size)]
            self._write_directory_page(old_root, pointers + pointers)
        else:
            new_root = self.file.allocate()
            for _ in range(new_pages - 1):
                self.file.allocate()
            source_pid, source_data = -1, None
            for page_offset in range(new_pages):
                start = page_offset * self.directory_capacity
                pointers = []
                for position in range(start, min(start + self.directory_capacity, new_size)):
                    original = position % old_size
                    old_page, old_slot = divmod(original, self.directory_capacity)
                    pid = old_root + old_page
                    if source_pid != pid:
                        source_data = self.file.read(pid)
                        _header(source_data, pid, _DIRECTORY)
                        source_pid = pid
                    pointers.append(_POINTER.unpack_from(source_data, PAGE_HEADER.size + old_slot * 4)[0])
                self._write_directory_page(new_root + page_offset, pointers)
            self._root = new_root
            self._directory_pages = new_pages
        self._height += 1

    def _read_bucket(self, pid):
        data = self.file.read(pid)
        count, free, nxt, tail, _, local_depth, token = _header(data, pid, _BUCKET)
        if count > self.capacity or free != PAGE_HEADER.size + count * self.entry_size:
            raise ValueError("Cantidad de entradas inválida en bucket")
        # El bit alto de profundidad marca un bucket homogéneo por hash.
        uniform = bool(local_depth & (1 << 31))
        local_depth &= 0x7FFFFFFF
        entries = [self.codec.read_entry(data, PAGE_HEADER.size + i * self.entry_size) for i in range(count)]
        return {"pid": pid, "entries": entries, "next": nxt, "tail": tail,
                "depth": local_depth, "uniform": uniform, "hash": token}

    def _write_bucket(self, pid, entries, depth, nxt=-1, tail=-1, uniform=None, token=None):
        if uniform is None:
            hashes = {self._hash(entry[0]) for entry in entries}
            uniform = len(hashes) <= 1
            token = next(iter(hashes), 0)
        data = _page(self.page_size, pid, len(entries), PAGE_HEADER.size + len(entries) * self.entry_size,
                     nxt, tail, _BUCKET, depth | ((1 << 31) if uniform else 0), token or 0)
        for index, entry in enumerate(entries):
            self.codec.write_entry(data, PAGE_HEADER.size + index * self.entry_size, entry)
        self.file.write(pid, data)

    def _store_bucket(self, bucket):
        self._write_bucket(bucket["pid"], bucket["entries"], bucket["depth"], bucket["next"], bucket["tail"], bucket["uniform"], bucket["hash"])

    def _bucket_chain(self, root):
        yield root
        while root["next"] != -1:
            root = self._read_bucket(root["next"])
            yield root

    def _write_chain(self, pid, entries, depth, reuse):
        chunks = [entries[start:start + self.capacity] for start in range(0, len(entries), self.capacity)] or [[]]
        ids = [pid]
        for _ in chunks[1:]:
            ids.append(reuse.pop() if reuse else self.file.allocate())
        hashes = {self._hash(entry[0]) for entry in entries}
        uniform, token = len(hashes) <= 1, next(iter(hashes), 0)
        for index, chunk in enumerate(chunks):
            self._write_bucket(ids[index], chunk, depth, ids[index + 1] if index + 1 < len(ids) else -1,
                               ids[-1] if index == 0 and len(ids) > 1 else -1, uniform, token)

    def _redirect_split(self, position, depth, right_pid):
        prefix = position & ((1 << depth) - 1)
        first = prefix | (1 << depth)
        stride = 1 << (depth + 1)
        current_pid, data = -1, None
        for slot in range(first, 1 << self._height, stride):
            page_offset, in_page = divmod(slot, self.directory_capacity)
            pid = self._root + page_offset
            if pid != current_pid:
                if data is not None:
                    self.file.write(current_pid, data)
                data = self.file.read(pid)
                _header(data, pid, _DIRECTORY)
                current_pid = pid
            _POINTER.pack_into(data, PAGE_HEADER.size + in_page * 4, right_pid)
        if data is not None:
            self.file.write(current_pid, data)

    def _split_bucket(self, bucket, position):
        depth = bucket["depth"]
        if depth >= 63:
            raise OverflowError("Se agotaron los bits de hash disponibles")
        if depth == self._height:
            self._double_directory()
        chain = list(self._bucket_chain(bucket))
        left, right = [], []
        for page in chain:
            for entry in page["entries"]:
                (right if self._hash(entry[0]) & (1 << depth) else left).append(entry)
        right_pid = self.file.allocate()
        reuse = [page["pid"] for page in chain[1:]]
        self._write_chain(bucket["pid"], left, depth + 1, reuse)
        self._write_chain(right_pid, right, depth + 1, reuse)
        self._redirect_split(position, depth, right_pid)
        self._bucket_count += 1

    def insert(self, key, rid: RID):
        entry = self.codec.normalize(key), _normalize_rid(rid)
        digest = self._hash(entry[0])
        while True:
            position = digest & ((1 << self._height) - 1)
            bucket = self._read_bucket(self._directory_pointer(position))
            if len(bucket["entries"]) < self.capacity:
                if not bucket["entries"] and bucket["next"] == -1:
                    bucket["hash"], bucket["uniform"] = digest, True
                elif bucket["hash"] != digest:
                    bucket["uniform"] = False
                bucket["entries"].append(entry)
                self._store_bucket(bucket)
                break
            if bucket["uniform"] and bucket["hash"] == digest:
                tail = self._read_bucket(bucket["tail"]) if bucket["tail"] != -1 else bucket
                if len(tail["entries"]) == self.capacity:
                    new_pid = self.file.allocate()
                    self._write_bucket(new_pid, [entry], bucket["depth"], uniform=True, token=digest)
                    if tail["pid"] == bucket["pid"]:
                        bucket["next"] = new_pid
                    else:
                        tail["next"] = new_pid
                        self._store_bucket(tail)
                    bucket["tail"] = new_pid
                    self._store_bucket(bucket)
                else:
                    tail["entries"].append(entry)
                    self._store_bucket(tail)
                break
            self._split_bucket(bucket, position)
        self._size += 1
        self._save_meta()

    def search(self, key) -> list[RID]:
        key = self.codec.normalize(key)
        position = self._hash(key) & ((1 << self._height) - 1)
        bucket = self._read_bucket(self._directory_pointer(position))
        return [rid for page in self._bucket_chain(bucket) for found, rid in page["entries"] if found == key]

    def delete(self, key, rid: RID) -> bool:
        entry = self.codec.normalize(key), _normalize_rid(rid)
        position = self._hash(entry[0]) & ((1 << self._height) - 1)
        bucket = self._read_bucket(self._directory_pointer(position))
        for page in self._bucket_chain(bucket):
            try:
                page["entries"].remove(entry)
            except ValueError:
                continue
            self._store_bucket(page)
            self._size -= 1
            self._save_meta()
            return True
        return False

    def stats(self):
        return {"type": "HASH", "size": self._size, "height": 1, "fanout": 1 << self._height,
                "global_depth": self._height, "buckets": self._bucket_count,
                "bucket_capacity": self.capacity, "directory_pages": self._directory_pages,
                "pages": self.file.page_count, "root_page_id": self._root, "page_size": self.page_size}
