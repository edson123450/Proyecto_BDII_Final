"""R-Tree dinámico y baseline espacial sobre páginas binarias propias.

Las coordenadas persistidas son float32. Las distancias se evalúan en float64
sobre esas mismas coordenadas y centros normalizados a float32. No se conserva
un caché de nodos ni una copia de los puntos en memoria entre operaciones.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path
import struct

import numpy as np

from .io import DiskCounter, DiskFile


HEADER = struct.Struct("<4siiiiiB7x")  # magic,pid,count,free,next,prev,kind
META = struct.Struct("<8sIIIIQQ")      # magic,version,D,B,root,size,height
RID = tuple[int, int]


def canonical_point(point, dimensions: int) -> np.ndarray:
    values = np.asarray(point, dtype=np.float64)
    if values.shape != (dimensions,) or not np.isfinite(values).all():
        raise ValueError(f"Se requieren {dimensions} coordenadas finitas")
    with np.errstate(over="ignore"):
        packed = values.astype(np.float32)
    if not np.isfinite(packed).all():
        raise ValueError("Coordenada fuera del dominio float32")
    return packed


def outward(values, dimensions: int, lower: bool) -> np.ndarray:
    """Redondeo conservador: una caja nunca se reduce al convertir a float32."""
    source = np.asarray(values, dtype=np.float64)
    packed = canonical_point(source, dimensions)
    wrong = packed.astype(np.float64) > source if lower else packed.astype(np.float64) < source
    return np.where(wrong, np.nextafter(packed, -np.inf if lower else np.inf, dtype=np.float32), packed)


def _rid(rid) -> RID:
    if len(rid) != 2 or any(not isinstance(x, (int, np.integer)) or not -(2**31) <= int(x) < 2**31 for x in rid):
        raise ValueError("RID debe contener dos enteros signed32")
    return int(rid[0]), int(rid[1])


def _areas(low, high):
    return np.prod(np.maximum(high.astype(np.float64) - low.astype(np.float64), 0), axis=-1)


def _mindist2(point, low, high):
    delta = np.maximum(np.maximum(low.astype(np.float64) - point, point - high.astype(np.float64)), 0)
    return np.einsum("...i,...i->...", delta, delta)


def _offer(heap, distance2: float, rid: RID, k: int):
    item = (-float(distance2), -int(rid[0]), -int(rid[1]))
    if len(heap) < k:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _answers(heap):
    ordered = sorted((-d, (-p, -s)) for d, p, s in heap)
    return [(rid, math.sqrt(distance2)) for distance2, rid in ordered]


@dataclass
class _Node:
    page_id: int
    leaf: bool
    low: np.ndarray
    high: np.ndarray
    refs: np.ndarray

    def bounds(self):
        return np.min(self.low, axis=0), np.max(self.high, axis=0)


class RTree:
    """R-Tree de puntos D-dimensionales con quadratic split y k-NN exacto."""

    def __init__(self, path, dimensions=2, page_size=4096, counter=None, create=False):
        if not isinstance(dimensions, int) or not 2 <= dimensions <= 128:
            raise ValueError("dimensions debe estar entre 2 y 128")
        self.path = Path(path)
        self.dimensions, self.page_size = dimensions, page_size
        self.counter = counter if counter is not None else DiskCounter()
        self.dtype = np.dtype([("low", "<f4", (dimensions,)), ("high", "<f4", (dimensions,)), ("ref", "<i4", (2,))])
        self.capacity = (page_size - HEADER.size) // self.dtype.itemsize
        if self.capacity < 3:
            raise ValueError("La página debe admitir al menos tres entradas")
        if create and self.path.exists():
            raise FileExistsError(self.path)
        self.file = DiskFile(self.path, page_size, self.counter, role="index")
        if create:
            self.root, self.size, self.height = 1, 0, 1
            self._save_meta()
            self._write(_Node(1, True, np.empty((0, dimensions), np.float32),
                              np.empty((0, dimensions), np.float32), np.empty((0, 2), np.int32)))
        else:
            try:
                page = self.file.read(0)
                magic, version, saved_d, saved_b, self.root, self.size, self.height = META.unpack_from(page, HEADER.size)
                if (magic, version, saved_d, saved_b) != (b"P2RTREE\0", 1, dimensions, page_size):
                    raise ValueError("Archivo R-Tree incompatible con dimensiones/tamaño de página")
            except Exception:
                self.file.close()
                raise

    def _save_meta(self):
        page = bytearray(self.page_size)
        HEADER.pack_into(page, 0, b"RTM1", 0, 1, HEADER.size + META.size, -1, -1, 2)
        META.pack_into(page, HEADER.size, b"P2RTREE\0", 1, self.dimensions, self.page_size, self.root, self.size, self.height)
        self.file.write(0, page)

    def _read(self, page_id):
        page = self.file.read(page_id)
        magic, saved_id, count, free, _, _, kind = HEADER.unpack_from(page)
        if magic != b"RTN1" or saved_id != page_id or not 0 <= count <= self.capacity or free != HEADER.size + count * self.dtype.itemsize or kind not in (0, 1):
            raise ValueError("Página R-Tree corrupta")
        records = np.frombuffer(page, dtype=self.dtype, count=count, offset=HEADER.size)
        return _Node(page_id, bool(kind), records["low"].copy(), records["high"].copy(), records["ref"].copy())

    def _write(self, node):
        count = len(node.refs)
        if count > self.capacity:
            raise ValueError("Nodo desbordado sin dividir")
        page = bytearray(self.page_size)
        HEADER.pack_into(page, 0, b"RTN1", node.page_id, count, HEADER.size + count * self.dtype.itemsize, -1, -1, int(node.leaf))
        records = np.ndarray(count, dtype=self.dtype, buffer=page, offset=HEADER.size)
        records["low"], records["high"], records["ref"] = node.low, node.high, node.refs
        self.file.write(node.page_id, page)

    @staticmethod
    def _append(node, low, high, ref):
        node.low = np.concatenate((node.low, np.asarray(low, np.float32)[None, :]))
        node.high = np.concatenate((node.high, np.asarray(high, np.float32)[None, :]))
        node.refs = np.concatenate((node.refs, np.asarray(ref, np.int32)[None, :]))

    def _split(self, node):
        """Guttman quadratic: PickSeeds, PickNext y ocupación mínima del 40 %."""
        count = len(node.refs)
        areas = _areas(node.low, node.high)
        pair_low = np.minimum(node.low[:, None, :], node.low[None, :, :])
        pair_high = np.maximum(node.high[:, None, :], node.high[None, :, :])
        waste = _areas(pair_low, pair_high) - areas[:, None] - areas[None, :]
        waste[np.tril_indices(count)] = -np.inf
        first, second = np.unravel_index(np.argmax(waste), waste.shape)
        groups = [[int(first)], [int(second)]]
        lows = [node.low[first].copy(), node.low[second].copy()]
        highs = [node.high[first].copy(), node.high[second].copy()]
        remaining = [i for i in range(count) if i not in (first, second)]
        minimum = max(1, math.ceil(self.capacity * .4))
        while remaining:
            forced = next((g for g in (0, 1) if len(groups[g]) + len(remaining) == minimum), None)
            if forced is not None:
                groups[forced].extend(remaining)
                break
            choices = np.asarray(remaining)
            area = [_areas(lows[g], highs[g]) for g in (0, 1)]
            growth = [_areas(np.minimum(lows[g], node.low[choices]), np.maximum(highs[g], node.high[choices])) - area[g] for g in (0, 1)]
            position = int(np.argmax(np.abs(growth[0] - growth[1])))
            chosen = remaining.pop(position)
            group = min((0, 1), key=lambda g: (growth[g][position], area[g], len(groups[g]), g))
            groups[group].append(chosen)
            lows[group], highs[group] = np.minimum(lows[group], node.low[chosen]), np.maximum(highs[group], node.high[chosen])
        right_id = self.file.page_count
        left = _Node(node.page_id, node.leaf, node.low[groups[0]], node.high[groups[0]], node.refs[groups[0]])
        right = _Node(right_id, node.leaf, node.low[groups[1]], node.high[groups[1]], node.refs[groups[1]])
        self._write(left)
        self._write(right)
        return left, right

    def insert(self, point, rid):
        point, rid = canonical_point(point, self.dimensions), _rid(rid)
        node = self._read(self.root)
        path = []
        while not node.leaf:
            areas = _areas(node.low, node.high)
            growth = _areas(np.minimum(node.low, point), np.maximum(node.high, point)) - areas
            chosen = int(np.lexsort((np.arange(len(areas)), areas, growth))[0])
            path.append((node, chosen))
            node = self._read(int(node.refs[chosen, 0]))
        self._append(node, point, point, rid)
        while True:
            if len(node.refs) > self.capacity:
                node, sibling = self._split(node)
            else:
                self._write(node)
                sibling = None
            if not path:
                if sibling is not None:
                    low1, high1 = node.bounds()
                    low2, high2 = sibling.bounds()
                    root = _Node(self.file.page_count, False, np.asarray([low1, low2]), np.asarray([high1, high2]),
                                 np.asarray([(node.page_id, 0), (sibling.page_id, 0)], np.int32))
                    self._write(root)
                    self.root, self.height = root.page_id, self.height + 1
                break
            parent, child_position = path.pop()
            new_low, new_high = node.bounds()
            unchanged = np.array_equal(parent.low[child_position], new_low) and np.array_equal(parent.high[child_position], new_high)
            if sibling is None and unchanged:
                break
            parent.low[child_position], parent.high[child_position] = new_low, new_high
            if sibling is not None:
                self._append(parent, *sibling.bounds(), (sibling.page_id, 0))
            node = parent
        self.size += 1
        self._save_meta()

    def range_search(self, low, high):
        low, high = outward(low, self.dimensions, True), outward(high, self.dimensions, False)
        if np.any(low > high):
            raise ValueError("Límites de ventana invertidos")
        pending, result = [self.root], []
        while pending:
            node = self._read(pending.pop())
            selected = np.flatnonzero(np.all(node.high >= low, axis=1) & np.all(node.low <= high, axis=1))
            if node.leaf:
                result.extend((int(node.refs[i, 0]), int(node.refs[i, 1])) for i in selected)
            else:
                pending.extend(int(node.refs[i, 0]) for i in selected)
        return sorted(result)

    def radius_search(self, center, radius):
        center = canonical_point(center, self.dimensions).astype(np.float64)
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("El radio debe ser finito y no negativo")
        pending, result = [self.root], []
        limit = float(radius) ** 2
        while pending:
            node = self._read(pending.pop())
            distances = _mindist2(center, node.low, node.high)
            selected = np.flatnonzero(distances <= limit)
            if node.leaf:
                result.extend(((int(node.refs[i, 0]), int(node.refs[i, 1])), math.sqrt(float(distances[i]))) for i in selected)
            else:
                pending.extend(int(node.refs[i, 0]) for i in selected)
        return sorted(result, key=lambda row: (row[1], row[0]))

    def knn(self, center, k):
        center = canonical_point(center, self.dimensions).astype(np.float64)
        if not isinstance(k, int) or k < 1:
            raise ValueError("k debe ser un entero positivo")
        pending, nearest = [(0.0, self.root)], []
        while pending:
            bound, page_id = heapq.heappop(pending)
            if len(nearest) == k and bound > -nearest[0][0]:
                break
            node = self._read(page_id)
            distances = _mindist2(center, node.low, node.high)
            if node.leaf:
                for i, distance in enumerate(distances):
                    _offer(nearest, float(distance), node.refs[i], k)
            else:
                for i, distance in enumerate(distances):
                    if len(nearest) < k or distance <= -nearest[0][0]:
                        heapq.heappush(pending, (float(distance), int(node.refs[i, 0])))
        return _answers(nearest)

    def stats(self):
        return {"type": "RTREE", "size": self.size, "dimensions": self.dimensions,
                "height": self.height, "capacity": self.capacity, "fanout": self.capacity,
                "pages": self.file.page_count, "page_size": self.page_size, "root_page_id": self.root,
                "coordinate_type": "float32", "split": "quadratic"}

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class PointStore:
    """Baseline exacto: puntos paginados, scan de una página por vez, Top-K heap."""

    def __init__(self, path, dimensions=2, page_size=4096, counter=None, create=False):
        if not isinstance(dimensions, int) or not 2 <= dimensions <= 128:
            raise ValueError("dimensions debe estar entre 2 y 128")
        self.path, self.dimensions, self.page_size = Path(path), dimensions, page_size
        self.counter = counter if counter is not None else DiskCounter()
        self.dtype = np.dtype([("point", "<f4", (dimensions,)), ("ref", "<i4", (2,))])
        self.capacity = (page_size - HEADER.size) // self.dtype.itemsize
        if self.capacity < 1:
            raise ValueError("Punto mayor que página")
        if create and self.path.exists():
            raise FileExistsError(self.path)
        self.file = DiskFile(self.path, page_size, self.counter, role="data")
        if create:
            self.size = 0
            self._save_meta()
        else:
            try:
                page = self.file.read(0)
                magic, version, saved_d, saved_b, _, self.size, _ = META.unpack_from(page, HEADER.size)
                if (magic, version, saved_d, saved_b) != (b"P2POINTS", 1, dimensions, page_size):
                    raise ValueError("Archivo de puntos incompatible")
            except Exception:
                self.file.close()
                raise

    def _save_meta(self):
        page = bytearray(self.page_size)
        HEADER.pack_into(page, 0, b"PTM1", 0, 1, HEADER.size + META.size, -1, -1, 2)
        META.pack_into(page, HEADER.size, b"P2POINTS", 1, self.dimensions, self.page_size, 0, self.size, 1)
        self.file.write(0, page)

    def insert(self, point, rid):
        point, rid = canonical_point(point, self.dimensions), _rid(rid)
        page_id, slot = self.size // self.capacity + 1, self.size % self.capacity
        page = self.file.read(page_id) if slot else bytearray(self.page_size)
        record = np.ndarray(1, dtype=self.dtype, buffer=page, offset=HEADER.size + slot * self.dtype.itemsize)
        record["point"], record["ref"] = point, rid
        HEADER.pack_into(page, 0, b"PTN1", page_id, slot + 1, HEADER.size + (slot + 1) * self.dtype.itemsize, -1, -1, 1)
        self.file.write(page_id, page)
        self.size += 1
        self._save_meta()

    def _pages(self):
        for page_id in range(1, self.file.page_count):
            page = self.file.read(page_id)
            magic, saved_id, count, free, _, _, _ = HEADER.unpack_from(page)
            if magic != b"PTN1" or saved_id != page_id or not 0 < count <= self.capacity or free != HEADER.size + count * self.dtype.itemsize:
                raise ValueError("Página de puntos corrupta")
            yield np.frombuffer(page, dtype=self.dtype, count=count, offset=HEADER.size)

    def knn(self, center, k):
        center = canonical_point(center, self.dimensions).astype(np.float64)
        if not isinstance(k, int) or k < 1:
            raise ValueError("k debe ser positivo")
        nearest = []
        for records in self._pages():
            delta = records["point"].astype(np.float64) - center
            distances = np.einsum("ij,ij->i", delta, delta)
            for i, distance in enumerate(distances):
                _offer(nearest, float(distance), records["ref"][i], k)
        return _answers(nearest)

    def radius_search(self, center, radius):
        center = canonical_point(center, self.dimensions).astype(np.float64)
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("El radio debe ser finito y no negativo")
        result = []
        for records in self._pages():
            delta = records["point"].astype(np.float64) - center
            distances = np.einsum("ij,ij->i", delta, delta)
            result.extend(((int(records["ref"][i, 0]), int(records["ref"][i, 1])), math.sqrt(float(distances[i])))
                          for i in np.flatnonzero(distances <= float(radius) ** 2))
        return sorted(result, key=lambda row: (row[1], row[0]))

    def range_search(self, low, high):
        low, high = outward(low, self.dimensions, True), outward(high, self.dimensions, False)
        if np.any(low > high):
            raise ValueError("Límites de ventana invertidos")
        result = []
        for records in self._pages():
            selected = np.flatnonzero(np.all(records["point"] >= low, axis=1) & np.all(records["point"] <= high, axis=1))
            result.extend((int(records["ref"][i, 0]), int(records["ref"][i, 1])) for i in selected)
        return sorted(result)

    def stats(self):
        return {"type": "POINT_SCAN", "size": self.size, "dimensions": self.dimensions,
                "capacity": self.capacity, "pages": self.file.page_count, "page_size": self.page_size}

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
