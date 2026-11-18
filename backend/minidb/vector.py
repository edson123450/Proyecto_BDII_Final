"""Vectores paginados, búsqueda exacta e IVF implementado sobre archivos propios."""

from __future__ import annotations

import heapq
import math
import struct
from pathlib import Path

import numpy as np

from .io import DiskFile, DiskCounter


HEADER = struct.Struct('<4sIIIIqq')
META = struct.Struct('<8sIIQII')
RID = struct.Struct('<qi')
CENTROID_PREFIX = struct.Struct('<Iqq')
MAGIC = b'VEC1'


def page_header(page, pid, count=0, free=None, dimension=0, next_page=-1, previous=-1):
    HEADER.pack_into(page, 0, MAGIC, pid, count, HEADER.size if free is None else free,
                     dimension, next_page, previous)
    return page


def vector_array(vector, dimension, metric):
    vector = np.asarray(vector, dtype='<f4')
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError(f'El vector debe tener {dimension} componentes finitos.')
    if metric == 'COSINE':
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError('La similitud de coseno requiere un vector no nulo.')
        vector = vector / norm
    return vector.astype('<f4', copy=False)


def distance(query, vector, metric):
    # La acumulación float64 evita diferencias por la alineación del slot en disco.
    q64 = query.astype(np.float64)
    v64 = vector.astype(np.float64)
    if metric == 'COSINE':
        return max(0.0, min(2.0, 1.0 - float(np.sum(q64 * v64, dtype=np.float64))))
    delta = q64 - v64
    return float(np.sqrt(np.sum(delta * delta, dtype=np.float64)))


def add_top(heap, rid, dist, k):
    entry = (-dist, -int(rid[0]), -int(rid[1]))
    if len(heap) < k:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def sorted_top(heap):
    return sorted([((int(-p), int(-s)), float(-d)) for d, p, s in heap], key=lambda item: (item[1], item[0]))


class VectorStore:
    """Este archivo conserva cada vector junto al RID de su registro de datos."""

    def __init__(self, path, dimension=512, metric='COSINE', page_size=4096, counter=None, create=False):
        self.path = Path(path)
        if create and self.path.exists():
            self.path.unlink()
        self.counter = counter if counter is not None else DiskCounter()
        self.file = DiskFile(self.path, page_size, self.counter, role='data')
        self.page_size = page_size
        self.dimension, self.metric = dimension, metric.upper()
        if self.metric not in ('COSINE', 'L2') or not 1 <= dimension <= 768:
            raise ValueError('Dimensión o métrica vectorial no válida.')
        self.count = 0
        if self.file.page_count:
            page = self.file.read(0)
            magic, self.dimension, stored_page, self.count, kind, _ = META.unpack_from(page, HEADER.size)
            if magic != b'VECTORS1' or stored_page != page_size:
                raise ValueError('Archivo vectorial incompatible.')
            self.metric = 'COSINE' if kind else 'L2'
        else:
            self._save_meta()
        self.entry_size = RID.size + self.dimension * 4
        self.capacity = (self.page_size - HEADER.size) // self.entry_size
        if self.capacity < 1:
            raise ValueError('El vector no cabe en una página.')

    def _save_meta(self):
        page = page_header(bytearray(self.page_size), 0, 1, HEADER.size + META.size, self.dimension)
        META.pack_into(page, HEADER.size, b'VECTORS1', self.dimension, self.page_size, self.count,
                       self.metric == 'COSINE', 0)
        self.file.write(0, page)

    def append(self, rid, vector):
        vector = vector_array(vector, self.dimension, self.metric)
        pid = 1 + self.count // self.capacity
        slot = self.count % self.capacity
        if slot == 0:
            page = page_header(bytearray(self.page_size), pid, dimension=self.dimension, previous=pid-1 if pid > 1 else -1)
        else:
            page = self.file.read(pid)
        offset = HEADER.size + slot * self.entry_size
        RID.pack_into(page, offset, int(rid[0]), int(rid[1]))
        page[offset + RID.size:offset + self.entry_size] = vector.tobytes()
        page_header(page, pid, slot + 1, offset + self.entry_size, self.dimension, previous=pid-1 if pid > 1 else -1)
        self.file.write(pid, page)
        self.count += 1
        self._save_meta()

    def iter_vectors(self):
        for pid in range(1, self.file.page_count):
            page = self.file.read(pid)
            magic, actual, count, _, dimension, _, _ = HEADER.unpack_from(page)
            if magic != MAGIC or actual != pid or dimension != self.dimension or count > self.capacity:
                raise ValueError('Página vectorial inválida.')
            for slot in range(count):
                offset = HEADER.size + slot * self.entry_size
                rid = RID.unpack_from(page, offset)
                vector = np.frombuffer(page, dtype='<f4', count=self.dimension, offset=offset + RID.size)
                yield rid, vector

    def knn(self, query, k=10, allowed_rids=None, exclude_rids=None):
        if k < 1:
            raise ValueError('k debe ser positivo.')
        query = vector_array(query, self.dimension, self.metric)
        heap = []
        excluded = exclude_rids or set()
        for rid, vector in self.iter_vectors():
            if rid in excluded or (allowed_rids is not None and rid not in allowed_rids):
                continue
            add_top(heap, rid, distance(query, vector, self.metric), k)
        return sorted_top(heap)

    def range_search(self, query, radius, allowed_rids=None):
        if radius < 0 or not math.isfinite(radius):
            raise ValueError('El radio debe ser finito y no negativo.')
        query = vector_array(query, self.dimension, self.metric)
        result = []
        for rid, vector in self.iter_vectors():
            if allowed_rids is not None and rid not in allowed_rids:
                continue
            dist = distance(query, vector, self.metric)
            if dist <= radius:
                result.append((rid, dist))
        return sorted(result, key=lambda item:(item[1], item[0]))

    def close(self):
        self.file.close()


class IVFIndex:
    """Este índice explora únicamente las listas de los centroides seleccionados."""

    def __init__(self, directory, counter=None, page_size=4096):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.counter = counter if counter is not None else DiskCounter()
        self.page_size = page_size
        self.centroid_path = self.directory / 'centroids.dir'
        self.list_path = self.directory / 'lists.ivf'
        self.centroid_file = None
        self.lists = None
        self.dimension = self.count = self.clusters = 0
        self.metric = 'COSINE'
        if self.centroid_path.exists() and self.centroid_path.stat().st_size:
            self._open()

    def _open(self):
        self.centroid_file = DiskFile(self.centroid_path, self.page_size, self.counter, role='index')
        self.lists = DiskFile(self.list_path, self.page_size, self.counter, role='index')
        page = self.centroid_file.read(0)
        magic, self.dimension, stored_page, self.count, kind, self.clusters = META.unpack_from(page, HEADER.size)
        if magic != b'IVFFLAT1' or stored_page != self.page_size:
            raise ValueError('Formato IVF incompatible.')
        self.metric = 'COSINE' if kind else 'L2'
        self.entry_size = RID.size + self.dimension * 4
        self.capacity = (self.page_size - HEADER.size) // self.entry_size

    def build(self, store, clusters=None, seed=42):
        from sklearn.cluster import MiniBatchKMeans
        from threadpoolctl import threadpool_limits
        if not store.count:
            raise ValueError('No se puede construir IVF sin vectores.')
        self.close()
        self.dimension, self.metric, self.count = store.dimension, store.metric, store.count
        self.clusters = max(1, int(math.sqrt(self.count))) if clusters is None else int(clusters)
        if not 1 <= self.clusters <= self.count:
            raise ValueError('La cantidad de centroides debe estar entre 1 y N.')
        self.entry_size = RID.size + self.dimension * 4
        self.capacity = (self.page_size - HEADER.size) // self.entry_size
        sample_size = min(self.count, 20000)
        rng = np.random.default_rng(seed)
        sample = np.empty((sample_size, self.dimension), dtype=np.float32)
        for i, (_, vector) in enumerate(store.iter_vectors()):
            if i < sample_size:
                sample[i] = vector
            else:
                selected = rng.integers(i + 1)
                if selected < sample_size:
                    sample[selected] = vector
        with threadpool_limits(limits=2):
            model = MiniBatchKMeans(n_clusters=self.clusters, random_state=seed, n_init=3,
                                    batch_size=min(1024, sample_size), max_iter=100,
                                    reassignment_ratio=0.01).fit(sample)
        centroids = model.cluster_centers_.astype('<f4')
        for path in (self.centroid_path, self.list_path):
            if path.exists():
                path.unlink()
        self.centroid_file = DiskFile(self.centroid_path, self.page_size, self.counter, role='index')
        self.lists = DiskFile(self.list_path, self.page_size, self.counter, role='index')
        meta = page_header(bytearray(self.page_size), 0, 1, HEADER.size + META.size, self.dimension)
        META.pack_into(meta, HEADER.size, b'IVFFLAT1', self.dimension, self.page_size, self.count,
                       self.metric == 'COSINE', self.clusters)
        self.centroid_file.allocate(meta)
        self.lists.allocate(meta)
        heads = [-1] * self.clusters
        tails = [-1] * self.clusters
        counts = [0] * self.clusters
        for rid, vector in store.iter_vectors():
            delta = centroids - vector
            cluster = int(np.argmin(np.einsum('ij,ij->i', delta, delta)))
            slot = counts[cluster] % self.capacity
            previous = tails[cluster]
            if slot == 0:
                pid = self.lists.page_count
                page = page_header(bytearray(self.page_size), pid, dimension=self.dimension, previous=previous)
                if previous >= 0:
                    last_page = self.lists.read(previous)
                    header = list(HEADER.unpack_from(last_page))
                    header[5] = pid
                    HEADER.pack_into(last_page, 0, *header)
                    self.lists.write(previous, last_page)
                else:
                    heads[cluster] = pid
                tails[cluster] = pid
            else:
                pid = tails[cluster]
                page = self.lists.read(pid)
                previous = HEADER.unpack_from(page)[6]
            offset = HEADER.size + slot * self.entry_size
            RID.pack_into(page, offset, *rid)
            page[offset + RID.size:offset + self.entry_size] = vector.tobytes()
            page_header(page, pid, slot + 1, offset + self.entry_size, self.dimension, previous=previous)
            self.lists.write(pid, page)
            counts[cluster] += 1
        for cluster, center in enumerate(centroids):
            page = page_header(bytearray(self.page_size), cluster + 1, 1,
                               HEADER.size + CENTROID_PREFIX.size + center.nbytes, self.dimension)
            CENTROID_PREFIX.pack_into(page, HEADER.size, cluster, heads[cluster], counts[cluster])
            page[HEADER.size + CENTROID_PREFIX.size:HEADER.size + CENTROID_PREFIX.size + center.nbytes] = center.tobytes()
            self.centroid_file.allocate(page)
        return self.stats()

    def _centroids(self):
        for cluster in range(self.clusters):
            page = self.centroid_file.read(cluster + 1)
            cid, head, count = CENTROID_PREFIX.unpack_from(page, HEADER.size)
            center = np.frombuffer(page, dtype='<f4', count=self.dimension, offset=HEADER.size + CENTROID_PREFIX.size)
            yield cid, head, count, center

    def knn(self, query, k=10, probes=5, allowed_rids=None, exclude_rids=None):
        if k < 1 or probes < 1:
            raise ValueError('k y probes deben ser positivos.')
        query = vector_array(query, self.dimension, self.metric)
        routes = []
        for cid, head, count, center in self._centroids():
            delta = query - center
            routes.append((float(np.dot(delta, delta)), cid, head))
        routes.sort()
        heap, excluded = [], exclude_rids or set()
        for _, _, pid in routes[:min(probes, self.clusters)]:
            while pid >= 0:
                page = self.lists.read(pid)
                magic, actual, count, _, dimension, next_page, _ = HEADER.unpack_from(page)
                if magic != MAGIC or actual != pid or dimension != self.dimension or count > self.capacity:
                    raise ValueError('Lista IVF inválida.')
                for slot in range(count):
                    offset = HEADER.size + slot * self.entry_size
                    rid = RID.unpack_from(page, offset)
                    if rid in excluded or (allowed_rids is not None and rid not in allowed_rids):
                        continue
                    vector = np.frombuffer(page, dtype='<f4', count=self.dimension, offset=offset + RID.size)
                    add_top(heap, rid, distance(query, vector, self.metric), k)
                pid = next_page
        return sorted_top(heap)

    def stats(self):
        return dict(vectors=self.count, dimension=self.dimension, clusters=self.clusters,
                    metric=self.metric, page_size=self.page_size, entries_per_page=self.capacity,
                    index_bytes=sum(p.stat().st_size for p in (self.centroid_path, self.list_path) if p.exists()))

    def close(self):
        if self.centroid_file is not None:
            self.centroid_file.close()
            self.centroid_file = None
        if self.lists is not None:
            self.lists.close()
            self.lists = None
