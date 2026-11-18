"""DTW propio y LB_Keogh multivariado sobre ventanas de longitud fija.

La garantía se refiere a todas las ventanas de MFCC13 de 200 ms del corpus.
Coste: suma de distancias euclidianas al cuadrado, sin dividir por longitud
variable del camino. La banda y la normalización son iguales en filtro/refino.
"""
from __future__ import annotations

import heapq
import math
from time import perf_counter

import numpy as np
from numba import njit

STEP_SECONDS = .2


def _sequence(value):
    value = np.ascontiguousarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
        raise ValueError('La secuencia debe ser una matriz finita no vacía')
    return value


def band_radius(length, radius=None):
    if radius is None:
        return max(2, int(.1 * length))
    if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError('El radio DTW debe ser un entero no negativo')
    return int(radius)


@njit(cache=True)
def _dtw_cost(query, candidate, radius, cutoff):
    length, dimensions = query.shape
    previous = np.full(length + 1, np.inf, dtype=np.float64)
    current = np.full(length + 1, np.inf, dtype=np.float64)
    previous[0] = 0.
    for i in range(1, length + 1):
        current[:] = np.inf
        row_minimum = np.inf
        for j in range(max(1, i - radius), min(length, i + radius) + 1):
            local = 0.
            for dimension in range(dimensions):
                difference = query[i - 1, dimension] - candidate[j - 1, dimension]
                local += difference * difference
            value = local + min(previous[j], current[j - 1], previous[j - 1])
            current[j] = value
            row_minimum = min(row_minimum, value)
        if row_minimum > cutoff + 1e-10 * (1. + abs(cutoff)):
            return np.inf
        previous, current = current, previous
    return previous[length]


@njit(cache=True)
def _envelope(query, radius):
    length, dimensions = query.shape
    lower = np.empty_like(query)
    upper = np.empty_like(query)
    for i in range(length):
        for dimension in range(dimensions):
            lo, hi = np.inf, -np.inf
            for j in range(max(0, i - radius), min(length, i + radius + 1)):
                lo = min(lo, query[j, dimension])
                hi = max(hi, query[j, dimension])
            lower[i, dimension], upper[i, dimension] = lo, hi
    return lower, upper


@njit(cache=True)
def _lower_bound(candidate, lower, upper, cutoff):
    cost = 0.
    for i in range(len(candidate)):
        for dimension in range(candidate.shape[1]):
            value = candidate[i, dimension]
            delta = max(lower[i, dimension] - value, 0., value - upper[i, dimension])
            cost += delta * delta
        if cost > cutoff + 1e-10 * (1. + abs(cutoff)):
            return cost
    return cost


@njit(cache=True)
def _best_window(query, track, lower, upper, radius, cutoff, use_lb):
    best, best_start = np.inf, -1
    windows = max(0, len(track) - len(query) + 1)
    pruned, refined, abandoned = 0, 0, 0
    for start in range(windows):
        window = track[start:start + len(query)]
        threshold = min(cutoff, best)
        tolerance = 1e-10 * (1. + abs(threshold))
        if use_lb and _lower_bound(window, lower, upper, threshold) > threshold + tolerance:
            pruned += 1
            continue
        refined += 1
        cost = _dtw_cost(query, window, radius, threshold)
        if np.isinf(cost):
            abandoned += 1
        if cost < best:
            best, best_start = cost, start
    return best, best_start, windows, pruned, refined, abandoned


def dtw_distance(query, candidate, radius=None, cutoff=math.inf):
    """Esta función devuelve el coste DTW cuadrático con dos filas de memoria."""
    query, candidate = _sequence(query), _sequence(candidate)
    if query.shape != candidate.shape:
        raise ValueError('DTW de este motor requiere ventanas de la misma forma')
    if cutoff < 0 or math.isnan(cutoff):
        raise ValueError('El umbral debe ser no negativo')
    return float(_dtw_cost(query, candidate, band_radius(len(query), radius), float(cutoff)))


def lb_keogh(query, candidate, radius=None):
    """Esta función calcula una cota inferior con la envolvente de la consulta."""
    query, candidate = _sequence(query), _sequence(candidate)
    if query.shape != candidate.shape:
        raise ValueError('LB_Keogh requiere ventanas de la misma forma')
    lower, upper = _envelope(query, band_radius(len(query), radius))
    return float(_lower_bound(candidate, lower, upper, math.inf))


def best_subsequence(query, track, radius=None, cutoff=math.inf, use_lb=True):
    """Esta función examina cada inicio discreto de una pista, sin saltar ventanas."""
    query, track = _sequence(query), _sequence(track)
    if query.shape[1] != track.shape[1]:
        raise ValueError('Las secuencias deben tener igual dimensión')
    if cutoff < 0 or math.isnan(cutoff):
        raise ValueError('El umbral debe ser no negativo')
    radius = band_radius(len(query), radius)
    lower, upper = _envelope(query, radius)
    cost, start, windows, pruned, refined, abandoned = _best_window(query, track, lower, upper, radius, float(cutoff), use_lb)
    return dict(cost=float(cost), start_frame=int(start), start_seconds=float(start * STEP_SECONDS) if start >= 0 else None,
                windows=int(windows), pruned=int(pruned), refined=int(refined), abandoned=int(abandoned))


def search_tracks(query, tracks, k=5, radius=None, seed_tracks=(), use_lb=True):
    """Esta función recorre todo el corpus; los candidatos ANN solo siembran el heap.

    tracks y seed_tracks producen (track_id entero, secuencia normalizada).
    Nunca se omiten pistas por pertenencia a una lista aproximada.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError('k debe ser positivo')
    query = _sequence(query)
    radius = band_radius(len(query), radius)
    lower, upper = _envelope(query, radius)
    start = perf_counter()
    heap, visited = [], set()
    stats = dict(tracks=0, windows=0, pruned=0, refined=0, abandoned=0,
                 exact_universe='all_equal_length_windows_step_200ms', radius=radius)
    for source in (seed_tracks, tracks):
        for track_id, track in source:
            track_id = int(track_id)
            if track_id in visited:
                continue
            visited.add(track_id)
            values = _sequence(track)
            if values.shape[1] != query.shape[1]:
                raise ValueError('Las secuencias deben tener igual dimensión')
            cutoff = -heap[0][0] if len(heap) == k else math.inf
            cost, frame, windows, pruned, refined, abandoned = _best_window(query, values, lower, upper, radius, cutoff, use_lb)
            stats['tracks'] += 1
            for name, amount in zip(('windows', 'pruned', 'refined', 'abandoned'), (windows, pruned, refined, abandoned)):
                stats[name] += int(amount)
            if not math.isfinite(cost):
                continue
            entry = (-float(cost), -track_id, -int(frame))
            if len(heap) < k:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
    results = [dict(track_id=-track_id, cost=-cost, start_frame=-frame,
                    start_seconds=-frame * STEP_SECONDS) for cost, track_id, frame in heap]
    results.sort(key=lambda item: (item['cost'], item['track_id'], item['start_frame']))
    stats['elapsed_ms'] = (perf_counter() - start) * 1000
    return results, stats


def warmup():
    """Esta función separa la compilación inicial de las mediciones de consultas."""
    values = np.zeros((4, 2), np.float64)
    best_subsequence(values, values)
