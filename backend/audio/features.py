"""Frontends reproducibles de audio y vocabulario BoAW entrenado solo con training."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

FEATURE_VERSION = 'mfcc20-no-c0-16k-512-320-pool10-panns512-v1'
SAMPLE_RATE = 16000
HOP_LENGTH = 320
SEQUENCE_STEP_SECONDS = .2


def load_audio(path, max_seconds=30):
    """Esta función decodifica una pista acotada sin guardar WAV intermedios."""
    import soundfile as sf
    with sf.SoundFile(path) as source:
        sample_rate = source.samplerate
        frame_limit = -1 if max_seconds is None else int(max_seconds * sample_rate)
        waveform = source.read(frames=frame_limit, dtype='float32', always_2d=True).mean(axis=1)
    if waveform.size < max(1, sample_rate // 2) or not np.isfinite(waveform).all():
        raise ValueError('El audio debe contener al menos 0.5 segundos finitos')
    return np.ascontiguousarray(waveform, dtype=np.float32), sample_rate


def classical_features(waveform, sample_rate):
    """Esta función obtiene MFCC20 y una secuencia MFCC13 cada 200 ms."""
    import librosa
    waveform = np.asarray(waveform, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    if len(waveform) < 512:
        waveform = np.pad(waveform, (0, 512 - len(waveform)))
    descriptors = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=21,
                                       n_fft=512, hop_length=HOP_LENGTH, n_mels=64,
                                       fmin=50, fmax=8000, center=False)[1:].T
    descriptors = np.ascontiguousarray(descriptors, dtype=np.float32)
    sequence = np.stack([descriptors[start:start + 10, :13].mean(axis=0)
                         for start in range(0, len(descriptors), 10)]).astype(np.float32)
    return descriptors, sequence


def segment_audio(waveform, sample_rate, seconds=10, stride=5):
    """Esta función produce cinco ventanas de 10 s para una pista de 30 s."""
    length = min(len(waveform), int(seconds * sample_rate))
    if length < 1:
        raise ValueError('Audio vacío')
    step = int(stride * sample_rate)
    starts = list(range(0, len(waveform) - length + 1, step))
    last = len(waveform) - length
    if last - starts[-1] > sample_rate // 4:
        starts.append(last)
    return [np.ascontiguousarray(waveform[start:start + length]) for start in starts], np.asarray(starts, dtype=np.float32) / sample_rate


class AudioExtractor:
    def __init__(self, cache_dir='data/cache', device='cpu', batch_size=2, model=None, threads=2):
        if model is None:
            from .model import PANNsEmbedder
            model = PANNsEmbedder(cache_dir, device=device, batch_size=batch_size, threads=threads)
        self.model = model

    def extract_waveform(self, waveform, sample_rate):
        """Esta función aplica el mismo frontend a pistas y consultas recortadas."""
        import librosa
        start = perf_counter()
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError('Se requiere un audio mono finito')
        duration = len(waveform) / sample_rate
        descriptors, sequence = classical_features(waveform, sample_rate)
        mfcc_seconds = perf_counter() - start
        deep_start = perf_counter()
        audio_32k = waveform if sample_rate == 32000 else librosa.resample(waveform, orig_sr=sample_rate, target_sr=32000)
        segments, starts = segment_audio(audio_32k, 32000)
        segment_embeddings = self.model.embed(segments)
        embedding = segment_embeddings.mean(axis=0)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return dict(mfcc_descriptors=descriptors, sequence=sequence,
                    embedding=embedding.astype(np.float32), segment_embeddings=segment_embeddings,
                    segment_starts=starts, duration=float(duration), sample_rate=int(sample_rate),
                    feature_version=FEATURE_VERSION,
                    timings=dict(mfcc_ms=mfcc_seconds * 1000,
                                 embedding_ms=(perf_counter() - deep_start) * 1000,
                                 extraction_ms=(perf_counter() - start) * 1000))

    def extract(self, path, max_seconds=30):
        start = perf_counter()
        waveform, sample_rate = load_audio(path, max_seconds)
        decode_ms = (perf_counter() - start) * 1000
        result = self.extract_waveform(waveform, sample_rate)
        result['timings']['decode_ms'] = decode_ms
        result['timings']['total_ms'] = (perf_counter() - start) * 1000
        return result


def extract_audio(path, model=None, max_seconds=30, cache_dir='data/cache', device='cpu'):
    """Esta función expone la extracción simple; reutilizar AudioExtractor en lotes."""
    return AudioExtractor(cache_dir=cache_dir, device=device, model=model).extract(path, max_seconds)


class AudioCodebook:
    """Vocabulario acústico: normalización aprendida y K-Means de MFCC20."""

    def __init__(self, mean, std, centers, training_ids=(), seed=2026):
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        self.centers = np.asarray(centers, np.float32)
        self.training_ids = tuple(int(value) for value in training_ids)
        self.seed = int(seed)
        if self.mean.shape != (20,) or self.std.shape != (20,) or self.centers.ndim != 2 or self.centers.shape[1] != 20:
            raise ValueError('Dimensiones inválidas del vocabulario MFCC20')
        if np.any(self.std <= 0) or not all(np.isfinite(a).all() for a in (self.mean, self.std, self.centers)):
            raise ValueError('Vocabulario inválido')

    @classmethod
    def fit(cls, descriptors, words=256, training_ids=(), seed=2026):
        """Esta función ajusta K-Means sobre descriptores procedentes de training."""
        from sklearn.cluster import MiniBatchKMeans
        from threadpoolctl import threadpool_limits
        values = np.asarray(descriptors, np.float32)
        if values.ndim != 2 or values.shape[1] != 20 or len(values) < words:
            raise ValueError('No hay suficientes descriptores MFCC20 para el vocabulario')
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(values.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
        normalized = (values - mean) / std
        estimator = MiniBatchKMeans(n_clusters=words, batch_size=2048, random_state=seed,
                                    n_init=3, max_iter=100, reassignment_ratio=.01)
        with threadpool_limits(limits=2):
            estimator.fit(normalized)
        return cls(mean, std, estimator.cluster_centers_, training_ids, seed)

    def counts(self, descriptors):
        """Esta función cuantiza por lotes sin expandir el histograma en tokens."""
        values = np.asarray(descriptors, np.float32)
        if values.ndim != 2 or values.shape[1] != 20:
            raise ValueError('Se requieren descriptores MFCC20')
        frequencies = np.zeros(len(self.centers), dtype=np.int64)
        center_norm = np.einsum('ij,ij->i', self.centers, self.centers)
        for start in range(0, len(values), 1024):
            batch = (values[start:start + 1024] - self.mean) / self.std
            distances = np.einsum('ij,ij->i', batch, batch)[:, None] + center_norm - 2 * batch @ self.centers.T
            frequencies += np.bincount(distances.argmin(axis=1), minlength=len(self.centers))
        return Counter({f'aw{word:03d}': int(count) for word, count in enumerate(frequencies) if count})

    def normalize_sequence(self, sequence):
        """Esta función aplica a DTW la normalización fija aprendida en training."""
        values = np.asarray(sequence, np.float32)
        if values.ndim != 2 or values.shape[1] != 13:
            raise ValueError('Se requiere una secuencia MFCC13')
        return np.ascontiguousarray((values - self.mean[:13]) / self.std[:13], dtype=np.float32)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as handle:
            np.savez_compressed(handle, mean=self.mean, std=self.std, centers=self.centers,
                                training_ids=np.asarray(self.training_ids, np.int64), seed=self.seed,
                                feature_version=FEATURE_VERSION)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data['feature_version']) != FEATURE_VERSION:
                raise ValueError('Versión incompatible del frontend acústico')
            return cls(data['mean'], data['std'], data['centers'], data['training_ids'], data['seed'])
