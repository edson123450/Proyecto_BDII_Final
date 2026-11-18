"""Inferencia Cnn6 de PANNs, adaptada del código MIT de Qiuqiang Kong.

Referencia: https://github.com/qiuqiangkong/audioset_tagging_cnn
Licencia completa: data/panns_LICENSE.txt. No se entrena una red nueva.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

MODEL_URL = 'https://zenodo.org/api/records/3987831/files/Cnn6_mAP=0.343.pth/content'
MODEL_BYTES = 23698321
MODEL_MD5 = 'e25e26b84585b14c7754c91e48efc9be'
MODEL_NAME = 'Cnn6_mAP=0.343.pth'
MODEL_VERSION = 'panns-cnn6-audioset-512-v1'


def _md5(path):
    """Esta función verifica el checkpoint sin cargar el archivo completo en RAM."""
    digest = hashlib.md5()
    with open(path, 'rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_checkpoint(cache_dir='data/cache'):
    """Esta función descarga únicamente el checkpoint oficial y comprueba su MD5."""
    path = Path(cache_dir) / MODEL_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _md5(path) != MODEL_MD5:
            raise ValueError(f'El checkpoint no coincide con el MD5 oficial: {path}')
        return path
    temporary = path.with_suffix('.downloading')
    try:
        request = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'ProyectoBDII/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response, open(temporary, 'wb') as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        if temporary.stat().st_size != MODEL_BYTES or _md5(temporary) != MODEL_MD5:
            raise ValueError('El checkpoint descargado no coincide con el MD5 oficial')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, value):
        return F.avg_pool2d(F.relu_(self.bn1(self.conv1(value))), (2, 2))


class Cnn6(nn.Module):
    """Arquitectura y nombres de parámetros compatibles con el checkpoint oficial."""

    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(n_fft=1024, hop_length=320, win_length=1024,
                                                 window='hann', center=True, pad_mode='reflect',
                                                 freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=32000, n_fft=1024, n_mels=64, fmin=50,
                                                 fmax=14000, ref=1.0, amin=1e-10, top_db=None,
                                                 freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock5x5(1, 64)
        self.conv_block2 = ConvBlock5x5(64, 128)
        self.conv_block3 = ConvBlock5x5(128, 256)
        self.conv_block4 = ConvBlock5x5(256, 512)
        self.fc1 = nn.Linear(512, 512)
        self.fc_audioset = nn.Linear(512, 527)

    def forward(self, waveform):
        value = self.logmel_extractor(self.spectrogram_extractor(waveform))
        value = self.bn0(value.transpose(1, 3)).transpose(1, 3)
        for block in (self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4):
            value = block(value)
        value = value.mean(dim=3)
        value = value.max(dim=2).values + value.mean(dim=2)
        return F.relu_(self.fc1(value))


class PANNsEmbedder:
    """Inferencia en lotes limitados; devuelve el embedding de 512D, no etiquetas."""

    def __init__(self, cache_dir='data/cache', device='cpu', batch_size=2, threads=2):
        if batch_size < 1:
            raise ValueError('batch_size debe ser positivo')
        if threads:
            torch.set_num_threads(threads)
        if device == 'auto':
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device(device)
        self.batch_size = batch_size
        checkpoint = torch.load(ensure_checkpoint(cache_dir), map_location='cpu', weights_only=True)
        self.network = Cnn6()
        self.network.load_state_dict(checkpoint['model'], strict=True)
        self.network.eval().to(self.device)
        del checkpoint

    def embed(self, segments):
        """Esta función procesa segmentos mono de 32 kHz y normaliza sus vectores."""
        segments = [np.asarray(segment, dtype=np.float32) for segment in segments]
        if not segments:
            return np.empty((0, 512), np.float32)
        output = []
        with torch.inference_mode():
            for start in range(0, len(segments), self.batch_size):
                batch = segments[start:start + self.batch_size]
                lengths = {len(segment) for segment in batch}
                if len(lengths) != 1:
                    # No mezclar longitudes mediante padding: alteraría el max/mean temporal.
                    output.extend(self.embed([segment])[0] for segment in batch)
                    continue
                value = torch.from_numpy(np.stack(batch)).to(self.device)
                vectors = F.normalize(self.network(value), dim=1)
                output.extend(vectors.cpu().numpy())
        return np.asarray(output, dtype=np.float32)
