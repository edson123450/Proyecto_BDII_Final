"""Rutas locales configurables; los datos y modelos no se publican en Git."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env():
    """Esta función lee rutas de .env sin ejecutar instrucciones del archivo."""
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()
DATA_DIR = Path(os.environ.get('MINIDB_DATA_DIR', ROOT / 'data/runtime'))
AUDIO_DIR = Path(os.environ.get('FMA_AUDIO_DIR', ROOT / 'data/fma_small'))
METADATA_DIR = Path(os.environ.get('FMA_METADATA_DIR', ROOT / 'data/fma_metadata'))
AUDIO_CACHE = ROOT / 'data/cache/audio'
