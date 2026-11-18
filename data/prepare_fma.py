"""Descarga oficial verificable; conserva las pistas fuera de Git."""
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.config import AUDIO_DIR,METADATA_DIR

ARCHIVES={
    'fma_small':('ade154f733639d52e35e32f5593efe5be76c6d70',18*1024**3),
    'fma_metadata':('f0df49ffe5f2a6008d7dc83c6915b31835dfe733',2*1024**3),
}


def download(name,destination):
    """Esta función valida el SHA-1 oficial antes de extraer dentro de la carpeta de datos."""
    destination=Path(destination).resolve()
    default=(ROOT/'data'/name).resolve()
    if destination!=default:
        raise ValueError(f'La ruta configurada para {name} no contiene el dataset. Revisar .env.')
    expected,required=ARCHIVES[name]
    if shutil.disk_usage(ROOT).free<required:
        raise ValueError(f'Se requieren {required/1024**3:.0f} GB libres para descargar y extraer {name}. También se puede usar una copia existente mediante .env.')
    archive=ROOT/'data/cache'/(name+'.zip')
    archive.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://os.unil.cloud.switch.ch/fma/{name}.zip'
    print(f'Descargando {name} del sitio oficial...',flush=True)
    request=urllib.request.Request(url,headers={'User-Agent':'SonoraDB/1.0 (academic project)'})
    digest=hashlib.sha1()
    temporary=archive.with_suffix('.download')
    try:
        with urllib.request.urlopen(request,timeout=120) as response, temporary.open('wb') as target:
            while chunk:=response.read(1024*1024):
                digest.update(chunk);target.write(chunk)
        if digest.hexdigest()!=expected:
            raise ValueError(f'Checksum inválido para {name}; no se extraerá el archivo.')
        temporary.replace(archive)
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                relative=Path(item.filename)
                if not relative.parts or relative.parts[0]!=name:
                    continue
                output=(ROOT/'data'/relative).resolve()
                if not output.is_relative_to(destination):
                    raise ValueError('El ZIP contiene una ruta fuera de su carpeta.')
                source.extract(item,ROOT/'data')
    finally:
        temporary.unlink(missing_ok=True)
    archive.unlink()


def ensure_data():
    if not (METADATA_DIR/'tracks.csv').is_file():
        download('fma_metadata',METADATA_DIR)
    if not AUDIO_DIR.is_dir() or len(list(AUDIO_DIR.glob('*/*.mp3')))<8000:
        download('fma_small',AUDIO_DIR)
    from backend.audio.prepare import fma_metadata
    records=fma_metadata(AUDIO_DIR,METADATA_DIR)
    missing=[r['id'] for r in records if not Path(r['path']).is_file()]
    if len(records)!=8000 or missing:
        raise ValueError(f'FMA Small incompleto: {len(records)} registros; {len(missing)} audios ausentes.')
    print('FMA Small: 8000 pistas y metadatos oficiales disponibles.',flush=True)


if __name__=='__main__':
    ensure_data()
