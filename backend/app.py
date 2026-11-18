"""API REST del gestor y publicación del cliente web."""

import os
import math
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .minidb.multimodal import MultimodalDatabase
from .audio.service import AudioService
from .config import ROOT, DATA_DIR


database = None
audio_service = None


@asynccontextmanager
async def lifespan(app):
    global database, audio_service
    database = MultimodalDatabase(DATA_DIR)
    audio_service = AudioService(database)
    try:
        yield
    finally:
        audio_service.close()
        database.close()


app = FastAPI(title='SonoraDB', description='Motor multimodal construido sobre páginas binarias.', lifespan=lifespan)


class QueryRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=200000)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=25, ge=1, le=100)


class ReorganizeRequest(BaseModel):
    table: str


class SpatialRequest(BaseModel):
    lon: float = Field(default=-77.04,ge=-180,le=180)
    lat: float = Field(default=-12.05,ge=-90,le=90)
    mode: str = 'radius'
    radius: float = Field(default=2.5,gt=0,le=100)
    k: int = Field(default=20,ge=1,le=100)
    width: float = Field(default=2.5,gt=0,le=100)


class TextRequest(BaseModel):
    query: str = Field(min_length=1,max_length=4000)
    k: int = Field(default=10,ge=1,le=100)


@app.post('/api/spatial/search')
def spatial_search(request: SpatialRequest):
    point=f'({request.lon},{request.lat})'
    if request.mode=='radius':
        predicate=f'ubicacion WITHIN_RADIUS ({point},{request.radius})'
    elif request.mode=='knn':
        predicate=f'ubicacion KNN ({point},{request.k})'
    elif request.mode=='window':
        dy=request.width/2/111.32;dx=dy/math.cos(math.radians(-12.05))
        predicate=f'ubicacion WITHIN_WINDOW (({request.lon-dx},{request.lat-dy}),({request.lon+dx},{request.lat+dy}))'
    else:
        raise HTTPException(400,'Modo espacial desconocido.')
    sql=f'SELECT id, nombre, ubicacion FROM locales WHERE {predicate};'
    try:
        result=database.execute(sql,page_size=100000)
        result.update(sql=sql,geometry=dict(center=[request.lon,request.lat],radius_km=request.radius,width_km=request.width,mode=request.mode))
        return result
    except ValueError as error:
        raise HTTPException(400,str(error)) from error


@app.post('/api/text/search')
def text_search(request: TextRequest):
    escaped=request.query.replace("'","''")
    sql=f"SELECT id, titulo, cuerpo, COSINE_SCORE() AS relevancia FROM documentos WHERE cuerpo MATCH '{escaped}' ORDER BY relevancia DESC LIMIT {request.k};"
    try:
        result=database.execute(sql,page_size=request.k)
        return dict(**result,sql=sql)
    except ValueError as error:
        raise HTTPException(400,str(error)) from error


@app.get('/api/audio/info')
def audio_info():
    with database.lock:
        return audio_service.info()


@app.get('/api/audio/catalog')
def audio_catalog(limit:int=Query(default=100,ge=1,le=8000)):
    with database.lock:
        return audio_service.catalog(limit)


@app.get('/api/audio/track/{identifier}')
def audio_track(identifier:int):
    with database.lock:
        if not audio_service.ready() or database.get_by_id('pistas',identifier) is None:
            raise HTTPException(404,'Pista fuera de la colección.')
    path=audio_service.path(identifier)
    if not path.is_file():
        raise HTTPException(404,'El archivo de audio no está disponible en la ruta configurada.')
    return FileResponse(path,media_type='audio/mpeg')


@app.get('/api/audio/sql/{identifier}')
def audio_sql(identifier:int):
    """Esta función expone la consulta vectorial de la pista completa, no sus segmentos."""
    try:
        with database.lock:
            if 'pistas' not in database.tables:
                raise HTTPException(404,'La colección de pistas todavía no está preparada.')
            found=database.get_by_id('pistas',identifier)
            if found is None:
                raise HTTPException(404,'Pista fuera de la colección.')
            row=found[1]
            vector=database.value('pistas',row,'embedding')
            import json
            literal=json.dumps(vector,separators=(',',':'),allow_nan=False)
            genre=str(row['genero']).replace("'","''")
            sql=(f"SELECT id, titulo, genero, embedding <=> '{literal}' AS distancia FROM pistas "
                 f"WHERE genero='{genre}' ORDER BY distancia ASC LIMIT 5;")
            return dict(sql=sql,scope='Vector de la pista completa; la comparación de audio utiliza segmentos.')
    except (ValueError,KeyError) as error:
        raise HTTPException(400,str(error)) from error


@app.get('/api/audio/query/{identifier}')
def query_audio(identifier:str):
    if not re_identifier(identifier):
        raise HTTPException(404,'Consulta no encontrada.')
    path=ROOT/'data/cache/queries'/(identifier+'.wav')
    if not path.is_file():
        raise HTTPException(404,'Consulta no encontrada.')
    return FileResponse(path,media_type='audio/wav')


def re_identifier(value):
    return len(value)==32 and all(c in '0123456789abcdef' for c in value)


def audio_fragment(path,start,duration):
    """Esta función lee solo el recorte solicitado, incluso después de los 30 s."""
    import soundfile as sf
    import numpy as np
    with sf.SoundFile(path) as source:
        sr=source.samplerate
        if start>=len(source)/sr:
            raise ValueError('El inicio queda fuera del archivo de audio.')
        source.seek(int(start*sr))
        waveform=source.read(frames=int(duration*sr),dtype='float32',always_2d=True).mean(axis=1)
    if len(waveform)<sr/2 or not np.isfinite(waveform).all():
        raise ValueError('El recorte debe contener al menos 0.5 segundos de audio finito.')
    return np.ascontiguousarray(waveform,dtype=np.float32),sr


@app.post('/api/audio/search')
def audio_search(file:UploadFile|None=File(default=None),track_id:str=Form(default=''),start:float=Form(default=0),
                 duration:float=Form(default=10),noise_db:str=Form(default=''),k:int=Form(default=5),
                 probes:int=Form(default=5),include_dtw:bool=Form(default=True)):
    import numpy as np
    import soundfile as sf
    with database.lock:
        if not audio_service.ready():
            raise HTTPException(400,'Completar la preparación de las pistas antes de consultar audio.')
    if not 1<=k<=20 or not 1<=probes<=1000 or not math.isfinite(start) or start<0 or not math.isfinite(duration) or not .5<=duration<=30:
        raise HTTPException(400,'Usar k entre 1–20, probes entre 1–1000 y fragmentos de 0.5–30 segundos.')
    directory=ROOT/'data/cache/queries';directory.mkdir(parents=True,exist_ok=True)
    identifier=uuid.uuid4().hex
    temporary=None
    query_path=directory/(identifier+'.wav')
    completed=False
    try:
        if file is not None and file.filename:
            temporary=directory/(identifier+'.upload')
            size=0
            with temporary.open('wb') as target:
                while chunk:=file.file.read(1024*1024):
                    size+=len(chunk)
                    if size>30*1024*1024:
                        raise ValueError('El audio de consulta debe ocupar como máximo 30 MB.')
                    target.write(chunk)
            source=temporary;source_id=None
        elif track_id:
            source_id=int(track_id)
            with database.lock:
                audio_service._track(source_id)
            source=audio_service.path(source_id)
        else:
            raise ValueError('Seleccionar una pista o subir un archivo de audio.')
        waveform,sr=audio_fragment(source,start,duration)
        if noise_db:
            snr=float(noise_db)
            if not math.isfinite(snr) or not 0<=snr<=60:
                raise ValueError('La relación señal/ruido debe estar entre 0 y 60 dB.')
            power=float(np.mean(waveform.astype(np.float64)**2))
            noise=np.random.default_rng(42).normal(0,np.sqrt(power/10**(snr/10)),len(waveform))
            waveform=(waveform+noise).astype(np.float32)
        sf.write(query_path,waveform,sr,subtype='FLOAT')
        with database.lock:
            result=audio_service.search(waveform,sr,k,probes,include_dtw)
        result['query'].update(source_id=source_id,audio_url=f'/api/audio/query/{identifier}')
        completed=True
        return result
    except (ValueError,RuntimeError,OSError) as error:
        raise HTTPException(400,str(error)) from error
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
        if not completed:
            query_path.unlink(missing_ok=True)


@app.post('/api/query')
def query(request: QueryRequest):
    try:
        return database.execute(request.sql, request.page, request.page_size)
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(400, str(error)) from error


@app.get('/api/tables')
def tables():
    with database.lock:
        return database.describe()


@app.post('/api/tables/reorganize')
def reorganize(request: ReorganizeRequest):
    try:
        return database.reorganize(request.table)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.get('/api/health')
def health():
    return {'status': 'ok', 'engine': 'SonoraDB', 'page_size': database.page_size}


@app.get('/api/benchmarks')
def benchmarks():
    directory = ROOT / 'benchmarks' / 'results'
    return [dict(name=p.name, url='/benchmarks/' + p.relative_to(directory).as_posix())
            for p in sorted(directory.rglob('*.png')) if 'smoke' not in p.parts]


app.mount('/assets', StaticFiles(directory=ROOT / 'frontend'), name='assets')
app.mount('/benchmarks', StaticFiles(directory=ROOT / 'benchmarks' / 'results'), name='benchmarks')


@app.get('/')
def index():
    return FileResponse(ROOT / 'frontend' / 'index.html')
