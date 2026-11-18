"""API REST del gestor y publicación del cliente web."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .minidb.engine import Database


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('MINIDB_DATA_DIR', ROOT / 'data' / 'runtime'))
database = None


@asynccontextmanager
async def lifespan(app):
    global database
    database = Database(DATA_DIR)
    yield
    database.close()


app = FastAPI(title='SonoraDB', description='Motor multimodal construido sobre páginas binarias.', lifespan=lifespan)


class QueryRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=200000)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=25, ge=1, le=100)


class ReorganizeRequest(BaseModel):
    table: str


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
