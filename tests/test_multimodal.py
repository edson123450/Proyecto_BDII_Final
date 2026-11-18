import json

import numpy as np
import pytest

from backend.minidb.blob import BlobStore
from backend.minidb.multimodal import MultimodalDatabase, project
from backend.minidb.vector import VectorStore
from backend.prepare_multimodal import column


@pytest.fixture
def collection(tmp_path):
    db=MultimodalDatabase(tmp_path)
    db.create_table('locales',[column('id',primary=True),column('nombre','CHAR',30),column('lon','FLOAT'),column('lat','FLOAT')])
    for identifier,offset in enumerate([0,.001,.01,.1],1):
        db.insert('locales',dict(id=identifier,nombre=f'Local {identifier}',lon=-77.04+offset,lat=-12.05))
    db.register('locales',virtual_columns={'ubicacion':'POINT'})
    db.execute('CREATE INDEX geo ON locales(ubicacion) USING RTREE;')
    db.create_table('documentos',[column('id',primary=True),column('titulo','CHAR',30),column('cuerpo_ptr')])
    blobs=BlobStore(tmp_path/'text.blob')
    for identifier,body in enumerate(['music jazz piano','music jazz drums','music electronic drums','bird tree garden'],1):
        db.insert('documentos',dict(id=identifier,titulo=f'Texto {identifier}',cuerpo_ptr=blobs.append(body.encode())))
    blobs.close()
    db.create_index('documentos','doc_id','id','BTREE')
    db.register('documentos',virtual_columns={'cuerpo':'TEXT'},blobs='text.blob')
    db.execute("CREATE INDEX txt ON documentos(cuerpo) USING INVERTED WITH(scoring='TFIDF_COSINE');")
    db.create_table('pistas',[column('id',primary=True),column('genero','CHAR',30),column('embedding_ptr')])
    blobs=BlobStore(tmp_path/'vec.blob')
    store=VectorStore(tmp_path/'vec.bin',dimension=4,create=True)
    rng=np.random.default_rng(42)
    for identifier in range(30):
        vector=rng.normal(size=4).astype('<f4')
        rid=db.insert('pistas',dict(id=identifier,genero='jazz' if identifier%2 else 'rock',embedding_ptr=blobs.append(vector.tobytes())))
        store.append(rid,vector)
    blobs.close();store.close()
    db.create_index('pistas','genre','genero','HASH')
    db.register('pistas',virtual_columns={'embedding':'VECTOR(4)'},blobs='vec.blob',vectors='vec.bin')
    db.execute("CREATE INDEX vec ON pistas(embedding) USING IVF WITH(lists=5,metric='COSINE');")
    yield db
    db.close()


def test_spatial_radius_knn_window_and_reopen(collection):
    db=collection
    result=db.execute('SELECT id, ubicacion FROM locales WHERE ubicacion WITHIN_RADIUS((-77.04,-12.05),0.2);')
    assert {r['id'] for r in result['rows']}=={1,2}
    assert result['metrics']['by_role']['index']['disk_reads']>0
    result=db.execute('SELECT id FROM locales WHERE ubicacion KNN((-77.04,-12.05),3);')
    assert [r['id'] for r in result['rows']]==[1,2,3]
    result=db.execute('SELECT id FROM locales WHERE ubicacion WITHIN_WINDOW((-77.05,-12.06),(-77.035,-12.04));')
    assert {r['id'] for r in result['rows']}=={1,2}
    db.close()
    reopened=MultimodalDatabase(db.directory)
    assert reopened.execute('SELECT id FROM locales WHERE ubicacion KNN((-77.04,-12.05),1);')['rows']==[{'id':1}]
    reopened.close()


def test_text_boolean_ranking_source_and_filters(collection):
    db=collection
    result=db.execute("SELECT id, cuerpo, COSINE_SCORE() AS relevance FROM documentos WHERE cuerpo MATCH 'music AND NOT electronic' ORDER BY relevance DESC LIMIT 10;")
    assert {r['id'] for r in result['rows']}=={1,2}
    assert all('music' in r['cuerpo'] and 0<=r['relevance']<=1.0000001 for r in result['rows'])
    result=db.execute("SELECT id, COSINE_SCORE() AS relevance FROM documentos WHERE cuerpo MATCH 'jazz' AND id=2 ORDER BY relevance DESC LIMIT 10;")
    assert [r['id'] for r in result['rows']]==[2]
    assert result['plan']['filter']['operation']=='IndexScan'
    assert result['metrics']['by_role']['data']['disk_reads']>0


def test_hybrid_ivf_matches_exact_after_prefilter(collection):
    db=collection
    db.execute('SET ivf.probes=5;')
    query=[1.,.2,.3,.4]
    sql=f"SELECT id, genero, embedding <=> '{json.dumps(query)}' AS distancia FROM pistas WHERE genero='jazz' ORDER BY distancia ASC LIMIT 5;"
    result=db.execute(sql)
    allowed={rid for rid,row in db.tables['pistas'].scan() if row['genero']=='jazz'}
    truth=db.resources['pistas']['vectors'].knn(query,5,allowed_rids=allowed)
    assert [r['id'] for r in result['rows']]==[db.tables['pistas'].get(rid)['id'] for rid,_ in truth]
    assert result['plan']['strategy']=='Pre-filtering mediante IndexScan'
    assert result['plan']['filter']['index']=='genre'
    assert result['metrics']['by_role']['index']['disk_reads']>0
    assert all(r['genero']=='jazz' for r in result['rows'])
    with pytest.raises(ValueError):
        db.execute(sql.replace('<=>','<->'))


def test_managed_collections_reject_unsynchronized_writes(collection):
    with pytest.raises(ValueError,match='sincronizadas'):
        collection.execute('DELETE FROM pistas WHERE id=1;')
    with pytest.raises(ValueError):
        collection.execute('SELECT inexistente FROM pistas;')
    with pytest.raises(ValueError):
        collection.execute('SELECT id, COSINE_SCORE() AS score FROM pistas;')
