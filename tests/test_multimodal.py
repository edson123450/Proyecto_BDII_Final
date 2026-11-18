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


def test_relational_or_inside_managed_collection(collection):
    result=collection.execute('SELECT id FROM locales WHERE (id=1 OR id=3) AND nombre!=\'missing\' ORDER BY id;')
    assert [row['id'] for row in result['rows']]==[1,3]
    result=collection.execute("SELECT id FROM documentos WHERE cuerpo MATCH 'music' AND (id=1 OR id=3) ORDER BY id;")
    assert [row['id'] for row in result['rows']]==[1,3]


def test_boolean_operators_are_case_insensitive(collection):
    result=collection.execute("SELECT id FROM documentos WHERE cuerpo MATCH 'music and not electronic' ORDER BY id;")
    assert [row['id'] for row in result['rows']]==[1,2]


def test_text_pagination_reads_only_visible_originals(collection,monkeypatch):
    blobs=collection.resources['documentos']['blobs']
    original=blobs.read
    calls=[]
    def read(pointer):
        calls.append(pointer)
        return original(pointer)
    monkeypatch.setattr(blobs,'read',read)
    result=collection.execute("SELECT id,cuerpo,COSINE_SCORE() AS score FROM documentos WHERE cuerpo MATCH 'music OR bird' ORDER BY score DESC LIMIT 3;",page=2,page_size=1)
    assert result['total']==3
    assert len(result['rows'])==1
    assert len(calls)==1
    assert result['rows'][0]['cuerpo']


def test_vector_limit_zero_and_ambiguous_projections(collection):
    query="SELECT id,embedding <=> '[1,2,3,4]' AS d FROM pistas ORDER BY d ASC LIMIT 0;"
    result=collection.execute(query)
    assert result['rows']==[] and result['total']==0
    assert result['metrics']['disk_reads']==0
    with pytest.raises(ValueError,match='sola expresión'):
        collection.execute("SELECT embedding <=> '[1,0,0,0]' AS a,embedding <=> '[0,1,0,0]' AS b FROM pistas ORDER BY a ASC LIMIT 3")
    for page,size in [(0,25),(-1,25),(True,25),(1,0),(1,2.5)]:
        with pytest.raises(ValueError):
            collection.execute('SELECT id FROM locales',page=page,page_size=size)


def test_multimodal_index_construction_failure_leaves_catalog_intact(collection,monkeypatch):
    before=collection.catalog_path.read_bytes()
    original=collection.tables['locales'].scan
    def broken_scan():
        yield next(original())
        raise RuntimeError('injected construction failure')
    monkeypatch.setattr(collection.tables['locales'],'scan',broken_scan)
    with pytest.raises(RuntimeError,match='injected'):
        collection.execute('CREATE INDEX failed_geo ON locales(ubicacion) USING RTREE')
    assert 'failed_geo' not in collection.mm
    assert all(index['name']!='failed_geo' for index in collection.catalog['locales']['multimodal']['indexes'])
    assert not (collection.directory/'failed_geo').exists()
    assert not (collection.directory/'failed_geo.building').exists()
    assert collection.catalog_path.read_bytes()==before
    assert collection.execute('SELECT id FROM locales WHERE ubicacion KNN((-77.04,-12.05),1)')['rows']==[{'id':1}]


def test_multimodal_catalog_publication_failure_rolls_back_files(collection,monkeypatch):
    before=collection.catalog_path.read_bytes()
    def fail():
        raise OSError('injected catalog failure')
    monkeypatch.setattr(collection,'_save_catalog',fail)
    with pytest.raises(OSError,match='injected'):
        collection.execute('CREATE INDEX failed_publish ON locales(ubicacion) USING RTREE')
    assert 'failed_publish' not in collection.mm
    assert not (collection.directory/'failed_publish').exists()
    assert not (collection.directory/'failed_publish.building').exists()
    assert collection.catalog_path.read_bytes()==before


def test_registry_rejects_missing_files_and_index_name_collisions(collection):
    collection.create_table('broken',[column('id'),column('cuerpo_ptr')])
    with pytest.raises(ValueError,match='Falta el archivo'):
        collection.register('broken',virtual_columns={'cuerpo':'TEXT'},blobs='missing.blob')
    assert 'multimodal' not in collection.catalog['broken']
    assert not (collection.directory/'missing.blob').exists()
    with pytest.raises(ValueError,match='ya existe'):
        collection.execute('CREATE INDEX geo ON locales(id) USING BTREE')
    with pytest.raises(ValueError,match='sincronizadas'):
        collection.insert('locales',dict(id=8,nombre='Nuevo',lon=0.,lat=0.))


def test_preparation_does_not_silently_accept_partial_tables(tmp_path):
    from backend.prepare_multimodal import prepare_space,prepare_text
    db=MultimodalDatabase(tmp_path)
    try:
        db.create_table('locales',[column('id')])
        db.create_table('documentos',[column('id')])
        with pytest.raises(ValueError,match='incompleta'):
            prepare_space(db,100000)
        with pytest.raises(ValueError,match='incompleta'):
            prepare_text(db,tmp_path,20000)
        assert set(db.catalog)=={'locales','documentos'}
    finally:
        db.close()


@pytest.mark.parametrize('filename', ['geo', 'txt/vocab.dic', 'txt/postings.inv', 'vec/centroids.dir', 'vec/lists.ivf'])
def test_reopening_missing_multimodal_index_fails_without_creating_files(collection,filename):
    directory=collection.directory
    collection.close()
    missing=directory/filename
    missing.unlink()
    with pytest.raises(ValueError,match='Faltan archivos del índice'):
        MultimodalDatabase(directory)
    assert not missing.exists()
