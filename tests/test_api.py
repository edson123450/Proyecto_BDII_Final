from fastapi.testclient import TestClient
from backend import app as module


def test_web_api_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'DATA_DIR', tmp_path)
    with TestClient(module.app) as client:
        assert client.get('/').status_code == 200
        assert 'Editor SQL' in client.get('/').text
        assert client.get('/api/health').json()['status'] == 'ok'
        statements = [
            'CREATE TABLE empleados (id INT PRIMARY KEY, nombre CHAR(30)) USING HEAP;',
            "INSERT INTO empleados VALUES (1, 'Joaquín');",
            'CREATE INDEX idx ON empleados (id) USING HASH;',
        ]
        for sql in statements:
            response = client.post('/api/query', json={'sql': sql})
            assert response.status_code == 200, response.text
        result = client.post('/api/query', json={'sql': 'SELECT * FROM empleados WHERE id = 1;'}).json()
        assert result['rows'] == [{'id': 1, 'nombre': 'Joaquín'}]
        assert result['plan']['operation'] == 'IndexScan'
        assert result['metrics']['disk_reads'] > 0
        assert result['metrics']['by_role']['index']['disk_reads'] > 0
        assert client.post('/api/query', json={'sql': 'DROP DATABASE cualquiera;'}).status_code == 400
        assert client.post('/api/query', json={'sql': 'SELECT * FROM empleados;', 'page_size': 1000}).status_code == 422
        assert client.post('/api/tables/reorganize', json={'table': 'empleados'}).status_code == 400
    with TestClient(module.app) as client:
        assert client.get('/api/tables').json()[0]['record_count'] == 1
        result = client.post('/api/query', json={'sql': 'SELECT * FROM empleados WHERE id=1;'}).json()
        assert result['total'] == 1


def test_audio_sql_uses_stored_vector_and_escapes_genre(tmp_path,monkeypatch):
    import json
    import numpy as np
    from backend.minidb.blob import BlobStore
    from backend.minidb.multimodal import MultimodalDatabase
    from backend.prepare_multimodal import column
    from backend.minidb.sql_multimodal import parse_multimodal_sql
    db=MultimodalDatabase(tmp_path)
    db.create_table('pistas',[column('id',primary=True),column('titulo','CHAR',30),column('genero','CHAR',30),column('embedding_ptr')])
    blobs=BlobStore(tmp_path/'pistas.blob')
    vector=np.asarray([.1,.2,.3,.4],dtype='<f4')
    pointer=blobs.append(vector.tobytes());blobs.close()
    db.insert('pistas',dict(id=7,titulo='Pista',genero="rock 'n' roll",embedding_ptr=pointer))
    db.create_index('pistas','pistas_id','id','BTREE')
    db.register('pistas',virtual_columns={'embedding':'VECTOR(4)'},blobs='pistas.blob')
    db.close()
    monkeypatch.setattr(module,'DATA_DIR',tmp_path)
    with TestClient(module.app) as client:
        response=client.get('/api/audio/sql/7')
        assert response.status_code==200,response.text
        ast=parse_multimodal_sql(response.json()['sql'])
        assert ast['where']==('=','genero',"rock 'n' roll")
        np.testing.assert_allclose(ast['columns'][3]['vector'],vector)
        assert ast['limit']==5 and ast['order']==('distancia',False)
        assert client.get('/api/audio/sql/999').status_code==404
        assert client.get('/api/audio/info').json()['ready'] is False


def test_uploaded_fragment_after_thirty_seconds_and_failure_cleanup(tmp_path,monkeypatch):
    import io
    import numpy as np
    import soundfile as sf
    monkeypatch.setattr(module,'DATA_DIR',tmp_path/'database')
    monkeypatch.setattr(module,'ROOT',tmp_path)
    sr=8000
    waveform=np.concatenate([np.zeros(35*sr),np.ones(5*sr)*.2]).astype(np.float32)
    payload=io.BytesIO();sf.write(payload,waveform,sr,format='WAV',subtype='FLOAT')
    with TestClient(module.app) as client:
        def fake_search(samples,rate,k,probes,include_dtw):
            assert rate==sr and len(samples)==5*sr
            np.testing.assert_allclose(samples,.2,atol=1e-7)
            return dict(query={'duration':5},methods=[],extraction_ms={},corpus_count=0)
        monkeypatch.setattr(module.audio_service,'search',fake_search)
        response=client.post('/api/audio/search',data={'start':'35','duration':'5','include_dtw':'false'},files={'file':('long.wav',payload.getvalue(),'audio/wav')})
        assert response.status_code==200,response.text
        assert client.get(response.json()['query']['audio_url']).status_code==200
        files_before=set((tmp_path/'data/cache/queries').iterdir())
        def failed(*args,**kwargs):
            raise ValueError('injected search failure')
        monkeypatch.setattr(module.audio_service,'search',failed)
        response=client.post('/api/audio/search',data={'start':'35','duration':'5'},files={'file':('long.wav',payload.getvalue(),'audio/wav')})
        assert response.status_code==400
        assert set((tmp_path/'data/cache/queries').iterdir())==files_before
        response=client.post('/api/audio/search',data={'start':'1e300','duration':'5'},files={'file':('long.wav',payload.getvalue(),'audio/wav')})
        assert response.status_code==400
