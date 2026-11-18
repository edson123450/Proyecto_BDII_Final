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
