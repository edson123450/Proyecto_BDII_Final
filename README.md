# SonoraDB · Proyecto de Base de Datos II

Motor de base de datos que almacena registros e índices en archivos binarios paginados. La primera entrega implementa Heap File, Sequential File, árbol B+, hashing extendible, SQL básico, API REST y un cliente web con métricas reales de acceso a disco.

## Iniciar

Requisito: Python 3.11 o superior. Desde la carpeta del repositorio:

```bash
./start.sh
```

El script prepara el entorno, instala las dependencias si faltan y genera 100 000 empleados sintéticos tanto en Heap como en Sequential. Luego inicia la aplicación en **http://127.0.0.1:8014**. La primera preparación tarda más porque construye los archivos y sus índices; las siguientes ejecuciones reutilizan esos datos.

```bash
git clone https://github.com/edson123450/Proyecto_BDII_Final.git
cd Proyecto_BDII_Final
./start.sh
```

## Consultas de la primera entrega

```sql
SELECT * FROM empleados WHERE id = 42000;
SELECT id, nombre, salario FROM empleados WHERE id >= 100 AND id <= 150;
SELECT id, nombre FROM empleados WHERE dept = 'Sistemas' LIMIT 100;
SELECT * FROM empleados_seq WHERE salario >= 13500 LIMIT 100;
```

En una tabla propia se pueden probar creación, inserción, índices y eliminación:

```sql
CREATE TABLE prueba (id INT PRIMARY KEY, nombre CHAR(30), salario FLOAT) USING HEAP;
INSERT INTO prueba VALUES (1, 'Ada Lovelace', 5200.0);
CREATE INDEX idx_prueba ON prueba (id) USING BTREE;
SELECT * FROM prueba WHERE id = 1;
DELETE FROM prueba WHERE id = 1;
```

Ejecutar una sentencia por vez. `USING SEQUENTIAL` selecciona la organización secuencial; `USING HASH` crea un hash extendible. Las columnas admiten `INT`, `FLOAT` y `CHAR(n)`. Los filtros admiten comparaciones, `AND`, `OR` y paréntesis; las consultas permiten proyección, `ORDER BY` y `LIMIT`.

## Organización

- `backend/minidb/`: páginas, registros, índices, parser y motor.
- `backend/app.py`: API REST y publicación del cliente.
- `frontend/`: explorador, editor SQL, resultados paginados y plan/métricas.
- `data/`: preparación de datos; `runtime/` contiene los archivos generados.
- `benchmarks/`: experimentos reproducibles y sus resultados.
- `tests/`: pruebas de persistencia, estructuras y consultas.

## Medición y experimentos

```bash
.venv/bin/python -m pytest -q
.venv/bin/python benchmarks/parte1.py --scale full
```

Los experimentos 1–4 usan hasta 500 000 registros, 1 000 consultas de igualdad, cinco selectividades y páginas de 1, 2, 4 y 8 KB. Las mediciones completas se guardan en `benchmarks/results/full/`, incluyendo datos crudos, resúmenes, metodología y gráficos. El modo `--scale smoke` permite comprobar el procedimiento con menos datos y no reemplaza los resultados completos.

Cada lectura/escritura cuenta una transferencia de página realizada por el gestor. No hay un buffer de páginas oculto; el sistema operativo sí puede conservar páginas en su caché. Por ello, estas cifras son transferencias del gestor al archivo, no mediciones de accesos eléctricos al SSD. El desglose separa datos e índices; los tiempos corresponden al equipo y condiciones de la ejecución.

El Sequential mantiene un área principal ordenada y un overflow enlazado. Los enlaces de salto del overflow evitan un recorrido lineal por cada inserción masiva. La reorganización fusiona ambas áreas con un factor de llenado del 75 % y reconstruye los índices porque cambian los RID.

## API

- `POST /api/query`: `{"sql":"SELECT * FROM empleados WHERE id = 1;","page":1,"page_size":25}`.
- `GET /api/tables`: tablas, columnas, índices y cantidad de registros.
- `POST /api/tables/reorganize`: `{"table":"empleados_seq"}`.
- `GET /api/benchmarks`: gráficos de los experimentos completos.

La documentación interactiva de la API está en `/docs`. El servidor se inicia en la interfaz local del equipo.

## Referencias de implementación

Las estructuras de almacenamiento y los índices están implementados en Python usando `seek`, lecturas de tamaño fijo y `struct`; no utilizan otro motor de bases de datos ni una librería de índices. FastAPI se utiliza para la API y NumPy/Matplotlib para generar y analizar los experimentos.
