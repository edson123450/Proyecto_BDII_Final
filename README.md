# SonoraDB · Proyecto de Base de Datos II

Gestor multimodal construido sobre archivos binarios propios. Integra almacenamiento relacional, consultas espaciales, recuperación textual y una aplicación de identificación de fragmentos musicales con **FMA Small**.

## Iniciar en un paso

Requisitos: Python 3.11–3.13, conexión para la primera descarga y espacio libre para FMA Small. El proyecto se verificó en macOS con Python 3.13.7. Desde la carpeta del repositorio:

```bash
./start.sh
```

Abrir **http://127.0.0.1:8014**. El script prepara el entorno, instala dependencias, obtiene los datos si faltan, extrae las representaciones y construye los archivos e índices. La primera preparación puede tardar una hora o más; la extracción se puede reanudar. Las siguientes aperturas reutilizan los archivos existentes.

Si FMA ya está descargado, copiar `.env.example` como `.env` y especificar `FMA_AUDIO_DIR` y `FMA_METADATA_DIR` antes de iniciar. Así se utiliza la copia existente. `.env`, audios, modelos, cachés y bases generadas quedan fuera de Git. La descarga oficial verifica el checksum antes de extraer y requiere aproximadamente 18 GB libres durante la preparación.

```bash
git clone https://github.com/edson123450/Proyecto_BDII_Final.git
cd Proyecto_BDII_Final
./start.sh
```

PostgreSQL **solo se necesita para reproducir la comparación experimental E6**. La aplicación y sus índices funcionan con el motor propio.

## Recorrido de la aplicación

- **Consola SQL:** explorador de tablas, editor, resultados paginados y plan con tiempos y transferencias de páginas.
- **Mapa:** consultas de radio, ventana y vecinos más cercanos sobre 100 000 puntos simulados de Lima. El mapa base requiere conexión; Leaflet y los puntos se sirven localmente.
- **Texto:** búsqueda booleana y TF-IDF/coseno en 20 000 documentos formados por metadatos originales de FMA.
- **Audio:** subir un archivo o recortar una pista del catálogo, añadir ruido de prueba y comparar BoAW, IVF, búsqueda vectorial exacta y DTW con LB_Keogh.
- **Experimentos:** gráficos obtenidos de las mediciones, con sus datos crudos en `benchmarks/results/`.

Las colecciones preparadas se mantienen de solo lectura desde SQL para conservar sincronizados sus textos, vectores y secuencias. Se pueden crear índices adicionales. Las tablas relacionales propias admiten inserciones y eliminaciones.

## SQL relacional

Ejecutar una sentencia por vez:

```sql
SELECT * FROM empleados WHERE id = 42000;
SELECT id, nombre, salario FROM empleados WHERE id >= 100 AND id <= 150;
SELECT id, nombre FROM empleados WHERE dept = 'Sistemas' LIMIT 100;
SELECT * FROM empleados_seq WHERE salario >= 13500 LIMIT 100;
```

```sql
CREATE TABLE prueba (id INT PRIMARY KEY, nombre CHAR(30), salario FLOAT) USING HEAP;
INSERT INTO prueba VALUES (1, 'Ada Lovelace', 5200.0);
CREATE INDEX idx_prueba ON prueba (id) USING BTREE;
SELECT * FROM prueba WHERE id = 1;
DELETE FROM prueba WHERE id = 1;
```

`USING SEQUENTIAL` selecciona almacenamiento secuencial. `USING HASH` crea hashing extendible. Las columnas relacionales admiten `INT`, `FLOAT` y `CHAR(n)`; los filtros, comparaciones, `AND`, `OR` y paréntesis. También se implementan proyección, `ORDER BY` y `LIMIT`.

## SQL multimodal

Los índices de las colecciones ya se construyen durante la preparación. Estos ejemplos crean otros índices con nombres diferentes:

```sql
CREATE INDEX geo_demo ON locales (ubicacion) USING RTREE;
CREATE INDEX texto_demo ON documentos (cuerpo) USING INVERTED WITH (scoring = 'TFIDF_COSINE');
CREATE INDEX vector_demo ON pistas (embedding) USING IVF WITH (lists = 100, metric = 'COSINE');
```

```sql
SELECT id, nombre, ubicacion FROM locales
WHERE ubicacion WITHIN_RADIUS ((-77.0282, -12.1354), 2.5);

SELECT id, nombre, ubicacion FROM locales
WHERE ubicacion KNN ((-77.04, -12.05), 20);

SELECT id, titulo, COSINE_SCORE() AS relevancia FROM documentos
WHERE cuerpo MATCH 'jazz AND piano AND NOT electronic'
ORDER BY relevancia DESC LIMIT 10;

SET ivf.probes = 5;
```

En Audio, **Ver consulta vectorial** prepara una consulta ejecutable usando los 512 componentes reales de la pista seleccionada. Su forma es `embedding <=> 'vector JSON' AS distancia`, ordenada de menor a mayor y combinada con un filtro por género. El plan informa el prefiltrado relacional mediante hash y la exploración IVF. El operador `<->` corresponde a L2 cuando la colección se ha preparado con esa métrica; las pistas usan coseno.

El radio espacial se expresa en kilómetros. Se usa una proyección plana local de Lima, no distancias geodésicas globales. En consultas mixtas se combina una operación multimodal con filtros relacionales mediante `AND`; los operadores booleanos de texto se escriben dentro de `MATCH`.

## Diseño implementado

- **Heap:** registros fijos, bitmap y lista de páginas libres, RID `(page_id, slot_id)`.
- **Sequential:** área principal ordenada y overflow enlazado con enlaces de salto. La reorganización usa un llenado del 75 % y reconstruye los índices cuyos RID cambian.
- **B+:** varios niveles, división al 50 %, hojas doblemente enlazadas, igualdad y rango.
- **Hash extendible:** directorio paginado, profundidades global/local y división de buckets.
- **R-Tree:** MBR, menor ampliación de área, división cuadrática, ventanas, radio y kNN con cola por MINDIST.
- **Invertido:** `vocab.dic`, `postings.inv` y normas persistidas. Normalización, stopwords de español/inglés, stemming Snowball, mezcla de listas booleanas y TF-IDF/coseno con max-heap Top-K.
- **IVF:** `floor(sqrt(N))` centroides por defecto, directorio y listas en páginas de 4 KB. MiniBatch K-Means ajusta los centroides; el índice y sus recorridos son propios. Explorar todas las listas reproduce la búsqueda exacta sobre la misma representación.
- **Objetos variables:** cadenas de páginas para textos, embeddings y secuencias. Los MP3 se leen de la carpeta FMA configurada.

Los índices no usan FAISS, Rtree, PostgreSQL ni otro motor como almacenamiento interno. Las consultas leen los archivos propios mediante `seek` y transferencias de página. Las bibliotecas de aprendizaje se usan para extracción y clustering, no para reemplazar los índices.

## Representación acústica y alcance

FMA Small contiene 8000 pistas de aproximadamente 30 segundos y ocho géneros: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop y Rock. Se conservan las divisiones oficiales de entrenamiento, validación y prueba. Los errores de decodificación se registran explícitamente durante la preparación.

En la preparación incluida se revisaron los 8000 archivos de la copia local: **7994 fueron utilizables** (6394 de entrenamiento, 800 de validación y 800 de prueba). Los IDs 98565, 98567, 98569, 99134, 108925 y 133297 no pudieron decodificarse; sus originales se conservaron y no se sustituyeron. La aplicación utiliza las pistas válidas y los experimentos mantienen las consultas de prueba predefinidas.

**BoAW:** MFCC20 por ventanas, normalización y vocabulario de 256 palabras acústicas ajustados solo con pistas de entrenamiento. Los histogramas se consultan mediante el mismo índice invertido con TF-IDF/coseno.

**Embeddings:** PANNs Cnn6 preentrenada en AudioSet genera vectores de 512 dimensiones. Se extraen ventanas de 10 segundos con paso de 5 segundos; el promedio normalizado representa la pista para comparar géneros. La identificación utiliza los segmentos y agrupa sus vecinos por pista.

**DTW:** secuencias MFCC13 agregadas cada 200 ms, con normalización aprendida en entrenamiento. Se comparan todas las ventanas de igual duración con una banda del 10 % de su longitud, de al menos dos pasos. LB_Keogh descarta ventanas cuya cota supera el candidato actual; los resultados ANN solo adelantan evaluaciones y no excluyen pistas del recorrido exacto. La garantía de ausencia de falsos negativos corresponde a este universo discreto y esta distancia, no a reconocer cualquier grabación del mundo real.

La aplicación devuelve candidatos y metadatos de licencia; una coincidencia acústica no determina jurídicamente la titularidad de derechos.

## Experimentos reproducibles

```bash
.venv/bin/python -m pytest -q
.venv/bin/python benchmarks/parte1.py --scale full
.venv/bin/python benchmarks/parte2_espacial.py --scale full
.venv/bin/python benchmarks/parte2_texto.py --output benchmarks/results/parte2_texto/full
.venv/bin/python benchmarks/parte2_audio.py --scale full
.venv/bin/python benchmarks/identificacion_audio.py
```

E6 requiere PostgreSQL disponible y credenciales locales mediante `.pgpass` o `SONORADB_PG_DSN`. El programa usa una base de comparación dedicada. No se incluyen contraseñas. E7 y E8 se ejecutan después de la extracción completa.

| Experimentos | Evaluación |
|---|---|
| E1–E4 | Hasta 500 000 registros; inserción, 1000 igualdades, cinco selectividades y tamaños de página de 1/2/4/8 KB |
| E5 | R-Tree frente a scan, 10 000/50 000/100 000 puntos y dimensiones 2–128 |
| E6 | 1000/5000/10 000/20 000 documentos; booleanos y coseno frente a scan y GIN real |
| E7 | P@5 y P@10 por género, BoAW frente a vectores densos; entrenamiento y consultas de prueba separados |
| E8 | Recall@10 frente a kNN exacto al variar `probes`, con latencia y lecturas de páginas |
| Identificación | Recortes y ruido, acierto de la pista original Top-1/Top-5, separado de la evaluación semántica |

Cada directorio de resultados completos conserva tablas crudas, resúmenes, metodología y gráficos. `--scale smoke`, cuando está disponible, verifica el procedimiento con menos datos y no reemplaza la ejecución completa.

Una lectura/escritura cuenta una transferencia de página realizada por el gestor. El sistema operativo puede mantener páginas en caché: estas cifras no son accesos eléctricos al SSD. El desglose separa datos e índices. En PostgreSQL se reportan por separado bloques leídos y encontrados en buffers; tampoco son métricas físicamente idénticas a las del gestor propio. Las latencias dependen del equipo y de las condiciones de ejecución.

## Archivos y API

`backend/minidb/` contiene almacenamiento, índices, SQL y planificación. `backend/audio/` contiene extracción, vocabulario, DTW y búsqueda. `frontend/` contiene el cliente web. `data/` prepara y configura los datos. `benchmarks/` conserva experimentos y resultados. `tests/` verifica persistencia, equivalencia y casos límite.

Endpoints principales: `POST /api/query`, `GET /api/tables`, `POST /api/tables/reorganize`, `POST /api/spatial/search`, `POST /api/text/search`, `POST /api/audio/search` y `GET /api/benchmarks`. La documentación interactiva está en `/docs`. El servidor se publica únicamente en la interfaz local del equipo.

## Fuentes y licencias

- [FMA, dataset y checksums oficiales](https://github.com/mdeff/fma). Defferrard et al., ISMIR 2017. Los metadatos usan CC BY 4.0; cada audio conserva la licencia elegida por su artista.
- [PANNs, implementación original](https://github.com/qiuqiangkong/audioset_tagging_cnn) y [pesos Cnn6 oficiales](https://zenodo.org/records/3987831). Kong et al. Código MIT y checkpoint CC BY 4.0; atribución en `data/panns_LICENSE.txt`.
- [Leaflet](https://leafletjs.com/), distribuido localmente con su licencia. Cartografía de OpenStreetMap con atribución visible.
