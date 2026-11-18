# Datos

Al iniciar el sistema se generan 100 000 empleados sintéticos reproducibles en dos tablas: Heap y Sequential. Los archivos binarios del motor se guardan en `runtime/` y se regeneran localmente; no se versionan.

La segunda parte añade 100 000 puntos simulados de Lima, 20 000 documentos de los metadatos de FMA y las pistas de FMA Small. `prepare_fma.py` descarga y verifica los archivos oficiales si faltan. También se puede utilizar una copia existente configurando `.env`, como se explica en el README principal.

Los MP3 y metadatos originales, los pesos de PANNs y las características intermedias se conservan localmente. La carpeta `cache/` permite reanudar la extracción. Los índices de la aplicación se construyen en `runtime/`; las consultas leen sus páginas binarias, no los archivos de características intermedias.

Los experimentos generan sus propios conjuntos y dejan las tablas de mediciones y gráficos en `benchmarks/results/`. Los datos voluminosos, modelos, cachés y archivos binarios generados no se suben al repositorio.
