# Análisis y línea base secuencial — Mia Fuentes


## Entregables

| Archivo | Contenido |
|---|---|
| `operaciones.md` | Flujo secuencial, operaciones identificadas, repeticiones y costo de CNN2. |
| `contar_operaciones.py` | Conteo automático de imágenes, lotes, repeticiones y FLOPs en CPU. |
| `bench_secuencial.py` | Calentamiento, cinco corridas por variante, monitor, CSV y figuras. |
| `tiempos_secuenciales.md` | Procedimiento, 15 resultados reales, estadísticas e interpretación. |
| `oportunidades.md` | Operaciones paralelizables y asignación técnica CPU/GPU. |
| `discusion_p3.md` | Factores que justifican o limitan CUDA. |

Los resultados de Mia se encuentran en:

- `resultados/tiempos_mia.csv`: 15 filas medidas, cinco por variante.
- `resultados/logs/mia_*.csv`: 18 logs, incluidos tres calentamientos.
- `resultados/figuras/mia_*.png`: barras, boxplot y tres heatmaps de CPU.

## Ejecución desde PowerShell

Desde la raíz del repositorio:

```powershell
py -3.12 -m pip install -r requirements-torch-cpu.txt
py -3.12 -m pip install -r requirements-cpu.txt
py -3.12 -m analisis.contar_operaciones
py -3.12 -m analisis.bench_secuencial
```

El benchmark completo ejecuta una corrida de calentamiento y cinco medidas para cada una de estas variantes:

- `A_preparacion/secuencial`.
- `T_original/cpu_1hilo`.
- `T_original/cpu_nhilos`.

Es seguro ejecutar el último comando nuevamente: detecta cada combinación de fase, variante y corrida ya registrada, no duplica filas y regenera las figuras finales. Para reanudar o ejecutar una sola parte:

```powershell
py -3.12 -m analisis.bench_secuencial --solo preparacion
py -3.12 -m analisis.bench_secuencial --solo cpu_1hilo
py -3.12 -m analisis.bench_secuencial --solo cpu_nhilos
py -3.12 -m analisis.bench_secuencial --solo figuras
```

Si el dataset ya existe en otra ubicación, puede evitarse la descarga de KaggleHub durante la inicialización:

```powershell
$env:ASL_ROOT = 'C:\ruta\a\asl_alphabet_train\asl_alphabet_train'
py -3.12 -m analisis.bench_secuencial
```

La descarga, la instalación y la resolución de rutas ocurren fuera de los bloques cronometrados.

## Estado

- [x] Análisis de `data.py`, `preprocessing.py`, `cnn.py` y `evaluation.py`.
- [x] Dos tablas de operaciones con conteos reales.
- [x] Conteo automático y `FlopCounterMode` en CPU.
- [x] Una corrida de calentamiento y cinco medidas por variante.
- [x] Cuatro épocas completas, semilla 3084 y parámetros compartidos.
- [x] CSV con 15 registros y sin duplicados.
- [x] Logs de CPU/RAM y cinco figuras reales.
- [x] Informe de tiempos con media, desviación, CV y speedup.
- [x] Oportunidades de paralelización.
- [x] Discusión técnica de CUDA.
