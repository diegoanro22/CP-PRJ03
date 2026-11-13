# Proyecto 3: Mejorando algoritmos con CUDA (CC3069)

Aceleración del Laboratorio 3 de Data Science (clasificación de ASL con CNN2) separando el pipeline:
**CPU multinúcleo para preparar las imágenes** y **GPU con CUDA para la matemática de la red**.

Integrantes: Diego Lopez, Diego Rosales, Mia Fuentes.

## Estructura

| Carpeta | Responsable | Qué contiene |
|---|---|---|
| `secuencial/` | (copia, no se toca) | Código original del Lab 3: línea base |
| `comun/` | (contrato) | Configuración, rutas del dataset, caché, medición y monitor de recursos |
| `analisis/` | Mia | Pasos 1, 2 y 3: operaciones, tiempos secuenciales, oportunidades |
| `fase_a_preparacion_cpu/` | Rosales | Preparación de imágenes con multiprocessing |
| `fase_b_entrenamiento_gpu/` | Lopez | Entrenamiento de CNN2 en CUDA (paso 5) |
| `fase_c_kernels_numba/` | Rosales | Diseño (paso 4) y kernels CUDA propios con Numba |
| `resultados/` | todos | `tiempos_<persona>.csv`, `logs/`, `figuras/`, `capturas/` |
| `Proyecto3_CUDA.ipynb` | Rosales (integración) | Junta resultados, tabla comparativa y speedups |

Cada quien trabaja en su rama y **solo en su carpeta** (más su `resultados/tiempos_<persona>.csv`).
`comun/` y `secuencial/` no se modifican sin avisar al equipo.

## Instalación

```bash
uv venv --python 3.12
source .venv/bin/activate
# GPU (Lopez, Rosales):
uv pip install -r requirements-torch-gpu.txt
uv pip install -r requirements-gpu.txt
# Solo CPU (Mia):
uv pip install -r requirements-torch-cpu.txt
uv pip install -r requirements-cpu.txt
```

Dataset: la primera vez se descarga con kagglehub (requiere sesión de Kaggle). Si ya lo tienen en otra
carpeta: `export ASL_ROOT=/ruta/a/asl_alphabet_train`.

Todos los scripts se corren desde la raíz como módulos:

```bash
python -m comun.datos_seq                              # caché secuencial (cache/*.npy)
python -m analisis.bench_secuencial                    # Mia
python -m fase_b_entrenamiento_gpu.bench_entrenamiento # Lopez
python -m fase_a_preparacion_cpu.bench_preparacion     # Rosales
python -m fase_c_kernels_numba.bench_kernels           # Rosales
```

## Contrato de resultados

`resultados/tiempos_<persona>.csv`, escrito con `comun.medicion.registrar`:

`fase, variante, dispositivo, corrida, tiempo_s, n_imagenes, workers_o_hilos, batch, epocas, hardware, fecha`

Fases y variantes **obligatorias** (`comun/config.py::VOCABULARIO`; `registrar` rechaza cualquier otro nombre):

| Fase | Variantes | Quién | Base del speedup |
|---|---|---|---|
| `T_original` | `cpu_1hilo`, `cpu_nhilos` | Mia | `cpu_1hilo` |
| `A_preparacion` | `secuencial`, `procesos_<p>` | Mia (secuencial), Rosales | `secuencial` |
| `B_entrenamiento` | `cpu_1hilo`, `cpu_nhilos`, `cuda`, `cuda_amp` | Lopez | `cpu_1hilo` |
| `B_transferencia`, `B_computo` | `cuda`, `cuda_amp` | Lopez | (desglose) |
| `B_barrido_batch` | `batch_<n>` | Lopez | (curva) |
| `C_inferencia` | `numpy_cpu`, `numba_cuda`, `numba_cuda_con_transferencias`, `torch_cuda` | Rosales | `numpy_cpu` |

- `T_original` es el pipeline original tal cual (JPG en cada época). En `B_entrenamiento` todas las variantes
  leen la misma caché en memoria, así el speedup mide solo la matemática de la red.
- Los speedups se calculan **dentro de cada máquina** (columna `hardware`); nunca se mezclan equipos.
- Speedup total = `T_original/cpu_1hilo` ÷ (mejor `A_preparacion/procesos_<p>` + mejor `B_entrenamiento` en GPU).
  La tabla oficial se mide en la laptop de Rosales.
