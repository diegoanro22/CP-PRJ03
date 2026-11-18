# Fase C: kernels CUDA propios con Numba — Diego Rosales (pasos 4 y 5)

Forward completo de CNN2 escrito a mano con `numba.cuda`, usando los pesos entrenados de
`modelos/cnn2.pt`. Muestra explícitamente cómo se reparten hilos, bloques y grids (paso 4),
algo que PyTorch esconde.

## Archivos

| Archivo | Contenido |
|---|---|
| `DISENO.md` | **Paso 4**: mapeo de cada operación a hilos, bloques y grids, memoria compartida, resultados |
| `kernels.py` | `k_normalizar`, `k_conv3x3_bn_relu` (tile 18×18 en memoria compartida), `k_maxpool2x2`, `k_avgpool_global`, `k_lineal` (tiling) |
| `pesos.py` | Carga `cnn2.pt` y fusiona BatchNorm en los pesos de cada convolución |
| `inferencia_numba.py` | Forward en GPU con buffers reservados una vez; validación contra PyTorch |
| `referencia_numpy.py` | El mismo forward en NumPy (im2col + BLAS) en CPU: el "antes" |
| `bench_kernels.py` | NumPy CPU vs Numba CUDA vs PyTorch/cuDNN, 5 corridas, y desglose por kernel con eventos CUDA |
| `discusion_p1.md` | **Pregunta 1**: por qué los archivos van al CPU y la red al GPU |

```bash
python -m fase_c_kernels_numba.inferencia_numba   # validación
python -m fase_c_kernels_numba.bench_kernels      # tiempos
```

## Resultados (RTX 5070 Laptop, 2,610 imágenes de test, lotes de 256)

| Variante | Media (s) | Imágenes/s | Speedup vs NumPy |
|---|---:|---:|---:|
| `numpy_cpu` (32 hilos BLAS) | 46.31 | 56 | 1× |
| `numba_cuda` (solo cómputo) | 1.60 | 1,636 | **29×** |
| `numba_cuda_con_transferencias` | 1.60 | 1,628 | 29× |
| `torch_cuda` (cuDNN, fp32) | 0.22 | 11,674 | 207× |

- **Corrección:** logits iguales a PyTorch (diferencia máxima 1e-5), 100 % de predicciones iguales,
  accuracy en test 0.8977 (el mismo del Laboratorio 3).
- Las transferencias casi no pesan: la entrada viaja como `uint8` y entre capas no hay copias.
- Las convoluciones son el 99.4 % del tiempo; `conv7` y `conv8` (mapas de 8×8 con bloques de 16×16,
  75 % de hilos ociosos) toman el 45 %. Análisis completo en `DISENO.md`.
- Detalle: `resultados/rosales_kernels.csv`, `resultados/rosales_kernels_desglose.csv` y figuras `rosales_kernels_*.png`.
