# Fase C: kernels CUDA propios con Numba — Diego Rosales (pasos 4 y 5)

PyTorch esconde cómo se reparten hilos, bloques y grids. Aquí se escribe a mano el forward de CNN2
con `numba.cuda` para mostrar ese diseño (paso 4) y medirlo (paso 5). Usa `modelos/cnn2.pt`, así que no
depende del entrenamiento de la fase B.

## Archivos

| Archivo | Contenido |
|---|---|
| `DISENO.md` | Paso 4: mapeo de cada operación a hilos/bloques/grids, con diagrama |
| `kernels.py` | `conv3x3_bn_relu`, `maxpool2x2`, `avgpool_global`, `matmul_tiled` (memoria compartida), `relu` |
| `inferencia_numba.py` | Carga pesos, fusiona BatchNorm en la convolución, corre el forward sobre test |
| `referencia_numpy.py` | Mismo forward en NumPy puro (CPU) para tener un "antes" |
| `bench_kernels.py` | NumPy CPU vs Numba CUDA vs PyTorch/cuDNN, con y sin transferencias |

## Mapeo propuesto

| Operación | Hilo calcula | Bloque | Grid |
|---|---|---|---|
| Conv 3x3 + BN + ReLU | 1 píxel de salida (todos los canales de entrada × 9) | 16×16 | (⌈W/16⌉, ⌈H/16⌉, B·C_out) |
| MaxPool 2x2 | 1 píxel de salida (máximo de 4) | 16×16 | (⌈W_out/16⌉, ⌈H_out/16⌉, B·C) |
| AvgPool global | 1 (imagen, canal) con reducción | 256 | ⌈B·C/256⌉ |
| Lineal (matmul) | 1 elemento de salida, tiles 16×16 en memoria compartida | 16×16 | (⌈out/16⌉, ⌈B/16⌉) |

Notas: CNN2 tiene Dropout (en inferencia es identidad) y BatchNorm (en inferencia es una afinidad que se
fusiona en pesos y sesgo de la conv). Los tensores se quedan en GPU entre capas (`cuda.device_array`);
solo hay una copia H2D al inicio y una D2H de logits al final.

## Checklist

- [ ] `DISENO.md` con el razonamiento de cada kernel (coalescencia: el índice x del hilo recorre el ancho).
- [ ] Validación: logits vs `secuencial.cnn.CNN2` cargado desde `cnn2.pt` (`np.allclose(atol=1e-3)`), predicciones ≥ 99.9 % iguales.
- [ ] Bench con 5 corridas; registrar fase `C_inferencia` con variantes `numpy_cpu`, `numba_cuda`, `torch_cuda`, `numba_cuda_con_transferencias`.
- [ ] Pregunta 1 de la discusión: `discusion_p1.md`.
