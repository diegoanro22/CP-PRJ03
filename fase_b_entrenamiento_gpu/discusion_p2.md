# Discusión técnica, pregunta 2: ¿cómo CUDA permitió reducir o cambiar el tiempo?

Diego Lopez, fase B (paso 5). Todo lo de abajo sale de `resultados/tiempos_lopez.csv`,
`resultados/lopez_metricas.csv`, `resultados/lopez_barrido_batch.csv` y los logs de `resultados/logs/`.

## Condiciones de la medición

| | |
|---|---|
| Hardware | Intel (20 hilos lógicos), NVIDIA GeForce RTX 5070 Laptop 8 GB, driver 596.49, Windows 11 |
| Software | Python 3.12, PyTorch 2.14.1+cu130 (CUDA 13.0, cuDNN 9.24) |
| Carga | CNN2, 4 épocas fijas, batch 256, lr 1e-3, semilla 3084, 12,180 imágenes de train + 2,610 de validación por época |
| Método | 1 corrida de calentamiento (descartada) + 5 medidas por variante; mismo bucle y misma caché en memoria en las cuatro variantes, solo cambian dispositivo, hilos y AMP |

La caché en memoria significa que **no se abre ningún JPG dentro del bucle**: este speedup mide solo la matemática de la red.
La preparación de imágenes es la fase A (otra persona) y el pipeline original con JPG en cada época es `T_original` (Mia, otra máquina); no se mezclan.

## Resultado

| Variante | Media (s) | Desv. (s) | CV | Speedup vs `cpu_1hilo` | Speedup vs `cpu_nhilos` |
|---|---:|---:|---:|---:|---:|
| `cpu_1hilo` | 723.78 | 5.21 | 0.7 % | 1.0x | n/a |
| `cpu_nhilos` (20 hilos) | 192.84 | 18.03 | 9.3 % | 3.8x | 1.0x |
| `cuda` (fp32) | 11.80 | 0.05 | 0.4 % | **61.3x** | **16.3x** |
| `cuda_amp` (bf16) | 6.99 | 0.01 | 0.1 % | **103.5x** | **27.6x** |

`cuda_amp` es 1.69x más rápida que `cuda` fp32.

Dos lecturas del speedup, y conviene reportar las dos: 61x es contra un solo hilo, la base fijada en el plan;
**16x contra la CPU usando todos sus hilos es la comparación más honesta** de "qué ganó la GPU".

## Por qué CUDA cambió el tiempo

**1. El trabajo es el que la GPU sabe hacer.** Las convoluciones y las capas lineales son miles de multiplicaciones-sumas
idénticas sobre datos independientes (cada píxel de salida, cada imagen del lote). Una CPU las reparte en 20 hilos; la GPU
las reparte en miles de hilos CUDA a la vez.

**2. El cuello de botella cambió de lugar.** En GPU, el 99.6 % del tiempo es cómputo y la transferencia es casi nula:

| | `cuda` | `cuda_amp` |
|---|---:|---:|
| Transferencia H2D (eventos CUDA) | 0.044 s | 0.041 s |
| Cómputo (forward + backward + step, eventos CUDA) | 11.749 s | 6.942 s |

Las copias cuestan ~0.4 % del tiempo. Las dos decisiones de diseño que lo explican son que los lotes viajan como `uint8`
(el `/255` se hace ya en la GPU) y que se copian desde memoria *pinned* con `non_blocking=True`, solapadas con el
cómputo del lote anterior. Con las cifras medidas: unos 0.73 GB por entrenamiento en 0.044 s, del orden de 16 GB/s
(cálculo mío a partir de los tamaños de los tensores). No medí la alternativa en `float32` sin pinned, así que el efecto
exacto de cada decisión por separado **no está medido**.

**3. La GPU quedó saturada y la CPU casi libre.** `nvidia-smi dmon` durante un entrenamiento de 15 épocas dio `sm` medio
de 98.2 % (52 muestras de 1 s) y el volcado de `nvidia-smi` mostró 99 % de utilización con 2,356 MiB.
Sumando el % de los 20 núcleos lógicos en los logs del monitor (`resultados/lopez_uso_recursos.csv`,
`resultados/figuras/lopez_uso_cpu_gpu.png`), promedio de las 5 corridas:

| Variante | Núcleos equivalentes ocupados (de 20) | Util. GPU media |
|---|---:|---:|
| `cpu_1hilo` | 1.79 | 8.7 % |
| `cpu_nhilos` | 19.99 | 1.0 % |
| `cuda` | 1.91 | 98.8 % |
| `cuda_amp` | 1.88 | 98.9 % |

Con `cuda` la CPU equivale a unos 2 núcleos ocupados mientras la GPU trabaja al ~99 %: es el pipeline pedido, la CPU
junta y reparte datos y la GPU hace la matemática. Dos aclaraciones sobre los heatmaps (`monitor_lopez_*.png`):
en `cpu_nhilos` los 20 núcleos están al ~100 % toda la corrida, pero en `cpu_1hilo` **no** se ve "un núcleo al 100 %":
Windows va moviendo el único hilo entre núcleos, por eso la carga aparece repartida en varias bandas a niveles medios
(el 1.79 incluye además ruido de fondo del sistema). La utilización de GPU de 8.7 % en `cpu_1hilo` es del escritorio y
otras aplicaciones, no del entrenamiento.

**4. AMP agrega 1.69x.** Con `autocast` bf16 las convoluciones y matmuls usan Tensor Cores y mueven la mitad de bytes.
El costo está en la sección de abajo.

## Lo que no salió como se esperaba (y no se debe esconder)

**El barrido de batch contradice la intuición "más lote, más ocupación, más rápido".**

| Batch | s por época | Util. GPU media | Memoria máx. | Test acc (media de 5) |
|---:|---:|---:|---:|---:|
| 64 | 2.72 | 96.6 % | 1,135 MiB | 0.941 |
| 128 | 2.91 | 98.8 % | 1,381 MiB | 0.895 |
| 256 | 3.06 | 98.7 % | 2,393 MiB | 0.887 |
| 512 | 3.10 | 99.8 % | 3,826 MiB | 0.764 |

La GPU ya está casi al 97 % con batch 64, así que no había ocupación que ganar: CNN2 sobre 64x64 es cómputo puro y los
lanzamientos de kernels no dominan. El tiempo incluso sube un poco con el batch. Con lr fija, el batch grande hace menos
pasos de optimización en 4 épocas, y por eso el accuracy cae (0.764 con batch 512). Una limitación del barrido: los
tamaños corrieron en orden creciente en una laptop, y no descarto que la temperatura contribuya a la subida del tiempo;
no lo medí.

**AMP no es gratis en precisión.** Con la semilla fija el resultado de `cuda_amp` es determinista (0.8333 en las 5
corridas del benchmark; 0.7897 en otro proceso: probablemente por la elección de algoritmos de cuDNN, **no verificado**).
En una prueba aparte con otras semillas, `cuda_amp` dio 0.822 a 0.900 y fp32 dio 0.889 a 0.933: en 4 épocas AMP queda
unos puntos por debajo en promedio. Son pocas muestras y con mucho ruido, así que es una tendencia, no una conclusión.
Los 103x deben citarse junto con esta salvedad.

**La CPU multihilo escala mal.** 20 hilos dan solo 3.8x sobre 1 hilo (eficiencia de ~19 %). No investigué por qué:
que las capas sean pequeñas y que haya sobrecosto de sincronización entre hilos son hipótesis, no medidas.

## Equivalencia de resultados

| Variante | Test accuracy | F1 macro |
|---|---:|---:|
| Original (Lab 3) | 0.898 | n/d |
| `cpu_1hilo` | 0.9123 | 0.9117 |
| `cpu_nhilos` | 0.8981 | 0.8961 |
| `cuda` (media de 5; mín–máx) | 0.8986 (0.862–0.924) | 0.8965 |
| `cuda_amp` | 0.8333 | 0.8264 |

`cuda` fp32 queda en el rango del original, pero la variación entre corridas (0.86 a 0.92) es grande: con 4 épocas el
modelo aún no converge. Con 15 épocas, una sola corrida en GPU llegó a **0.9981** de accuracy en test
(`resultados/logs/lopez_entrenamiento_cuda_evidencia.txt`). Las CPU dan resultados distintos entre sí con la misma
semilla (0.9123 vs 0.8981); la causa probable es que otro número de hilos cambia el orden de las sumas en coma flotante,
pero no lo verifiqué. "Mismo resultado" aquí significa "mismo rango de accuracy", no bit a bit.

## Notas de procedimiento

- **Una corrida de `cpu_1hilo` se descartó:** la laptop se suspendió ~50 min durante ella y el cronómetro contó ese tiempo
  (el log del monitor mostraba un salto de 3,009 s). Se repitió la variante completa con bloqueo de suspensión y se
  verificó que ningún log de las 5 corridas finales tenga saltos mayores a 0.53 s.
- **`cpu_nhilos` tiene CV de 9.3 %** por la corrida 5 (224.6 s contra 181–188 s en las otras cuatro), que coincidió con
  un periodo de poca memoria libre en el equipo. Se conservó en la tabla. Sin ella la media sería 184.9 s y los speedups
  de GPU contra `cpu_nhilos` serían 15.7x (fp32) y 26.5x (AMP).
- Todas las medidas son de **una sola máquina** (laptop); los speedups no se comparan con los de otros equipos.
- Evidencia de ejecución (todo sale de datos reales medidos durante el entrenamiento):
  `resultados/logs/lopez_dmon.txt` y `resultados/figuras/lopez_dmon_gpu.png` (`nvidia-smi dmon`, 1 muestra por segundo),
  `resultados/capturas/lopez_nvidia_smi_entrenando.txt` y `.png`, `resultados/figuras/monitor_lopez_*.png`,
  `resultados/figuras/lopez_uso_cpu_gpu.png` y `resultados/figuras/lopez_barrido_batch.png`.
  **Captura de pantalla real:** `resultados/capturas/lopez_taskmgr_gpu_entrenando.png` (Administrador de tareas,
  2026-11-17 00:06:44, durante un entrenamiento `cuda` de 20 épocas que duró 66.2 s y llegó a 0.9981 de accuracy en test,
  log en `resultados/logs/lopez_entrenamiento_captura.txt`): GPU NVIDIA al 100 %, 3.2 de 8.0 GB de memoria dedicada y CPU
  al 7 %. Esa memoria (3.2 GB) es la de todo el sistema, así que no es comparable con los 2,356 MiB de `nvidia-smi` de otro
  entrenamiento. La temperatura de la GPU marcó 80 °C en un entrenamiento sostenido de ~66 s (62 °C en el volcado de
  `nvidia-smi` de la corrida anterior); no medí si hubo reducción de frecuencia por temperatura.
  `lopez_nvidia_smi_entrenando.png` es la salida real de `nvidia-smi` renderizada como imagen (sin la tabla de procesos),
  **no una captura de pantalla**. Las gráficas se generaron desde los logs del monitor.
