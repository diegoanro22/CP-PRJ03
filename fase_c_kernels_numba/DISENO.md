# Paso 4. Diseño del algoritmo paralelo con CUDA

## Qué se paraleliza y por qué

El paso 3 (`analisis/oportunidades.md`) concluyó que casi todo el costo de CNN2 está en las
ocho convoluciones (unos 380 millones de FLOPs por imagen), y que son operaciones regulares,
con muchos datos independientes y sin control irregular. Este diseño lleva **todo el forward de
la red** a la GPU con kernels propios escritos en Numba CUDA, manteniendo la preparación de
imágenes en la CPU (fase A).

PyTorch y cuDNN deciden internamente cómo repartir el trabajo. Aquí cada decisión se toma a
mano para que quede visible cómo se asignan hilos, bloques y grids.

## Flujo de datos

```
CPU (fase A)                         GPU (fase C)
─────────────                        ──────────────────────────────────────────────────────────
caché uint8 (N,3,64,64) ──H2D──▶ k_normalizar ─▶ conv1 ─▶ conv2 ─▶ maxpool  (64→32)
  4x menos bytes que float32                    ─▶ conv3 ─▶ conv4 ─▶ maxpool  (32→16)
                                                ─▶ conv5 ─▶ conv6 ─▶ maxpool  (16→8)
                                                ─▶ conv7 ─▶ conv8 ─▶ avgpool global
                                                ─▶ lineal 192→128 + ReLU ─▶ lineal 128→29
logits (N,29) ◀──────────D2H───────────────────────────────────────────────────────── ┘
```

Tres decisiones globales:

1. **Los datos se quedan en la GPU entre capas.** Solo hay dos copias por lote: la entrada uint8
   y los logits. Las activaciones intermedias (hasta 128 MB por lote en `conv2`) nunca cruzan PCIe.
2. **Todo se reserva una vez.** Pesos y buffers de activación se crean al inicio y se reutilizan
   en cada lote, así no hay `cudaMalloc` dentro del bucle.
3. **BatchNorm se fusiona en la convolución.** En inferencia BN es una transformación afín fija por
   canal, así que se absorbe en los pesos y el sesgo (`pesos.py`). Cada bloque conv + BN + ReLU es
   **un solo kernel**: se ahorra leer y escribir el tensor completo dos veces por capa. Dropout es la
   identidad en inferencia y desaparece.

## Distribución del trabajo por kernel

| Kernel | Qué calcula un hilo | Bloque | Grid | Memoria compartida |
|---|---|---|---|---|
| `k_normalizar` | 1 elemento: `u8 / 255` | 256 (1D) | ⌈N·3·64·64 / 256⌉ | no |
| `k_conv3x3_bn_relu` | 1 píxel de salida `(n, co, y, x)`, recorre Cin × 3 × 3 | 16 × 16 | (⌈W/16⌉, ⌈H/16⌉, N·Cout) | tile 18 × 18 de entrada + 9 pesos |
| `k_maxpool2x2` | 1 píxel de salida, máximo de 4 entradas | 16 × 16 | (⌈Wo/16⌉, ⌈Ho/16⌉, N·C) | no |
| `k_avgpool_global` | 1 par (imagen, canal): promedio de 8 × 8 = 64 valores | 256 (1D) | ⌈N·C / 256⌉ | no |
| `k_lineal` | 1 elemento de salida `(n, m)` | 16 × 16 | (⌈M/16⌉, ⌈N/16⌉) | 2 tiles 16 × 16 |

### Convolución (el kernel que domina)

* **Un hilo por píxel de salida.** Con lotes de 256 imágenes la primera convolución lanza
  256 × 32 × 64 × 64 = 33.5 millones de hilos, y la GPU tiene trabajo de sobra para ocultar latencias.
* **Bloque de 16 × 16 = 256 hilos (8 warps).** Cubre un tile de 16 × 16 píxeles de **un** canal de
  salida de **una** imagen. 256 hilos es múltiplo de 32 (warps completos) y deja que varios bloques
  convivan en el mismo SM.
* **Grid 3D.** `x` e `y` recorren el tile espacial y `z` codifica la pareja (imagen, canal de salida)
  como `n·Cout + co`. Como `gridDim.z` no puede pasar de 65,535, el lote máximo para la capa de 192
  canales es ⌊65,535 / 192⌋ = 341 imágenes. Se usan lotes de 256, igual que en el entrenamiento.
* **Memoria compartida.** Para cada canal de entrada el bloque carga en cooperación un tile de
  18 × 18 (16 × 16 más un borde de 1 píxel para el kernel 3 × 3 con padding 1), sincroniza con
  `cuda.syncthreads()` y cada hilo hace sus 9 multiplicaciones desde memoria compartida. Cada valor
  se lee **una vez** de memoria global y se reutiliza hasta 9 veces. Los 9 pesos `(co, ci)` son iguales
  para todo el bloque, así que también se suben a memoria compartida y se leen sin divergencia.
* **Coalescencia.** `threadIdx.x` recorre el ancho, que es la dimensión contigua en NCHW: los 32 hilos
  de un warp leen 32 direcciones consecutivas en una sola transacción.
* **Padding sin ramas en el cálculo.** Los bordes se resuelven al cargar el tile (fuera de la imagen se
  escribe 0). El bucle de 9 multiplicaciones no tiene condicionales.
* Los hilos que caen fuera de la imagen **también** cargan y sincronizan; solo se omite su escritura
  final. Si salieran antes, `syncthreads` quedaría bloqueado.

### Capas lineales

Multiplicación de matrices con *tiling*: cada bloque de 16 × 16 calcula un tile de la salida.
En cada iteración carga un tile de la entrada y otro de los pesos en memoria compartida, sincroniza
y acumula 16 productos. La matriz de pesos de PyTorch `(M, K)` se transpone **una vez** en la CPU a
`(K, M)` para que la carga del tile también sea coalescida.

### Pooling

Son operaciones limitadas por memoria con poca aritmética: un hilo por salida, sin memoria compartida.
El promedio global usa un hilo por (imagen, canal) porque solo suma 64 valores.

## Resultados del diseño (RTX 5070 Laptop, lote de 256)

Medido con eventos CUDA (`bench_kernels.py`, `resultados/rosales_kernels_desglose.csv`):

| Kernel | Grid × bloque | ms por lote | % |
|---|---|---:|---:|
| conv1 3→32 @64×64 | (4, 4, 8192) × (16, 16) | 2.49 | 1.6 |
| conv2 32→32 @64×64 | (4, 4, 8192) × (16, 16) | 21.38 | 13.5 |
| conv3 32→64 @32×32 | (2, 2, 16384) × (16, 16) | 10.38 | 6.5 |
| conv4 64→64 @32×32 | (2, 2, 16384) × (16, 16) | 20.72 | 13.1 |
| conv5 64→128 @16×16 | (1, 1, 32768) × (16, 16) | 10.27 | 6.5 |
| conv6 128→128 @16×16 | (1, 1, 32768) × (16, 16) | 20.97 | 13.2 |
| conv7 128→192 @8×8 | (1, 1, 49152) × (16, 16) | 29.43 | 18.6 |
| conv8 192→192 @8×8 | (1, 1, 49152) × (16, 16) | 41.89 | 26.4 |
| normalizar, 3 maxpool, avgpool, 2 lineales | | 1.02 | 0.6 |

Las convoluciones son el 99.4 % del tiempo, como predijo el análisis de FLOPs del paso 1.

**Hallazgo: el bloque fijo de 16 × 16 desperdicia hilos en las capas pequeñas.** `conv7` y `conv8`
trabajan sobre mapas de 8 × 8, pero el bloque es de 16 × 16: solo 64 de los 256 hilos de cada bloque
calculan algo y el **75 %** queda ocioso (aunque sí carga el tile y sincroniza). Por eso esas dos capas
toman el 45 % del tiempo, aunque solo hacen el 19 % de los FLOPs (unos 71 de 380 millones por imagen).
La mejora natural sería un bloque que se adapte al tamaño del mapa (8 × 8 en esas capas) o que cada
bloque calcule varios canales de salida a la vez, reutilizando el mismo tile de entrada.

**Comparación con cuDNN.** El forward completo con estos kernels tarda 1.60 s para las 2,610 imágenes
de test (29× más rápido que NumPy en la CPU con 32 hilos de BLAS). PyTorch con cuDNN tarda 0.22 s, unas
7× menos. cuDNN usa algoritmos que estos kernels no implementan: transformaciones Winograd/GEMM implícita,
varios canales de salida por bloque, uso de registros para reutilizar datos y selección del mejor
algoritmo según la forma de cada capa (`cudnn.benchmark`). La lección del diseño es doble: CUDA reduce el
tiempo en un orden de magnitud aun con kernels sencillos, y las librerías especializadas aportan otro
orden de magnitud. Por eso la fase B entrena con PyTorch/cuDNN en vez de con estos kernels.

**Corrección.** Con los mismos pesos, los logits coinciden con la CNN2 original de PyTorch (diferencia
máxima de 1e-5), el 100 % de las predicciones son iguales y el accuracy en test es 0.8977, el mismo del
Laboratorio 3. La paralelización cambió el tiempo, no el resultado.
