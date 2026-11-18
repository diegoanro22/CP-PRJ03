"""Kernels CUDA escritos a mano con Numba para el forward de CNN2 (paso 5).

Cada kernel documenta qué calcula UN hilo, cómo se agrupan los hilos en bloques y
cómo se arma el grid. El diseño completo y su justificación están en ``DISENO.md``.

Convenciones:
* Tensores en formato NCHW, ``float32``, contiguos en memoria de la GPU.
* ``threadIdx.x`` recorre el ANCHO (la dimensión contigua en memoria): hilos vecinos
  de un warp leen direcciones vecinas, así las lecturas a memoria global se coalescen.
* Los lanzadores ``lanzar_*`` calculan grid y bloque y no copian nada a la CPU.
"""

from __future__ import annotations

import math

import numpy as np
from numba import cuda, float32

TILE = 16                   # bloque 2D de 16x16 = 256 hilos (8 warps)
HALO = TILE + 2             # tile de entrada con 1 píxel de borde por lado (kernel 3x3, padding 1)
MAX_GRID_Z = 65535          # límite de gridDim.z en CUDA


# ---------------------------------------------------------------------------
# 1. Normalización: uint8 [0, 255] -> float32 [0, 1]
#    1 hilo = 1 elemento.  Bloque 1D de 256.  Grid 1D de ceil(n/256).
# ---------------------------------------------------------------------------
@cuda.jit
def k_normalizar(entrada_u8, salida):
    i = cuda.grid(1)
    if i < salida.size:
        salida[i] = float32(entrada_u8[i]) / float32(255.0)


# ---------------------------------------------------------------------------
# 2. Convolución 3x3 (stride 1, padding 1) + BatchNorm fusionado + ReLU
#    1 hilo  = 1 píxel de salida (n, co, y, x); recorre Cin x 3 x 3.
#    Bloque  = 16x16 hilos = un tile de 16x16 píxeles de UN canal de salida.
#    Grid    = (ceil(W/16), ceil(H/16), N * Cout).
#    Memoria compartida: por cada canal de entrada el bloque carga juntos un tile
#    de 18x18 (16x16 + borde). Cada píxel de entrada se lee 1 vez de memoria global
#    y se reutiliza hasta 9 veces desde memoria compartida.
#    Los 9 pesos de (co, ci) son iguales para todo el bloque: el bloque también los
#    sube a memoria compartida (lectura difundida, sin divergencia).
# ---------------------------------------------------------------------------
@cuda.jit
def k_conv3x3_bn_relu(entrada, pesos, sesgo, salida, cin, cout):
    tile = cuda.shared.array((HALO, HALO), dtype=float32)
    w9 = cuda.shared.array(9, dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    x = cuda.blockIdx.x * TILE + tx
    y = cuda.blockIdx.y * TILE + ty
    n = cuda.blockIdx.z // cout
    co = cuda.blockIdx.z % cout
    alto = entrada.shape[2]
    ancho = entrada.shape[3]
    x0 = cuda.blockIdx.x * TILE - 1      # esquina superior izquierda del tile con borde
    y0 = cuda.blockIdx.y * TILE - 1
    hilo = ty * TILE + tx

    acc = float32(0.0)
    for ci in range(cin):
        # Carga cooperativa del tile 18x18 (324 valores con 256 hilos: algunos cargan 2).
        for k in range(hilo, HALO * HALO, TILE * TILE):
            yy = y0 + k // HALO
            xx = x0 + k % HALO
            if 0 <= yy < alto and 0 <= xx < ancho:
                tile[k // HALO, k % HALO] = entrada[n, ci, yy, xx]
            else:
                tile[k // HALO, k % HALO] = float32(0.0)   # padding = 1 con ceros
        if hilo < 9:
            w9[hilo] = pesos[co, ci, hilo // 3, hilo % 3]
        cuda.syncthreads()                                 # el tile está completo
        for ky in range(3):
            for kx in range(3):
                acc += tile[ty + ky, tx + kx] * w9[ky * 3 + kx]
        cuda.syncthreads()                                 # nadie pisa el tile hasta que todos terminen

    if x < ancho and y < alto:
        v = acc + sesgo[co]
        salida[n, co, y, x] = v if v > 0 else float32(0.0)   # ReLU


# ---------------------------------------------------------------------------
# 3. MaxPool 2x2, stride 2
#    1 hilo = 1 píxel de salida (máximo de 4 entradas).
#    Bloque 16x16.  Grid (ceil(Wo/16), ceil(Ho/16), N * C).
# ---------------------------------------------------------------------------
@cuda.jit
def k_maxpool2x2(entrada, salida, canales):
    x = cuda.blockIdx.x * TILE + cuda.threadIdx.x
    y = cuda.blockIdx.y * TILE + cuda.threadIdx.y
    n = cuda.blockIdx.z // canales
    c = cuda.blockIdx.z % canales
    if x < salida.shape[3] and y < salida.shape[2]:
        a = entrada[n, c, 2 * y, 2 * x]
        b = entrada[n, c, 2 * y, 2 * x + 1]
        d = entrada[n, c, 2 * y + 1, 2 * x]
        e = entrada[n, c, 2 * y + 1, 2 * x + 1]
        salida[n, c, y, x] = max(max(a, b), max(d, e))


# ---------------------------------------------------------------------------
# 4. AdaptiveAvgPool(1): promedio global de cada canal
#    1 hilo = 1 par (n, c); suma los H*W valores (8x8 = 64 en CNN2).
#    Bloque 1D de 256.  Grid 1D de ceil(N*C/256).
# ---------------------------------------------------------------------------
@cuda.jit
def k_avgpool_global(entrada, salida):
    i = cuda.grid(1)
    canales = entrada.shape[1]
    if i < entrada.shape[0] * canales:
        n = i // canales
        c = i % canales
        s = float32(0.0)
        for yy in range(entrada.shape[2]):
            for xx in range(entrada.shape[3]):
                s += entrada[n, c, yy, xx]
        salida[n, c] = s / float32(entrada.shape[2] * entrada.shape[3])


# ---------------------------------------------------------------------------
# 5. Capa lineal: salida[N, M] = entrada[N, K] @ Wt[K, M] + b  (opcional ReLU)
#    Wt es la matriz de pesos de PyTorch (M, K) ya TRANSPUESTA en la CPU una sola vez:
#    así, al cargar el tile, hilos con tx consecutivo leen columnas consecutivas
#    (direcciones contiguas) y la lectura se coalesce.
#    Multiplicación de matrices con tiling en memoria compartida.
#    1 hilo  = 1 elemento de salida (n, m).
#    Bloque  = 16x16; cada iteración el bloque carga un tile 16x16 de la entrada y
#              otro de W, sincroniza y acumula 16 productos desde memoria compartida.
#    Grid    = (ceil(M/16), ceil(N/16)).
# ---------------------------------------------------------------------------
@cuda.jit
def k_lineal(entrada, pesos_t, sesgo, salida, aplicar_relu):
    sa = cuda.shared.array((TILE, TILE), dtype=float32)
    sw = cuda.shared.array((TILE, TILE), dtype=float32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    m = cuda.blockIdx.x * TILE + tx          # columna de salida (neurona)
    n = cuda.blockIdx.y * TILE + ty          # fila de salida (imagen)
    filas = entrada.shape[0]
    k_total = entrada.shape[1]
    neuronas = pesos_t.shape[1]

    acc = float32(0.0)
    for t in range((k_total + TILE - 1) // TILE):
        ka = t * TILE + tx
        sa[ty, tx] = entrada[n, ka] if (n < filas and ka < k_total) else float32(0.0)
        kw = t * TILE + ty
        mw = cuda.blockIdx.x * TILE + tx
        sw[ty, tx] = pesos_t[kw, mw] if (mw < neuronas and kw < k_total) else float32(0.0)
        cuda.syncthreads()
        for j in range(TILE):
            acc += sa[ty, j] * sw[j, tx]
        cuda.syncthreads()

    if n < filas and m < neuronas:
        v = acc + sesgo[m]
        if aplicar_relu and v < 0:
            v = float32(0.0)
        salida[n, m] = v


# ---------------------------------------------------------------------------
# Lanzadores: calculan la configuración de ejecución <<<grid, bloque>>>
# ---------------------------------------------------------------------------
def config_normalizar(n_elementos: int) -> tuple[int, int]:
    return math.ceil(n_elementos / 256), 256


def config_conv(n: int, cout: int, alto: int, ancho: int) -> tuple[tuple[int, int, int], tuple[int, int]]:
    z = n * cout
    if z > MAX_GRID_Z:
        raise ValueError(f"N*Cout = {z} supera gridDim.z = {MAX_GRID_Z}; use lotes más pequeños")
    return (math.ceil(ancho / TILE), math.ceil(alto / TILE), z), (TILE, TILE)


def lanzar_normalizar(x_u8, salida) -> None:
    grid, bloque = config_normalizar(salida.size)
    k_normalizar[grid, bloque](x_u8.reshape(-1), salida.reshape(-1))


def lanzar_conv(entrada, pesos, sesgo, salida) -> None:
    n, cin, alto, ancho = entrada.shape
    cout = pesos.shape[0]
    grid, bloque = config_conv(n, cout, alto, ancho)
    k_conv3x3_bn_relu[grid, bloque](entrada, pesos, sesgo, salida, cin, cout)


def lanzar_maxpool(entrada, salida) -> None:
    n, c, ho, wo = salida.shape
    grid = (math.ceil(wo / TILE), math.ceil(ho / TILE), n * c)
    k_maxpool2x2[grid, (TILE, TILE)](entrada, salida, c)


def lanzar_avgpool(entrada, salida) -> None:
    total = entrada.shape[0] * entrada.shape[1]
    k_avgpool_global[math.ceil(total / 256), 256](entrada, salida)


def lanzar_lineal(entrada, pesos_t, sesgo, salida, relu: bool) -> None:
    grid = (math.ceil(pesos_t.shape[1] / TILE), math.ceil(entrada.shape[0] / TILE))
    k_lineal[grid, (TILE, TILE)](entrada, pesos_t, sesgo, salida, relu)


def a_dispositivo(arr: np.ndarray):
    return cuda.to_device(np.ascontiguousarray(arr, dtype=np.float32))
