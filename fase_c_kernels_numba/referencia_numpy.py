"""Forward de CNN2 en NumPy puro sobre CPU: el "antes" de la fase C.

Usa los mismos pesos fusionados que los kernels (``pesos.py``). La convolución se
hace con im2col (``sliding_window_view``) seguida de un producto de matrices
(``np.tensordot`` → BLAS). Es la forma razonable de escribirla en NumPy: un bucle
Python por píxel sería absurdamente lento y no sería una comparación honesta.

Nota: el BLAS de NumPy puede usar varios hilos; el número usado queda registrado
en ``workers_o_hilos`` del CSV.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from fase_c_kernels_numba.pesos import POOL_DESPUES, PesosCNN2


def conv3x3_bn_relu(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x (N,Cin,H,W), w (Cout,Cin,3,3) -> (N,Cout,H,W) con padding 1 y ReLU."""
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    ventanas = sliding_window_view(xp, (3, 3), axis=(2, 3))          # (N,Cin,H,W,3,3)
    y = np.tensordot(ventanas, w, axes=([1, 4, 5], [1, 2, 3]))       # (N,H,W,Cout)
    y += b
    np.maximum(y, 0, out=y)
    return np.ascontiguousarray(y.transpose(0, 3, 1, 2))


def maxpool2x2(x: np.ndarray) -> np.ndarray:
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def forward(x: np.ndarray, p: PesosCNN2) -> np.ndarray:
    """x float32 (N,3,64,64) en [0,1] -> logits (N,29)."""
    for i, (w, b) in enumerate(p.convs):
        x = conv3x3_bn_relu(x, w, b)
        if i % 2 == 1 and POOL_DESPUES[i // 2]:
            x = maxpool2x2(x)
    x = x.mean(axis=(2, 3))                                           # AdaptiveAvgPool(1)
    (w1, b1), (w2, b2) = p.lineales
    x = np.maximum(x @ w1.T + b1, 0)
    return x @ w2.T + b2


def predecir(x_u8_nchw: np.ndarray, p: PesosCNN2, lote: int = 256) -> np.ndarray:
    """Logits para todo el conjunto, procesado por lotes desde la caché uint8."""
    salidas = []
    for i in range(0, len(x_u8_nchw), lote):
        x = x_u8_nchw[i:i + lote].astype(np.float32) / np.float32(255.0)
        salidas.append(forward(x, p))
    return np.concatenate(salidas)
