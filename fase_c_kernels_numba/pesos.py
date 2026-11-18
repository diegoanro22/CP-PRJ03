"""Carga los pesos de ``modelos/cnn2.pt`` y fusiona BatchNorm en las convoluciones.

En inferencia, BatchNorm es una transformación afín fija por canal:

    bn(z) = gamma * (z - media) / sqrt(var + eps) + beta

Como ``z = conv(x) + b`` es lineal, se puede absorber en los pesos de la convolución:

    escala = gamma / sqrt(var + eps)
    W' = W * escala            (por canal de salida)
    b' = (b - media) * escala + beta

Así cada bloque conv + BN + ReLU se vuelve UN solo kernel (``k_conv3x3_bn_relu``)
y se ahorra una lectura y escritura completa del tensor por capa. Dropout y
Dropout2d son la identidad en inferencia y desaparecen.

Ambas implementaciones (NumPy CPU y Numba CUDA) usan exactamente estos pesos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from comun import config

EPS = 1e-5
# Índices dentro de CNN2.features de cada bloque doble (conv,bn,relu,conv,bn,relu)
BLOQUES = ("features.0", "features.2", "features.5", "features.8")
POOL_DESPUES = (True, True, True, False)   # tras el último bloque va el promedio global


@dataclass
class PesosCNN2:
    convs: list[tuple[np.ndarray, np.ndarray]]   # 8 x (W' (Cout,Cin,3,3), b' (Cout,))
    lineales: list[tuple[np.ndarray, np.ndarray]]  # 2 x (W (M,K), b (M,))
    labels: list[str]


def _fusionar(sd: dict, conv: str, bn: str) -> tuple[np.ndarray, np.ndarray]:
    w = sd[f"{conv}.weight"].double()
    b = sd[f"{conv}.bias"].double()
    escala = sd[f"{bn}.weight"].double() / torch.sqrt(sd[f"{bn}.running_var"].double() + EPS)
    w_f = w * escala[:, None, None, None]
    b_f = (b - sd[f"{bn}.running_mean"].double()) * escala + sd[f"{bn}.bias"].double()
    return w_f.float().numpy(), b_f.float().numpy()


def cargar_pesos(ruta=config.MODELO_CNN2) -> PesosCNN2:
    ck = torch.load(ruta, map_location="cpu")
    sd = ck["state_dict"]
    convs = []
    for bloque in BLOQUES:
        convs.append(_fusionar(sd, f"{bloque}.0", f"{bloque}.1"))
        convs.append(_fusionar(sd, f"{bloque}.3", f"{bloque}.4"))
    lineales = [(sd[f"classifier.{i}.weight"].numpy(), sd[f"classifier.{i}.bias"].numpy()) for i in (2, 5)]
    return PesosCNN2(convs, lineales, list(ck["labels"]))


def modelo_torch(ruta=config.MODELO_CNN2):
    """CNN2 original con los pesos entrenados, en modo evaluación (referencia de validación)."""
    from secuencial.cnn import CNN2

    ck = torch.load(ruta, map_location="cpu")
    m = CNN2(len(ck["labels"]))
    m.load_state_dict(ck["state_dict"])
    return m.eval()
