"""Preparación SECUENCIAL de imágenes hacia la caché compartida.

Formato de la caché (contrato, la fase A debe producir exactamente lo mismo,
LOS DOS archivos por split, con el sufijo "_par"):
    cache/<split>_x.npy  uint8   (N, 64, 64, 3)   NHWC, píxeles RGB sin normalizar
    cache/<split>_y.npy  int64   (N,)             índice en config.CLASES

Ojo: la caché es NHWC (así la entrega PIL) y CNN2 espera NCHW. Para entrenar
use ``cargar_tensores``, que ya hace el permute y deja uint8 (la división /255
se hace en el dispositivo).

Se guarda en uint8 (4x menos memoria que float32); la división /255 se hace
en el dispositivo de entrenamiento. El resize es el mismo del original:
RGB + BILINEAR a 64x64.

Uso:  python -m comun.datos_seq
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

from comun import config
from comun.rutas import cargar_manifiesto


def preparar_imagen(ruta: str, lado: int = config.IMG) -> np.ndarray:
    """Abre, convierte a RGB y redimensiona. Misma operación que el original."""
    with Image.open(ruta) as img:
        return np.asarray(img.convert("RGB").resize((lado, lado), Image.Resampling.BILINEAR), dtype=np.uint8)


def etiquetas_a_indices(labels) -> np.ndarray:
    idx = {c: i for i, c in enumerate(config.CLASES)}
    return np.array([idx[l] for l in labels], dtype=np.int64)


def preparar_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    df = cargar_manifiesto(split)
    x = np.stack([preparar_imagen(r) for r in df["filepath"]])
    return x, etiquetas_a_indices(df["label"])


def guardar(split: str, x: np.ndarray, y: np.ndarray, sufijo: str = "") -> None:
    config.CACHE.mkdir(parents=True, exist_ok=True)
    np.save(config.CACHE / f"{split}{sufijo}_x.npy", x)
    np.save(config.CACHE / f"{split}{sufijo}_y.npy", y)


def cargar_cache(split: str, sufijo: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Lee la caché; ``sufijo`` permite elegir otra versión (p. ej. la de la fase A)."""
    x_path = config.CACHE / f"{split}{sufijo}_x.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Falta {x_path}. Corra primero: python -m comun.datos_seq")
    return np.load(x_path), np.load(config.CACHE / f"{split}{sufijo}_y.npy")


def cargar_tensores(split: str, sufijo: str = ""):
    """Caché como tensores de torch listos para CNN2.

    Devuelve ``x`` uint8 (N, 3, 64, 64) contiguo en NCHW e ``y`` int64 (N,).
    En el dispositivo: ``x.float().div_(255)`` reproduce la normalización original.
    """
    import torch

    x, y = cargar_cache(split, sufijo)
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(y)


def main() -> None:
    for split in config.SPLITS:
        t0 = time.perf_counter()
        x, y = preparar_split(split)
        guardar(split, x, y)
        print(f"{split}: {x.shape} en {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
