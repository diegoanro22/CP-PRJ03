"""Ubica el dataset ASL en la máquina de cada integrante.

Los manifiestos guardan rutas relativas (p. ej. ``S/S369.jpg``). La raíz se
toma de la variable de entorno ``ASL_ROOT`` o, si no existe, se descarga con
kagglehub (queda en caché y no se vuelve a bajar).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from comun import config


@lru_cache(maxsize=1)
def raiz_dataset() -> Path:
    """Carpeta que contiene las subcarpetas A, B, ..., SPACE."""
    env = os.environ.get("ASL_ROOT")
    if env:
        base = Path(env).expanduser()
    else:
        import kagglehub
        base = Path(kagglehub.dataset_download(config.KAGGLE_HANDLE))
    for candidata in (base, base / "asl_alphabet_train" / "asl_alphabet_train", base / "asl_alphabet_train"):
        if (candidata / "A").is_dir() and (candidata / "Z").is_dir():
            return candidata.resolve()
    raise FileNotFoundError(f"No se encontraron las carpetas de clases dentro de {base}")


def cargar_manifiesto(split: str) -> pd.DataFrame:
    """Devuelve el manifiesto con columnas ``filepath`` (absoluta) y ``label``.

    Es el mismo formato que espera el código de ``secuencial/``.
    """
    df = pd.read_csv(config.MANIFIESTOS / f"{split}.csv")
    raiz = raiz_dataset()
    df["filepath"] = [str(raiz / r) for r in df["ruta_relativa"]]
    return df[["filepath", "label"]]


def cargar_splits() -> dict[str, pd.DataFrame]:
    return {s: cargar_manifiesto(s) for s in config.SPLITS}
