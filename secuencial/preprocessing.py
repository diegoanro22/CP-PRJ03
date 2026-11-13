"""Preprocesamiento reproducible para las imágenes de ASL Alphabet.

Las funciones de este módulo consumen las particiones creadas en ``src.data``.
No crean un split nuevo ni aplican aumentación: esta última se reserva para los
experimentos de entrenamiento, exclusivamente sobre el conjunto de train.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_IMAGE_SIZE = (64, 64)
SPLIT_NAMES = ("train", "validation", "test")
REQUIRED_COLUMNS = {"filepath", "label"}


def _path_key(raw_path: str | Path) -> str:
    """Normaliza una ruta para detectar duplicados también en Windows."""
    return str(Path(raw_path).expanduser().resolve()).casefold()


def validate_split_integrity(splits: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Verifica columnas, clases y ausencia de fuga entre particiones."""
    missing_splits = set(SPLIT_NAMES).difference(splits)
    if missing_splits:
        raise ValueError(f"Faltan particiones: {sorted(missing_splits)}")

    normalized_paths: dict[str, set[str]] = {}
    class_sets: dict[str, set[str]] = {}
    for name in SPLIT_NAMES:
        frame = splits[name]
        missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"{name}.csv no contiene: {sorted(missing_columns)}")
        if frame.empty:
            raise ValueError(f"{name}.csv está vacío.")
        if frame[["filepath", "label"]].isna().any().any():
            raise ValueError(f"{name}.csv contiene rutas o etiquetas vacías.")

        keys = frame["filepath"].map(_path_key)
        if keys.duplicated().any():
            raise ValueError(f"{name}.csv contiene imágenes duplicadas.")
        normalized_paths[name] = set(keys)
        class_sets[name] = set(frame["label"].astype(str))

    for index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[index + 1:]:
            overlap = normalized_paths[left_name].intersection(normalized_paths[right_name])
            if overlap:
                raise ValueError(
                    f"Hay {len(overlap)} imágenes compartidas entre {left_name} y {right_name}."
                )

    reference_classes = class_sets["train"]
    for name in SPLIT_NAMES[1:]:
        if class_sets[name] != reference_classes:
            raise ValueError(f"Las clases de {name} no coinciden con las de train.")

    return {
        "split_sizes": {name: len(splits[name]) for name in SPLIT_NAMES},
        "n_classes": len(reference_classes),
        "has_overlap": False,
    }


def load_split_manifests(processed_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Carga los tres CSV oficiales y comprueba que sean compatibles."""
    processed_dir = Path(processed_dir)
    splits: dict[str, pd.DataFrame] = {}
    for name in SPLIT_NAMES:
        csv_path = processed_dir / f"{name}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"No se encontró el manifiesto: {csv_path}")
        splits[name] = pd.read_csv(csv_path)
    validate_split_integrity(splits)
    return splits


def preprocess_image(
    image_path: str | Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Convierte una imagen a RGB, la redimensiona y normaliza a ``[0, 1]``.

    El resultado tiene forma ``(alto, ancho, 3)`` y tipo ``float32``. No se
    ajusta ningún parámetro con el dataset, de modo que la misma transformación
    puede aplicarse sin fuga a train, validation, test y fotos externas.
    """
    if len(image_size) != 2 or any(size <= 0 for size in image_size):
        raise ValueError("image_size debe contener ancho y alto positivos.")

    with Image.open(image_path) as image:
        resized = image.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32)
    return pixels / np.float32(255.0)


def preprocess_batch(
    image_paths: Iterable[str | Path],
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Preprocesa una colección pequeña de rutas y la apila en un lote."""
    images = [preprocess_image(path, image_size=image_size) for path in image_paths]
    if not images:
        raise ValueError("image_paths debe contener al menos una ruta.")
    return np.stack(images)
