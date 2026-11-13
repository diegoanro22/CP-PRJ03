"""Carga, validación y división reproducible del dataset ASL Alphabet.

Este módulo no entrena modelos: prepara los CSV de manifiesto que utilizarán
los demás notebooks y scripts. Así se evita que cada modelo use una partición
distinta o que haya fuga de datos entre entrenamiento y prueba.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXPECTED_CLASSES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ("SPACE", "DELETE", "NOTHING")
KAGGLE_DATASET_HANDLE = "grassknoted/asl-alphabet"
LABEL_ALIASES = {"DEL": "DELETE"}


def canonical_label(folder_name: str) -> str:
    """Normaliza los nombres de carpetas de Kaggle al vocabulario del informe."""
    name = folder_name.upper()
    return LABEL_ALIASES.get(name, name)


def download_asl_alphabet(output_dir: str | Path = "data/raw") -> Path:
    """Descarga ASL Alphabet con KaggleHub y devuelve la ruta local.

    KaggleHub utiliza su caché local extraída y no vuelve a descargar el dataset
    si ya existe. ``output_dir`` se conserva por compatibilidad con llamadas
    anteriores, pero no se pasa a KaggleHub: con ``output_dir`` KaggleHub hace
    una descarga directa como archivo y no garantiza una carpeta de clases
    lista para el manifiesto.
    Las credenciales, cuando sean necesarias, deben configurarse con el método
    oficial de Kaggle; nunca se guardan en este repositorio.
    """
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError("Falta kagglehub. Ejecute: python -m pip install -r requirements.txt") from error
    try:
        path = kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)
    except Exception as error:
        raise RuntimeError(
            "No fue posible descargar ASL Alphabet. Verifique su conexión y autenticación de Kaggle. "
            "No incluya credenciales en el repositorio."
        ) from error
    return Path(path).resolve()


def _image_files(directory: Path) -> list[Path]:
    """Devuelve imágenes directamente contenidas en una carpeta de clase."""
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_class_root(dataset_dir: str | Path) -> Path:
    """Encuentra la carpeta cuyas subcarpetas son las clases ASL.

    Funciona con la estructura usual de Kaggle, incluso cuando el archivo ZIP
    deja un nivel adicional como ``asl_alphabet_train/asl_alphabet_train``.
    """
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No existe el directorio del dataset: {dataset_dir}")

    candidates: list[tuple[int, int, Path]] = []
    for directory in [dataset_dir, *[p for p in dataset_dir.rglob("*") if p.is_dir()]]:
        children = [child for child in directory.iterdir() if child.is_dir()]
        names = {canonical_label(child.name) for child in children}
        class_count = len(names.intersection(EXPECTED_CLASSES))
        image_count = sum(len(_image_files(child)) for child in children if child.name.upper() in EXPECTED_CLASSES)
        if class_count >= 2 and image_count > 0:
            candidates.append((class_count, image_count, directory))

    if not candidates:
        raise ValueError(
            "No se encontraron carpetas de clases con imágenes. "
            "Indique la carpeta que contiene subcarpetas como A, B, ..., SPACE."
        )
    # Prioriza las 29 clases esperadas y, en caso de empate, más imágenes.
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def build_manifest(dataset_dir: str | Path, sample_per_class: int | None, seed: int) -> tuple[pd.DataFrame, Path]:
    """Crea un manifiesto de rutas y etiquetas, con submuestra estratificada opcional."""
    class_root = find_class_root(dataset_dir)
    rows: list[dict[str, str]] = []

    for class_dir in sorted((path for path in class_root.iterdir() if path.is_dir()), key=lambda p: p.name):
        label = canonical_label(class_dir.name)
        if label not in EXPECTED_CLASSES:
            continue
        files = _image_files(class_dir)
        if not files:
            continue
        if sample_per_class is not None:
            if sample_per_class < 1:
                raise ValueError("sample_per_class debe ser mayor que cero o None.")
            rng = np.random.default_rng(seed + sum(ord(char) for char in label))
            chosen = sorted(rng.choice(files, size=min(sample_per_class, len(files)), replace=False))
        else:
            chosen = files
        rows.extend({"filepath": str(path.resolve()), "label": label} for path in chosen)

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError(f"No se encontraron imágenes válidas en {class_root}")
    counts = manifest["label"].value_counts()
    if counts.min() < 3:
        raise ValueError("Cada clase necesita al menos tres imágenes para crear train/validation/test.")
    return manifest.sample(frac=1, random_state=seed).reset_index(drop=True), class_root


def validate_images(paths: Iterable[str], max_images: int | None = None) -> pd.DataFrame:
    """Inspecciona formato, dimensiones y archivos corruptos sin cargar todo en memoria."""
    rows: list[dict[str, object]] = []
    for index, raw_path in enumerate(paths):
        if max_images is not None and index >= max_images:
            break
        path = Path(raw_path)
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                rows.append({
                    "filepath": str(path),
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "format": image.format,
                    "is_valid": True,
                    "error": "",
                })
        except (UnidentifiedImageError, OSError, ValueError) as error:
            rows.append({"filepath": str(path), "width": None, "height": None, "mode": None,
                         "format": None, "is_valid": False, "error": str(error)})
    return pd.DataFrame(rows)


def stratified_split(manifest: pd.DataFrame, validation_size: float, test_size: float, seed: int) -> dict[str, pd.DataFrame]:
    """Genera particiones estratificadas y sin solapamiento.

    ``validation_size`` y ``test_size`` son proporciones respecto al total.
    """
    if not 0 < validation_size < 1 or not 0 < test_size < 1 or validation_size + test_size >= 1:
        raise ValueError("validation_size y test_size deben ser positivos y sumar menos de 1.")
    train, holdout = train_test_split(
        manifest, test_size=validation_size + test_size, random_state=seed, stratify=manifest["label"]
    )
    validation_share_of_holdout = validation_size / (validation_size + test_size)
    validation, test = train_test_split(
        holdout, test_size=1 - validation_share_of_holdout, random_state=seed, stratify=holdout["label"]
    )
    splits = {"train": train.copy(), "validation": validation.copy(), "test": test.copy()}
    all_paths = pd.concat(splits.values())["filepath"]
    if all_paths.duplicated().any():
        raise RuntimeError("Se detectó solapamiento entre particiones; se cancela para evitar fuga de datos.")
    return {name: frame.sample(frac=1, random_state=seed).reset_index(drop=True) for name, frame in splits.items()}


def save_splits(splits: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def build_summary(manifest: pd.DataFrame, splits: dict[str, pd.DataFrame], inspection: pd.DataFrame, class_root: Path, seed: int) -> dict:
    """Crea el resumen auditable que se citará en el informe."""
    valid = inspection[inspection["is_valid"]] if not inspection.empty else inspection
    dimensions = Counter(zip(valid.get("width", []), valid.get("height", [])))
    return {
        "class_root": str(class_root),
        "seed": seed,
        "n_images_selected": int(len(manifest)),
        "n_classes": int(manifest["label"].nunique()),
        "class_counts": {key: int(value) for key, value in manifest["label"].value_counts().sort_index().items()},
        "split_sizes": {name: int(len(frame)) for name, frame in splits.items()},
        "split_class_counts": {name: {key: int(value) for key, value in frame["label"].value_counts().sort_index().items()} for name, frame in splits.items()},
        "inspection": {
            "n_checked": int(len(inspection)),
            "n_valid": int(inspection["is_valid"].sum()) if not inspection.empty else 0,
            "n_invalid": int((~inspection["is_valid"]).sum()) if not inspection.empty else 0,
            "formats": {str(key): int(value) for key, value in valid["format"].value_counts(dropna=False).items()} if not valid.empty else {},
            "modes": {str(key): int(value) for key, value in valid["mode"].value_counts(dropna=False).items()} if not valid.empty else {},
            "dimensions": {f"{width}x{height}": int(count) for (width, height), count in dimensions.items()},
        },
    }


def save_summary(summary: dict, output_dir: str | Path) -> None:
    path = Path(output_dir) / "dataset_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
