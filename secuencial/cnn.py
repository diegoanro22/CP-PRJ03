"""Entrenamiento reproducible de las CNN del Laboratorio 3.

El módulo consume exclusivamente los manifiestos oficiales generados durante
el EDA. Mantiene el código de entrenamiento fuera del notebook para que este
pueda concentrarse en explicar decisiones y resultados.
"""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ASLDataset(Dataset):
    """Dataset de rutas de imagen con normalización fija a [0, 1]."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        labels: list[str],
        image_size: int = 64,
        augment: Callable[[Image.Image], Image.Image] | None = None,
    ):
        self.paths = manifest["filepath"].tolist()
        label_to_index = {label: index for index, label in enumerate(labels)}
        self.targets = [label_to_index[label] for label in manifest["label"].tolist()]
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            if self.augment is not None:
                image = self.augment(image)
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1), self.targets[index]


def seed_everything(seed: int = 3084) -> None:
    """Configura aleatoriedad reproducible para Python, NumPy y PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loaders(
    splits: dict[str, pd.DataFrame],
    labels: list[str],
    batch_size: int = 64,
    image_size: int = 64,
    train_augment: Callable[[Image.Image], Image.Image] | None = None,
) -> dict[str, DataLoader]:
    """Crea cargadores; la aumentación se limita explícitamente a ``train``."""
    datasets = {
        name: ASLDataset(
            frame,
            labels,
            image_size=image_size,
            augment=train_augment if name == "train" else None,
        )
        for name, frame in splits.items()
    }
    return {
        name: DataLoader(dataset, batch_size=batch_size, shuffle=name == "train", num_workers=0)
        for name, dataset in datasets.items()
    }


class ASLAugmentation:
    """Aumentación fotográfica moderada para ASL, aplicada en memoria.

    No incluye flip horizontal: invertir una mano puede modificar orientación o
    significado de una seña. Los parámetros se expresan en la escala original
    de cada fotografía y son deliberadamente pequeños para conservar la clase.
    """

    def __init__(
        self,
        max_rotation_degrees: float = 8.0,
        max_translation_fraction: float = 0.08,
        max_zoom: float = 1.10,
        brightness_range: tuple[float, float] = (0.85, 1.15),
    ) -> None:
        if max_rotation_degrees < 0 or not 0 <= max_translation_fraction < 0.5:
            raise ValueError("Los límites de rotación y traslación no son válidos.")
        if max_zoom < 1.0 or brightness_range[0] <= 0 or brightness_range[0] > brightness_range[1]:
            raise ValueError("Los límites de zoom o brillo no son válidos.")
        self.max_rotation_degrees = max_rotation_degrees
        self.max_translation_fraction = max_translation_fraction
        self.max_zoom = max_zoom
        self.brightness_range = brightness_range

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        angle = random.uniform(-self.max_rotation_degrees, self.max_rotation_degrees)
        # El color del borde se aproxima al de la esquina para no introducir marcos negros.
        fill = image.getpixel((0, 0))
        transformed = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=fill)

        shift_x = round(random.uniform(-self.max_translation_fraction, self.max_translation_fraction) * width)
        shift_y = round(random.uniform(-self.max_translation_fraction, self.max_translation_fraction) * height)
        transformed = transformed.transform(
            (width, height),
            Image.Transform.AFFINE,
            (1, 0, -shift_x, 0, 1, -shift_y),
            resample=Image.Resampling.BILINEAR,
            fillcolor=fill,
        )

        zoom = random.uniform(1.0, self.max_zoom)
        crop_width, crop_height = max(1, round(width / zoom)), max(1, round(height / zoom))
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        transformed = transformed.crop((left, top, left + crop_width, top + crop_height))
        transformed = transformed.resize((width, height), Image.Resampling.BILINEAR)
        return ImageEnhance.Brightness(transformed).enhance(random.uniform(*self.brightness_range))


class CNN1(nn.Module):
    """CNN compacta: referencia con tres bloques convolucionales."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, n_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class CNN2(nn.Module):
    """CNN más profunda con dos convoluciones por bloque y dropout mayor."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            _double_conv(3, 32), nn.MaxPool2d(2),
            _double_conv(32, 64), nn.MaxPool2d(2), nn.Dropout2d(0.10),
            _double_conv(64, 128), nn.MaxPool2d(2), nn.Dropout2d(0.15),
            _double_conv(128, 192), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(192, 128), nn.ReLU(), nn.Dropout(0.20), nn.Linear(128, n_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(),
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    learning_rate: float = 1e-3,
    max_epochs: int = 8,
    patience: int = 3,
    device: str = "cpu",
) -> tuple[nn.Module, pd.DataFrame, float]:
    """Entrena usando validación para early stopping; nunca consulta test."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_state, best_loss, wait = copy.deepcopy(model.state_dict()), float("inf"), 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        row: dict[str, float] = {"epoch": epoch}
        for split_name in ("train", "validation"):
            is_train = split_name == "train"
            model.train(is_train)
            loss_sum = correct = total = 0
            with torch.set_grad_enabled(is_train):
                for inputs, targets in loaders[split_name]:
                    inputs, targets = inputs.to(device), targets.to(device)
                    if is_train:
                        optimizer.zero_grad()
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                    if is_train:
                        loss.backward()
                        optimizer.step()
                    loss_sum += loss.item() * targets.size(0)
                    correct += (logits.argmax(dim=1) == targets).sum().item()
                    total += targets.size(0)
            row[f"{split_name}_loss"] = loss_sum / total
            row[f"{split_name}_accuracy"] = correct / total
        history.append(row)
        if row["validation_loss"] < best_loss - 1e-5:
            best_loss, best_state, wait = row["validation_loss"], copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), time.perf_counter() - started


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, labels: list[str], device: str = "cpu") -> tuple[list[str], list[str]]:
    model.eval().to(device)
    actual, predicted = [], []
    for inputs, targets in loader:
        outputs = model(inputs.to(device)).argmax(dim=1).cpu().tolist()
        actual.extend(labels[index] for index in targets.tolist())
        predicted.extend(labels[index] for index in outputs)
    return actual, predicted


def save_model(model: nn.Module, path: str | Path, labels: list[str], config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "labels": labels, "config": config}, path)
