"""Cuenta las repeticiones y estima los FLOPs de la CNN2 en CPU.

Este programa no abre las imágenes ni ejecuta entrenamiento. Usa los
manifiestos oficiales para contar datos y una entrada sintética únicamente
para inspeccionar la forma y el costo del *forward* de CNN2.

Uso desde la raíz del repositorio:
    py -3.12 -m analisis.contar_operaciones
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd
import torch
from torch import nn

from comun import config
from secuencial.cnn import CNN2, parameter_count


def contar_splits() -> dict[str, int]:
    """Obtiene los tamaños directamente de los manifiestos compartidos."""
    return {
        split: len(pd.read_csv(config.MANIFIESTOS / f"{split}.csv"))
        for split in config.SPLITS
    }


def contar_repeticiones(tamanos: dict[str, int]) -> dict[str, int]:
    """Calcula repeticiones por imagen, lote y época sin aproximar lotes."""
    lotes = {split: math.ceil(n / config.BATCH) for split, n in tamanos.items()}
    return {
        "lotes_train_epoca": lotes["train"],
        "lotes_validation_epoca": lotes["validation"],
        "lotes_test_prediccion": lotes["test"],
        "lotes_train_total": lotes["train"] * config.EPOCAS,
        "lotes_validation_total": lotes["validation"] * config.EPOCAS,
        "lotes_forward_entrenamiento": (lotes["train"] + lotes["validation"]) * config.EPOCAS,
        "pasos_backward": lotes["train"] * config.EPOCAS,
        "pasos_adam": lotes["train"] * config.EPOCAS,
        "imagenes_train_total": tamanos["train"] * config.EPOCAS,
        "imagenes_validation_total": tamanos["validation"] * config.EPOCAS,
        "imagenes_preprocesadas_entrenamiento":
            (tamanos["train"] + tamanos["validation"]) * config.EPOCAS,
        "imagenes_preparacion_tres_splits": sum(tamanos.values()),
        "imagenes_test_prediccion": tamanos["test"],
    }


def _flops_capa(modulo: nn.Module, entrada: tuple[torch.Tensor, ...], salida: torch.Tensor) -> int:
    """Estimación aritmética con una multiplicación y una suma como 2 FLOPs."""
    x = entrada[0]
    if isinstance(modulo, nn.Conv2d):
        elementos_salida = salida.numel()
        productos = (modulo.in_channels // modulo.groups) * modulo.kernel_size[0] * modulo.kernel_size[1]
        return elementos_salida * (2 * productos + (1 if modulo.bias is not None else 0))
    if isinstance(modulo, nn.BatchNorm2d):
        # En inferencia: centrar, escalar y desplazar; aproximación de 4 FLOPs/elemento.
        return 4 * salida.numel()
    if isinstance(modulo, nn.ReLU):
        return salida.numel()
    if isinstance(modulo, nn.MaxPool2d):
        kernel = modulo.kernel_size if isinstance(modulo.kernel_size, tuple) else (modulo.kernel_size,) * 2
        return salida.numel() * (kernel[0] * kernel[1] - 1)
    if isinstance(modulo, nn.AdaptiveAvgPool2d):
        area_entrada = x.shape[-2] * x.shape[-1]
        area_salida = salida.shape[-2] * salida.shape[-1]
        elementos_por_salida = area_entrada // area_salida
        # Sumas y una división por cada elemento de salida.
        return salida.numel() * elementos_por_salida
    if isinstance(modulo, nn.Linear):
        return salida.numel() * (2 * modulo.in_features + (1 if modulo.bias is not None else 0))
    return 0


def flops_por_capa(modelo: nn.Module) -> list[dict[str, object]]:
    """Ejecuta un forward de una imagen y recoge forma y FLOPs por capa."""
    filas: list[dict[str, object]] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []
    tipos = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear)

    def crear_hook(nombre: str) -> Callable:
        def hook(modulo: nn.Module, entrada: tuple[torch.Tensor, ...], salida: torch.Tensor) -> None:
            filas.append({
                "capa": nombre,
                "tipo": modulo.__class__.__name__,
                "salida": "×".join(str(d) for d in salida.shape),
                "flops": _flops_capa(modulo, entrada, salida),
            })
        return hook

    for nombre, modulo in modelo.named_modules():
        if isinstance(modulo, tipos):
            handles.append(modulo.register_forward_hook(crear_hook(nombre)))

    try:
        with torch.inference_mode():
            modelo(torch.zeros(1, 3, config.IMG, config.IMG))
    finally:
        for handle in handles:
            handle.remove()
    return filas


def medir_con_flop_counter(modelo: nn.Module) -> int | None:
    """Consulta FlopCounterMode cuando la versión instalada lo ofrece."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except (ImportError, ModuleNotFoundError):
        return None

    entrada = torch.zeros(1, 3, config.IMG, config.IMG)
    try:
        contador = FlopCounterMode(display=False)
        with torch.inference_mode(), contador:
            modelo(entrada)
        return int(contador.get_total_flops())
    except (AttributeError, RuntimeError, TypeError) as error:
        print(f"FlopCounterMode no es compatible con este entorno: {error}")
        return None


def formato_entero(valor: int) -> str:
    return f"{valor:,}".replace(",", " ")


def main() -> None:
    tamanos = contar_splits()
    repeticiones = contar_repeticiones(tamanos)
    modelo = CNN2(len(config.CLASES)).eval().cpu()
    capas = flops_por_capa(modelo)
    flops_aproximados = sum(int(fila["flops"]) for fila in capas)
    flops_contador = medir_con_flop_counter(modelo)

    print("CNN2: conteos del proyecto")
    print("=" * 72)
    for split in config.SPLITS:
        lotes = math.ceil(tamanos[split] / config.BATCH)
        print(f"{split:12s}: {formato_entero(tamanos[split]):>8s} imágenes, {lotes:2d} lotes")
    print(f"total       : {formato_entero(sum(tamanos.values())):>8s} imágenes, {len(config.CLASES)} clases")
    print(f"batch={config.BATCH}, épocas={config.EPOCAS}, parámetros entrenables={formato_entero(parameter_count(modelo))}")

    print("\nRepeticiones del flujo original")
    print("-" * 72)
    for nombre, valor in repeticiones.items():
        print(f"{nombre:39s} {formato_entero(valor):>12s}")

    print("\nFLOPs aproximados por capa para un forward de una imagen")
    print("-" * 92)
    print(f"{'capa':29s} {'tipo':20s} {'salida':15s} {'FLOPs':>15s}")
    for fila in capas:
        print(f"{str(fila['capa']):29.29s} {str(fila['tipo']):20s} "
              f"{str(fila['salida']):15s} {formato_entero(int(fila['flops'])):>15s}")
    print("-" * 92)
    print(f"Total aproximado (incluye BN/ReLU/pooling): {formato_entero(flops_aproximados)} FLOPs/imagen")
    if flops_contador is not None:
        print(f"FlopCounterMode (operadores que PyTorch soporta): {formato_entero(flops_contador)} FLOPs/imagen")

    forward_batch = flops_aproximados * config.BATCH
    forward_cuatro_epocas = flops_aproximados * repeticiones["imagenes_preprocesadas_entrenamiento"]
    proxy_entrenamiento = flops_aproximados * (
        3 * repeticiones["imagenes_train_total"] + repeticiones["imagenes_validation_total"]
    )
    print("\nCosto computacional")
    print("-" * 72)
    print(f"Forward de un lote completo de {config.BATCH}: {formato_entero(forward_batch)} FLOPs")
    print(f"Forward de train+validation durante 4 épocas: {formato_entero(forward_cuatro_epocas)} FLOPs")
    print(f"Proxy de entrenamiento completo (3× forward en train + forward en validation): "
          f"{formato_entero(proxy_entrenamiento)} FLOPs")
    print("El proxy de entrenamiento no es una medición de FlopCounterMode: backward, "
          "CrossEntropy y Adam agregan trabajo y dependen de sus implementaciones.")


if __name__ == "__main__":
    main()
