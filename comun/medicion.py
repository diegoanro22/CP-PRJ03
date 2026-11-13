"""Medición de tiempos con un esquema de CSV común para todo el equipo.

Cada integrante escribe en su propio archivo ``resultados/tiempos_<persona>.csv``
para evitar conflictos de git. El notebook final los concatena.

Las fases y variantes válidas están en ``comun.config.VOCABULARIO``.

Ejemplo:
    from comun.medicion import Cronometro, registrar
    with Cronometro(cuda=True) as t:
        entrenar(...)
    registrar("lopez", fase="B_entrenamiento", variante="cuda", dispositivo="cuda",
              corrida=1, tiempo_s=t.segundos, n_imagenes=12180, workers_o_hilos=1)
"""

from __future__ import annotations

import csv
import os
import re
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

from comun import config

COLUMNAS = [
    "fase", "variante", "dispositivo", "corrida", "tiempo_s", "n_imagenes",
    "workers_o_hilos", "batch", "epocas", "hardware", "fecha",
]


def _sincronizar_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class Cronometro:
    """Context manager con perf_counter. Con ``cuda=True`` sincroniza la GPU
    antes y después, porque los kernels CUDA son asíncronos y sin esto se
    mediría solo el tiempo de lanzarlos."""

    def __init__(self, cuda: bool = False):
        self.cuda = cuda
        self.segundos = float("nan")

    def __enter__(self):
        if self.cuda:
            _sincronizar_cuda()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            _sincronizar_cuda()
        self.segundos = time.perf_counter() - self._t0
        return False


def descripcion_hardware() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    gpu = "sin GPU"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return f"{cpu} ({os.cpu_count()} hilos) | {gpu}"


def validar_variante(fase: str, variante: str) -> None:
    """Lanza ValueError si la fase o la variante no están en config.VOCABULARIO."""
    if fase not in config.VOCABULARIO:
        raise ValueError(f"Fase desconocida {fase!r}. Opciones: {sorted(config.VOCABULARIO)}")
    for patron in config.VOCABULARIO[fase]:
        if re.fullmatch(re.escape(patron).replace(r"\{p\}", r"[1-9]\d*"), variante):
            return
    raise ValueError(f"Variante {variante!r} no válida para {fase}. Opciones: {config.VOCABULARIO[fase]}")


def registrar(persona: str, *, fase: str, variante: str, dispositivo: str, corrida: int,
              tiempo_s: float, n_imagenes: int, workers_o_hilos: int = 1,
              batch: int = config.BATCH, epocas: int | str = "", archivo: Path | None = None) -> None:
    """Agrega una fila al CSV de la persona (lo crea con encabezado si no existe).

    Valida fase y variante contra ``config.VOCABULARIO`` antes de escribir.
    """
    validar_variante(fase, variante)
    archivo = archivo or config.RESULTADOS / f"tiempos_{persona}.csv"
    archivo.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not archivo.exists()
    with open(archivo, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        if nuevo:
            w.writeheader()
        w.writerow({
            "fase": fase, "variante": variante, "dispositivo": dispositivo, "corrida": corrida,
            "tiempo_s": f"{tiempo_s:.6f}", "n_imagenes": n_imagenes, "workers_o_hilos": workers_o_hilos,
            "batch": batch, "epocas": epocas, "hardware": descripcion_hardware(),
            "fecha": datetime.now().isoformat(timespec="seconds"),
        })


def resumen(tiempos: list[float]) -> dict[str, float]:
    """Media, desviación estándar y coeficiente de variación."""
    media = statistics.fmean(tiempos)
    desv = statistics.stdev(tiempos) if len(tiempos) > 1 else 0.0
    return {"media_s": media, "desv_s": desv, "cv": desv / media if media else float("nan")}
