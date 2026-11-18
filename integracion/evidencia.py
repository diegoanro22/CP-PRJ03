"""Corridas de evidencia de uso de recursos para las fases A y C (rúbrica: 30 pts).

Las corridas medidas de estas fases duran ~1 s, demasiado poco para que el monitor (o el
Administrador de tareas / nvidia-smi) muestre la carga real. Aquí cada operación se repite
durante ~10-15 s con muestreo cada 0.1 s. NO son mediciones de tiempo (esas están en
``tiempos_rosales.csv``); solo sirven como evidencia de qué hardware trabaja en cada fase.

Mientras corre es el momento de tomar capturas de ``btop``/``htop`` y ``nvidia-smi``.

    python -m integracion.evidencia
"""

from __future__ import annotations

import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

from comun import config
from comun.monitor import Monitor, graficar
from comun.rutas import cargar_manifiesto
from fase_a_preparacion_cpu.bench_preparacion import preparar_secuencial
from fase_a_preparacion_cpu.preparar_paralelo import preparar_todo
from fase_c_kernels_numba.bench_kernels import Preparado, ejecutar

warnings.simplefilter("ignore", NumbaPerformanceWarning)
DURACION = 12.0  # segundos aproximados por evidencia


def repetir(nombre: str, funcion, minimo: int = 1) -> None:
    with Monitor(f"evidencia_{nombre}", intervalo=0.1):
        t0, n = time.perf_counter(), 0
        while n < minimo or time.perf_counter() - t0 < DURACION:
            funcion()
            n += 1
    print(f"  {nombre}: {n} repeticiones en {time.perf_counter() - t0:.1f} s", flush=True)
    graficar(config.LOGS / f"evidencia_{nombre}.csv", config.FIGURAS / f"evidencia_{nombre}.png")
    plt.close("all")


def main() -> None:
    manifiestos = {s: cargar_manifiesto(s) for s in config.SPLITS}
    print("Fase A (CPU):", flush=True)
    repetir("A_secuencial", lambda: preparar_secuencial(manifiestos))
    repetir("A_procesos_16", lambda: preparar_todo(16, manifiestos))
    print("Fase C (GPU):", flush=True)
    d = Preparado()
    ejecutar("numba_cuda", d)  # compila los kernels fuera de la evidencia
    ejecutar("torch_cuda", d)
    repetir("C_numba_cuda", lambda: ejecutar("numba_cuda_con_transferencias", d))
    repetir("C_numpy_cpu", lambda: ejecutar("numpy_cpu", d))


if __name__ == "__main__":
    main()
