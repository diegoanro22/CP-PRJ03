"""Fase A: preparación de imágenes en CPU multinúcleo.

Hace exactamente la misma operación que ``comun.datos_seq`` (abrir JPG, convertir a
RGB, redimensionar a 64x64 con BILINEAR y guardar uint8), pero reparte las rutas
entre varios procesos con ``concurrent.futures.ProcessPoolExecutor``.

Decisiones de diseño:

* **Procesos, no hilos.** Decodificar JPEG y redimensionar es trabajo de CPU; con
  procesos cada worker tiene su propio intérprete y no compite por el GIL.
* **Bloques grandes.** Las rutas se dividen en bloques (~4 por worker y split) y cada
  worker devuelve un único arreglo ``uint8`` por bloque. Así se envían pocos mensajes
  grandes entre procesos en vez de 17,400 objetos pequeños.
* **Orden conservado.** ``executor.map`` devuelve los bloques en el orden en que se
  enviaron, de modo que las filas quedan en el orden del manifiesto y la caché es
  idéntica bit a bit a la secuencial (se verifica con ``np.array_equal``).
* **Un solo pool para los tres splits.** El costo de crear los procesos se paga una
  vez y se incluye en el tiempo medido (es parte del costo real de paralelizar).

Uso, desde la raíz del repositorio:

    python -m fase_a_preparacion_cpu.preparar_paralelo --workers 16
    python -m fase_a_preparacion_cpu.preparar_paralelo --workers 16 --verificar
"""

from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from comun import config
from comun.datos_seq import cargar_cache, etiquetas_a_indices, guardar, preparar_imagen
from comun.rutas import cargar_manifiesto

SUFIJO = "_par"
BLOQUES_POR_WORKER = 4


def procesar_bloque(rutas: list[str]) -> np.ndarray:
    """Trabajo de un worker: prepara un bloque de imágenes y lo devuelve apilado."""
    return np.stack([preparar_imagen(r) for r in rutas])


def dividir(rutas: list[str], workers: int) -> list[list[str]]:
    """Divide las rutas en bloques contiguos (contiguos = el orden se conserva)."""
    tam = max(1, math.ceil(len(rutas) / (BLOQUES_POR_WORKER * workers)))
    return [rutas[i:i + tam] for i in range(0, len(rutas), tam)]


def preparar_todo(workers: int, manifiestos: dict | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Prepara los tres splits con un pool de ``workers`` procesos.

    ``manifiestos`` permite pasar los DataFrames ya cargados (el benchmark lo hace
    para no medir la lectura de los CSV).
    """
    manifiestos = manifiestos or {s: cargar_manifiesto(s) for s in config.SPLITS}
    resultado = {}
    with ProcessPoolExecutor(max_workers=workers) as ejecutor:
        for split, df in manifiestos.items():
            bloques = dividir(list(df["filepath"]), workers)
            x = np.concatenate(list(ejecutor.map(procesar_bloque, bloques)))
            resultado[split] = (x, etiquetas_a_indices(df["label"]))
    return resultado


def verificar(datos: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Compara contra la caché secuencial (``python -m comun.datos_seq``)."""
    for split, (x, y) in datos.items():
        x_seq, y_seq = cargar_cache(split)
        if not (np.array_equal(x, x_seq) and np.array_equal(y, y_seq)):
            raise AssertionError(f"{split}: la caché paralela NO coincide con la secuencial")
        print(f"  {split}: idéntica a la secuencial {x.shape} {x.dtype}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--verificar", action="store_true", help="comparar contra cache/<split>_x.npy")
    a = p.parse_args()

    t0 = time.perf_counter()
    datos = preparar_todo(a.workers)
    print(f"{a.workers} procesos: {sum(len(y) for _, y in datos.values())} imágenes en {time.perf_counter() - t0:.2f} s")
    for split, (x, y) in datos.items():
        guardar(split, x, y, sufijo=SUFIJO)
    print(f"Guardado en {config.CACHE}/<split>{SUFIJO}_x.npy y _y.npy")
    if a.verificar:
        verificar(datos)


if __name__ == "__main__":
    main()
