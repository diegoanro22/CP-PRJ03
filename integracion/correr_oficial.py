"""Corridas oficiales de las fases de Mia y Lopez en la laptop de Diego Rosales.

Los speedups solo son válidos entre tiempos medidos en la MISMA máquina. Mia midió en un
i5-1135G7 sin GPU y Lopez en otra laptop; para el speedup total (pipeline original vs
preparación paralela + entrenamiento en GPU) hace falta todo en un solo equipo.

Este script reutiliza SIN modificar ``analisis/bench_secuencial.py`` y
``fase_b_entrenamiento_gpu/bench_entrenamiento.py``; solo redirige sus salidas para no
pisar los resultados originales de cada integrante:

* tiempos -> ``resultados/tiempos_oficial_mia.csv`` y ``resultados/tiempos_oficial_lopez.csv``
* logs del monitor -> ``resultados/logs/oficial_*.csv``

Las fases A y C (Rosales) ya se miden en esta misma máquina con sus propios benchmarks.
Es reanudable: ambos scripts saltan las corridas ya registradas.

    systemd-inhibit --what=sleep:idle python -m integracion.correr_oficial
"""

from __future__ import annotations

import os
import sys
import time

from comun import config


def preparar_mia():
    import analisis.bench_secuencial as mia

    mia.ARCHIVO_RESULTADOS = config.RESULTADOS / "tiempos_oficial_mia.csv"
    original = mia.nombre_monitor
    mia.nombre_monitor = lambda variante, indice, calentamiento: "oficial_" + original(variante, indice, calentamiento)
    return mia


def preparar_lopez():
    import fase_b_entrenamiento_gpu.bench_entrenamiento as lopez

    lopez.PERSONA = "oficial_lopez"
    lopez.CSV_TIEMPOS = config.RESULTADOS / "tiempos_oficial_lopez.csv"
    lopez.CSV_METRICAS = config.RESULTADOS / "oficial_lopez_metricas.csv"
    return lopez


def main() -> None:
    t0 = time.perf_counter()
    lopez = preparar_lopez()
    print("=== Fase B (bench de Lopez): cuda, cuda_amp, cpu_nhilos, cpu_1hilo ===", flush=True)
    sys.argv = [sys.argv[0]]
    lopez.main()

    mia = preparar_mia()
    print("\n=== T_original (bench de Mia): pipeline original en CPU ===", flush=True)
    mia.ejecutar_entrenamiento("cpu_nhilos", os.cpu_count() or 2)
    mia.ejecutar_entrenamiento("cpu_1hilo", 1)
    mia.mostrar_resumen()
    print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
