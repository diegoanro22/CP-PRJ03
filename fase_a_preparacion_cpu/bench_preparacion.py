"""Benchmark de la fase A: preparación secuencial vs pool de 1, 2, 4, 8, 16 y 32 procesos.

Por variante: ``N_CALENTAMIENTO`` corridas descartadas (dejan los JPG en la caché de
archivos del sistema, así ninguna variante se beneficia de llegar después) y
``N_CORRIDAS`` medidas. Los manifiestos se leen una sola vez, fuera del cronómetro.

Escribe:
* ``resultados/tiempos_rosales.csv``: fase ``A_preparacion``, variantes ``secuencial`` y ``procesos_<p>``.
* ``resultados/rosales_preparacion.csv``: media, speedup, eficiencia y fracción serial (Karp-Flatt).
* ``resultados/figuras/rosales_preparacion_speedup.png``: speedup medido vs ideal vs Amdahl.
* ``resultados/logs/rosales_prep_<variante>_corrida<n>.csv``: monitor de CPU por corrida.

Es reanudable: salta las corridas que ya están en el CSV.

    python -m fase_a_preparacion_cpu.bench_preparacion
    python -m fase_a_preparacion_cpu.bench_preparacion --solo-resumen
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comun import config
from comun.datos_seq import etiquetas_a_indices, preparar_imagen
from comun.medicion import Cronometro, registrar, resumen
from comun.monitor import Monitor, graficar
from comun.rutas import cargar_manifiesto
from fase_a_preparacion_cpu.preparar_paralelo import preparar_todo, verificar

PERSONA = "rosales"
CSV_TIEMPOS = config.RESULTADOS / f"tiempos_{PERSONA}.csv"
CSV_TABLA = config.RESULTADOS / f"{PERSONA}_preparacion.csv"
WORKERS = [p for p in (1, 2, 4, 8, 16, 32) if p <= (os.cpu_count() or 1)]


def preparar_secuencial(manifiestos: dict) -> dict:
    """Mismo bucle de ``comun.datos_seq`` pero con los manifiestos ya cargados."""
    return {s: (np.stack([preparar_imagen(r) for r in df["filepath"]]), etiquetas_a_indices(df["label"]))
            for s, df in manifiestos.items()}


def registradas() -> set[tuple[str, int]]:
    if not CSV_TIEMPOS.exists():
        return set()
    df = pd.read_csv(CSV_TIEMPOS)
    df = df[df["fase"] == "A_preparacion"]
    return set(zip(df["variante"], df["corrida"].astype(int)))


def medir(variante: str, workers: int, manifiestos: dict, n_imagenes: int) -> None:
    hechas = registradas()
    pendientes = [c for c in range(1, config.N_CORRIDAS + 1) if (variante, c) not in hechas]
    if not pendientes:
        print(f"{variante}: ya tiene {config.N_CORRIDAS} corridas.")
        return
    plan = [0] * config.N_CALENTAMIENTO + pendientes  # 0 = calentamiento
    for corrida in plan:
        etiqueta = "calentamiento" if corrida == 0 else f"corrida{corrida}"
        nombre = f"{PERSONA}_prep_{variante}_{etiqueta}"
        with Monitor(nombre, intervalo=0.2), Cronometro() as t:
            datos = preparar_secuencial(manifiestos) if variante == "secuencial" else preparar_todo(workers, manifiestos)
        print(f"  {variante:<12} {etiqueta:<13} {t.segundos:7.3f} s", flush=True)
        if corrida == 0:
            if variante != "secuencial":
                verificar(datos)  # el resultado paralelo debe ser idéntico al secuencial
            continue
        registrar(PERSONA, fase="A_preparacion", variante=variante, dispositivo="cpu", corrida=corrida,
                  tiempo_s=t.segundos, n_imagenes=n_imagenes, workers_o_hilos=workers, batch="", epocas="")
        if corrida == 1:
            graficar(config.LOGS / f"{nombre}.csv")
            plt.close("all")


def tabla_y_figura() -> pd.DataFrame:
    df = pd.read_csv(CSV_TIEMPOS)
    df = df[df["fase"] == "A_preparacion"]
    t_seq = df[df["variante"] == "secuencial"]["tiempo_s"].mean()
    filas = []
    for p in WORKERS:
        t = df[df["variante"] == f"procesos_{p}"]["tiempo_s"].tolist()
        if not t:
            continue
        s = resumen(t)
        speedup = t_seq / s["media_s"]
        # Karp-Flatt: fracción serial experimental e = (1/S - 1/p) / (1 - 1/p)
        serial = (1 / speedup - 1 / p) / (1 - 1 / p) if p > 1 else float("nan")
        filas.append({"procesos": p, "media_s": s["media_s"], "desv_s": s["desv_s"], "cv": s["cv"],
                      "speedup": speedup, "eficiencia": speedup / p, "fraccion_serial": serial})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CSV_TABLA, index=False)

    # Amdahl con la fracción serial media estimada en p >= 4 (donde ya pesa el overhead)
    f = tabla.loc[tabla["procesos"] >= 4, "fraccion_serial"].mean()
    ps = np.array(WORKERS)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(ps, ps, "--", color="gray", label="ideal (S = p)")
    ax.plot(ps, 1 / (f + (1 - f) / ps), ":", color="#b5651d", label=f"Amdahl, fracción serial {f:.3f}")
    ax.plot(tabla["procesos"], tabla["speedup"], "o-", color="#31688e", label="medido")
    for _, r in tabla.iterrows():
        ax.annotate(f"{r.speedup:.1f}x", (r.procesos, r.speedup), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ps, [str(p) for p in ps])
    ax.set_xlabel("procesos")
    ax.set_ylabel("speedup vs secuencial")
    ax.set_title(f"Fase A: preparación de 17,400 imágenes (secuencial = {t_seq:.2f} s)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURAS / f"{PERSONA}_preparacion_speedup.png", dpi=150)
    plt.close(fig)
    return tabla


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--solo-resumen", action="store_true")
    a = p.parse_args()
    if not a.solo_resumen:
        manifiestos = {s: cargar_manifiesto(s) for s in config.SPLITS}
        n = sum(len(df) for df in manifiestos.values())
        print(f"{n} imágenes, {os.cpu_count()} CPUs lógicas; variantes: secuencial + {WORKERS}")
        medir("secuencial", 1, manifiestos, n)
        for w in WORKERS:
            medir(f"procesos_{w}", w, manifiestos, n)
    print(tabla_y_figura().to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
