"""Barrido de tamaño de lote en GPU (64 / 128 / 256 / 512): ocupación y tiempo por época.

Cada lote grande da más trabajo por lanzamiento de kernels y mejor ocupación de la GPU,
pero hace menos pasos de optimización por época (con lr fija el accuracy puede cambiar).
Por tamaño: 1 calentamiento (autotune de cuDNN para esa forma) y ``N_CORRIDAS`` medidas.

Escribe ``B_barrido_batch`` (variante ``batch_<n>``) en ``resultados/tiempos_lopez.csv``,
la tabla ``resultados/lopez_barrido_batch.csv`` (con la utilización media de GPU leída
de los logs del monitor) y ``resultados/figuras/lopez_barrido_batch.png``. Reanudable.

    python -m fase_b_entrenamiento_gpu.barrido_batch
"""

from __future__ import annotations

import argparse

import pandas as pd

from comun import config
from comun.medicion import registrar, resumen
from fase_b_entrenamiento_gpu.bench_entrenamiento import CSV_TIEMPOS, PERSONA, MonitorRobusto, registradas
from fase_b_entrenamiento_gpu.entrenar_cuda import Datos, entrenar, evaluar, informar_dispositivo

BATCHES = (64, 128, 256, 512)
VARIANTE_MODELO = "cuda"
CSV_TABLA = config.RESULTADOS / f"{PERSONA}_barrido_batch.csv"
CSV_ACC = config.LOGS / f"{PERSONA}_barrido_batch_acc.csv"


def nombre_log(batch: int, etiqueta: str) -> str:
    return f"{PERSONA}_batch_{batch}_{etiqueta}"


def correr(batch: int, corrida: int, datos: Datos) -> None:
    calentamiento = corrida == 0
    etiqueta = "calentamiento" if calentamiento else f"corrida{corrida}"
    with MonitorRobusto(nombre_log(batch, etiqueta)):
        r = entrenar(VARIANTE_MODELO, datos, batch=batch)
    m = evaluar(r["modelo"], datos, "cuda", batch)
    print(f"  batch={batch:<4} {etiqueta:<13} {r['tiempo_s']:7.2f} s  ({r['tiempo_s'] / config.EPOCAS:.2f} s/época)  "
          f"test acc={m['accuracy']:.4f}", flush=True)
    if calentamiento:
        return
    registrar(PERSONA, fase="B_barrido_batch", variante=f"batch_{batch}", dispositivo="cuda", corrida=corrida,
              tiempo_s=r["tiempo_s"], n_imagenes=len(datos["train"][1]), workers_o_hilos=1, batch=batch,
              epocas=config.EPOCAS)
    with open(CSV_ACC, "a", encoding="utf-8") as f:
        f.write(f"{batch},{corrida},{m['accuracy']:.4f}\n")


def tabla() -> pd.DataFrame:
    df = pd.read_csv(CSV_TIEMPOS)
    df = df[df["fase"] == "B_barrido_batch"]
    acc = pd.read_csv(CSV_ACC, names=["batch", "corrida", "acc"]) if CSV_ACC.exists() else None
    filas = []
    for b in BATCHES:
        t = df[df["batch"] == b]["tiempo_s"].tolist()
        if not t:
            continue
        s = resumen(t)
        utils, mem = [], []
        for c in range(1, len(t) + 1):
            ruta = config.LOGS / f"{nombre_log(b, f'corrida{c}')}.csv"
            if ruta.exists():
                log = pd.read_csv(ruta)
                utils.append(log["gpu_util"].mean())
                mem.append(log["gpu_mem_mb"].max())
        filas.append({
            "batch": b, "n": len(t), "media_s": s["media_s"], "desv_s": s["desv_s"], "cv": s["cv"],
            "s_por_epoca": s["media_s"] / config.EPOCAS,
            "gpu_util_media_pct": sum(utils) / len(utils) if utils else float("nan"),
            "gpu_mem_max_mb": max(mem) if mem else float("nan"),
            "test_acc_media": acc[acc["batch"] == b]["acc"].mean() if acc is not None else float("nan"),
        })
    return pd.DataFrame(filas)


def graficar_tabla(t: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    x = [str(b) for b in t["batch"]]
    a1.bar(x, t["s_por_epoca"], yerr=t["desv_s"] / config.EPOCAS, color="#4c78a8", capsize=4)
    a1.set_xlabel("tamaño de lote"), a1.set_ylabel("s por época"), a1.set_title("Tiempo por época (GPU, fp32)")
    a2.bar(x, t["gpu_util_media_pct"], color="#59a14f")
    a2.set_xlabel("tamaño de lote"), a2.set_ylabel("% utilización media"), a2.set_ylim(0, 100)
    a2.set_title("Utilización de GPU (NVML)")
    fig.tight_layout()
    salida = config.FIGURAS / f"{PERSONA}_barrido_batch.png"
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=150)
    print(f"figura: {salida}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corridas", type=int, default=config.N_CORRIDAS)
    a = p.parse_args()
    informar_dispositivo()
    datos = Datos()
    hechas = registradas("B_barrido_batch")
    for b in BATCHES:
        pendientes = [c for c in range(1, a.corridas + 1) if (f"batch_{b}", c) not in hechas]
        if not pendientes:
            continue
        if not (config.LOGS / f"{nombre_log(b, 'calentamiento')}.csv").exists():
            correr(b, 0, datos)
        for c in pendientes:
            correr(b, c, datos)
    t = tabla()
    t.to_csv(CSV_TABLA, index=False, float_format="%.4f")
    print("\n" + t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    graficar_tabla(t)


if __name__ == "__main__":
    main()
