"""Benchmark de la fase C: inferencia de CNN2 sobre las 2,610 imágenes de test.

Variantes (``fase="C_inferencia"``), todas con los mismos pesos y lotes de 256:

* ``numpy_cpu``: forward en NumPy (im2col + BLAS) en la CPU.
* ``numba_cuda``: kernels propios; la entrada ya está en la GPU (solo cómputo).
* ``numba_cuda_con_transferencias``: kernels propios desde la CPU, con la copia H2D de
  la entrada uint8 y la copia D2H de los logits incluidas.
* ``torch_cuda``: la misma CNN2 en PyTorch/cuDNN (fp32, TF32 desactivado), entrada en la GPU.
  Es la referencia de "qué tan lejos está un kernel escrito a mano de una librería optimizada".

1 calentamiento (compila los kernels con Numba / autotune de cuDNN) + ``N_CORRIDAS`` medidas.
Además mide con eventos CUDA cuánto tarda cada kernel en un lote de 256 (desglose).

Escribe ``resultados/tiempos_rosales.csv``, ``resultados/rosales_kernels.csv``,
``resultados/rosales_kernels_desglose.csv``, figuras y logs del monitor.

    python -m fase_c_kernels_numba.bench_kernels
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
from threadpoolctl import threadpool_info

from comun import config
from comun.medicion import Cronometro, registrar, resumen
from comun.monitor import Monitor, graficar
from fase_c_kernels_numba import kernels as K
from fase_c_kernels_numba import referencia_numpy as R
from fase_c_kernels_numba.inferencia_numba import CNN2Numba, cargar_test_nchw
from fase_c_kernels_numba.pesos import POOL_DESPUES, cargar_pesos, modelo_torch

warnings.simplefilter("ignore", NumbaPerformanceWarning)  # el lote final (50 imgs) es chico a propósito

PERSONA = "rosales"
CSV_TIEMPOS = config.RESULTADOS / f"tiempos_{PERSONA}.csv"
VARIANTES = ["numpy_cpu", "numba_cuda", "numba_cuda_con_transferencias", "torch_cuda"]


def hilos_blas() -> int:
    return max((i.get("num_threads", 1) for i in threadpool_info() if i.get("user_api") == "blas"), default=1)


class Preparado:
    """Todo lo que se carga UNA vez, fuera de las mediciones."""

    def __init__(self):
        self.x, self.y = cargar_test_nchw()
        self.pesos = cargar_pesos()
        self.numba = CNN2Numba(self.pesos)
        lote = config.BATCH
        # Lotes de entrada ya residentes en la GPU (para medir solo cómputo)
        self.lotes_dev = [cuda.to_device(np.ascontiguousarray(self.x[i:i + lote])) for i in range(0, len(self.x), lote)]
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.torch_modelo = modelo_torch().cuda()
        xt = torch.from_numpy(self.x).cuda()
        self.lotes_torch = list(xt.split(lote))


def ejecutar(variante: str, d: Preparado) -> np.ndarray:
    if variante == "numpy_cpu":
        return R.predecir(d.x, d.pesos)
    if variante == "numba_cuda":
        for xb in d.lotes_dev:  # los logits se quedan en la GPU: aquí se mide solo cómputo
            d.numba.forward_dispositivo(xb)
        cuda.synchronize()
        return None
    if variante == "numba_cuda_con_transferencias":
        logits = d.numba.predecir(d.x)
        cuda.synchronize()
        return logits
    if variante == "torch_cuda":
        with torch.inference_mode():
            for xb in d.lotes_torch:
                d.torch_modelo(xb.float().div_(255))
        torch.cuda.synchronize()
        return None
    raise ValueError(variante)


def registradas() -> set[tuple[str, int]]:
    if not CSV_TIEMPOS.exists():
        return set()
    df = pd.read_csv(CSV_TIEMPOS)
    df = df[df["fase"] == "C_inferencia"]
    return set(zip(df["variante"], df["corrida"].astype(int)))


def medir(variante: str, d: Preparado) -> None:
    hechas = registradas()
    pendientes = [c for c in range(1, config.N_CORRIDAS + 1) if (variante, c) not in hechas]
    if not pendientes:
        print(f"{variante}: ya tiene {config.N_CORRIDAS} corridas.")
        return
    gpu = variante != "numpy_cpu"
    hilos = 1 if gpu else hilos_blas()
    for corrida in [0] * config.N_CALENTAMIENTO + pendientes:
        etiqueta = "calentamiento" if corrida == 0 else f"corrida{corrida}"
        nombre = f"{PERSONA}_inf_{variante}_{etiqueta}"
        if gpu:
            cuda.synchronize()
        with Monitor(nombre, intervalo=0.05), Cronometro(cuda=gpu) as t:
            ejecutar(variante, d)
            if gpu:
                cuda.synchronize()
        print(f"  {variante:<30} {etiqueta:<13} {t.segundos:9.4f} s", flush=True)
        if corrida == 0:
            continue
        registrar(PERSONA, fase="C_inferencia", variante=variante, dispositivo="cuda" if gpu else "cpu",
                  corrida=corrida, tiempo_s=t.segundos, n_imagenes=len(d.y), workers_o_hilos=hilos,
                  batch=config.BATCH, epocas="")
        if corrida == 1 and variante in ("numpy_cpu", "numba_cuda_con_transferencias"):
            graficar(config.LOGS / f"{nombre}.csv")
            plt.close("all")


def desglose(d: Preparado, repeticiones: int = 20) -> pd.DataFrame:
    """Tiempo de cada kernel en un lote de 256, medido con eventos CUDA (media de ``repeticiones``)."""
    m = d.numba
    xb = d.lotes_dev[0]
    b = m._reservar(xb.shape[0])
    pasos = [("normalizar", lambda: K.lanzar_normalizar(xb, b["x"]), "1 hilo/elemento", K.config_normalizar(b["x"].size))]
    actual, j = b["x"], 0
    for i, (w, s) in enumerate(m.convs):
        entrada, salida = actual, b["act"][j]
        n, cin, h, wd = entrada.shape
        nombre = f"conv{i + 1} {cin}->{w.shape[0]} @{h}x{wd}"
        pasos.append((nombre, (lambda e=entrada, w=w, s=s, o=salida: K.lanzar_conv(e, w, s, o)),
                      "1 hilo/píxel salida", K.config_conv(n, w.shape[0], h, wd)))
        actual, j = salida, j + 1
        if i % 2 == 1 and POOL_DESPUES[i // 2]:
            e, o = actual, b["act"][j]
            pasos.append((f"maxpool {o.shape[1]}c @{o.shape[2]}x{o.shape[3]}", (lambda e=e, o=o: K.lanzar_maxpool(e, o)),
                          "1 hilo/píxel salida", None))
            actual, j = o, j + 1
    a = actual
    pasos.append(("avgpool global", lambda a=a: K.lanzar_avgpool(a, b["gap"]), "1 hilo/(imagen,canal)", None))
    (w1, b1), (w2, b2) = m.lineales
    pasos.append(("lineal 192->128 +ReLU", lambda: K.lanzar_lineal(b["gap"], w1, b1, b["fc1"], True), "1 hilo/salida, tiling", None))
    pasos.append(("lineal 128->29", lambda: K.lanzar_lineal(b["fc1"], w2, b2, b["logits"], False), "1 hilo/salida, tiling", None))

    m.forward_dispositivo(xb)  # asegura que todo esté compilado
    cuda.synchronize()
    filas = []
    for nombre, f, mapeo, cfg in pasos:
        ini, fin = cuda.event(), cuda.event()
        ini.record()
        for _ in range(repeticiones):
            f()
        fin.record()
        fin.synchronize()
        ms = cuda.event_elapsed_time(ini, fin) / repeticiones
        filas.append({"kernel": nombre, "mapeo": mapeo, "grid_bloque": str(cfg) if cfg else "", "ms_por_lote": ms})
    df = pd.DataFrame(filas)
    df["porcentaje"] = 100 * df["ms_por_lote"] / df["ms_por_lote"].sum()
    df.to_csv(config.RESULTADOS / f"{PERSONA}_kernels_desglose.csv", index=False)
    return df


def tabla_y_figuras(desg: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(CSV_TIEMPOS)
    df = df[df["fase"] == "C_inferencia"]
    base = df[df["variante"] == "numpy_cpu"]["tiempo_s"].mean()
    filas = []
    for v in VARIANTES:
        t = df[df["variante"] == v]["tiempo_s"].tolist()
        if t:
            s = resumen(t)
            filas.append({"variante": v, "media_s": s["media_s"], "desv_s": s["desv_s"], "cv": s["cv"],
                          "imagenes_por_s": len(cargar_test_nchw()[1]) / s["media_s"], "speedup_vs_numpy": base / s["media_s"]})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(config.RESULTADOS / f"{PERSONA}_kernels.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    barras = ax.bar(tabla["variante"].str.replace("_con_", "\ncon ").str.replace("_", " "), tabla["media_s"],
                    yerr=tabla["desv_s"], capsize=4, color=["#b5651d", "#31688e", "#4a90c2", "#6b6b6b"])
    ax.set_yscale("log")
    ax.set_ylabel("tiempo (s, escala log)")
    ax.set_title("Fase C: inferencia de CNN2 sobre 2,610 imágenes de test")
    for bar, (_, r) in zip(barras, tabla.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, r.media_s, f"{r.media_s:.3g} s\n{r.speedup_vs_numpy:.0f}x",
                ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(config.FIGURAS / f"{PERSONA}_kernels_tiempos.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(desg["kernel"][::-1], desg["ms_por_lote"][::-1], color="#31688e")
    ax.set_xlabel("ms por lote de 256 imágenes (eventos CUDA)")
    ax.set_title("Fase C: tiempo de cada kernel propio")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(config.FIGURAS / f"{PERSONA}_kernels_desglose.png", dpi=150)
    plt.close(fig)
    return tabla


def main() -> None:
    print(f"GPU: {cuda.get_current_device().name} | torch {torch.__version__} | BLAS hilos: {hilos_blas()}")
    d = Preparado()
    for v in VARIANTES:
        medir(v, d)
    desg = desglose(d)
    print(desg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(tabla_y_figuras(desg).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
