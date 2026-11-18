"""Monitor de recursos en segundo plano: evidencia de ejecución (30 pts).

Registra cada ``intervalo`` segundos el % de uso de CADA núcleo de CPU y, si
hay GPU NVIDIA, su utilización, memoria usada y temperatura (vía NVML).

    from comun.monitor import Monitor
    with Monitor("lopez_gpu_corrida1"):
        entrenar(...)
    # -> resultados/logs/lopez_gpu_corrida1.csv

Con ``graficar(ruta_csv)`` se obtiene un heatmap de núcleos y la curva de GPU
para el informe (complementa las capturas de htop / nvidia-smi).
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import psutil

from comun import config


class Monitor:
    def __init__(self, nombre: str, intervalo: float = 0.5):
        self.ruta = config.LOGS / f"{nombre}.csv"
        self.intervalo = intervalo
        self._parar = threading.Event()
        self._gpu = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None

    def _muestra_gpu(self) -> list:
        if self._gpu is None:
            return ["", "", ""]
        try:
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._gpu)
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._gpu)
            temp = self._nvml.nvmlDeviceGetTemperature(self._gpu, 0)
        except Exception:
            # Una lectura fallida de NVML (p. ej. NVMLError_Unknown) no debe matar el
            # hilo del monitor: esa muestra de GPU queda en blanco y se sigue midiendo.
            return ["", "", ""]
        return [util.gpu, round(mem.used / 2**20), temp]

    def _bucle(self) -> None:
        n = psutil.cpu_count()
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ruta, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", *[f"cpu{i}" for i in range(n)], "ram_mb", "gpu_util", "gpu_mem_mb", "gpu_temp_c"])
            psutil.cpu_percent(percpu=True)  # la primera lectura siempre es 0
            t0 = time.perf_counter()

            def muestra() -> None:
                w.writerow([round(time.perf_counter() - t0, 2), *psutil.cpu_percent(percpu=True),
                            round(psutil.virtual_memory().used / 2**20), *self._muestra_gpu()])
                f.flush()

            while not self._parar.wait(self.intervalo):
                muestra()
            muestra()  # muestra final: garantiza al menos una fila aunque la corrida sea corta

    def __enter__(self):
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()
        return self

    def __exit__(self, *exc):
        self._parar.set()
        self._hilo.join()
        return False


def graficar(ruta_csv: str | Path, salida: str | Path | None = None):
    """Heatmap de % por núcleo y curva de utilización de GPU."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(ruta_csv)
    if not {"t_s", "gpu_util"}.issubset(df.columns):
        print(f"{ruta_csv}: no es un log de comun.monitor; se omite.")
        return None
    if len(df) < 2:
        print(f"{ruta_csv}: muy pocas muestras para graficar ({len(df)}); use un intervalo menor o una corrida más larga.")
        return None
    cpus = [c for c in df.columns if c.startswith("cpu")]
    tiene_gpu = df["gpu_util"].notna().any()
    fig, ejes = plt.subplots(2 if tiene_gpu else 1, 1, figsize=(10, 6 if tiene_gpu else 3.5), sharex=True, squeeze=False)
    im = ejes[0, 0].imshow(df[cpus].T, aspect="auto", cmap="magma", vmin=0, vmax=100,
                           extent=[df.t_s.min(), df.t_s.max(), len(cpus) - 0.5, -0.5])
    ejes[0, 0].set_ylabel("núcleo")
    ejes[0, 0].set_title("Uso por núcleo de CPU (%)")
    fig.colorbar(im, ax=ejes[0, 0])
    if tiene_gpu:
        ax = ejes[1, 0]
        ax.plot(df.t_s, df.gpu_util, color="#31688e", label="GPU utilización (%)")
        ax.set_ylim(-3, 103)
        ax.set_ylabel("GPU utilización (%)")
        ax2 = ax.twinx()
        ax2.plot(df.t_s, df.gpu_mem_mb, color="#b5651d", label="GPU memoria usada (MiB)")
        ax2.set_ylabel("memoria (MiB)")
        ax2.set_ylim(0, max(1.0, df.gpu_mem_mb.max()) * 1.15)
        lineas = ax.get_lines() + ax2.get_lines()
        ax.legend(lineas, [l.get_label() for l in lineas], loc="lower right")
        ejes[0, 0].set_title(f"Uso por núcleo de CPU (%), promedio {df[cpus].to_numpy().mean():.0f} %")
    ejes[-1, 0].set_xlabel("tiempo (s)")
    fig.tight_layout()
    salida = Path(salida or config.FIGURAS / f"monitor_{Path(ruta_csv).stem}.png")
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=150)
    return salida
