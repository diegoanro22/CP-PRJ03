"""Benchmark de la fase B: CPU 1 hilo vs CPU N hilos vs GPU vs GPU+AMP.

Todas las variantes usan el mismo bucle y la misma caché en memoria
(``entrenar_cuda.entrenar``); solo cambian dispositivo, hilos y AMP. Por variante:
1 corrida de calentamiento (descartada; en CUDA absorbe el contexto y el autotune
de cuDNN) y ``N_CORRIDAS`` medidas, con épocas fijas y semilla común.

Escribe ``resultados/tiempos_lopez.csv`` (``B_entrenamiento``; en CUDA además
``B_transferencia`` y ``B_computo``, medidos con eventos CUDA), el desempeño en test
de cada corrida en ``resultados/lopez_metricas.csv`` y un log del monitor por corrida
en ``resultados/logs/lopez_<variante>_corrida<n>.csv``.

Es reanudable: salta las corridas ya registradas (una corrida de CPU 1 hilo dura
~15 min). Uso, desde la raíz del repositorio:

    python -m fase_b_entrenamiento_gpu.bench_entrenamiento
    python -m fase_b_entrenamiento_gpu.bench_entrenamiento --variantes cuda cuda_amp
    python -m fase_b_entrenamiento_gpu.bench_entrenamiento --solo-resumen
"""

from __future__ import annotations

import argparse
import csv

import pandas as pd

from comun import config
from comun.medicion import registrar, resumen
from comun.monitor import Monitor, graficar
from fase_b_entrenamiento_gpu.entrenar_cuda import VARIANTES, Datos, entrenar, evaluar, informar_dispositivo


class MonitorRobusto(Monitor):
    """``comun.monitor.Monitor`` que sobrevive a un fallo puntual de NVML.

    Una lectura de GPU fallida (``NVMLError_Unknown``) mataba el hilo del monitor y
    dejaba el log truncado a mitad de la corrida; aquí esa muestra queda en blanco y
    el muestreo de CPU/RAM continúa. (``comun/`` es del equipo, por eso no se edita.)
    """

    def _muestra_gpu(self) -> list:
        try:
            return super()._muestra_gpu()
        except Exception:
            return ["", "", ""]


PERSONA = "lopez"
CSV_TIEMPOS = config.RESULTADOS / f"tiempos_{PERSONA}.csv"
CSV_METRICAS = config.RESULTADOS / f"{PERSONA}_metricas.csv"
ORDEN = ["cuda", "cuda_amp", "cpu_nhilos", "cpu_1hilo"]  # lo rápido primero, lo lento al final


def registradas(fase: str) -> set[tuple[str, int]]:
    if not CSV_TIEMPOS.exists():
        return set()
    df = pd.read_csv(CSV_TIEMPOS)
    df = df[df["fase"] == fase]
    return set(zip(df["variante"], df["corrida"].astype(int)))


def guardar_metricas(variante: str, corrida: int, r: dict, m: dict) -> None:
    nuevo = not CSV_METRICAS.exists()
    with open(CSV_METRICAS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["variante", "corrida", "tiempo_s", "val_acc", "test_accuracy", "test_f1_macro"])
        w.writerow([variante, corrida, f"{r['tiempo_s']:.6f}", f"{r['historia'][-1]['val_acc']:.4f}",
                    f"{m['accuracy']:.4f}", f"{m['f1_macro']:.4f}"])


def correr(variante: str, corrida: int, datos: Datos, calentamiento: bool) -> None:
    dispositivo, _, hilos = VARIANTES[variante]
    etiqueta = "calentamiento" if calentamiento else f"corrida{corrida}"
    nombre = f"{PERSONA}_{variante}_{etiqueta}"
    with MonitorRobusto(nombre):
        r = entrenar(variante, datos)
    m = evaluar(r["modelo"], datos, dispositivo)  # fuera del cronómetro
    print(f"  {variante:<11} {etiqueta:<13} {r['tiempo_s']:8.2f} s  (H2D {r['transferencia_s']:.2f} s, "
          f"cómputo {r['computo_s']:.2f} s)  test acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f}", flush=True)
    if calentamiento:
        return
    comun = dict(variante=variante, dispositivo=dispositivo, corrida=corrida, n_imagenes=len(datos["train"][1]),
                 workers_o_hilos=hilos, batch=config.BATCH, epocas=config.EPOCAS)
    registrar(PERSONA, fase="B_entrenamiento", tiempo_s=r["tiempo_s"], **comun)
    if dispositivo == "cuda":
        registrar(PERSONA, fase="B_transferencia", tiempo_s=r["transferencia_s"], **comun)
        registrar(PERSONA, fase="B_computo", tiempo_s=r["computo_s"], **comun)
    guardar_metricas(variante, corrida, r, m)
    if corrida == 1:
        graficar(config.LOGS / f"{nombre}.csv")


def mostrar_resumen() -> None:
    if not CSV_TIEMPOS.exists():
        print("Aún no hay tiempos registrados.")
        return
    df = pd.read_csv(CSV_TIEMPOS)
    df = df[df["fase"] == "B_entrenamiento"]
    base = None
    filas = []
    for v in ORDEN[::-1]:
        t = df[df["variante"] == v]["tiempo_s"].tolist()
        if not t:
            continue
        s = resumen(t)
        if v == config.BASE_SPEEDUP["B_entrenamiento"]:
            base = s["media_s"]
        filas.append((v, len(t), s))
    print("\nResumen B_entrenamiento (4 épocas, batch 256):")
    for v, n, s in filas[::-1]:
        sp = f"{base / s['media_s']:.1f}x" if base else "n/d (falta cpu_1hilo)"
        print(f"  {v:<11} n={n}  media={s['media_s']:8.2f} s  desv={s['desv_s']:6.2f}  cv={100 * s['cv']:4.1f}%  speedup vs cpu_1hilo: {sp}")


def evitar_suspension() -> None:
    """Pide a Windows que no suspenda el equipo mientras corre el benchmark.

    Si la laptop se suspende a mitad de una corrida, el cronómetro (que sigue
    contando) registra el tiempo dormido como tiempo de entrenamiento. Solo vale
    mientras vive este proceso; no cambia ningún ajuste de energía del sistema.
    No evita la suspensión por cerrar la tapa.
    """
    try:
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    except (AttributeError, OSError):
        print("Aviso: no se pudo bloquear la suspensión; evite que el equipo se duerma durante el benchmark.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variantes", nargs="+", choices=sorted(VARIANTES), default=ORDEN)
    p.add_argument("--corridas", type=int, default=config.N_CORRIDAS)
    p.add_argument("--solo-resumen", action="store_true")
    a = p.parse_args()
    if a.solo_resumen:
        mostrar_resumen()
        return

    informar_dispositivo()
    evitar_suspension()
    datos = Datos()  # la caché se carga una sola vez, fuera de toda medición
    hechas = registradas("B_entrenamiento")
    for v in [x for x in ORDEN if x in a.variantes]:
        pendientes = [c for c in range(1, a.corridas + 1) if (v, c) not in hechas]
        if not pendientes:
            print(f"{v}: ya tiene {a.corridas} corridas registradas.")
            continue
        print(f"{v}: pendientes {pendientes}", flush=True)
        # El log del monitor no prueba que el calentamiento terminó (puede quedar de una
        # ejecución cortada); solo cuenta el marcador, o que la variante ya tenga corridas.
        marca = config.LOGS / f"{PERSONA}_{v}_calentamiento.ok"
        if not marca.exists() and not any(var == v for var, _ in hechas):
            for _ in range(config.N_CALENTAMIENTO):
                correr(v, 0, datos, calentamiento=True)
            marca.write_text("calentamiento completado\n", encoding="utf-8")
        for c in pendientes:
            correr(v, c, datos, calentamiento=False)
    mostrar_resumen()


if __name__ == "__main__":
    main()
