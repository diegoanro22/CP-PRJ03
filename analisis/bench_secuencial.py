"""Benchmark de la preparación y del entrenamiento secuencial de Mia Fuentes.

Ejecuta una corrida de calentamiento y cinco medidas para cada variante. Los
resultados ya existentes se reconocen por fase, variante y número de corrida,
por lo que es seguro reanudar el programa sin duplicar filas.

Uso desde la raíz:
    py -3.12 -m analisis.bench_secuencial
    py -3.12 -m analisis.bench_secuencial --solo preparacion
    py -3.12 -m analisis.bench_secuencial --solo cpu_1hilo
    py -3.12 -m analisis.bench_secuencial --solo cpu_nhilos
    py -3.12 -m analisis.bench_secuencial --solo figuras
"""

from __future__ import annotations

import argparse
import csv
import gc
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from comun import config
from comun.datos_seq import preparar_split
from comun.medicion import COLUMNAS, Cronometro, registrar, resumen
from comun.monitor import Monitor, graficar
from comun.rutas import cargar_splits
from secuencial.cnn import CNN2, make_loaders, seed_everything, train_model


ARCHIVO_RESULTADOS = config.RESULTADOS / "tiempos_mia.csv"
ORDEN = [
    ("A_preparacion", "secuencial"),
    ("T_original", "cpu_1hilo"),
    ("T_original", "cpu_nhilos"),
]
ETIQUETAS = {
    ("A_preparacion", "secuencial"): "Preparación\nsecuencial",
    ("T_original", "cpu_1hilo"): "Entrenamiento\nCPU, 1 hilo",
    ("T_original", "cpu_nhilos"): "Entrenamiento\nCPU, varios hilos",
}


def corridas_registradas() -> set[tuple[str, str, int]]:
    """Lee las claves ya guardadas y verifica el contrato del CSV."""
    if not ARCHIVO_RESULTADOS.exists():
        return set()
    with ARCHIVO_RESULTADOS.open(newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        if lector.fieldnames != COLUMNAS:
            raise ValueError(
                f"{ARCHIVO_RESULTADOS} no tiene las columnas compartidas: {COLUMNAS}"
            )
        return {
            (fila["fase"], fila["variante"], int(fila["corrida"]))
            for fila in lector
        }


def nombre_monitor(variante: str, indice: int, calentamiento: bool) -> str:
    sufijo = "calentamiento" if calentamiento else f"corrida{indice}"
    return f"mia_{variante}_{sufijo}"


def ejecutar_preparacion() -> None:
    """Mide la lectura, RGB y resize secuencial de los tres conjuntos."""
    faltantes = [
        corrida for corrida in range(1, config.N_CORRIDAS + 1)
        if ("A_preparacion", "secuencial", corrida) not in corridas_registradas()
    ]
    if not faltantes:
        print("A_preparacion/secuencial: las cinco corridas ya están registradas.", flush=True)
        return

    total_imagenes = sum(len(pd.read_csv(config.MANIFIESTOS / f"{s}.csv")) for s in config.SPLITS)
    plan = [(0, True), *[(corrida, False) for corrida in faltantes]]
    for corrida, es_calentamiento in plan:
        nombre = nombre_monitor("secuencial", corrida, es_calentamiento)
        tipo = "calentamiento" if es_calentamiento else f"corrida {corrida}"
        print(f"A_preparacion/secuencial — {tipo}", flush=True)
        with Monitor(nombre), Cronometro() as cronometro:
            for split in config.SPLITS:
                x, y = preparar_split(split)
                del x, y
        gc.collect()
        print(f"  tiempo: {cronometro.segundos:.3f} s", flush=True)
        if not es_calentamiento:
            registrar(
                "mia",
                fase="A_preparacion",
                variante="secuencial",
                dispositivo="cpu",
                corrida=corrida,
                tiempo_s=cronometro.segundos,
                n_imagenes=total_imagenes,
                workers_o_hilos=1,
                batch="",
                epocas="",
                archivo=ARCHIVO_RESULTADOS,
            )


def ejecutar_entrenamiento(variante: str, hilos: int) -> None:
    """Mide cuatro épocas del pipeline original que vuelve a abrir cada JPG."""
    faltantes = [
        corrida for corrida in range(1, config.N_CORRIDAS + 1)
        if ("T_original", variante, corrida) not in corridas_registradas()
    ]
    if not faltantes:
        print(f"T_original/{variante}: las cinco corridas ya están registradas.", flush=True)
        return

    torch.set_num_threads(hilos)
    splits = cargar_splits()
    plan = [(0, True), *[(corrida, False) for corrida in faltantes]]
    for corrida, es_calentamiento in plan:
        seed_everything(config.SEED)
        loaders = make_loaders(
            splits,
            config.CLASES,
            batch_size=config.BATCH,
            image_size=config.IMG,
        )
        modelo = CNN2(len(config.CLASES))
        nombre = nombre_monitor(variante, corrida, es_calentamiento)
        tipo = "calentamiento" if es_calentamiento else f"corrida {corrida}"
        print(f"T_original/{variante} ({hilos} hilos) — {tipo}", flush=True)
        with Monitor(nombre), Cronometro() as cronometro:
            modelo_entrenado, historial, _ = train_model(
                modelo,
                loaders,
                learning_rate=config.LR,
                max_epochs=config.EPOCAS,
                patience=config.EPOCAS,
                device="cpu",
            )
        if len(historial) != config.EPOCAS:
            raise RuntimeError(
                f"Se ejecutaron {len(historial)} épocas; se esperaban {config.EPOCAS}."
            )
        print(f"  tiempo: {cronometro.segundos:.3f} s", flush=True)
        if not es_calentamiento:
            registrar(
                "mia",
                fase="T_original",
                variante=variante,
                dispositivo="cpu",
                corrida=corrida,
                tiempo_s=cronometro.segundos,
                n_imagenes=len(splits["train"]),
                workers_o_hilos=hilos,
                batch=config.BATCH,
                epocas=config.EPOCAS,
                archivo=ARCHIVO_RESULTADOS,
            )
        del modelo, modelo_entrenado, historial, loaders
        gc.collect()


def _datos_ordenados() -> pd.DataFrame:
    if not ARCHIVO_RESULTADOS.exists():
        raise FileNotFoundError(f"No existe {ARCHIVO_RESULTADOS}; ejecute primero las mediciones.")
    df = pd.read_csv(ARCHIVO_RESULTADOS)
    df = df[df.apply(lambda fila: (fila["fase"], fila["variante"]) in ORDEN, axis=1)].copy()
    if df.empty:
        raise ValueError("El CSV no contiene mediciones de Mia.")
    df["clave"] = list(zip(df["fase"], df["variante"]))
    df["etiqueta"] = df["clave"].map(ETIQUETAS)
    return df


def generar_figuras() -> None:
    """Crea las dos comparaciones y una figura de monitor por variante."""
    df = _datos_ordenados()
    config.FIGURAS.mkdir(parents=True, exist_ok=True)
    disponibles = [clave for clave in ORDEN if clave in set(df["clave"])]
    etiquetas = [ETIQUETAS[clave] for clave in disponibles]
    grupos = [df.loc[df["clave"] == clave, "tiempo_s"].astype(float).tolist() for clave in disponibles]
    medias = [sum(valores) / len(valores) for valores in grupos]
    desviaciones = [resumen(valores)["desv_s"] for valores in grupos]

    fig, eje = plt.subplots(figsize=(9, 5.5))
    barras = eje.bar(etiquetas, medias, yerr=desviaciones, capsize=5, color="#31688e")
    eje.set_yscale("log")
    eje.set_ylabel("Tiempo promedio (s, escala logarítmica)")
    eje.set_title("Tiempo promedio por variante (media ± desviación estándar)")
    eje.grid(axis="y", alpha=0.25)
    for barra, media in zip(barras, medias):
        eje.text(barra.get_x() + barra.get_width() / 2, media, f"{media:.2f} s",
                 ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURAS / "mia_promedios.png", dpi=150)
    plt.close(fig)

    fig, eje = plt.subplots(figsize=(9, 5.5))
    eje.boxplot(grupos, tick_labels=etiquetas, showmeans=True)
    eje.set_yscale("log")
    eje.set_ylabel("Tiempo (s, escala logarítmica)")
    eje.set_title("Distribución de las cinco corridas medidas")
    eje.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(config.FIGURAS / "mia_boxplot.png", dpi=150)
    plt.close(fig)

    for _, variante in disponibles:
        ruta_log = config.LOGS / f"mia_{variante}_corrida1.csv"
        if ruta_log.exists():
            salida = config.FIGURAS / f"mia_uso_cpu_{variante}.png"
            figura = graficar(ruta_log, salida)
            if figura is not None:
                plt.close("all")

    print(f"Figuras generadas en {config.FIGURAS}", flush=True)


def mostrar_resumen() -> None:
    df = _datos_ordenados()
    print("\nResumen de corridas medidas", flush=True)
    for fase, variante in ORDEN:
        valores = df.loc[(df["fase"] == fase) & (df["variante"] == variante), "tiempo_s"].astype(float).tolist()
        if valores:
            datos = resumen(valores)
            print(
                f"{fase}/{variante}: n={len(valores)}, media={datos['media_s']:.3f} s, "
                f"desv={datos['desv_s']:.3f} s, CV={100 * datos['cv']:.2f}%",
                flush=True,
            )


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solo",
        choices=("todo", "preparacion", "cpu_1hilo", "cpu_nhilos", "figuras"),
        default="todo",
        help="Permite ejecutar o reanudar una sola parte (por defecto: todo).",
    )
    return parser.parse_args()


def main() -> None:
    args = argumentos()
    hilos_disponibles = max(2, os.cpu_count() or torch.get_num_threads())
    print(
        f"Parámetros: seed={config.SEED}, imagen={config.IMG}×{config.IMG}, "
        f"batch={config.BATCH}, lr={config.LR}, épocas={config.EPOCAS}",
        flush=True,
    )
    print(f"CPU lógica: {os.cpu_count()}; variante multihilo: {hilos_disponibles}", flush=True)

    if args.solo in ("todo", "preparacion"):
        ejecutar_preparacion()
    if args.solo in ("todo", "cpu_1hilo"):
        ejecutar_entrenamiento("cpu_1hilo", 1)
    if args.solo in ("todo", "cpu_nhilos"):
        ejecutar_entrenamiento("cpu_nhilos", hilos_disponibles)

    if args.solo in ("todo", "figuras"):
        claves = corridas_registradas()
        if all((fase, variante, corrida) in claves
               for fase, variante in ORDEN
               for corrida in range(1, config.N_CORRIDAS + 1)):
            generar_figuras()
        else:
            print("Aún faltan corridas; las figuras finales se generarán al completar las 15.", flush=True)
    if ARCHIVO_RESULTADOS.exists():
        mostrar_resumen()


if __name__ == "__main__":
    main()
