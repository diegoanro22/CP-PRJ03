"""CONTRATO DEL EQUIPO. No cambiar valores sin avisar a los tres integrantes.

Todos los benchmarks leen estos valores para que los tiempos sean comparables.
Son los mismos hiperparámetros que usó CNN2 en el Laboratorio 3 de Data Science.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
MANIFIESTOS = RAIZ / "data" / "manifests"
CACHE = RAIZ / "cache"
RESULTADOS = RAIZ / "resultados"
LOGS = RESULTADOS / "logs"
FIGURAS = RESULTADOS / "figuras"
MODELO_CNN2 = RAIZ / "modelos" / "cnn2.pt"

SEED = 3084
IMG = 64                 # lado de la imagen tras el resize (64x64)
BATCH = 256              # batch del notebook original
LR = 1e-3
EPOCAS = 4               # épocas FIJAS: sin early stopping al medir tiempos
N_CORRIDAS = 5           # corridas medidas por variante
N_CALENTAMIENTO = 1      # corridas extra al inicio que se descartan

SPLITS = ("train", "validation", "test")
CLASES = [
    "A", "B", "C", "D", "DELETE", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
    "NOTHING", "O", "P", "Q", "R", "S", "SPACE", "T", "U", "V", "W", "X", "Y", "Z",
]  # orden alfabético, igual que sorted() en el notebook original

KAGGLE_HANDLE = "grassknoted/asl-alphabet"

# ---------------------------------------------------------------------------
# VOCABULARIO OBLIGATORIO de fases y variantes para resultados/tiempos_*.csv.
# comun.medicion.registrar() rechaza cualquier valor fuera de esta lista, así
# el notebook siempre encuentra la variante base de cada fase.
#   "{p}" = número de procesos / tamaño de lote (entero).
# ---------------------------------------------------------------------------
VOCABULARIO = {
    # Mia: pipeline original TAL CUAL (lee y redimensiona JPG en cada época dentro
    # del DataLoader, num_workers=0). Es la línea base de extremo a extremo.
    "T_original": ["cpu_1hilo", "cpu_nhilos"],
    # Preparación de imágenes a la caché (Mia mide "secuencial"; Rosales mide ambas).
    "A_preparacion": ["secuencial", "procesos_{p}"],
    # Entrenamiento leyendo SIEMPRE desde la caché en memoria (sin JPG ni PIL en
    # el bucle), para que el speedup sea solo de la matemática. Lopez.
    "B_entrenamiento": ["cpu_1hilo", "cpu_nhilos", "cuda", "cuda_amp"],
    "B_transferencia": ["cuda", "cuda_amp"],   # copias H2D medidas con eventos CUDA
    "B_computo": ["cuda", "cuda_amp"],         # forward + backward + step
    "B_barrido_batch": ["batch_{p}"],
    # Inferencia de CNN2 sobre test con kernels propios. Rosales.
    "C_inferencia": ["numpy_cpu", "numba_cuda", "numba_cuda_con_transferencias", "torch_cuda"],
}

# Variante contra la que se calcula el speedup de cada fase.
BASE_SPEEDUP = {
    "A_preparacion": "secuencial",
    "B_entrenamiento": "cpu_1hilo",
    "C_inferencia": "numpy_cpu",
    "T_original": "cpu_1hilo",
}
