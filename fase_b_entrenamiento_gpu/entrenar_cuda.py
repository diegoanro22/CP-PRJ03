"""Fase B: entrenamiento de CNN2 con datos en memoria y cómputo en CUDA (paso 5).

Mismo modelo (``secuencial.cnn.CNN2``) e hiperparámetros (``comun.config``) que el
original. Lo que cambia respecto a ``secuencial.cnn.train_model``:

* Los datos salen de la caché en memoria (``cargar_tensores``): no se abre ningún JPG
  dentro del bucle. La preparación de imágenes pertenece a la CPU y ocurre antes.
* Los lotes viajan como ``uint8`` por buffers *pinned* con ``non_blocking=True``
  (copia por DMA, solapada con el cómputo del lote anterior) y el ``/255`` se hace
  ya en el dispositivo.
* ``cudnn.benchmark=True`` en CUDA; variante ``cuda_amp`` con ``autocast`` bf16.
* Con eventos CUDA se separa el tiempo de transferencia H2D del de cómputo.

Las cuatro variantes (``cpu_1hilo``, ``cpu_nhilos``, ``cuda``, ``cuda_amp``) usan
exactamente el mismo bucle y la misma caché; solo cambian dispositivo, hilos y AMP.
Así el speedup mide únicamente la matemática de la red.

Uso:
    python -m fase_b_entrenamiento_gpu.entrenar_cuda --variante cuda
"""

from __future__ import annotations

import argparse
import os

import torch
from torch import nn

from comun import config
from comun.datos_seq import cargar_tensores
from comun.medicion import Cronometro
from secuencial.cnn import CNN2
from secuencial.evaluation import evaluate_classification

# variante -> (dispositivo, amp, hilos de CPU de PyTorch)
VARIANTES = {
    "cpu_1hilo": ("cpu", False, 1),
    "cpu_nhilos": ("cpu", False, os.cpu_count() or 1),
    "cuda": ("cuda", False, 1),
    "cuda_amp": ("cuda", True, 1),
}


class Datos:
    """Los tres splits de la caché como tensores uint8 NCHW en memoria de CPU."""

    def __init__(self, sufijo: str = ""):
        self.splits = {s: cargar_tensores(s, sufijo) for s in config.SPLITS}

    def __getitem__(self, split: str):
        return self.splits[split]


class Alimentador:
    """Reparte lotes de la caché en memoria al dispositivo de entrenamiento.

    En CUDA usa dos buffers *pinned* alternados: mientras la GPU calcula el lote
    ``k`` la CPU ya junta el lote ``k+1`` y lo copia con ``non_blocking=True``.
    Un evento por buffer evita sobrescribirlo antes de que termine su copia.
    Cada copia queda entre dos eventos CUDA para medir la transferencia H2D.
    """

    def __init__(self, dispositivo: str, batch: int):
        self.dispositivo = dispositivo
        self.batch = batch
        self.cuda = dispositivo == "cuda"
        self.h2d: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        if self.cuda:
            lado = config.IMG
            self._bx = [torch.empty((batch, 3, lado, lado), dtype=torch.uint8).pin_memory() for _ in range(2)]
            self._by = [torch.empty(batch, dtype=torch.int64).pin_memory() for _ in range(2)]
            self._listo: list[torch.cuda.Event | None] = [None, None]

    def lotes(self, x: torch.Tensor, y: torch.Tensor, orden: torch.Tensor):
        for k, i in enumerate(range(0, len(orden), self.batch)):
            idx = orden[i:i + self.batch]
            if not self.cuda:
                yield x.index_select(0, idx), y.index_select(0, idx)
                continue
            b = k % 2
            if self._listo[b] is not None:
                self._listo[b].synchronize()  # la copia anterior de este buffer ya terminó
            xb, yb = self._bx[b][:len(idx)], self._by[b][:len(idx)]
            torch.index_select(x, 0, idx, out=xb)
            torch.index_select(y, 0, idx, out=yb)
            inicio, fin = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            inicio.record()
            xg = xb.to(self.dispositivo, non_blocking=True)
            yg = yb.to(self.dispositivo, non_blocking=True)
            fin.record()
            self._listo[b] = fin
            self.h2d.append((inicio, fin))
            yield xg, yg


def _pasada(modelo, alim, x, y, orden, entrenar, optimizador, criterio, amp, eventos_computo):
    """Una pasada completa (train o validación). Devuelve (pérdida, accuracy) sin sincronizar."""
    dispositivo = alim.dispositivo
    modelo.train(entrenar)
    perdida = torch.zeros((), device=dispositivo)
    aciertos = torch.zeros((), dtype=torch.int64, device=dispositivo)
    with torch.set_grad_enabled(entrenar):
        for xb, yb in alim.lotes(x, y, orden):
            if alim.cuda:
                inicio = torch.cuda.Event(enable_timing=True)
                fin = torch.cuda.Event(enable_timing=True)
                inicio.record()
            xb = xb.float().div_(255)  # misma normalización del original, ya en el dispositivo
            if entrenar:
                optimizador.zero_grad()
            with torch.autocast(device_type=dispositivo, dtype=torch.bfloat16, enabled=amp):
                logits = modelo(xb)
                loss = criterio(logits, yb)
            if entrenar:
                loss.backward()
                optimizador.step()
            perdida += loss.detach().float() * yb.size(0)
            aciertos += (logits.argmax(dim=1) == yb).sum()
            if alim.cuda:
                fin.record()
                eventos_computo.append((inicio, fin))
    n = len(orden)
    return perdida / n, aciertos / n


def entrenar(variante: str, datos: Datos, batch: int = config.BATCH, epocas: int = config.EPOCAS,
             seed: int = config.SEED, verbose: bool = False) -> dict:
    """Entrena CNN2 ``epocas`` épocas fijas (sin early stopping), validando cada una.

    Devuelve el modelo (con los pesos de menor pérdida de validación, como el
    original), la historia por época y los tiempos total / transferencia / cómputo.
    """
    dispositivo, amp, hilos = VARIANTES[variante]
    cuda = dispositivo == "cuda"
    torch.set_num_threads(hilos)
    torch.backends.cudnn.benchmark = cuda
    torch.manual_seed(seed)
    modelo = CNN2(len(config.CLASES)).to(dispositivo)  # pesos iniciales idénticos en todas las variantes
    optimizador = torch.optim.Adam(modelo.parameters(), lr=config.LR)
    criterio = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)  # mismo orden de lotes en todas las variantes
    x_tr, y_tr = datos["train"]
    x_va, y_va = datos["validation"]
    orden_va = torch.arange(len(y_va))
    alim = Alimentador(dispositivo, batch)
    eventos_computo: list = []
    historia: list[dict] = []
    mejor_perdida, mejor_estado = float("inf"), None

    with Cronometro(cuda=cuda) as total:
        for epoca in range(1, epocas + 1):
            tl, ta = _pasada(modelo, alim, x_tr, y_tr, torch.randperm(len(y_tr), generator=gen), True,
                             optimizador, criterio, amp, eventos_computo)
            vl, va = _pasada(modelo, alim, x_va, y_va, orden_va, False, None, criterio, amp, eventos_computo)
            fila = {"epoca": epoca, "train_loss": tl.item(), "train_acc": ta.item(),
                    "val_loss": vl.item(), "val_acc": va.item()}
            historia.append(fila)
            if fila["val_loss"] < mejor_perdida - 1e-5:
                mejor_perdida = fila["val_loss"]
                mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
            if verbose:
                print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in fila.items()),
                      flush=True)
    modelo.load_state_dict(mejor_estado)

    transferencia = computo = 0.0
    if cuda:  # Cronometro ya sincronizó la GPU: todos los eventos están completos
        transferencia = sum(a.elapsed_time(b) for a, b in alim.h2d) / 1000
        computo = sum(a.elapsed_time(b) for a, b in eventos_computo) / 1000
    return {"modelo": modelo, "historia": historia, "tiempo_s": total.segundos,
            "transferencia_s": transferencia, "computo_s": computo}


@torch.inference_mode()
def evaluar(modelo: nn.Module, datos: Datos, dispositivo: str, batch: int = config.BATCH, split: str = "test") -> dict:
    """Accuracy y F1 macro con la misma función de métricas del Laboratorio 3 (en fp32)."""
    modelo.eval()
    x, y = datos[split]
    alim = Alimentador(dispositivo, batch)
    pred = []
    for xb, _ in alim.lotes(x, y, torch.arange(len(y))):
        pred.append(modelo(xb.float().div_(255)).argmax(dim=1).cpu())
    pred = torch.cat(pred)
    nombres = lambda t: [config.CLASES[i] for i in t.tolist()]  # noqa: E731
    return evaluate_classification(nombres(y), nombres(pred), config.CLASES)["summary"]


def informar_dispositivo() -> None:
    """Evidencia de que se usó la GPU (se imprime siempre al inicio)."""
    print(f"torch {torch.__version__} | CUDA runtime {torch.version.cuda} | cuda disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | cuDNN {torch.backends.cudnn.version()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variante", choices=sorted(VARIANTES), default="cuda")
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--epocas", type=int, default=config.EPOCAS)
    p.add_argument("--cache-sufijo", default="", help='"_par" para usar la caché de la fase A')
    a = p.parse_args()

    informar_dispositivo()
    datos = Datos(a.cache_sufijo)
    print(f"variante={a.variante} batch={a.batch} épocas={a.epocas} hilos CPU={VARIANTES[a.variante][2]}")
    r = entrenar(a.variante, datos, a.batch, a.epocas, verbose=True)
    m = evaluar(r["modelo"], datos, VARIANTES[a.variante][0], a.batch)
    print(f"tiempo total: {r['tiempo_s']:.2f} s | transferencia H2D: {r['transferencia_s']:.2f} s | cómputo: {r['computo_s']:.2f} s")
    print(f"test: accuracy={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
