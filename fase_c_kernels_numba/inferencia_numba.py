"""Forward completo de CNN2 en la GPU con los kernels propios de ``kernels.py``.

Flujo por lote (todo en la GPU, sin regresar a la CPU entre capas):

    uint8 (N,3,64,64) --H2D--> k_normalizar --> 8 x k_conv3x3_bn_relu (+3 x k_maxpool2x2)
        --> k_avgpool_global --> k_lineal(+ReLU) --> k_lineal --> logits --D2H--> CPU

* Los pesos se suben a la GPU UNA vez (``CNN2Numba.__init__``).
* Los tensores intermedios se reservan UNA vez por tamaño de lote y se reutilizan.
* Solo hay dos copias por lote: la entrada uint8 (4x menos bytes que float32) y los
  logits (N x 29). Todo lo demás vive en memoria de la GPU.

Validación:
    python -m fase_c_kernels_numba.inferencia_numba
compara contra la CNN2 original de PyTorch (CPU, fp32) sobre las 2,610 imágenes de test.
"""

from __future__ import annotations

import numpy as np
from numba import cuda

from comun import config
from fase_c_kernels_numba import kernels as K
from fase_c_kernels_numba.pesos import POOL_DESPUES, PesosCNN2, cargar_pesos


class CNN2Numba:
    def __init__(self, pesos: PesosCNN2, lote: int = config.BATCH, lado: int = config.IMG):
        self.lote = lote
        self.lado = lado
        self.convs = [(K.a_dispositivo(w), K.a_dispositivo(b)) for w, b in pesos.convs]
        # Pesos de las capas lineales transpuestos (K, M) para lecturas coalescidas
        self.lineales = [(K.a_dispositivo(w.T), K.a_dispositivo(b)) for w, b in pesos.lineales]
        self._buffers: dict[int, dict] = {}

    def _reservar(self, n: int) -> dict:
        """Reserva (una sola vez por tamaño de lote) todos los tensores intermedios."""
        if n in self._buffers:
            return self._buffers[n]
        f32 = np.float32
        lado = self.lado
        b = {"x_u8": cuda.device_array((n, 3, lado, lado), dtype=np.uint8),
             "x": cuda.device_array((n, 3, lado, lado), dtype=f32), "act": []}
        for i, (w, _) in enumerate(self.convs):
            cout = w.shape[0]
            b["act"].append(cuda.device_array((n, cout, lado, lado), dtype=f32))
            if i % 2 == 1 and POOL_DESPUES[i // 2]:
                lado //= 2
                b["act"].append(cuda.device_array((n, cout, lado, lado), dtype=f32))
        canales = self.convs[-1][0].shape[0]
        b["gap"] = cuda.device_array((n, canales), dtype=f32)
        b["fc1"] = cuda.device_array((n, self.lineales[0][0].shape[1]), dtype=f32)
        b["logits"] = cuda.device_array((n, self.lineales[1][0].shape[1]), dtype=f32)
        self._buffers[n] = b
        return b

    def forward_dispositivo(self, x_u8_dev) -> "cuda.devicearray.DeviceNDArray":
        """Forward de un lote que YA está en la GPU (uint8 NCHW). Devuelve logits en la GPU."""
        b = self._reservar(x_u8_dev.shape[0])
        K.lanzar_normalizar(x_u8_dev, b["x"])
        actual = b["x"]
        j = 0
        for i, (w, s) in enumerate(self.convs):
            K.lanzar_conv(actual, w, s, b["act"][j])
            actual = b["act"][j]
            j += 1
            if i % 2 == 1 and POOL_DESPUES[i // 2]:
                K.lanzar_maxpool(actual, b["act"][j])
                actual = b["act"][j]
                j += 1
        K.lanzar_avgpool(actual, b["gap"])
        (w1, b1), (w2, b2) = self.lineales
        K.lanzar_lineal(b["gap"], w1, b1, b["fc1"], True)
        K.lanzar_lineal(b["fc1"], w2, b2, b["logits"], False)
        return b["logits"]

    def predecir(self, x_u8_nchw: np.ndarray) -> np.ndarray:
        """Flujo completo desde la CPU: copia H2D por lote, forward y copia D2H de logits."""
        salidas = []
        for i in range(0, len(x_u8_nchw), self.lote):
            parte = np.ascontiguousarray(x_u8_nchw[i:i + self.lote])
            b = self._reservar(len(parte))
            b["x_u8"].copy_to_device(parte)
            salidas.append(self.forward_dispositivo(b["x_u8"]).copy_to_host())
        return np.concatenate(salidas)


def cargar_test_nchw(sufijo: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Test de la caché como uint8 NCHW contiguo (mismo layout que espera CNN2)."""
    from comun.datos_seq import cargar_cache

    x, y = cargar_cache("test", sufijo)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2)), y


def main() -> None:
    import torch

    from fase_c_kernels_numba.pesos import modelo_torch
    from secuencial.evaluation import evaluate_classification

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"GPU: {cuda.get_current_device().name}")
    x, y = cargar_test_nchw()
    pesos = cargar_pesos()
    logits_numba = CNN2Numba(pesos).predecir(x)

    with torch.inference_mode():
        ref = modelo_torch()(torch.from_numpy(x).float().div_(255)).numpy()

    dif = np.abs(logits_numba - ref)
    iguales = (logits_numba.argmax(1) == ref.argmax(1)).mean()
    nombres = lambda idx: [config.CLASES[i] for i in idx]  # noqa: E731
    m = evaluate_classification(nombres(y), nombres(logits_numba.argmax(1)), config.CLASES)["summary"]
    print(f"diferencia máxima de logits vs PyTorch: {dif.max():.2e} (media {dif.mean():.2e})")
    print(f"np.allclose(atol=1e-3): {np.allclose(logits_numba, ref, atol=1e-3)}")
    print(f"predicciones iguales a PyTorch: {100 * iguales:.2f} %")
    print(f"test con kernels Numba: accuracy={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
