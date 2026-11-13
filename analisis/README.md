# Análisis y línea base secuencial — Mia Fuentes (pasos 1, 2 y 3)

Todo corre en CPU. No necesita GPU ni código de nadie más: solo `secuencial/` y `comun/`.

## Archivos que se entregan aquí

| Archivo | Paso | Contenido |
|---|---|---|
| `operaciones.md` | 1 | Partes del programa + Tabla "No. / Operación / Descripción" + Tabla "Operación repetitiva / Datos / Repeticiones" |
| `contar_operaciones.py` | 1 | Cuenta lotes, decodificaciones JPG y FLOPs por capa de CNN2 (`torch.utils.flop_counter.FlopCounterMode`) |
| `bench_secuencial.py` | 2 | Mide el pipeline original en CPU y escribe `resultados/tiempos_mia.csv` |
| `tiempos_secuenciales.md` | 2 | Procedimiento, tabla media/desv/CV, gráficas y factores que influyen |
| `oportunidades.md` | 3 | Qué se puede paralelizar, tipo de paralelismo, CPU o GPU y por qué |
| `discusion_p3.md` | 6 | Pregunta 3: ¿qué factores justifican o limitan la utilización de CUDA? |

## Checklist

- [ ] Leer `secuencial/preprocessing.py` y `secuencial/cnn.py` y describir cada función.
- [ ] Tabla 1 con todas las operaciones, de la lectura del CSV hasta la predicción.
- [ ] Tabla 2 con repeticiones reales (12,180 train / 2,610 val / 2,610 test, batch 256, 4 épocas).
- [ ] `bench_secuencial.py` con dos variantes de entrenamiento: `torch.set_num_threads(1)` y hilos por defecto.
- [ ] Dos mediciones (nombres obligatorios de `comun/config.py::VOCABULARIO`, `registrar` rechaza otros):
      - `fase="A_preparacion", variante="secuencial"`: bucle PIL de `comun.datos_seq.preparar_split` para los 3 splits.
      - `fase="T_original", variante="cpu_1hilo"` y `"cpu_nhilos"`: el pipeline original **tal cual**
        (`make_loaders` + `secuencial.cnn.train_model` con `device="cpu"` y `patience=EPOCAS` para que no corte antes).
        Aquí cada época vuelve a abrir los JPG: es la línea base de extremo a extremo, no la de la fase B.
- [ ] 1 corrida de calentamiento + 5 medidas (`config.N_CALENTAMIENTO`, `config.N_CORRIDAS`).
- [ ] Envolver cada corrida en `comun.monitor.Monitor("mia_<variante>_<n>")` y generar la figura con `graficar`.
- [ ] Capturas de `htop`/`btop` durante la preparación (se ve 1 solo núcleo trabajando).
- [ ] Gráficas: barras con media ± desviación por fase y boxplot de corridas.

## Esqueleto sugerido de `bench_secuencial.py`

```python
import torch
from comun import config
from comun.medicion import Cronometro, registrar
from comun.monitor import Monitor
from comun.datos_seq import preparar_split
from comun.rutas import cargar_splits
from secuencial.cnn import CNN2, make_loaders, seed_everything, train_model

splits = cargar_splits()
for hilos in (1, torch.get_num_threads()):
    torch.set_num_threads(hilos)
    for n in range(config.N_CALENTAMIENTO + config.N_CORRIDAS):
        seed_everything(config.SEED)
        loaders = make_loaders(splits, config.CLASES, batch_size=config.BATCH, image_size=config.IMG)
        variante = "cpu_1hilo" if hilos == 1 else "cpu_nhilos"
        with Monitor(f"mia_{variante}_{n}"), Cronometro() as t:
            train_model(CNN2(len(config.CLASES)), loaders, learning_rate=config.LR,
                        max_epochs=config.EPOCAS, patience=config.EPOCAS, device="cpu")
        if n >= config.N_CALENTAMIENTO:
            registrar("mia", fase="T_original", variante=variante, dispositivo="cpu",
                      corrida=n, tiempo_s=t.segundos, n_imagenes=len(splits["train"]),
                      workers_o_hilos=hilos, epocas=config.EPOCAS)
```

Ojo: el entrenamiento en CPU con 1 hilo puede tardar bastante; si tarda demasiado se puede bajar
`N_CORRIDAS` solo para esa variante, pero se documenta en el informe.
