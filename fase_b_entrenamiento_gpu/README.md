# Fase B: entrenamiento de CNN2 en GPU con CUDA — Diego Lopez (paso 5)

Necesita GPU NVIDIA. No depende de las otras carpetas: usa `secuencial/` (modelo CNN2 original) y `comun/`.

## Archivos que se entregan aquí

| Archivo | Contenido |
|---|---|
| `entrenar_cuda.py` | Entrenamiento de CNN2 con datos en memoria y cómputo en `cuda` |
| `bench_entrenamiento.py` | CPU 1 hilo vs CPU N hilos vs GPU vs GPU+AMP, 5 corridas, `resultados/tiempos_lopez.csv` (`fase="B_entrenamiento"`, variantes `cpu_1hilo`, `cpu_nhilos`, `cuda`, `cuda_amp`) |
| `barrido_batch.py` | Batch 64/128/256/512 en GPU: tiempo por época y utilización |
| `discusion_p2.md` | Pregunta 2: ¿cómo CUDA permitió reducir o cambiar el tiempo? |

## Qué cambia respecto al original (`secuencial/cnn.py`)

| Original | Versión CUDA | Por qué |
|---|---|---|
| `ASLDataset` abre y redimensiona cada JPG en **cada época**, con `num_workers=0` | `TensorDataset` sobre `comun.datos_seq.cargar_tensores(split)`: caché cargada **una sola vez**, ya en NCHW | La GPU no queda esperando al disco y a PIL |
| tensores en memoria paginable | `pin_memory=True` + `.to("cuda", non_blocking=True)` | Copias H2D por DMA, solapables con cómputo |
| `device="cpu"` | `device="cuda"`, `torch.backends.cudnn.benchmark = True` | cuDNN elige el algoritmo de convolución más rápido |
| float32 | variante extra con `torch.autocast("cuda", dtype=torch.bfloat16)` | Usa Tensor Cores |
| `/255` en CPU | `x.float().div_(255)` ya en GPU, datos viajan como uint8 | 4x menos bytes por PCIe |

Separación del pipeline (lo que más pesa en la rúbrica): la preparación de imágenes ocurre **antes**
y en CPU (`comun.datos_seq` o la caché paralela de la fase A con `cargar_cache(split, sufijo="_par")`);
dentro del bucle de entrenamiento solo hay matemática en GPU.

## Checklist

- [ ] `python -m comun.datos_seq` una vez para tener la caché.
- [ ] `entrenar_cuda.py` reutiliza `secuencial.cnn.CNN2` (misma arquitectura) e hiperparámetros de `comun/config.py`.
- [ ] Cargar con `cargar_tensores(split)`: la caché en disco es NHWC (N, 64, 64, 3) y la función ya hace
      `permute(0, 3, 1, 2)`. No usar `np.load` directo sin permutar, porque `Conv2d` espera (N, 3, 64, 64).
- [ ] **Comparación justa**: las variantes `cpu_1hilo` y `cpu_nhilos` usan **el mismo bucle y la misma caché en memoria**
      que `cuda`; lo único que cambia es `device`. Así el speedup de B mide solo la matemática.
      (El pipeline original con JPG en cada época lo mide Mia como `T_original`.)
- [ ] Medir con **eventos CUDA** (`torch.cuda.Event(enable_timing=True)`) tres componentes por época:
      transferencia H2D, forward+backward+step, y total. Registrar fases `B_transferencia` y `B_computo` además de `B_entrenamiento`.
- [ ] Barrido de batch registrado como `fase="B_barrido_batch"`, `variante="batch_<n>"`.
- [ ] `Cronometro(cuda=True)` para el tiempo total (sincroniza la GPU).
- [ ] 1 calentamiento + 5 corridas por variante; 4 épocas fijas (`patience` desactivado).
- [ ] Accuracy y F1 de test con `secuencial.evaluation.evaluate_classification` (debe rondar 0.90 como el original).
- [ ] `Monitor("lopez_<variante>_<n>")` en cada corrida + `graficar`.
- [ ] En otra terminal durante el entrenamiento: `nvidia-smi dmon -s um -o T > resultados/logs/lopez_dmon.txt`
      y capturas de `nvidia-smi` y `nvtop` en `resultados/capturas/`.
- [ ] Imprimir en el log `torch.cuda.get_device_name(0)` y `torch.version.cuda` (evidencia de que se usó la GPU).
