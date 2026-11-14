# Paso 1. Análisis de la solución secuencial

## Flujo completo

La solución comienza en `secuencial/data.py`. El módulo localiza las 29 carpetas de clases, normaliza sus nombres, enumera los archivos de imagen y selecciona de forma reproducible 600 imágenes por clase con la semilla 3084. Luego crea particiones estratificadas, verifica que no compartan rutas y guarda sus manifiestos. El proyecto ya contiene esos manifiestos oficiales: 12,180 imágenes de entrenamiento, 2,610 de validación y 2,610 de prueba, para un total de 17,400.

`secuencial/preprocessing.py` verifica que los tres manifiestos tengan las columnas y clases esperadas y que no haya duplicados ni fuga entre conjuntos. Para cada ruta, Pillow abre y decodifica el JPG, convierte la imagen a RGB, la redimensiona a 64 × 64 con interpolación bilineal y NumPy convierte los píxeles a `float32` y los divide entre 255. El resultado queda en el intervalo `[0, 1]` y con forma HWC.

En el entrenamiento, `secuencial/cnn.py::ASLDataset.__getitem__` repite ese preprocesamiento cada vez que el `DataLoader` solicita una imagen. Después convierte el arreglo a tensor y cambia HWC a CHW. `make_loaders` agrupa los ejemplos en lotes de 256, mezcla únicamente entrenamiento y mantiene `num_workers=0`; por ello la lectura y decodificación de los JPG ocurre en el proceso principal y no se solapa con el cálculo de la red.

CNN2 recibe tensores `N × 3 × 64 × 64`. Tiene cuatro bloques dobles de convolución 3 × 3; cada convolución está seguida por BatchNorm y ReLU. Los tres primeros bloques reducen la resolución con MaxPool 2 × 2, y el cuarto termina con promedio global adaptativo. El clasificador aplana los 192 canales, aplica dos capas lineales (192 → 128 → 29), una ReLU intermedia y dropout durante entrenamiento. La salida contiene 29 *logits*, uno por clase.

`train_model` calcula entropía cruzada. Para entrenamiento limpia gradientes, ejecuta *forward*, `backward` y un paso de Adam por lote. Para validación ejecuta solo *forward* y pérdida con gradientes desactivados. Conserva el estado con menor pérdida de validación; en el benchmark se usa `patience=4`, por lo que siempre se completan las cuatro épocas. Finalmente, `predict` pone el modelo en evaluación, clasifica el conjunto de prueba sin gradientes y traduce los índices a etiquetas. `secuencial/evaluation.py` calcula exactitud, precisión, *recall*, F1 macro, métricas por clase y matriz de confusión.

## Tabla 1. Operaciones identificadas

| Número | Operación identificada | Descripción de su funcionamiento |
|---:|---|---|
| 1 | Localización del dataset | Busca la carpeta que contiene las 29 subcarpetas de clases ASL. |
| 2 | Lectura del dataset | Enumera archivos JPG/JPEG/PNG y asocia cada ruta con la etiqueta de su carpeta. |
| 3 | Selección y mezcla | Selecciona 600 imágenes por clase y mezcla las 17,400 rutas usando la semilla 3084. |
| 4 | División estratificada | Produce train, validation y test conservando 29 clases y sin compartir imágenes. |
| 5 | Lectura y validación de manifiestos | Lee los CSV y comprueba columnas, rutas únicas, clases compatibles y ausencia de fuga. |
| 6 | Apertura y decodificación JPG | `Image.open` lee el archivo comprimido y Pillow reconstruye sus píxeles. |
| 7 | Conversión RGB | Convierte cualquier modo de entrada a tres canales RGB uniformes. |
| 8 | Redimensionamiento | Reduce la imagen original de 200 × 200 a 64 × 64 mediante interpolación bilineal. |
| 9 | Normalización | Convierte a `float32` y divide cada canal entre 255 para obtener valores en `[0, 1]`. |
| 10 | Conversión y transposición | Convierte NumPy HWC a tensor PyTorch CHW. |
| 11 | Preparación de lotes | `DataLoader` agrupa hasta 256 ejemplos; mezcla train y usa `num_workers=0`. |
| 12 | Convoluciones | Ocho convoluciones 3 × 3 extraen rasgos locales y aumentan canales 3 → 32 → 64 → 128 → 192. |
| 13 | BatchNorm | Ocho capas normalizan activaciones por canal; durante train actualizan estadísticas móviles. |
| 14 | ReLU | Sustituye valores negativos por cero después de cada convolución y de la primera capa lineal. |
| 15 | Pooling | Tres MaxPool reducen 64 → 32 → 16 → 8; AdaptiveAvgPool resume cada canal a 1 × 1. |
| 16 | Capas lineales | Multiplican el vector de 192 rasgos por matrices 192 × 128 y 128 × 29 para producir logits. |
| 17 | Pérdida | `CrossEntropyLoss` combina *log-softmax* y entropía cruzada frente a la clase real. |
| 18 | Backpropagation | Calcula gradientes de la pérdida respecto de los 870,461 parámetros entrenables. |
| 19 | Actualización Adam | Mantiene momentos de primer y segundo orden y actualiza cada parámetro con LR `1e-3`. |
| 20 | Validación | Ejecuta *forward* sin gradientes, calcula pérdida y exactitud y conserva el mejor estado. |
| 21 | Predicción | Ejecuta el modelo en modo evaluación, toma `argmax` y traduce índices a las 29 etiquetas. |
| 22 | Evaluación | Calcula exactitud, métricas macro y por clase, y una matriz de confusión 29 × 29. |

## Tabla 2. Operaciones repetitivas

Los lotes son exactos: `ceil(12180/256) = 48` y `ceil(2610/256) = 11`. El último lote contiene 148 imágenes en train y 50 en validation/test. La columna distingue explícitamente repeticiones por imagen y por lote.

| Operación repetitiva | Datos que procesa | Cantidad de repeticiones |
|---|---|---:|
| Preparación secuencial de los tres conjuntos | 17,400 imágenes, una vez cada una | 17,400 por imagen |
| Apertura/decodificación, RGB, resize, normalización y CHW durante train | 12,180 imágenes por época × 4 | 48,720 por imagen |
| Apertura/decodificación, RGB, resize, normalización y CHW durante validation | 2,610 imágenes por época × 4 | 10,440 por imagen |
| Preprocesamiento dentro del entrenamiento original | Train + validation durante 4 épocas | 59,160 por imagen |
| Preparación de lotes de train | 48 lotes por época × 4 | 192 por lote |
| Preparación de lotes de validation | 11 lotes por época × 4 | 44 por lote |
| *Forward* durante el entrenamiento medido | 48 + 11 lotes por época × 4 | 236 por lote; 59,160 por imagen |
| Cálculo de pérdida | Un cálculo por lote de train y validation | 236 por lote |
| Limpieza de gradientes | Un paso por lote de train | 192 por lote |
| Backpropagation | Un paso por lote de train | 192 por lote |
| Actualización con Adam | Un paso por lote de train | 192 por lote |
| Validación | 11 lotes por época × 4 | 44 por lote; 10,440 por imagen |
| Predicción final de test | 2,610 imágenes agrupadas en 11 lotes | 11 por lote; 2,610 por imagen |
| *Forward* del flujo completo, incluyendo una predicción de test | 236 lotes de train/validation + 11 de test | 247 por lote; 61,770 por imagen |
| Cada capa convolucional de CNN2 en el flujo completo | Ocho convoluciones por cada uno de 61,770 ejemplos | 494,160 invocaciones capa-imagen |
| Cálculo de métricas de test | 2,610 pares de etiqueta real/predicha y 29 clases | 2,610 comparaciones; una agregación final |

## Costo de CNN2

Se ejecutó `py -3.12 -m analisis.contar_operaciones` en CPU con PyTorch 2.14.1. Con la convención de contar una multiplicación y una suma como dos FLOPs, `FlopCounterMode` contabilizó **379,903,232 FLOPs por imagen** en los operadores que soporta. El contador propio, que añade aproximaciones para BatchNorm, ReLU y pooling, obtuvo **382,987,805 FLOPs por imagen**. Esto equivale aproximadamente a 98,044,878,080 FLOPs para un lote completo de 256.

La suma de todos los *forward* de train y validation durante cuatro épocas es aproximadamente **22,657,558,543,800 FLOPs**. No representa el entrenamiento completo: *backward*, entropía cruzada y Adam agregan trabajo. Como referencia de orden de magnitud, el programa imprime por separado un proxy de **59,975,890,263,000 FLOPs**, calculado como tres veces el *forward* para las imágenes de train más un *forward* para validation; no se presenta como una medición exacta de PyTorch.

| Capa de CNN2 | Forma de salida para una imagen | FLOPs aproximados |
|---|---:|---:|
| Conv 3 → 32 | 32 × 64 × 64 | 7,208,960 |
| Conv 32 → 32 | 32 × 64 × 64 | 75,628,544 |
| Conv 32 → 64 | 64 × 32 × 32 | 37,814,272 |
| Conv 64 → 64 | 64 × 32 × 32 | 75,563,008 |
| Conv 64 → 128 | 128 × 16 × 16 | 37,781,504 |
| Conv 128 → 128 | 128 × 16 × 16 | 75,530,240 |
| Conv 128 → 192 | 192 × 8 × 8 | 28,323,840 |
| Conv 192 → 192 | 192 × 8 × 8 | 42,479,616 |
| Linear 192 → 128 | 128 | 49,280 |
| Linear 128 → 29 | 29 | 7,453 |

Los valores incluyen el sesgo de cada Conv/Linear. El desglose completo de BatchNorm, ReLU y pooling se obtiene directamente al ejecutar `contar_operaciones.py`.
