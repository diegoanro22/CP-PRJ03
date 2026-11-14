# Paso 3. Oportunidades de paralelización

La solución secuencial mezcla dos tipos de trabajo. La preparación de archivos tiene tareas independientes, control irregular y acceso a disco; conviene distribuirla entre núcleos de CPU. La red neuronal aplica la misma aritmética sobre muchos píxeles, canales y ejemplos; conviene mantener lotes en GPU y explotar miles de hilos CUDA. Esta sección identifica las oportunidades, pero no implementa los kernels ni el entrenamiento CUDA asignados a los otros integrantes.

| Operación | ¿Es paralelizable? | Tipo de paralelismo | Hardware recomendado | Justificación técnica |
|---|---|---|---|---|
| Lectura de imágenes | Sí, con límite | Datos independientes por archivo e I/O concurrente | CPU | Cada ruta puede leerse por separado. Varios procesos pueden ocultar latencia, pero demasiadas lecturas compiten por el mismo disco y reducen el beneficio. |
| Decodificación JPEG | Sí | Datos independientes por imagen | CPU | La decodificación tiene control y tamaños de bloques irregulares; bibliotecas optimizadas en procesos CPU evitan el costo de transferir bytes comprimidos a GPU. |
| Conversión RGB | Sí | Datos independientes por imagen y píxel | CPU | Es sencilla y se puede fusionar con la decodificación/preparación en cada trabajador CPU. Su baja intensidad aritmética no justifica una transferencia aislada a GPU. |
| Redimensionamiento bilineal | Sí | Datos independientes por imagen y píxel de salida | CPU | Cada imagen es independiente. En este pipeline los datos nacen en disco y el resize acompaña a Pillow, por lo que CPU multinúcleo evita transferencias adicionales. |
| Normalización `/255` | Sí | Paralelismo de datos por elemento | CPU o GPU; GPU si el lote ya reside allí | Cada canal se divide de forma independiente. Es una operación de poco cálculo respecto del movimiento de memoria; conviene ejecutarla donde ya estén los datos. |
| Convoluciones | Sí, ampliamente | Paralelismo por lote, canal de salida y píxel; reducción sobre canal/kernel | GPU | Son el costo dominante: ocho capas y alrededor de 380 millones de FLOPs soportados por imagen. La alta regularidad y reutilización de pesos favorecen CUDA/cuDNN. |
| Funciones de activación | Sí | Paralelismo por elemento | GPU | ReLU no tiene dependencias entre elementos y puede ejecutarse con muchos hilos; su beneficio aumenta al quedar fusionada o encadenada sin regresar a CPU. |
| Pooling | Sí | Paralelismo por elemento de salida con reducción local | GPU | Cada ventana o canal puede reducirse en paralelo. La sincronización es local y los tensores ya provienen de convoluciones en GPU. |
| Multiplicación de matrices | Sí, ampliamente | Paralelismo por filas/columnas con reducción del producto punto | GPU | Las capas lineales y las convoluciones transformadas en GEMM permiten bloques y *tiling*, alta reutilización y librerías CUDA optimizadas. |
| Cálculo de pérdida | Sí | Paralelismo por ejemplo y reducción del lote | GPU | El *softmax* y la pérdida se calculan por ejemplo; al final solo se reduce el lote. Mantener logits y etiquetas en GPU evita una copia intermedia. |
| Cálculo de gradientes | Sí, ampliamente | Paralelismo de datos y reducciones | GPU | *Backward* contiene convoluciones y multiplicaciones similares al *forward*. Los gradientes de ejemplos se acumulan mediante reducciones eficientes. |
| Actualización de parámetros | Parcialmente | Paralelismo por parámetro | GPU | Adam actualiza elementos independientes de pesos y momentos, pero requiere que el gradiente esté listo. Es menos costoso que convolución, aunque evita copiar gradientes a CPU. |
| Orden entre capas | No en general | Dependencia secuencial | CPU/GPU | La salida de una capa es la entrada de la siguiente. Se paraliza dentro de cada capa y entre ejemplos, no se ejecutan arbitrariamente todas las capas a la vez. |

## Asignación propuesta por tipo de trabajo

La preparación de archivos debe permanecer en CPU multinúcleo: cada trabajador puede abrir, decodificar, convertir y redimensionar una imagen completa. Esto se ajusta al acceso al sistema de archivos y evita usar la GPU para tareas irregulares de baja intensidad aritmética. El resultado preparado puede almacenarse una sola vez y entregarse por lotes al entrenamiento.

Las operaciones matemáticas deben ir a GPU. Las ocho convoluciones de CNN2 concentran casi todos los FLOPs y operan sobre arreglos densos y regulares. Una vez transferido un lote, convolución, BatchNorm, ReLU, pooling, capas lineales, pérdida y *backward* deben permanecer en la GPU para amortizar la transferencia y evitar sincronizaciones CPU–GPU entre capas.

## Relación con los cuellos de botella secuenciales

El análisis encontró dos cuellos de botella diferentes:

1. `num_workers=0` hace que el proceso principal abra y decodifique 59,160 imágenes durante cuatro épocas. La primera preparación medida fue mucho más lenta con caché fría, lo cual confirma que el disco y la caché afectan esta fase. Procesar imágenes independientes en varios núcleos y reutilizar una caché preparada reduce esa parte serial.
2. CNN2 realiza cerca de 380 millones de FLOPs por imagen y repite *forward*, *backward* y Adam en 192 lotes de entrenamiento. Aumentar hilos CPU puede acelerar bibliotecas matemáticas, pero la estructura masivamente regular de convoluciones y matrices tiene mejor correspondencia con una GPU CUDA.

La aceleración total estará limitada por la parte que permanezca secuencial: lectura de manifiestos, coordinación del `DataLoader`, acceso a archivos, lanzamiento de trabajo y transferencias. Por eso conviene paralelizar cada fase con el hardware apropiado y medirlas por separado.
