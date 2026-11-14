# Paso 2. Tiempos de la solución secuencial

## Objetivo

El objetivo fue establecer una línea base reproducible en CPU para dos partes del proyecto: la preparación secuencial de los 17,400 archivos y el entrenamiento original de CNN2, que mantiene la lectura y transformación de JPG dentro del `DataLoader`. Se comparó el entrenamiento con un hilo de PyTorch frente al mismo entrenamiento con los ocho hilos lógicos disponibles. No se midió descarga, instalación ni generación de manifiestos.

## Hardware y software

Las mediciones se realizaron en una Dell Inspiron 5406 2‑in‑1 con Intel Core i5‑1135G7 a 2.40 GHz, 4 núcleos físicos y 8 hilos lógicos, 7.73 GiB de RAM y Windows 10 Home Single Language 10.0.19045. Se usó Python 3.12, PyTorch 2.14.1 CPU, Pillow 11.3.0 y los paquetes declarados en `requirements-cpu.txt`. No se utilizó GPU.

## Parámetros

| Parámetro | Valor |
|---|---:|
| Semilla | 3084 |
| Imagen | 64 × 64 RGB |
| Clases | 29 |
| Train / validation / test | 12,180 / 2,610 / 2,610 |
| Tamaño de lote | 256 |
| Tasa de aprendizaje | `1e-3` |
| Épocas | 4 completas |
| Paciencia | 4 |
| Trabajadores del `DataLoader` | 0 |
| Corridas de calentamiento | 1 por variante, descartada |
| Corridas medidas | 5 por variante |

## Procedimiento experimental

`analisis/bench_secuencial.py` ejecutó las variantes en CPU. Antes de cada entrenamiento restableció Python, NumPy y PyTorch con la semilla 3084, creó nuevos cargadores y una nueva CNN2, y fijó `patience=EPOCAS` para completar cuatro épocas. La variante `cpu_1hilo` usó `torch.set_num_threads(1)` y `cpu_nhilos` usó ocho hilos; ambas conservaron `num_workers=0`.

Cada variante tuvo una corrida de calentamiento y cinco corridas medidas. `comun.medicion.Cronometro` tomó el tiempo de pared y `comun.monitor.Monitor` muestreó cada núcleo cada 0.5 s. `comun.medicion.registrar` escribió las quince filas con el vocabulario compartido. El script detecta claves ya presentes por fase, variante y corrida, por lo que puede reanudarse sin duplicarlas.

Los calentamientos duraron 149.579 s para preparación, 1,536.231 s para `cpu_1hilo` y 859.136 s para `cpu_nhilos`. Se conservaron sus logs como evidencia, pero ninguno se escribió en `tiempos_mia.csv` ni se incluyó en las estadísticas.

La preparación recorrió train, validation y test mediante `comun.datos_seq.preparar_split`; el cronómetro incluyó apertura, decodificación, RGB, resize y apilado, pero no guardado a `.npy`. El entrenamiento incluyó los JPG solicitados por el `DataLoader`, *forward*, pérdida, *backward*, Adam y validación de cada época.

## Resultados

Los tiempos son reales y están en `resultados/tiempos_mia.csv`. El calentamiento no aparece en esta tabla ni participa en las estadísticas.

| Fase | Variante | Corrida | Hilos | Tiempo (s) |
|---|---|---:|---:|---:|
| A_preparacion | secuencial | 1 | 1 | 16.372 |
| A_preparacion | secuencial | 2 | 1 | 16.629 |
| A_preparacion | secuencial | 3 | 1 | 15.969 |
| A_preparacion | secuencial | 4 | 1 | 16.341 |
| A_preparacion | secuencial | 5 | 1 | 16.313 |
| T_original | cpu_1hilo | 1 | 1 | 1,573.185 |
| T_original | cpu_1hilo | 2 | 1 | 1,373.901 |
| T_original | cpu_1hilo | 3 | 1 | 1,551.790 |
| T_original | cpu_1hilo | 4 | 1 | 1,582.510 |
| T_original | cpu_1hilo | 5 | 1 | 1,515.858 |
| T_original | cpu_nhilos | 1 | 8 | 844.101 |
| T_original | cpu_nhilos | 2 | 8 | 847.785 |
| T_original | cpu_nhilos | 3 | 8 | 825.344 |
| T_original | cpu_nhilos | 4 | 8 | 792.250 |
| T_original | cpu_nhilos | 5 | 8 | 787.809 |

### Resumen estadístico

La desviación es la desviación estándar muestral. El coeficiente de variación (CV) es `desviación / media × 100`.

| Fase / variante | Media (s) | Desviación (s) | CV | Media legible |
|---|---:|---:|---:|---:|
| A_preparacion / secuencial | 16.325 | 0.236 | 1.44 % | 16.325 s |
| T_original / cpu_1hilo | 1,519.449 | 85.304 | 5.61 % | 25 min 19 s |
| T_original / cpu_nhilos | 819.458 | 28.223 | 3.44 % | 13 min 39 s |

Para `T_original`, el speedup de ocho hilos frente a un hilo fue **1.854×** (`1519.449 / 819.458`) y la reducción del tiempo promedio fue **46.1 %**. No se calcula speedup entre preparación y entrenamiento porque son fases y cantidades de trabajo distintas.

## Interpretación

La corrida de calentamiento de preparación tardó 149.579 s, mientras que las cinco corridas medidas con caché caliente promediaron 16.325 s. La diferencia confirma que la primera lectura depende fuertemente del estado de la caché de archivos de Windows. Por diseño, el calentamiento prepara un estado comparable antes de medir y se excluye de la media.

La preparación secuencial utiliza un único flujo de Pillow y no se beneficia de los hilos matemáticos de PyTorch. En el entrenamiento, aumentar los hilos solo afecta a operadores internos como convoluciones y multiplicaciones; no paraleliza la lectura de JPG porque el `DataLoader` conserva `num_workers=0`. Por ello, el speedup multihilo de 1.854× queda lejos de 8× y no puede eliminar la parte serial de decodificación y preparación. También influyen que el procesador tiene cuatro núcleos físicos, aunque expone ocho hilos lógicos, y el costo de coordinación de las operaciones paralelas.

En los logs representativos de la corrida 1, el promedio sobre los ocho procesadores lógicos fue 30.23 % durante preparación, 33.59 % durante `cpu_1hilo` y 95.84 % durante `cpu_nhilos`. El valor de un hilo no equivale exactamente a 12.5 % porque incluye actividad ambiental de Windows y el hilo de monitoreo; el heatmap muestra además que el hilo de cómputo puede migrar. En cambio, la variante multihilo mantiene prácticamente todos los procesadores lógicos ocupados, lo que explica su menor tiempo y también su sensibilidad a temperatura y frecuencia sostenida.

## Factores que afectan el rendimiento

- **Lectura desde disco y caché del sistema:** la primera pasada debe recuperar muchos archivos; las siguientes pueden servirse desde RAM. La diferencia observada entre calentamiento y corridas medidas lo evidencia.
- **Decodificación JPEG:** Pillow descomprime cada imagen cada vez que `ASLDataset.__getitem__` se invoca. En cuatro épocas se repite 59,160 veces para train y validation.
- **Número de hilos:** un hilo limita el cálculo tensorial; ocho hilos permiten paralelismo intraoperación, pero agregan sincronización y compiten por cuatro núcleos físicos.
- **Utilización del procesador:** el monitor registra actividad por núcleo. En un hilo, el hilo de cómputo puede migrar entre núcleos; por eso no necesariamente aparece una sola columna fija al 100 %. En multihilo se espera actividad más distribuida.
- **Temperatura y frecuencia:** el turbo del procesador y la reducción de frecuencia por temperatura cambian durante ejecuciones largas. El entorno no expuso un sensor de temperatura al programa, así que no se atribuyen grados no medidos.
- **Carga del equipo:** Windows, antivirus y otros procesos pueden consumir CPU o disco. Las corridas largas mostraron actividad ambiental en el monitor; la desviación y el CV cuantifican su efecto combinado.
- **Memoria:** los lotes y activaciones provocan picos de memoria. Con 7.73 GiB disponibles, presión de memoria o paginación puede aumentar el tiempo.

## Figuras y evidencia

- `resultados/figuras/mia_promedios.png`: barras de media ± desviación estándar.
- `resultados/figuras/mia_boxplot.png`: distribución de las cinco corridas por variante.
- `resultados/figuras/mia_uso_cpu_secuencial.png`: uso por núcleo durante preparación.
- `resultados/figuras/mia_uso_cpu_cpu_1hilo.png`: uso por núcleo durante entrenamiento con un hilo.
- `resultados/figuras/mia_uso_cpu_cpu_nhilos.png`: uso por núcleo durante entrenamiento multihilo.
- `resultados/logs/mia_<variante>_<corrida>.csv`: muestras reales de CPU y RAM cada 0.5 s.

No se generó una captura de pantalla de Task Manager porque el entorno de ejecución no ofrece una sesión gráfica capturable. La evidencia automática sí queda en los CSV y *heatmaps* del monitor. Para añadir la evidencia manual, se debe abrir Task Manager → **Rendimiento** → **CPU**, cambiar la gráfica a **Procesadores lógicos** y capturarla mientras se ejecuta `--solo cpu_1hilo` o `--solo cpu_nhilos`; la imagen real puede guardarse en `resultados/capturas/` sin reemplazar los registros del monitor.
