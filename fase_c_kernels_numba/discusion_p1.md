# Discusión técnica, pregunta 1

## ¿Por qué es más eficiente procesar las imágenes con los múltiples núcleos del CPU y delegar exclusivamente la matemática de la red neuronal al GPU?

Porque las dos fases tienen formas de trabajo opuestas, y cada procesador está construido para una de ellas.

### La preparación de imágenes es trabajo de CPU

Preparar una imagen significa leer un archivo del disco, decodificar un JPEG, convertirlo a RGB y
redimensionarlo. Ese trabajo tiene tres características que favorecen al CPU:

1. **Es entrada/salida y llamadas al sistema.** Abrir 17,400 archivos pasa por el sistema operativo y
   el sistema de archivos. Una GPU no puede leer el disco: los bytes tienen que pasar primero por la CPU.
2. **La decodificación JPEG es irregular.** La decodificación entrópica (Huffman) es secuencial dentro de
   cada imagen y tiene muchas ramas que dependen de los datos. En una GPU eso produce divergencia de
   warps. En un núcleo de CPU, con su predicción de saltos y su caché grande, es eficiente.
3. **Las imágenes son independientes entre sí.** Ese es el paralelismo correcto para el CPU: una imagen
   completa por proceso, sin comunicación entre ellos. No hace falta paralelizar dentro de la imagen.

La fase A lo confirma. Con un pool de procesos la preparación de las 17,400 imágenes bajó de 11.7 s a
1.10 s (**10.6×** con 16 procesos) y el resultado es idéntico bit a bit al secuencial. La curva sigue
casi la recta ideal hasta 8 procesos (eficiencia del 84 %) y se aplana en 16. Con 32 no mejora, porque
el Ryzen 9 8940HX tiene 16 núcleos físicos: los 16 hilos SMT extra comparten las mismas unidades de
cálculo. La fracción serial estimada con Karp-Flatt es de entre 3 % y 6 % (crear los procesos y juntar
los resultados), y con ella Amdahl predice bien el aplanamiento.

Llevar esta fase a la GPU no tendría sentido en este proyecto. Cada imagen de 200 × 200 es un trabajo
muy pequeño, y la latencia de lanzar kernels y copiar por PCIe sería comparable al trabajo mismo.
Existen decodificadores JPEG en GPU (nvJPEG, DALI), pero rinden en flujos de video o imágenes grandes,
no en un conjunto que se prepara **una sola vez** y queda en memoria.

### La matemática de la red es trabajo de GPU

La red hace lo contrario: unos 380 millones de FLOPs por imagen (paso 1), casi todos en convoluciones
que repiten la misma multiplicación y suma sobre millones de píxeles, canales e imágenes, sin ramas que
dependan de los datos. Es exactamente para lo que existe una GPU: miles de hilos ejecutando la misma
instrucción (SIMT), memoria de alto ancho de banda y, en arquitecturas recientes, Tensor Cores.

Las mediciones lo confirman:

* **Fase B (entrenamiento, Lopez):** con los datos ya en memoria, 4 épocas de CNN2 tardan 724 s en un hilo
  de CPU, 193 s con 20 hilos y **11.8 s** en la GPU (61× contra un hilo, 16× contra todos los hilos del
  CPU). Con precisión mixta bf16 bajan a 7.0 s. La GPU se mantuvo al 99 % de uso.
* **Fase C (kernels propios):** incluso con kernels escritos a mano, sin las optimizaciones de cuDNN, la
  inferencia de las 2,610 imágenes de test bajó de 46.3 s (NumPy con 32 hilos de BLAS) a **1.60 s** (29×).

### Por qué separarlas, y no hacer todo en un solo lugar

El pipeline original mezclaba las dos fases: el `DataLoader` con `num_workers=0` volvía a abrir y
decodificar cada JPG en **cada época**, dentro del mismo bucle de entrenamiento (59,160 decodificaciones
en 4 épocas). Si ese bucle corre en la GPU, la GPU pasa la mayor parte del tiempo esperando a que un
solo núcleo de CPU termine de decodificar el siguiente lote.

Separar las fases resuelve los dos problemas a la vez:

* La CPU hace **una sola vez** el trabajo irregular, con todos sus núcleos, y deja una caché `uint8` en memoria.
* La GPU recibe lotes ya listos. La copia por PCIe es mínima (0.044 s de 11.8 s en la fase B, porque los
  datos viajan como `uint8`, 4 veces menos bytes que `float32`), y pesos, activaciones y gradientes nunca
  salen de la GPU.

Cada procesador hace lo que mejor sabe hacer, y ninguno espera al otro.
