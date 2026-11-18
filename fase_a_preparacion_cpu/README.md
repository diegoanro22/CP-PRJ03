# Fase A: preparación de imágenes en CPU multinúcleo — Diego Rosales

Abrir, decodificar, convertir a RGB y redimensionar las 17,400 imágenes, repartidas entre procesos.
Resultado idéntico bit a bit a `comun/datos_seq.py`.

## Archivos

| Archivo | Contenido |
|---|---|
| `preparar_paralelo.py` | `ProcessPoolExecutor` sobre bloques de rutas; guarda `cache/<split>_par_x.npy` y `cache/<split>_par_y.npy` |
| `bench_preparacion.py` | Secuencial vs 1, 2, 4, 8, 16 y 32 procesos; 1 calentamiento + 5 corridas por variante |

```bash
python -m fase_a_preparacion_cpu.preparar_paralelo --workers 16 --verificar
python -m fase_a_preparacion_cpu.bench_preparacion
```

## Diseño

- **Procesos, no hilos:** decodificar JPEG y redimensionar es trabajo de CPU; con procesos no hay GIL.
- **Bloques grandes:** ~4 bloques por worker y por split; cada worker devuelve un solo arreglo `uint8`
  por bloque (pocos mensajes grandes entre procesos, no 17,400 pequeños).
- **Orden conservado:** `executor.map` devuelve los bloques en orden; la caché queda en el orden del manifiesto.
- **Un solo pool** para los tres splits; su creación se incluye en el tiempo medido.
- **Verificación:** en cada calentamiento, `np.array_equal` de `x` e `y` contra la caché secuencial.

## Resultados (Ryzen 9 8940HX, 16 núcleos / 32 hilos)

| Procesos | Media (s) | Speedup | Eficiencia | Fracción serial (Karp-Flatt) |
|---:|---:|---:|---:|---:|
| secuencial | 11.73 | 1.00 | | |
| 1 | 12.52 | 0.94 | 0.94 | |
| 2 | 6.30 | 1.86 | 0.93 | 0.074 |
| 4 | 3.38 | 3.48 | 0.87 | 0.050 |
| 8 | 1.74 | 6.74 | 0.84 | 0.027 |
| 16 | 1.10 | **10.65** | 0.67 | 0.034 |
| 32 | 1.10 | 10.68 | 0.33 | 0.064 |

- Con 1 proceso es 6 % más lento que el bucle secuencial: es el costo de crear el proceso y enviar
  los resultados entre procesos.
- Escala casi lineal hasta 8 procesos. En 16 llega al número de núcleos físicos, y con 32 (SMT) ya no mejora.
- Detalle: `resultados/rosales_preparacion.csv`, figura `resultados/figuras/rosales_preparacion_speedup.png`,
  monitor por corrida en `resultados/logs/rosales_prep_*.csv`.
