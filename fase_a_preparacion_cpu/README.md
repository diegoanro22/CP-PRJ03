# Fase A: preparación de imágenes en CPU multinúcleo — Diego Rosales

## Archivos

| Archivo | Contenido |
|---|---|
| `preparar_paralelo.py` | Igual que `comun.datos_seq` pero repartiendo las rutas entre procesos; guarda **los dos** archivos por split, `cache/<split>_par_x.npy` y `cache/<split>_par_y.npy` (con `comun.datos_seq.guardar(split, x, y, sufijo="_par")`) |
| `bench_preparacion.py` | Secuencial vs 1, 2, 4, 8, 16, 32 workers, 5 corridas; `resultados/tiempos_rosales.csv` con `fase="A_preparacion"`, `variante="secuencial"` o `"procesos_<p>"` |

## Diseño

- `concurrent.futures.ProcessPoolExecutor(max_workers=w)`; las rutas se dividen en bloques
  (`chunksize` ≈ N / (4·w)) para que cada proceso devuelva un arreglo `uint8` grande y no miles de objetos pequeños.
- Cada worker llama a `comun.datos_seq.preparar_imagen` (misma operación, mismo resultado).
- Procesos y no hilos: decodificar JPEG y el resize de PIL son trabajo de CPU, y con procesos no hay GIL.
- Verificación: `np.array_equal` de `x` **y** de `y` contra la caché secuencial, para los tres splits.
- El orden de las filas debe ser el del manifiesto (usar `executor.map`, que conserva el orden).

## Checklist

- [ ] `preparar_paralelo.py` con argumento `--workers`.
- [ ] Verificación de igualdad con la caché secuencial.
- [ ] Barrido de workers, speedup S(p) = T1/Tp, eficiencia S(p)/p y comparación con Amdahl (fracción serial estimada).
- [ ] `Monitor("rosales_prep_w<p>_<n>")` + captura de `btop` con todos los núcleos ocupados.
