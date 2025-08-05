# Benchmarks

Este proyecto incluye utilidades para evaluar modelos de reconocimiento de se\xC3\xB1as.

## WER/CER en un subconjunto de validaci\xC3\xB3n

El script `scripts/evaluate_subset.py` usa `evaluate.py` para calcular **WER** y **CER**
sobre un subconjunto aleatorio de la partici\xC3\xB3n de validaci\xC3\xB3n. Se requieren los
archivos `HDF5` con landmarks y el `CSV` de etiquetas.

```bash
python scripts/evaluate_subset.py --h5_file data/val.h5 --csv_file data/val.csv \
    --checkpoint checkpoints/model.pt --model stgcn --subset 100
```

Los corpora pueden descargarse con `data/download.py`, que deja los datos en el
directorio indicado. Los ejemplos peque\xC3\xB1os para pruebas se encuentran en `tests/data/`.

## Comparaci\xC3\xB3n de arquitecturas

Para comparar arquitecturas como `stgcn` y `sttn` en el mismo corpus, ejecute el
test dedicado:

```bash
pytest tests/test_architecture_benchmark.py -s
```

La prueba genera un JSON con los valores de WER y CER por modelo y utiliza los
datos de ejemplo creados al vuelo. Para conjuntos reales, proporcione rutas a
los archivos correspondientes como en el script anterior.
