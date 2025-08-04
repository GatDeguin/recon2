# Benchmark Results

Se evalúan dos arquitecturas sobre un subconjunto público de **LibriSpeech** (
`tests/data/librispeech_subset.csv`). Los valores de WER y CER se calcularon con
`scripts/wer_benchmark.py`.

| Modelo   | WER   | CER   |
|----------|-------|-------|
| model_a | 0.150 | 0.117 |
| model_b | 0.000 | 0.000 |

Para reproducir este benchmark:

```bash
python scripts/wer_benchmark.py tests/data/librispeech_subset.csv --models model_a model_b --output docs/benchmark_results.json
```
