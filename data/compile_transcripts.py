"""Utilidades para unir transcripciones de múltiples corpora.

Este script recorre los archivos de texto de uno o varios directorios y
concatena todas las líneas en un único archivo de salida. Cada línea del
archivo resultante corresponde a una transcripción.

Ejemplo de uso::

    python data/compile_transcripts.py corpus1 corpus2 -o salida.txt

Donde ``corpus1`` y ``corpus2`` pueden ser directorios o archivos de texto.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def iter_transcripts(paths: Iterable[Path]) -> Iterable[str]:
    """Genera transcripciones encontradas en ``paths``.

    Si un elemento es un archivo, se leen todas sus líneas. Si es un directorio
    se buscan archivos ``.txt`` de forma recursiva.
    """

    for p in paths:
        if p.is_file():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        yield text
        elif p.is_dir():
            for txt in p.rglob("*.txt"):
                with txt.open("r", encoding="utf-8") as f:
                    for line in f:
                        text = line.strip()
                        if text:
                            yield text
        else:  # pragma: no cover - rutas inexistentes
            continue


def main() -> None:  # pragma: no cover - CLI fino
    parser = argparse.ArgumentParser(description="Compila transcripciones")
    parser.add_argument("paths", nargs="+", type=Path, help="Archivos o carpetas con transcripciones")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Archivo de salida")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for line in iter_transcripts(args.paths):
            out.write(f"{line}\n")


if __name__ == "__main__":  # pragma: no cover - invocación directa
    main()
