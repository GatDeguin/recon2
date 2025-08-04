import argparse
from typing import Tuple

import pandas as pd

try:
    import spacy
except Exception:  # pragma: no cover - optional dependency
    spacy = None


_DEF_MODEL = "es_core_news_sm"


def _extract_morph(text: str, nlp) -> Tuple[str, str, str, str, str]:
    if nlp is None:
        return ("none",) * 5
    doc = nlp(text)
    for tok in doc:
        if tok.pos_ == "VERB":
            m = tok.morph
            person = (m.get("Person") or ["none"])[0].lower()
            number = (m.get("Number") or ["none"])[0].lower()
            tense = (m.get("Tense") or ["none"])[0].lower()
            aspect = (m.get("Aspect") or ["none"])[0].lower()
            mood = (m.get("Mood") or ["none"])[0].lower()
            return person, number, tense, aspect, mood
    return ("none",) * 5


def main(inp: str, out: str, model: str = _DEF_MODEL, strict: bool = False) -> None:
    df = pd.read_csv(inp, sep=";")
    nlp = None
    if spacy is None:
        if strict:
            raise RuntimeError("spaCy is not installed")
        print("spaCy model not found; writing default morphology")
    else:  # pragma: no branch - optional dependency
        try:
            nlp = spacy.load(model)
        except Exception:  # pragma: no cover - optional dependency
            if strict:
                raise
            print("spaCy model not found; writing default morphology")
    rows = [
        _extract_morph(str(row["label"]), nlp)
        for _, row in df.iterrows()
    ]
    df[["person", "number", "tense", "aspect", "mode"]] = rows
    df.to_csv(out, sep=";", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate enriched metadata CSV")
    p.add_argument("input_csv")
    p.add_argument("output_csv")
    p.add_argument("--model", default=_DEF_MODEL, help="spaCy model")
    p.add_argument(
        "--strict",
        action="store_true",
        help="error if spaCy model cannot be loaded",
    )
    args = p.parse_args()
    main(args.input_csv, args.output_csv, args.model, args.strict)
