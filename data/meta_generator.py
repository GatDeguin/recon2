import argparse
import pandas as pd
from typing import Tuple

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


def main(inp: str, out: str, model: str = _DEF_MODEL) -> None:
    df = pd.read_csv(inp, sep=";")
    nlp = spacy.load(model) if spacy else None
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
    args = p.parse_args()
    main(args.input_csv, args.output_csv, args.model)
