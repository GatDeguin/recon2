import pandas as pd
import pytest
from data import meta_generator


def test_meta_generator_vocab(tmp_path, monkeypatch):
    raw_csv = tmp_path / "raw.csv"
    pd.DataFrame({
        "id": ["a", "b"],
        "label": ["first", "second"],
    }).to_csv(raw_csv, sep=";", index=False)
    out_csv = tmp_path / "meta.csv"

    def fake_extract(text, nlp):
        if text == "first":
            return ("1", "sg", "pres", "simple", "ind")
        return ("3", "pl", "past", "perf", "subj")

    monkeypatch.setattr(meta_generator, "_extract_morph", fake_extract)
    meta_generator.main(str(raw_csv), str(out_csv))

    df = pd.read_csv(out_csv, sep=";")
    assert df["person"].nunique() > 1
    assert df["number"].nunique() > 1
    assert df["tense"].nunique() > 1
    assert df["aspect"].nunique() > 1
    assert df["mode"].nunique() > 1


def test_meta_generator_strict_errors(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    pd.DataFrame({"id": ["a"], "label": ["x"]}).to_csv(
        raw_csv, sep=";", index=False
    )
    out_csv = tmp_path / "out.csv"
    with pytest.raises(RuntimeError):
        meta_generator.main(str(raw_csv), str(out_csv), strict=True)


def test_meta_generator_fallback_message(tmp_path, capsys):
    raw_csv = tmp_path / "raw.csv"
    pd.DataFrame({"id": ["a"], "label": ["x"]}).to_csv(
        raw_csv, sep=";", index=False
    )
    out_csv = tmp_path / "out.csv"
    meta_generator.main(str(raw_csv), str(out_csv))
    captured = capsys.readouterr().out
    assert "spaCy model not found" in captured
