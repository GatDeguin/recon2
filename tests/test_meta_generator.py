import numpy as np
import pandas as pd
import h5py
from data import meta_generator
from train import SignDataset


def test_meta_generator_vocab(tmp_path, monkeypatch):
    h5_file = tmp_path / "data.h5"
    with h5py.File(h5_file, "w") as h5:
        for vid in ["a.mp4", "b.mp4"]:
            grp = h5.create_group(vid)
            T = 1
            grp.create_dataset("pose", data=np.zeros((T, 33 * 3), np.float32))
            grp.create_dataset("left_hand", data=np.zeros((T, 21 * 3), np.float32))
            grp.create_dataset("right_hand", data=np.zeros((T, 21 * 3), np.float32))
            grp.create_dataset("face", data=np.zeros((T, 468 * 3), np.float32))
            grp.create_dataset("optical_flow", data=np.zeros((T, 2, 2, 2), np.float32))
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

    with SignDataset(str(h5_file), str(out_csv)) as ds:
        assert len(ds.person_vocab) > 1
        assert len(ds.number_vocab) > 1
        assert len(ds.tense_vocab) > 1
        assert len(ds.aspect_vocab) > 1
        assert len(ds.mode_vocab) > 1
