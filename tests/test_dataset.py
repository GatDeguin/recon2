import numpy as np
import pandas as pd
import h5py
from train import SignDataset, collate


def _create_data(h5_path, csv_path):
    with h5py.File(h5_path, "w") as h5:
        grp = h5.create_group("sample.mp4")
        T = 2
        grp.create_dataset("pose", data=np.zeros((T, 33 * 3), np.float32))
        grp.create_dataset("left_hand", data=np.zeros((T, 21 * 3), np.float32))
        grp.create_dataset("right_hand", data=np.zeros((T, 21 * 3), np.float32))
        grp.create_dataset("face", data=np.zeros((T, 468 * 3), np.float32))
        grp.create_dataset("optical_flow", data=np.zeros((T, 2, 2, 2), np.float32))
    pd.DataFrame({
        "id": ["sample"],
        "label": ["hello world"],
        "nmm": ["neutral"],
        "suffix": ["none"],
    }).to_csv(csv_path, sep=";", index=False)


def test_dataset_loading(tmp_path):
    h5_file = tmp_path / "data.h5"
    csv_file = tmp_path / "labels.csv"
    _create_data(h5_file, csv_file)
    with SignDataset(str(h5_file), str(csv_file)) as ds:
        assert len(ds) == 1
        x, y, d, nmm, suf = ds[0]
        assert x.shape == (3, 2, 544)
        assert d == 0
        assert y.tolist() == [ds.vocab["hello"], ds.vocab["world"]]
        assert nmm == ds.nmm_vocab["neutral"]
        assert suf == ds.suffix_vocab["none"]


def test_collate(tmp_path):
    h5_file = tmp_path / "data.h5"
    csv_file = tmp_path / "labels.csv"
    _create_data(h5_file, csv_file)
    with SignDataset(str(h5_file), str(csv_file)) as ds:
        batch = [ds[0], ds[0]]
        feats, labels, feat_lens, label_lens, domains, nmms, sufs = collate(batch)
        assert feats.shape[0] == 2
        assert feat_lens.tolist() == [2, 2]
        assert label_lens.tolist() == [2, 2]
        assert domains.tolist() == [0, 0]
        assert (nmms == ds.nmm_vocab["neutral"]).all()
        assert (sufs == ds.suffix_vocab["none"]).all()
