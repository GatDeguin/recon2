import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import DataLoader
import pytest
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from train import SignDataset, collate
from evaluate import compute_metrics


def _create_eval_data(h5_path, csv_path):
    with h5py.File(h5_path, "w") as h5:
        for vid in ["s1.mp4", "s2.mp4"]:
            grp = h5.create_group(vid)
            T = 2
            grp.create_dataset("pose", data=np.zeros((T, 33 * 3), np.float32))
            grp.create_dataset("left_hand", data=np.zeros((T, 21 * 3), np.float32))
            grp.create_dataset("right_hand", data=np.zeros((T, 21 * 3), np.float32))
            grp.create_dataset("face", data=np.zeros((T, 468 * 3), np.float32))
            grp.create_dataset("optical_flow", data=np.zeros((T, 2, 2, 2), np.float32))
    pd.DataFrame({
        "id": ["s1", "s2"],
        "label": ["hello", "world"],
        "nmm": ["neutral", "smile"],
    }).to_csv(csv_path, sep=";", index=False)


class DummyModel(torch.nn.Module):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds
        self.call = 0

    def forward(self, x):
        B, C, T, V = x.shape
        vocab = len(self.ds.vocab)
        nmm_classes = len(self.ds.nmm_vocab)
        gloss_logits = torch.full((B, T, vocab), -1e9)
        nmm_logits = torch.full((B, nmm_classes), -1e9)
        if self.call == 0:
            gloss_logits[0, 0, self.ds.vocab["hello"]] = 5.0
            gloss_logits[0, 1, self.ds.vocab["hello"]] = 5.0
            nmm_logits[0, self.ds.nmm_vocab["neutral"]] = 5.0
        else:
            gloss_logits[0, 0, self.ds.vocab["hello"]] = 5.0
            gloss_logits[0, 1, self.ds.vocab["hello"]] = 5.0
            nmm_logits[0, self.ds.nmm_vocab["neutral"]] = 5.0  # incorrect
        self.call += 1
        return (
            gloss_logits,
            nmm_logits,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def test_compute_metrics(tmp_path):
    h5_file = tmp_path / "data.h5"
    csv_file = tmp_path / "labels.csv"
    _create_eval_data(h5_file, csv_file)
    with SignDataset(str(h5_file), str(csv_file)) as ds:
        dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
        inv_vocab = {v: k for k, v in ds.vocab.items()}
        model = DummyModel(ds)
        wer, cer, acc = compute_metrics(model, dl, inv_vocab, torch.device("cpu"))
        assert wer == pytest.approx(0.5)
        assert cer == pytest.approx(0.4)
        assert acc["nmm"] == pytest.approx(0.5)
