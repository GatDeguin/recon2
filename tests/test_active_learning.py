import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import DataLoader

from train import SignDataset, collate
from active_learning import compute_scores


def _create_data(h5_path, csv_path):
    with h5py.File(h5_path, "w") as h5:
        T = 2
        for i in range(2):
            grp = h5.create_group(f"sample{i}.mp4")
            grp.create_dataset("pose", data=np.zeros((T, 33 * 3), np.float32))
            grp.create_dataset("left_hand", data=np.zeros((T, 21 * 3), np.float32))
            grp.create_dataset("right_hand", data=np.zeros((T, 21 * 3), np.float32))
            grp.create_dataset("face", data=np.zeros((T, 468 * 3), np.float32))
            grp.create_dataset("optical_flow", data=np.zeros((T, 2, 2, 2), np.float32))

    pd.DataFrame({
        "id": ["sample0", "sample1"],
        "label": ["hello world", "hello world"],
        "nmm": ["neutral", "neutral"],
        "suffix": ["none", "none"],
        "rnm": ["tipo1", "tipo1"],
        "person": ["1", "1"],
        "number": ["sg", "sg"],
        "tense": ["pres", "pres"],
        "aspect": ["simple", "simple"],
        "mode": ["ind", "ind"],
    }).to_csv(csv_path, sep=";", index=False)


class DummyModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        B, C, T, V = x.shape
        logits = torch.zeros(B, T, self.num_classes)
        # return tuple to mimic real model
        return logits.log_softmax(-1), torch.empty(B, 2), torch.empty(B, 3)


def test_compute_scores_with_collate(tmp_path):
    h5_file = tmp_path / "data.h5"
    csv_file = tmp_path / "labels.csv"
    _create_data(h5_file, csv_file)

    with SignDataset(str(h5_file), str(csv_file)) as ds:
        dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
        model = DummyModel(len(ds.vocab))
        scores = compute_scores(model, dl)

    assert len(scores) == 2
    assert [s[0] for s in scores] == ["sample0.mp4", "sample1.mp4"]
    assert all(isinstance(s[1], float) for s in scores)
