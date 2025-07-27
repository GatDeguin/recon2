import numpy as np
import pandas as pd
import h5py
import torch
from functools import partial

from train import SignDataset, collate, length_for_epoch


def _create_data(h5_path, csv_path):
    with h5py.File(h5_path, "w") as h5:
        grp = h5.create_group("sample.mp4")
        T = 2
        grp.create_dataset("pose", data=np.zeros((T, 33 * 3), np.float32))
        grp.create_dataset("left_hand", data=np.zeros((T, 21 * 3), np.float32))
        grp.create_dataset("right_hand", data=np.zeros((T, 21 * 3), np.float32))
        grp.create_dataset("face", data=np.zeros((T, 468 * 3), np.float32))
        grp.create_dataset("optical_flow", data=np.zeros((T, 2, 2, 2), np.float32))
    pd.DataFrame({"id": ["sample"], "label": ["hello world"]}).to_csv(csv_path, sep=";", index=False)


def test_length_schedule(tmp_path):
    h5_file = tmp_path / "data.h5"
    csv_file = tmp_path / "labels.csv"
    _create_data(h5_file, csv_file)
    with SignDataset(str(h5_file), str(csv_file)) as ds:
        schedule = [2, 3]
        expected = [1, 2, 3]
        for epoch in range(3):
            max_len = length_for_epoch(epoch, 1, schedule)
            dl = torch.utils.data.DataLoader(
                ds,
                batch_size=1,
                collate_fn=partial(collate, max_length=max_len),
            )
            feats, *_ = next(iter(dl))
            assert feats.shape[2] == expected[epoch]

