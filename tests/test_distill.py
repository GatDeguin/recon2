import os
import argparse
import numpy as np
import pandas as pd
import h5py
import torch

from train import SignDataset, build_model
from distill import distill


def _create_data(h5_path, csv_path):
    with h5py.File(h5_path, "w") as h5:
        grp = h5.create_group("sample.mp4")
        T = 2
        grp.create_dataset("pose", data=np.zeros((T, 33 * 3), np.float32))
        grp.create_dataset("left_hand", data=np.zeros((T, 21 * 3), np.float32))
        grp.create_dataset("right_hand", data=np.zeros((T, 21 * 3), np.float32))
        grp.create_dataset("face", data=np.zeros((T, 468 * 3), np.float32))
        grp.create_dataset("optical_flow", data=np.zeros((T, 2, 2, 2), np.float32))
    pd.DataFrame(
        {
            "id": ["sample"],
            "label": ["hello world"],
            "nmm": ["neutral"],
            "suffix": ["none"],
            "rnm": ["tipo1"],
            "person": ["1"],
            "number": ["sg"],
            "tense": ["pres"],
            "aspect": ["simple"],
            "mode": ["ind"],
        }
    ).to_csv(csv_path, sep=";", index=False)


def test_distill_single_step(tmp_path):
    h5_file = tmp_path / "data.h5"
    csv_file = tmp_path / "labels.csv"
    _create_data(h5_file, csv_file)

    # prepare teacher checkpoint
    with SignDataset(str(h5_file), str(csv_file)) as ds:
        teacher = build_model(
            "stgcn",
            len(ds.vocab),
            len(ds.nmm_vocab),
            len(ds.suffix_vocab),
            len(ds.rnm_vocab),
        )
        ckpt = {"model_state": teacher.state_dict()}
    ckpt_path = tmp_path / "teacher.pt"
    torch.save(ckpt, ckpt_path)

    args = argparse.Namespace(
        h5_file=str(h5_file),
        csv_file=str(csv_file),
        teacher_ckpt=str(ckpt_path),
        model="stgcn",
        batch_size=1,
        epochs=1,
        domain_csv=None,
    )

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        distill(args)
    finally:
        os.chdir(cwd)
