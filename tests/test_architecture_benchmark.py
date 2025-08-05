import json
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import DataLoader
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from train import SignDataset, collate, build_model
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


def _evaluate_model(name, ds, dl, inv_vocab):
    model = build_model(
        name,
        len(ds.vocab),
        len(ds.nmm_vocab),
        len(ds.suffix_vocab),
        len(ds.rnm_vocab),
        len(ds.person_vocab),
        len(ds.number_vocab),
        len(ds.tense_vocab),
        len(ds.aspect_vocab),
        len(ds.mode_vocab),
        num_nodes=ds.num_nodes,
    )
    wer, cer, _ = compute_metrics(model, dl, inv_vocab, torch.device("cpu"))
    return {"wer": wer, "cer": cer}


def test_compare_architectures(tmp_path):
    h5_file = tmp_path / "data.h5"
    csv_file = tmp_path / "labels.csv"
    _create_eval_data(h5_file, csv_file)
    with SignDataset(str(h5_file), str(csv_file)) as ds:
        dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
        inv_vocab = {v: k for k, v in ds.vocab.items()}
        results = {}
        for name in ["stgcn", "sttn"]:
            results[name] = _evaluate_model(name, ds, dl, inv_vocab)
        print(json.dumps(results, indent=2))
        assert set(results.keys()) == {"stgcn", "sttn"}
