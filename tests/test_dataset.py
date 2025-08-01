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
        "rnm": ["tipo1"],
        "person": ["1"],
        "number": ["sg"],
        "tense": ["pres"],
        "aspect": ["simple"],
        "mode": ["ind"],
    }).to_csv(csv_path, sep=";", index=False)


def _create_segment_data(h5_path, csv_path):
    """Create HDF5 file with segmented samples."""
    with h5py.File(h5_path, "w") as h5:
        grp = h5.create_group("sample.mp4")
        for i in range(2):
            seg = grp.create_group(f"segment_{i:03d}")
            T = 2
            seg.create_dataset("pose", data=np.zeros((T, 33 * 3), np.float32))
            seg.create_dataset("left_hand", data=np.zeros((T, 21 * 3), np.float32))
            seg.create_dataset("right_hand", data=np.zeros((T, 21 * 3), np.float32))
            seg.create_dataset("face", data=np.zeros((T, 468 * 3), np.float32))
        seg.create_dataset("optical_flow", data=np.zeros((T, 2, 2, 2), np.float32))
    pd.DataFrame({
        "id": ["sample"],
        "label": ["hello world"],
    }).to_csv(csv_path, sep=";", index=False)


def _create_openface_data(h5_path, csv_path):
    """Create HDF5 file including OpenFace features."""
    with h5py.File(h5_path, "w") as h5:
        grp = h5.create_group("sample.mp4")
        T = 2
        grp.create_dataset("pose", data=np.zeros((T, 33 * 3), np.float32))
        grp.create_dataset("left_hand", data=np.zeros((T, 21 * 3), np.float32))
        grp.create_dataset("right_hand", data=np.zeros((T, 21 * 3), np.float32))
        grp.create_dataset("face", data=np.zeros((T, 468 * 3), np.float32))
        grp.create_dataset("optical_flow", data=np.zeros((T, 2, 2, 2), np.float32))
        grp.create_dataset("head_pose", data=np.zeros((T, 3), np.float32))
        grp.create_dataset("torso_pose", data=np.zeros((T, 3), np.float32))
        grp.create_dataset("aus", data=np.zeros((T, 2), np.float32))
    pd.DataFrame({"id": ["sample"], "label": ["hello world"]}).to_csv(csv_path, sep=";", index=False)


def test_dataset_loading(tmp_path):
    h5_file = tmp_path / "data.h5"
    csv_file = tmp_path / "labels.csv"
    _create_data(h5_file, csv_file)
    with SignDataset(str(h5_file), str(csv_file)) as ds:
        assert len(ds) == 1
        x, y, d, nmm, suf, rnm, per, num, tense, aspect, mode = ds[0]
        assert x.shape == (3, 2, 544)
        assert d == 0
        assert y.tolist() == [ds.vocab["hello"], ds.vocab["world"]]
        assert nmm == ds.nmm_vocab["neutral"]
        assert suf == ds.suffix_vocab["none"]
        assert rnm == ds.rnm_vocab["tipo1"]
        assert per == ds.person_vocab["1"]
        assert num == ds.number_vocab["sg"]
        assert tense == ds.tense_vocab["pres"]
        assert aspect == ds.aspect_vocab["simple"]
        assert mode == ds.mode_vocab["ind"]


def test_collate(tmp_path):
    h5_file = tmp_path / "data.h5"
    csv_file = tmp_path / "labels.csv"
    _create_data(h5_file, csv_file)
    with SignDataset(str(h5_file), str(csv_file)) as ds:
        batch = [ds[0], ds[0]]
        (
            feats,
            labels,
            feat_lens,
            label_lens,
            domains,
            nmms,
            sufs,
            rnms,
            pers,
            nums,
            tenses,
            aspects,
            modes,
        ) = collate(batch)
        assert feats.shape[0] == 2
        assert feat_lens.tolist() == [2, 2]
        assert label_lens.tolist() == [2, 2]
        assert domains.tolist() == [0, 0]
        assert (nmms == ds.nmm_vocab["neutral"]).all()
        assert (sufs == ds.suffix_vocab["none"]).all()
        assert (rnms == ds.rnm_vocab["tipo1"]).all()
        assert (pers == ds.person_vocab["1"]).all()
        assert (nums == ds.number_vocab["sg"]).all()
        assert (tenses == ds.tense_vocab["pres"]).all()
        assert (aspects == ds.aspect_vocab["simple"]).all()
        assert (modes == ds.mode_vocab["ind"]).all()


def test_dataset_segments(tmp_path):
    h5_file = tmp_path / "data.h5"
    csv_file = tmp_path / "labels.csv"
    _create_segment_data(h5_file, csv_file)
    with SignDataset(str(h5_file), str(csv_file), segments=True) as ds:
        assert len(ds) == 2
        assert ds.samples[0][0].endswith("segment_000")
        x, *_ = ds[0]
        assert x.shape == (3, 2, 544)


def test_dataset_openface(tmp_path):
    h5_file = tmp_path / "data.h5"
    csv_file = tmp_path / "labels.csv"
    _create_openface_data(h5_file, csv_file)
    with SignDataset(str(h5_file), str(csv_file), include_openface=True) as ds:
        x, *_ = ds[0]
        assert x.shape[2] == 548
        assert ds.num_nodes == 548

