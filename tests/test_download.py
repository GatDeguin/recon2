import csv
import hashlib
import os
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from importlib import import_module

download = import_module("data.download")
_verify_checksum = download._verify_checksum
_extract_archive = download._extract_archive
_normalize = download._normalize
_write_splits = download._write_splits


def test_verify_checksum(tmp_path):
    file = tmp_path / "file.txt"
    content = b"hello"
    file.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    _verify_checksum(str(file), digest)
    with pytest.raises(RuntimeError):
        _verify_checksum(str(file), "bad")


def test_extract_archive(tmp_path):
    zpath = tmp_path / "archive.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("a/b.txt", "ok")
    out_dir = tmp_path / "out"
    _extract_archive(str(zpath), str(out_dir))
    assert (out_dir / "a" / "b.txt").read_text() == "ok"


def test_extract_unknown_archive(tmp_path):
    path = tmp_path / "file.bin"
    path.write_text("data")
    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError):
        _extract_archive(str(path), str(out_dir))


def test_normalize(tmp_path):
    dest = tmp_path / "data"
    sub = dest / "sub"
    sub.mkdir(parents=True)

    video = sub / "clip.mp4"
    video.write_text("video")

    csv_file = sub / "info.csv"
    csv_file.write_text("id;video\n1;clip")

    _normalize(str(dest))

    moved_video = dest / "videos" / "clip.mp4"
    assert moved_video.exists()
    assert (dest / "meta.csv").read_text() == "id;video\n1;clip"
    assert not video.exists()
    assert not csv_file.exists()


def test_write_splits(tmp_path):
    dest = tmp_path / "dataset"
    videos = dest / "videos"
    videos.mkdir(parents=True)

    for vid in ["a", "b", "c"]:
        (videos / f"{vid}.mp4").write_text("vid")

    meta = dest / "meta.csv"
    meta.write_text("id;video\n1;a\n2;b\n3;c\n")

    cfg = {"train": ["1"], "val": ["2"], "test": ["3"]}
    _write_splits(str(meta), str(dest), cfg)

    splits = {"train": "a", "val": "b", "test": "c"}
    for split, vid in splits.items():
        split_dir = dest / split
        assert (split_dir / "videos" / f"{vid}.mp4").exists()
        with open(split_dir / "meta.csv", newline="", encoding="utf8") as f:
            rows = list(csv.DictReader(f, delimiter=";"))
        assert rows == [{"id": next(iter(cfg[split])), "video": vid}]

    assert not videos.exists()
