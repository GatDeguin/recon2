import hashlib
import os
import sys
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from importlib import import_module

download = import_module("data.download")
_verify_checksum = download._verify_checksum
_extract_archive = download._extract_archive


def test_verify_checksum(tmp_path):
    file = tmp_path / "file.txt"
    content = b"hello"
    file.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    assert _verify_checksum(str(file), digest)


def test_extract_archive(tmp_path):
    zpath = tmp_path / "archive.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("a/b.txt", "ok")
    out_dir = tmp_path / "out"
    _extract_archive(str(zpath), str(out_dir))
    assert (out_dir / "a" / "b.txt").read_text() == "ok"
