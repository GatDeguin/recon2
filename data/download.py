import argparse
import hashlib
import os
import shutil
import tarfile
import zipfile
from typing import Optional

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None


_DATASETS = {
    "lsa_t": {
        "url": "https://example.com/lsa_t.tar.gz",
        "sha256": "",
    },
    "lsa64": {
        "url": "https://example.com/lsa64.zip",
        "sha256": "",
    },
    "phoenix": {
        "url": "https://example.com/phoenix.tar.gz",
        "sha256": "",
    },
    "col-sltd": {
        "url": "https://example.com/col_sltd.zip",
        "sha256": "",
    },
}


def _download_file(url: str, out_path: str, username: Optional[str] = None, password: Optional[str] = None) -> None:
    """Download ``url`` to ``out_path`` if it does not already exist."""
    if requests is None:
        raise RuntimeError("requests library is required for downloading")
    if os.path.exists(out_path):
        return
    auth = (username, password) if username or password else None
    with requests.get(url, stream=True, auth=auth) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def _verify_checksum(path: str, checksum: str) -> bool:
    if not checksum:
        return True
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest() == checksum


def _extract_archive(path: str, dest: str) -> None:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            tf.extractall(dest)


def _normalize(dest: str) -> None:
    videos_dir = os.path.join(dest, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    for root, _dirs, files in os.walk(dest):
        for fn in files:
            lower = fn.lower()
            full = os.path.join(root, fn)
            if root == dest and fn == "meta.csv":
                continue
            if lower.endswith((".mp4", ".mov", ".avi")):
                shutil.move(full, os.path.join(videos_dir, fn))
            elif lower.endswith(".csv") and fn != "meta.csv":
                shutil.move(full, os.path.join(dest, "meta.csv"))


def download_dataset(name: str, dest: str, url: Optional[str] = None, checksum: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None) -> None:
    info = _DATASETS.get(name)
    if info is None:
        raise ValueError(f"Unknown dataset {name}")
    url = url or info["url"]
    checksum = checksum or info.get("sha256", "")

    os.makedirs(dest, exist_ok=True)
    archive_path = os.path.join(dest, os.path.basename(url))
    _download_file(url, archive_path, username, password)
    if not _verify_checksum(archive_path, checksum):
        raise RuntimeError("Checksum mismatch for downloaded file")
    _extract_archive(archive_path, dest)
    _normalize(dest)


def main() -> None:
    p = argparse.ArgumentParser(description="Descarga y normaliza corpora")
    sub = p.add_subparsers(dest="dataset", required=True)
    for name in _DATASETS:
        sp = sub.add_parser(name)
        sp.add_argument("dest")
        sp.add_argument("--url", default=_DATASETS[name]["url"])
        sp.add_argument("--checksum", default=_DATASETS[name].get("sha256", ""))
        sp.add_argument("--username")
        sp.add_argument("--password")
    args = p.parse_args()
    download_dataset(args.dataset, args.dest, args.url, args.checksum, args.username, args.password)


if __name__ == "__main__":
    main()
