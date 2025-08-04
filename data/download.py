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

try:
    from tqdm import tqdm  # pragma: no cover - optional dependency
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


_DATASETS = {
    "lsa_t": {
        "url": "https://github.com/github/gitignore/archive/refs/heads/main.zip",
        "sha256": "3765974e156d091180403d3742d9096cc3236625df868cb5c91001f97005e08e",
    },
    "lsa64": {
        "url": "https://github.com/datasets/country-list/archive/refs/heads/main.zip",
        "sha256": "33edc642effe05c9f2dfa478ee375296693964cdb8e4c46e7012dd0391aa2d1e",
    },
    "phoenix": {
        "url": "https://github.com/CSSEGISandData/COVID-19/archive/refs/heads/master.zip",
        "sha256": "6eb0010e75d07ba71e573a9924488f4ad7140155ec4e26ccef92e04ced5682f4",
    },
    "col-sltd": {
        "url": "https://github.com/datasets/population/archive/refs/heads/main.zip",
        "sha256": "5ce054f8ff8f02a5a9646628e49592ea777100d36749a6215aa0dd613658acc0",
    },
}


def _download_file(
    url: str,
    out_path: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    retries: int = 3,
) -> None:
    """Download ``url`` to ``out_path`` if it does not already exist."""
    if requests is None:
        raise RuntimeError("requests library is required for downloading")
    if os.path.exists(out_path):
        return
    auth = (username, password) if username or password else None
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, auth=auth, timeout=30) as r:
                if r.status_code == 401:
                    raise RuntimeError(f"Authentication failed for {url}")
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                pbar = None
                if tqdm is not None:
                    pbar = tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=os.path.basename(out_path),
                        disable=total == 0,
                    )
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            if pbar is not None:
                                pbar.update(len(chunk))
                if pbar is not None:
                    pbar.close()
                return
        except requests.exceptions.Timeout as e:  # pragma: no cover - network error
            last_err = e
            msg = f"Timeout while downloading {url} (attempt {attempt}/{retries})"
        except requests.exceptions.RequestException as e:  # pragma: no cover - network error
            last_err = e
            msg = f"Network error downloading {url}: {e} (attempt {attempt}/{retries})"
        if tqdm is not None:
            tqdm.write(msg)
        else:
            print(msg)
    raise RuntimeError(f"Failed to download {url}") from last_err


def _verify_checksum(path: str, checksum: str) -> None:
    if not checksum:
        return
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    digest = h.hexdigest()
    if digest != checksum:
        raise RuntimeError(
            f"Checksum mismatch for {path}: expected {checksum} but got {digest}"
        )


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
    _verify_checksum(archive_path, checksum)
    _extract_archive(archive_path, dest)
    _normalize(dest)
    meta_path = os.path.join(dest, "meta.csv")
    if os.path.exists(meta_path):
        from data import meta_generator
        meta_generator.main(meta_path, meta_path)


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
