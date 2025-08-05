import argparse
import csv
import glob
import hashlib
import os
import random
import shutil
import tarfile
import zipfile
from typing import Dict, List, Optional

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None

try:
    from tqdm import tqdm  # pragma: no cover - optional dependency
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


_DATASETS: Dict[str, Dict[str, str]] = {
    # NOTE: These URLs correspond to the official hosting locations of the
    # corpora. Checksums must match the published files and should be
    # updated if the upstream releases change.
    "lsa_t": {
        "url": "https://www.dls.fceia.unr.edu.ar/lsat/LSA_T.tar.gz",
        "sha256": "",  # TODO: update with official checksum
    },
    "lsa64": {
        "url": "https://www.dls.fceia.unr.edu.ar/lsa64/LSA64.zip",
        "sha256": "",  # TODO: update with official checksum
    },
    "phoenix": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/phoenix/2014T/phoenix2014T.tar.gz",
        "sha256": "",  # TODO: update with official checksum
    },
    "col-sltd": {
        "url": "https://repository.udistrital.edu.co/col-sltd/COL-SLTD.tar.gz",
        "sha256": "",  # TODO: update with official checksum
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
    else:
        raise RuntimeError(f"Unsupported archive format: {path}")


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


def _load_split_cfg(path: Optional[str]) -> Dict[str, List[str] | float]:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to parse split configuration")
    with open(path, "r", encoding="utf8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _write_splits(meta_path: str, dest: str, cfg: Dict[str, List[str] | float]) -> None:
    if not os.path.exists(meta_path):
        return
    with open(meta_path, newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    if not rows:
        return
    splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    if any(isinstance(v, list) for v in cfg.values()):
        id_map = {row["id"]: row for row in rows}
        for split, ids in cfg.items():
            for vid in ids or []:
                row = id_map.get(vid)
                if row:
                    splits[split].append(row)
    else:
        ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
        ratios.update(cfg)
        random.shuffle(rows)
        n = len(rows)
        n_train = int(n * ratios.get("train", 0))
        n_val = int(n * ratios.get("val", 0))
        splits["train"] = rows[:n_train]
        splits["val"] = rows[n_train : n_train + n_val]
        splits["test"] = rows[n_train + n_val :]
    videos_root = os.path.join(dest, "videos")
    for split, srows in splits.items():
        split_dir = os.path.join(dest, split)
        split_vid = os.path.join(split_dir, "videos")
        os.makedirs(split_vid, exist_ok=True)
        seen: set[str] = set()
        for row in srows:
            vid = row.get("video")
            if not vid or vid in seen:
                continue
            seen.add(vid)
            for path in glob.glob(os.path.join(videos_root, vid + ".*")):
                shutil.move(path, os.path.join(split_vid, os.path.basename(path)))
        out_csv = os.path.join(split_dir, "meta.csv")
        with open(out_csv, "w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter=";")
            writer.writeheader()
            writer.writerows(srows)
    if os.path.isdir(videos_root) and not os.listdir(videos_root):
        os.rmdir(videos_root)


def download_dataset(
    name: str,
    dest: str,
    url: Optional[str] = None,
    checksum: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    split_config: Optional[str] = None,
) -> None:
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
        cfg = _load_split_cfg(split_config)
        _write_splits(meta_path, dest, cfg)


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
        sp.add_argument("--split-config", help="YAML file with split configuration")
    args = p.parse_args()
    download_dataset(
        args.dataset,
        args.dest,
        args.url,
        args.checksum,
        args.username,
        args.password,
        args.split_config,
    )


if __name__ == "__main__":
    main()
