"""Resumable PlantVillage download + extraction.

The Hugging Face CDN drops long transfers and `hf_hub_download` starts a fresh
temporary blob on every process launch, so a 2.2 GB archive never finishes.
This script instead does a plain HTTP GET with a Range header against a single
fixed part-file, so every retry continues from the bytes already on disk.

Usage:
    python download_dataset.py                # download + extract color/ and segmented/
    python download_dataset.py --keep-archive # keep the zip after extraction
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO_ID = "mohanty/PlantVillage"
ARCHIVE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/data.zip"
SPLIT_FILES = [
    "splits/color_train.txt",
    "splits/color_test.txt",
    "splits/segmented_train.txt",
    "splits/segmented_test.txt",
]
KEEP_PREFIXES = ("color", "segmented")
CHUNK = 1024 * 1024
USER_AGENT = "prism-leafsense/1.0"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        size = resp.headers.get("x-linked-size") or resp.headers.get("Content-Length")
        return int(size)


def download_splits(dataset_dir: Path) -> None:
    for name in SPLIT_FILES:
        dest = dataset_dir / Path(name).name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{name}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
            shutil.copyfileobj(resp, out)
    log("splits ready")


def seed_from_hf_cache(part: Path) -> None:
    """Reuse the largest partial blob left behind by a previous hf_hub_download."""
    if part.exists():
        return
    cache = Path.home() / ".cache/huggingface/hub/datasets--mohanty--PlantVillage/blobs"
    if not cache.is_dir():
        return
    blobs = sorted(cache.glob("*.incomplete"), key=lambda p: p.stat().st_size, reverse=True)
    if blobs and blobs[0].stat().st_size > CHUNK:
        log(f"seeding part file from cached blob ({blobs[0].stat().st_size} bytes)")
        shutil.move(str(blobs[0]), str(part))
        for extra in blobs[1:]:
            extra.unlink(missing_ok=True)


def download_archive(dataset_dir: Path, max_attempts: int = 500) -> Path:
    part = dataset_dir / "data.zip.part"
    final = dataset_dir / "data.zip"
    if final.exists():
        log(f"archive already present: {final}")
        return final

    seed_from_hf_cache(part)
    total = remote_size(ARCHIVE_URL)
    log(f"archive size {total} bytes")

    delay = 3
    for attempt in range(1, max_attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if have >= total:
            break
        headers = {"User-Agent": USER_AGENT, "Range": f"bytes={have}-"}
        try:
            req = urllib.request.Request(ARCHIVE_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                if have and resp.status != 206:
                    raise RuntimeError(f"server ignored Range (status {resp.status})")
                with open(part, "ab") as out:
                    last_report = time.time()
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                        have += len(chunk)
                        if time.time() - last_report > 20:
                            pct = 100 * have / total
                            log(f"{have}/{total} bytes ({pct:.1f}%)")
                            last_report = time.time()
            delay = 3
        except Exception as exc:
            log(f"attempt {attempt} interrupted at {have} bytes "
                f"({type(exc).__name__}: {exc}); retrying in {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 30)

    have = part.stat().st_size if part.exists() else 0
    if have != total:
        raise RuntimeError(f"incomplete download: {have}/{total} bytes")
    part.replace(final)
    log(f"archive complete: {final}")
    return final


def extract(archive: Path, dataset_dir: Path) -> None:
    target = dataset_dir / "plantvillage"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = []
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            parts = Path(member).parts
            for prefix in KEEP_PREFIXES:
                if prefix in parts:
                    members.append((member, Path(*parts[parts.index(prefix):])))
                    break
        log(f"extracting {len(members)} files into {target}")
        for i, (member, rel) in enumerate(members, 1):
            dest = target / rel
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            if i % 5000 == 0:
                log(f"extracted {i}/{len(members)}")
    log("extraction done")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(Path(__file__).resolve().parents[2] / "datasets"))
    ap.add_argument("--keep-archive", action="store_true")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    download_splits(dataset_dir)
    archive = download_archive(dataset_dir)
    extract(archive, dataset_dir)
    if not args.keep_archive:
        try:
            archive.unlink()
            log("removed archive to free disk space")
        except OSError as exc:
            log(f"could not remove archive: {exc}")
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
