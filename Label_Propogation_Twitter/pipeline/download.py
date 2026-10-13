"""Download and verify the SNAP ego-Twitter dataset.

Run from the project root:

    python -m pipeline.download

Files already present are skipped, so the script can be re-run after a failure.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://snap.stanford.edu/data/"
FILES = ("twitter.tar.gz", "twitter_combined.txt.gz", "readme-Ego.txt")
EGO_EXTS = ("edges", "feat", "egofeat", "featnames", "circles")

# Values counted from the SNAP files downloaded on 2026-10-01.
EXPECTED = {
    "egos_found": 973,
    "egos_missing_files": 0,
    "egos_empty_circles": 35,
    "combined_lines": 2_420_766,
    "combined_unique_edges": 1_768_149,
    "combined_unique_nodes": 81_306,
}

LABELS = {
    "egos_found": "Ego networks found",
    "egos_missing_files": "Egos missing files",
    "egos_empty_circles": "Egos with empty circles",
    "combined_lines": "Combined lines",
    "combined_unique_edges": "Combined unique edges",
    "combined_unique_nodes": "Combined unique nodes",
}


def download_file(url: str, dest: Path, force: bool = False) -> None:
    """Download url to dest via a .part file, skipping it if dest already exists."""
    if dest.exists() and not force:
        print(f"  skip  {dest.name} (already downloaded)")
        return
    part = dest.with_name(dest.name + ".part")
    print(f"  get   {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
        shutil.copyfileobj(resp, out)
    part.replace(dest)


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def extract_egos(archive: Path, raw_dir: Path) -> None:
    """Extract the ego archive into raw_dir, skipping members that would land outside it."""
    with tarfile.open(archive, "r:gz") as tar:
        members = []
        for member in tar.getmembers():
            if not (member.isfile() or member.isdir()) or not _is_within(raw_dir, raw_dir / member.name):
                print(f"  warn  skipping unsafe archive entry: {member.name}")
                continue
            members.append(member)
        if hasattr(tarfile, "data_filter"):
            tar.extractall(raw_dir, members=members, filter="data")
        else:
            tar.extractall(raw_dir, members=members)


def decompress(gz_path: Path) -> Path:
    """Decompress a .gz file next to itself, keeping the original."""
    out = gz_path.with_suffix("")
    with gzip.open(gz_path, "rb") as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return out


def verify(raw_dir: Path) -> dict[str, int]:
    """Count egos, missing files, empty circles, and combined edges/nodes."""
    twitter_dir = raw_dir / "twitter"
    ego_ids = {
        p.stem for p in twitter_dir.iterdir()
        if p.is_file() and p.suffix.lstrip(".") in EGO_EXTS and p.stem.isdigit()
    }
    missing = sum(
        1 for ego in ego_ids
        if any(not (twitter_dir / f"{ego}.{ext}").exists() for ext in EGO_EXTS)
    )
    empty_circles = sum(
        1 for ego in ego_ids
        if (p := twitter_dir / f"{ego}.circles").exists() and p.stat().st_size == 0
    )

    edges = pd.read_csv(
        raw_dir / "twitter_combined.txt", sep=" ", header=None,
        names=["src", "dst"], dtype="int64",
    )
    nodes = np.unique(np.concatenate([edges["src"].to_numpy(), edges["dst"].to_numpy()]))

    return {
        "egos_found": len(ego_ids),
        "egos_missing_files": missing,
        "egos_empty_circles": empty_circles,
        "combined_lines": len(edges),
        "combined_unique_edges": len(edges.drop_duplicates()),
        "combined_unique_nodes": len(nodes),
    }


def check(stats: dict[str, int]) -> list[str]:
    """Return one message per stat that differs from EXPECTED."""
    return [
        f"{key}: got {stats[key]:,}, expected {expected:,}"
        for key, expected in EXPECTED.items()
        if stats.get(key) != expected
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true", help="re-download and re-extract everything")
    args = parser.parse_args(argv)

    raw_dir: Path = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading")
    for name in FILES:
        download_file(BASE_URL + name, raw_dir / name, force=args.force)

    print("Extracting")
    twitter_dir = raw_dir / "twitter"
    if args.force or not twitter_dir.exists() or not any(twitter_dir.iterdir()):
        extract_egos(raw_dir / "twitter.tar.gz", raw_dir)
        print(f"  done  {twitter_dir}")
    else:
        print(f"  skip  {twitter_dir} (already extracted)")

    combined = raw_dir / "twitter_combined.txt"
    if args.force or not combined.exists():
        decompress(raw_dir / "twitter_combined.txt.gz")
        print(f"  done  {combined}")
    else:
        print(f"  skip  {combined} (already decompressed)")

    print("Verifying")
    stats = verify(raw_dir)
    for key, label in LABELS.items():
        print(f"  {label:<24}: {stats[key]:>10,}  (expected {EXPECTED[key]:,})")

    problems = check(stats)
    if problems:
        print("\nVerification FAILED:")
        for msg in problems:
            print(f"  - {msg}")
        print("Delete data/raw/twitter/ and re-run, or re-run with --force.")
        return 1
    print("\nVerification passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
