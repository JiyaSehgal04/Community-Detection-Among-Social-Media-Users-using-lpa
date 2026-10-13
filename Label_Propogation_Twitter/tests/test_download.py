import gzip
import io
import tarfile
from pathlib import Path

from pipeline import download

EGO_EXTS = ("edges", "feat", "egofeat", "featnames", "circles")


def _write_ego(twitter_dir: Path, ego_id: str, edges: str, circles: str = "0\t1\t2\n",
               skip: tuple[str, ...] = ()) -> None:
    contents = {
        "edges": edges,
        "feat": "1 1 0\n2 0 1\n",
        "egofeat": "1 1\n",
        "featnames": "0 #python\n1 @nasa\n",
        "circles": circles,
    }
    for ext in EGO_EXTS:
        if ext not in skip:
            (twitter_dir / f"{ego_id}.{ext}").write_text(contents[ext])


def test_verify_counts_egos_circles_and_combined_edges(tmp_path):
    twitter_dir = tmp_path / "twitter"
    twitter_dir.mkdir()
    _write_ego(twitter_dir, "100", "1 2\n2 3\n")
    _write_ego(twitter_dir, "200", "1 2\n", circles="")
    _write_ego(twitter_dir, "300", "4 5\n", skip=("feat",))
    # "1 2" appears twice: 4 lines, 3 unique edges, nodes {1,2,3,4}
    (tmp_path / "twitter_combined.txt").write_text("1 2\n2 3\n1 2\n3 4\n")

    stats = download.verify(tmp_path)

    assert stats["egos_found"] == 3
    assert stats["egos_missing_files"] == 1
    assert stats["egos_empty_circles"] == 1
    assert stats["combined_lines"] == 4
    assert stats["combined_unique_edges"] == 3
    assert stats["combined_unique_nodes"] == 4


def test_verify_ignores_non_ego_files(tmp_path):
    twitter_dir = tmp_path / "twitter"
    twitter_dir.mkdir()
    _write_ego(twitter_dir, "100", "1 2\n")
    (twitter_dir / ".DS_Store").write_bytes(b"\x00")
    (tmp_path / "twitter_combined.txt").write_text("1 2\n")

    stats = download.verify(tmp_path)

    assert stats["egos_found"] == 1
    assert stats["egos_missing_files"] == 0


def test_check_reports_mismatches(tmp_path):
    stats = dict(download.EXPECTED)
    assert download.check(stats) == []

    stats["combined_unique_edges"] = 5
    problems = download.check(stats)
    assert len(problems) == 1
    assert "combined_unique_edges" in problems[0]


def test_extract_skips_unsafe_paths(tmp_path):
    archive = tmp_path / "twitter.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in (("twitter/1.edges", b"1 2\n"), ("../evil.txt", b"x")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    download.extract_egos(archive, tmp_path)

    assert (tmp_path / "twitter" / "1.edges").read_text() == "1 2\n"
    assert not (tmp_path.parent / "evil.txt").exists()


def test_decompress_keeps_gz(tmp_path):
    gz = tmp_path / "twitter_combined.txt.gz"
    with gzip.open(gz, "wt") as f:
        f.write("1 2\n")

    out = download.decompress(gz)

    assert out.read_text() == "1 2\n"
    assert gz.exists()
