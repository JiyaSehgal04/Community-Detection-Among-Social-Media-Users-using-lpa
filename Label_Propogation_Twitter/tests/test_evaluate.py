# tests/test_evaluate.py
import pytest

from pipeline import evaluate

SPLIT = {"train": [1, 2], "val": [3], "test": [8, 9]}


def test_guards_block_leakage_wrong_egos_and_overwrites(tmp_path):
    ok = {"tuned_on": [1, 2, 3]}
    evaluate.check_guards(ok, SPLIT, [8, 9], tmp_path)  # passes silently

    with pytest.raises(ValueError, match="tuned on test egos"):
        evaluate.check_guards({"tuned_on": [1, 8]}, SPLIT, [8, 9], tmp_path)
    with pytest.raises(ValueError, match="exactly the test egos"):
        evaluate.check_guards(ok, SPLIT, [8], tmp_path)

    (tmp_path / "metrics.csv").write_text("x")
    with pytest.raises(FileExistsError):
        evaluate.check_guards(ok, SPLIT, [8, 9], tmp_path)
    evaluate.check_guards(ok, SPLIT, [8, 9], tmp_path, force=True)
