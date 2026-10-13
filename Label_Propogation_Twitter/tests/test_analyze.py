import numpy as np
import pytest

from pipeline import analyze


def test_bootstrap_is_seeded_and_resamples_egos():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    constant = analyze.paired_bootstrap(a, a - 1.0, n_boot=200, seed=1)
    assert constant["mean"] == pytest.approx(1.0)
    assert constant["lo"] == pytest.approx(1.0) and constant["hi"] == pytest.approx(1.0)

    c, d = np.array([0.0, 3.0, 1.0, 7.0, 2.0]), np.zeros(5)
    first = analyze.paired_bootstrap(c, d, n_boot=200, seed=1)
    assert first == analyze.paired_bootstrap(c, d, n_boot=200, seed=1)  # same seed, same answer
    assert first["lo"] <= first["mean"] <= first["hi"]
    assert first["lo"] < first["hi"]  # the interval widens when egos differ


def test_ablation_and_sensitivity_refuse_test_egos():
    split = {"train": [1], "val": [2], "test": [99]}
    with pytest.raises(ValueError, match="test egos"):
        analyze.run_ablation(None, split, [2, 99], {})
    with pytest.raises(ValueError, match="test egos"):
        analyze.run_sensitivity(None, split, [99], {})
