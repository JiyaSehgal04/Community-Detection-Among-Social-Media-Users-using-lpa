import pandas as pd
import pytest

from pipeline import tune


def test_infeasible_config_ranks_below_any_feasible_one():
    trials = pd.DataFrame({
        "trial": [0, 1, 2, 3],
        "objective": [1.50, 0.40, 0.5001, 0.5004],
        "feasible": [False, True, True, True],
        "circle_f1": [0.90, 0.10, 0.20, 0.60],
    })
    ranked = tune.rank_trials(trials)
    assert ranked.trial.tolist()[-1] == 0                 # infeasible last despite the best raw score
    assert ranked.trial.tolist()[:2] == [3, 2]            # 0.5001 and 0.5004 tie at 3 decimals; higher F1 wins
    nan_obj = pd.DataFrame({"trial": [0, 1], "objective": [float("nan"), 0.1],
                            "feasible": [True, True], "circle_f1": [0.5, 0.5]})
    assert tune.rank_trials(nan_obj).trial.tolist() == [1, 0]  # NaN objective ranks last


def test_tuning_refuses_test_egos():
    split = {"train": [1, 2], "val": [3], "test": [99]}
    with pytest.raises(ValueError, match="test egos"):
        tune.run_tuning(None, split, train_egos=[1, 99], val_egos=[3])  # raises before touching data
