import numpy as np
import pytest

from backend import overlap


def test_omega_index_known_values_and_random_baseline_below_perfect():
    nodes = {1, 2, 3}
    # pair (1,2) shares one group in A but all pairs share one group in B -> chance-level agreement
    assert overlap.omega_index([{1, 2}, {3}], [{1, 2, 3}], nodes) == pytest.approx(0.0)
    # hand-computed: 5/6 of pairs agree, 1/2 expected by chance -> (5/6 - 1/2) / (1 - 1/2)
    assert overlap.omega_index([{1, 2, 3}, {3, 4}], [{1, 2, 3}, {4}], {1, 2, 3, 4}) == pytest.approx(2 / 3)
    cover = [{1, 2}, {2, 3}]  # overlapping cover compared with itself
    assert overlap.omega_index(cover, cover, nodes) == pytest.approx(1.0)

    circles = {"0": [1, 2, 3, 4], "1": [5, 6, 7, 8], "2": [9, 10, 11, 12]}
    communities = [set(members) for members in circles.values()]
    assert overlap.circle_omega(communities, circles) == pytest.approx(1.0)
    random_scores = [
        overlap.circle_omega(overlap.random_partition_like(communities, np.random.default_rng(seed)), circles)
        for seed in range(20)
    ]
    assert np.mean(random_scores) < 0.2  # a size-matched random partition scores near zero
