import pytest
import numpy as np
from src.common import compute_partitions

@pytest.mark.parametrize("coordinates, partitions", [
    (np.array([[0,0], [1,2], [3,4], [5,6]]), 1),
    (np.array([[0,0], [1,2], [3,4], [5,6]]), 2),
    (np.array([[0,0], [1,2], [3,4], [5,6]]), 3)
    ])
def test_compute_partitions(coordinates, partitions):
    L = compute_partitions(coordinates, partitions)
    assert len(L) == partitions
    for arr in L:
        assert arr.size != 0

    # should fail due to only having 3 heliostats
    M = compute_partitions(np.array([[0,0], [1,2], [3,4], [5,6]]), 4)
    assert not M
