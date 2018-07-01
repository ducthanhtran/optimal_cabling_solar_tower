#!/usr/bin/env python3
# UTF-8 encoding
import pytest
import numpy as np
from src.solutions import PowerCableSolution

@pytest.mark.parametrize("n, partitions", [
    (0,0),
    (5,1),
    (5,5),
    ])
def test_power_cable_solution_init(n: int, partitions: int):
    solution =  PowerCableSolution(n, partitions)

    assert not solution.predecessor
    assert len(solution.successors) == n
    assert solution.connected_children == {i:0 for i in range(n)}

    # base class
    assert len(solution.edges) == partitions
