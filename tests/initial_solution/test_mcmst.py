#!/usr/bin/env python3
# UTF-8 encoding
import pytest
import numpy as np

from ..common import random_list
from src.initial_solution.mcmst_state import MCMSTState


COORDINATES = np.array([[  0.  ,   0.  ],
                        [ 73.33,  43.86],
                        [ 59.72,  61.12],
                        [ 42.15,  74.33],
                        [ 21.8 ,  82.62],
                        [  0.  ,  85.45],
                        [-21.8 ,  82.62],
                        [-42.15,  74.33],
                        [-59.72,  61.12],
                        [-73.33,  43.86]])

CABLE_COSTS = np.array([0.58,0.87,1.24,1.95,3.13,5.19,6.9])
CABLE_LENGTHS = np.array([38.36,47.08,56.04,69.3,84.87,102.79,120.31])
CABLE_CAPACITY = np.array([56,73,92,124,162,209,250])


@pytest.fixture
def mcmst_state():
    return MCMSTState(CABLE_COSTS, CABLE_LENGTHS, CABLE_CAPACITY, COORDINATES, np.arange(COORDINATES.shape[0]))


def test_init(mcmst_state):
    N = COORDINATES.shape[0]

    np.testing.assert_array_equal(mcmst_state.predecessor, np.zeros(N))
    assert mcmst_state.successors == {i:[] for i in range(N)}

    np.testing.assert_array_equal(mcmst_state.number_connected_heliostats, np.zeros(N))
    assert mcmst_state.heliostat_parents == {i:[0] for i in range(N)}

    assert mcmst_state.upgraded_vertices == [0]

    # Current
    np.testing.assert_array_equal(mcmst_state.current.capacity, np.ones(N))
    np.testing.assert_array_equal(mcmst_state.current.max_capacity, np.ones(N)*np.max(CABLE_CAPACITY))
    np.testing.assert_array_equal(mcmst_state.current.max_length, np.ones(N)*np.max(CABLE_LENGTHS))
    assert mcmst_state.current.cable_index_to_parent == np.argmax(CABLE_CAPACITY)


    for parent in range(1,N):
        # TODO
        assert mcmst_state._max_number_reconnections(parent) == CABLE_CAPACITY[-1] - 1


def test_candidate_indices(mcmst_state):
    N = COORDINATES.shape[0]

    # after init
    forbidden_candidates = random_list(np.random.randint(0,N), N)
    for parent in range(1,N):
        candidates = mcmst_state._candidate_indices(parent, forbidden_candidates)
        assert np.all(candidates != parent) # parent
        assert np.all(candidates != 0) # solar tower

        for forbidden in forbidden_candidates:
            assert np.all(candidates != forbidden)
