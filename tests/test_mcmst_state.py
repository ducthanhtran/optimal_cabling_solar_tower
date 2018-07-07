#!/usr/bin/env python3
# UTF-8 encoding
from copy import deepcopy
from itertools import chain

import pytest
import numpy as np

from src.common import Edge
from src.solutions import EdgeSolution
from src.mcmst_state import MCMSTState, Candidate


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
N = COORDINATES.shape[0]

CABLE_COSTS = np.array([0.58,0.87,1.24,1.95,3.13,5.19,6.9])
CABLE_LENGTHS = np.array([38.36,47.08,56.04,69.3,84.87,102.79,120.31])
CABLE_CAPACITIES = np.array([56,73,92,124,162,209,250])
EDGES = [[Edge(0,1)], [Edge(1,3)]]


@pytest.fixture
def mcmst_state():
    return MCMSTState(CABLE_COSTS, CABLE_LENGTHS, CABLE_CAPACITIES, COORDINATES, 25, EDGES, np.arange(1,COORDINATES.shape[0])) # Spain has 25 euro working cost


@pytest.fixture
def solution():
    return EdgeSolution(1)


def test_init(mcmst_state):
    np.testing.assert_array_equal(mcmst_state.predecessor, np.zeros(N))
    assert mcmst_state.successors == {i:[] for i in range(N)}
    assert mcmst_state.heliostat_parents == {i:[0] for i in range(N)}
    assert mcmst_state.unupgraded == [i for i in range(1,N)]

    assert len(mcmst_state.data_cables) == len(EDGES)

    assert mcmst_state.trench_costs.shape == mcmst_state.distance.shape
    assert (mcmst_state.trench_costs == 0).sum() - mcmst_state.trench_costs.shape[0] == 2*len(list(chain.from_iterable(EDGES))) # forward and backward edges

    for edge in mcmst_state.data_cables:
        assert mcmst_state.trench_costs[edge.v,edge.w] == 0
        assert mcmst_state.trench_costs[edge.w,edge.v] == 0

    # Current
    np.testing.assert_array_equal(mcmst_state.current.capacity, np.ones(N))
    np.testing.assert_array_equal(mcmst_state.current.max_capacity, np.ones(N)*np.max(CABLE_CAPACITIES))
    np.testing.assert_array_equal(mcmst_state.current.max_length, np.ones(N)*np.max(CABLE_LENGTHS))
    np.testing.assert_array_equal(mcmst_state.current.cable_cost, np.ones(N)*np.max(CABLE_COSTS))

    # forbidden_candidates = random_list(np.random.randint(0,N), N)
    # methods
    for parent in range(1,N):
        # test_max_number_reconnections
        number = mcmst_state._max_number_reconnections(parent)
        assert number == CABLE_CAPACITIES[-1] - 1

        # test_upgrade_cost and test_best_candidate
        ## cable length not sufficient
        for i in range(5):
            assert np.inf == mcmst_state._upgrade_cost(parent, i)

            best_candidate = mcmst_state._best_candidate(parent, i)
            assert best_candidate.parent == parent
            assert best_candidate.value == 0.0
            assert best_candidate.added_capacity == 0
            assert best_candidate.new_children == []
            assert best_candidate.new_cable_index == i
        ## cable length sufficient
        for j in range(5,7):
            assert np.inf != mcmst_state._upgrade_cost(parent, j)

            best_candidate = mcmst_state._best_candidate(parent, j)
            assert best_candidate.parent == parent
            assert best_candidate.value > 0
            assert best_candidate.added_capacity > 0
            assert len(best_candidate.new_children) > 0
            assert best_candidate.new_cable_index == j

        # test_candidate_indices
        candidates = mcmst_state._candidate_indices(parent)
        assert np.all(candidates != parent) # parent
        assert np.all(candidates != 0) # solar tower

        # for forbidden in forbidden_candidates:
        #     assert np.all(candidates != forbidden)

        # reconnecting_cost
        reconn_profit = mcmst_state._reconnecting_profit(parent, candidates)
        assert len(reconn_profit) == len(candidates)

        # parent_of_parents
        list(mcmst_state._parent_of_parents(parent)) == []


@pytest.mark.parametrize('cable_index', [(5),(6)])
def test_update(mcmst_state, cable_index, solution):
    # copy old data structures for comparisons
    old_current = deepcopy(mcmst_state.current)

    candidate = mcmst_state.next_candidate(cable_index, solution)
    mcmst_state.update(candidate)

    # parents of parent
    for p in mcmst_state._parent_of_parents(candidate.parent):
        assert mcmst_state.current.capacity[p] == candidate.added_capacity + 1

    # children
    for c in candidate.new_children:
        assert mcmst_state.predecessor[c] == candidate.parent

        # check invariant capacity, cable cost, max length and max capacity of all children
        assert mcmst_state.current.capacity[c] == old_current.capacity[c]
        assert mcmst_state.current.cable_cost[c] == old_current.cable_cost[c]
        assert mcmst_state.current.max_length[c] == old_current.max_length[c]
        assert mcmst_state.current.max_capacity[c] == old_current.max_capacity[c]

    # parent vertex
    assert mcmst_state.current.capacity[candidate.parent] == candidate.added_capacity + 1
    assert mcmst_state.current.cable_cost[candidate.parent] == CABLE_COSTS[candidate.new_cable_index]
    assert mcmst_state.current.max_length[candidate.parent] == CABLE_LENGTHS[candidate.new_cable_index]
    assert mcmst_state.current.max_capacity[candidate.parent] == CABLE_CAPACITIES[candidate.new_cable_index]

    ##########################
    # candidate = mcmst_state.next_candidate(1)
    # for i in range(7):
    #     assert mcmst_state.next_candidate(i) is not None
