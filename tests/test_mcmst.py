#!/usr/bin/env python3
# UTF-8 encoding
import pytest
import numpy as np

from src.common import Edge
from src.mcmst import MCMST
from src.mcmst_state import MCMSTState


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
def mcmst():
    return MCMST(COORDINATES, 1, 50, EDGES, CABLE_COSTS, CABLE_LENGTHS, CABLE_CAPACITIES)

# @pytest.mark.parametrize('cable_index', [(5), (6)])
# def test_mcmst_update(mcmst, cable_index):
#     # INIT
#     for i in np.arange(1,N):
#         mcmst.solution.add_edge(Edge(0,i), CABLE_COSTS[-1], 0, COORDINATES)
#
#     mcmst.mcmst_state = MCMSTState(CABLE_COSTS,
#                                   CABLE_LENGTHS,
#                                   CABLE_CAPACITIES,
#                                   COORDINATES,
#                                   50,
#                                   EDGES,
#                                   np.arange(1,N))
#
#     next_candidate = mcmst.mcmst_state.next_candidate(cable_index)
#     mcmst._update(next_candidate, 0)
#
#     for c in next_candidate.new_children:
#         assert Edge(next_candidate.parent, c) in mcmst.solution.edges[0]
#         assert Edge(0, c) not in mcmst.solution.edges[0]


def test_mcmst_compute(mcmst):
    mcmst.compute()
