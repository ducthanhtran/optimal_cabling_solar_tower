from itertools import chain
from typing import Dict, List

import numpy as np
from scipy.spatial.distance import cdist

from common import Edge


def get_successors(edges: List[List[Edge]], N: int) -> Dict[int, List[int]]:
    successor = {i: [] for i in range(N)}
    for e in chain.from_iterable(edges):
        successor[e.v].append(e.w)
    return successor


def get_successors_gen(v: int, successors: Dict[int, List[int]]) -> List[int]:
    res = [v]
    for s in successors[v]:
        res = res + get_successors_gen(s, successors)
    return res


def get_successor_list(edges: List[List[Edge]], N: int) -> Dict[int, List[int]]:
    succs = get_successors(edges, N)
    res = {i: [] for i in range(N)}
    for i in range(N):
        res[i] = get_successors_gen(i, succs)
    return res


def get_predecessor(edges: List[List[Edge]]) -> Dict[int, int]:
    predecessor = {}
    for e in chain.from_iterable(edges):
        predecessor[e.w] = e.v
    return predecessor


def obtain_parents(edges: List[List[Edge]], N: int) -> Dict[int, List[int]]:
    predecessor = get_predecessor(edges)
    parents = {i: [] for i in range(N)}
    for i in range(1, N):
        parents[i] = list(get_predecessors(i, predecessor))
    return parents


def get_predecessors(v: int, predecessor: Dict[int, int]):
    while predecessor[v] != 0:
        yield predecessor[v]
        v = predecessor[v]


def count_successors(vertex: int, successors: List[int]) -> int:
    i = 1
    for s in successors[vertex]:
        i += count_successors(s, successors)
    return i


def compute_capacities(edges: List[List[Edge]], N: int) -> np.ndarray:
    successor = get_successors(edges, N)
    return np.array([count_successors(i, successor) for i in range(N)])


def assign_power_cables(edges: List[List[Edge]],
                        coordinates: np.ndarray,
                        cable_lengths: np.ndarray,
                        cable_capacities: np.ndarray) -> Dict[Edge, int]:
    """
    Assigns power cables greedily according to their required capacity and length constraints

    :param edges: current edges
    :param coordinates: coordinates of heliostats and solar tower
    :param cable_lengths: maximum cable lengths of power cables - for an connecting edge
    :param cable_capacities: maximum cable capacities of power cables
    :return: assignment of edges to power cables in form of a dictionary
    """
    capacities = compute_capacities(edges, coordinates.shape[0])
    distances = cdist(coordinates, coordinates)

    assignment = {}

    for e in chain.from_iterable(edges):
        length = distances[e.v, e.w] <= cable_lengths
        capacity = capacities[e.w] <= cable_capacities

        best_cable_index = np.where(np.logical_and(length, capacity))
        if best_cable_index[0].size == 0:
            raise RuntimeError('No suitable cable found for {}'.format(e))

        assignment[e] = best_cable_index[0][0]
    return assignment


def assign_cable(power_cable_assignment: Dict[Edge, int], N: int) -> np.ndarray:
    """
    Converts the assignment-dictionary to an integer numpy array.

    :param power_cable_assignment: dictionary with cable assignment
    :param N: number heliostats
    :return: integer numpy array that specifies for each heliostats the power cable type it is connected to
    """
    cables = np.zeros(N)
    for edge, cable_index in power_cable_assignment.items():
        cables[edge.w] = cable_index
    cables[0] = np.nan
    return cables.astype(int)


# def compute_max_capacities(power_cable_assignment: Dict[Edge, int], N: int) -> np.ndarray:
#     max_capacities = np.zeros(N)
#     for edge, cable_index in power_cable_assignment.items():
#         max_capacities[edge.w] = CABLE_CAPACITIES[cable_index]  # TODO: BUGGY? might need to import this
#     max_capacities[0] = N
#     return max_capacities.astype(int)
#
#
# def obtain_leaves(edges: List[List[Edge]]) -> List[int]:
#     successor = get_successors(edges) ## TODO: BUGGY
#     return [v for v in successor if successor[v] == []]