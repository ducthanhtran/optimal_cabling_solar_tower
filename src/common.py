#!/usr/bin/env python3
# UTF-8 encoding
from itertools import chain
from typing import Dict, List, NamedTuple

import numpy as np

CABLE_COSTS = np.array([0.58, 0.87, 1.24, 1.95, 3.13, 5.19, 6.9])
CABLE_LENGTHS = np.array([38.36, 47.08, 56.04, 69.3, 84.87, 102.79, 120.31])
CABLE_CAPACITIES = np.array([56, 73, 92, 124, 162, 209, 250])

Cables = NamedTuple('Cables', [('costs', np.ndarray),
                               ('lengths', np.ndarray),
                               ('capacities', np.ndarray)])
Edge = NamedTuple('Edge', [('v', int), ('w', int)])  # edge contains two indices to coordinates-array


def compute_partitions(coordinates: np.ndarray, partitions: int) -> List[np.ndarray]:
    """
    Partitions the coordinates into an array of indices. For this matter we use the angles
    between the reference vector of the x-axis.

    :param coordinates: a 2D real array of x- and y-coordinates
    :param partitions: number of partitions
    :return: list of arrays that contain indices of heliostats with regards to the coordinates array
    """
    if partitions > len(coordinates) - 1:  # not enough heliostats
        return []

    degrees = np.degrees(np.arctan2(coordinates[1:, 1], coordinates[1:, 0]))
    indices = degrees.argsort()
    padding = (-len(indices)) % partitions
    lst = np.split(np.concatenate((indices, np.ones(padding) * -1)), partitions)  # padding value: -1
    # remove padding from last partition and increase index
    return [np.delete(a, np.where(a == -1)) + 1 for a in lst]


def solution_value(distances: np.ndarray, edges, edge_cost: float, degrees):
    cost = sum(distances[e.v, e.w] * edge_cost for e in chain.from_iterable(edges))
    # subtract switch costs of solar tower at the end
    return cost + sum(switch_cost(d) for d in degrees.values()) - switch_cost(degrees[0])


def switch_cost(degree: int) -> float:
    """Note: Hard-coded values for quick and dirty development"""
    if degree <= 2:
        return 100
    elif 3 <= degree <= 9:
        return 800
    elif 10 <= degree <= 17:
        return 1500
    else:
        raise ValueError('Degree is too high. No cost found.')


def cable_length(edges: List[Edge], distance: np.ndarray) -> float:
    return sum(distance[e.v, e.w] for e in chain.from_iterable(edges))


def total_cost(edges: List[Edge],
               degrees: Dict[int, int],
               data_cable_cost: float,
               power_cable_assignment: Dict[Edge, int],
               distance: np.ndarray) -> float:
    data_costs = sum(
            distance[e.v, e.w] * data_cable_cost for e in chain.from_iterable(edges))  # incl. trench cost + foil
    switch_costs = sum(switch_cost(d) for d in degrees.values()) - switch_cost(degrees[0])
    power_costs = sum(2 * distance[e.v, e.w] * CABLE_COSTS[power_cable_assignment[e.w]]
                      for e in chain.from_iterable(edges))
    return data_costs + switch_costs + power_costs
