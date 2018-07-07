#!/usr/bin/env python3
# UTF-8 encoding
from itertools import chain
from typing import List, NamedTuple

import numpy as np


Edge = NamedTuple('Edge', [('v', int), ('w', int)]) # edge contains two indices to coordinates-array


def compute_partitions(coordinates: np.ndarray, partitions: int) -> List[np.ndarray]:
    """
    Partitions the coordinates into an array of indices. For this matter we use the angles
    between the reference vector of the x-axis.

    :param coordinates: a 2D real array of x- and y-coordinates
    :param partitions: number of partitions
    :return: list of arrays that contain indices of heliostats with regards to the coordinates array
    """
    if partitions > len(coordinates) - 1: # not enough heliostats
        return []

    degrees = np.degrees(np.arctan2(coordinates[1:,1], coordinates[1:,0]))
    indices = degrees.argsort()
    padding = (-len(indices))%partitions
    L = np.split(np.concatenate((indices,np.ones(padding)*-1)),partitions) # padding value: -1
    # remove padding from last partition and increase index
    return [np.delete(a, np.where(a == -1)) + 1 for a in L]


def solution_value(distances: np.ndarray, edges, edge_cost: float, degrees):
    cost = sum(distances[e.v,e.w]*edge_cost for e in chain.from_iterable(edges))
    # subtract switch costs of solar tower at the end
    # NOTE: make it more readable or omit switch cost of solar tower immediately
    return cost + sum(_switch_cost(d) for d in degrees.values()) - _switch_cost(degrees[0])


def _switch_cost(degree: int) -> float:
    """Note: Hard-coded values for quick and dirty development"""
    if degree <= 2:
        return 100
    elif degree >= 3 and degree <= 9:
        return 800
    elif degree >= 10 and degree <= 17:
        return 1500
    else:
        raise ValueError('Degree is too high. No cost found.')
