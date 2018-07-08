import pickle
from typing import NamedTuple, Dict, List, Set
from common import *

from itertools import chain

import numpy as np
from scipy.spatial.distance import cdist


CABLE_COSTS = np.array([0.58,0.87,1.24,1.95,3.13,5.19,6.9])
CABLE_LENGTHS = np.array([38.36,47.08,56.04,69.3,84.87,102.79,120.31])
CABLE_CAPACITIES = np.array([56,73,92,124,162,209,250])

Cables = NamedTuple('Cables', [('costs', np.ndarray),
                               ('lengths', np.ndarray),
                               ('capacities', np.ndarray)])

cables = Cables(CABLE_COSTS, CABLE_LENGTHS, CABLE_CAPACITIES)

coordinates = np.loadtxt('../data/instances/PS10.csv', delimiter=';')
coordinates = np.vstack((np.array([0,0]), coordinates))

D = cdist(coordinates,coordinates)

def get_successors(edges: List[List[Edge]], N) -> Dict[int,List[int]]:
    successor = {i:[] for i in range(N)}
    for e in chain.from_iterable(edges):
        successor[e.v].append(e.w)
    return successor

def get_successors_gen(v, successors):
    res = [v]
    for s in successors[v]:
        res = res + get_successors_gen(s, successors)
    return res

def get_successor_list(edges: List[List[Edge]], N):
    succs = get_successors(edges,N)
    res = {i:[] for i in range(N)}
    for i in range(N):
        res[i] = get_successors_gen(i,succs)
    return res

def get_predecessor(edges: List[List[Edge]]) -> Dict[int,int]:
    predecessor = {}
    for e in chain.from_iterable(edges):
        predecessor[e.w] = e.v
    return predecessor

def obtain_parents(edges: List[List[Edge]], N) -> Dict[int,List[int]]:
    predecessor = get_predecessor(edges)
    parents = {i:[] for i in range(N)} # type: Dict[int,List[int]]
    for i in range(1,N):
        parents[i] = list(get_predecessors(i,predecessor))
    return parents

def get_predecessors(v, predecessor):
    while(predecessor[v] != 0):
        yield predecessor[v]
        v = predecessor[v]


def obtain_leaves(edges: List[List[Edge]]) -> List[int]:
    successor = get_successors(edges)
    return [v for v in successor if successor[v] == []]


def count_successors(vertex, successors) -> int:
    i = 1
    for s in successors[vertex]:
        i+= count_successors(s, successors)
    return i


def compute_capacities(edges, N) -> np.ndarray:
    successor = get_successors(edges,N)
    return np.array([count_successors(i,successor) for i in range(N)])


def assign_power_cables(edges: List[List[Edge]], coordinates, cable_lengths, cable_capacities) -> Dict[Edge,int]:
    capacities = compute_capacities(edges, coordinates.shape[0])
    distances = cdist(coordinates, coordinates)

    assignment = {}

    for e in chain.from_iterable(edges):
        length = distances[e.v,e.w] <= cable_lengths
        capacity = capacities[e.w] <= cable_capacities

        best_cable_index = np.where(np.logical_and(length,capacity) == True)
        if best_cable_index[0].size == 0:
            raise RuntimeError('No suitable cable found for {}'.format(e))

        assignment[e] = best_cable_index[0][0]
    return assignment

def assign_cable(power_cable_assignment,N):
    cables = np.zeros(N)
    for edge, cable_index in power_cable_assignment.items():
        cables[edge.w] = cable_index
    cables[0] = np.nan
    return cables.astype(int)

def compute_max_capacities(power_cable_assignment,N) -> np.ndarray:
    max_capacities = np.zeros(N)
    for edge, cable_index in power_cable_assignment.items():
        max_capacities[edge.w] = CABLE_CAPACITIES[cable_index]
    max_capacities[0] = N
    return max_capacities.astype(int)

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

def total_cost(edges, degrees, data_cable_cost, power_cable_assignment):
    data_cost = sum(D[e.v,e.w]*data_cable_cost for e in chain.from_iterable(edges)) #incl trench cost + foil
    switch_cost = sum(_switch_cost(d) for d in degrees.values()) - _switch_cost(degrees[0])
    power_cost = sum(2*D[e.v,e.w] * CABLE_COSTS[power_cable_assignment[e.w]] for e in chain.from_iterable(edges))
    return data_cost + switch_cost + power_cost


if __name__ == '__main__':


    cost = 54
    for i in range(3,7):
        with open('../pkl/after_ls_hamilton_{}_{}.png.pkl'.format(cost,i), 'rb') as o:
            edges,degrees = pickle.load(o)

            a = assign_power_cables(edges, coordinates, CABLE_LENGTHS, CABLE_CAPACITIES)
            ac = assign_cable(a,coordinates.shape[0])

            print("{}: {}".format(i,total_cost(edges,degrees,cost,ac)))
