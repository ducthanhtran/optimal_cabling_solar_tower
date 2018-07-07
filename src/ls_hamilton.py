import copy
from itertools import chain, product
from pickle import dump, HIGHEST_PROTOCOL
from typing import Dict, List, Set, Tuple, NamedTuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from initial_hamilton import Hamilton
from common import Edge
from intersection import is_edge_intersecting
from solutions import DataCableSolution

TriangulationEdges = NamedTuple('TriangulationEdges', [('add', List[Edge]), ('remove', List[Edge]), ('indices', List[int])])


def perform_local_search_hamilton(initial_solution: DataCableSolution,
                                  min_improvement: float,
                                  upper_cap: float):
    # perform local search onto initial Hamilton solution
    degrees = copy.deepcopy(initial_solution.solution.degrees)
    edges = copy.deepcopy(initial_solution.solution.edges)
    edge_coords = copy.deepcopy(initial_solution.solution.edge_coords)
    d = obtain_initial_partition(edges)
    coordinates = initial_solution.coordinates[:]
    incident = compute_incident_edges(edges,coordinates)
    T = compute_incident_vertices(edges,coordinates)

    solution_edges = []
    solution_edges, degrees = improve(edges, degrees, d, incident, edge_coords, T, coordinates, distances, min_improvement, upper_cap)

    return solution_edges, degrees

def improve(edges: List[List[Edge]], degrees, d, incident, edge_coords, T, coordinates, distance, min_improvement, upper_cap):
    while True:
        edges_copy = edges[:]
        was_improved=False
        a = False
        for e in chain.from_iterable(edges_copy):
            if e.v != 0:
                was_improved, edges, degrees, d, incident, edge_coords, T= improve_edge(e, edges, degrees, d, incident, edge_coords, T, coordinates, distance, min_improvement, upper_cap)
                if was_improved:
                    a = True
                    break
        if not a:
            return edges, degrees


def improve_edge(e, edges, degrees, d, incident, edge_coords, T, coordinates, distance, min_improvement, upper_cap):
    for new_cost,t in get_candidates(edges, d, incident, T, coordinates, distance, min_improvement, upper_cap):
        tmp_coords = remove_coords(t.remove, edge_coords, coordinates) # remove t.remove temporarily
        if not is_edge_intersecting_single(t.add, tmp_coords, coordinates):
            for f in t.remove:
                remove_edge(f, edges, degrees, d, incident, edge_coords, T, coordinates)
            for e, partition_index in zip(t.add, t.indices):
                add_edge(e, partition_index, edges, degrees, d, incident, edge_coords, T, coordinates)
            return True, edges, degrees, d, incident, edge_coords, T

    # could not improve e
    return False, edges, degrees, d, incident, edge_coords, T


def obtain_initial_partition(edges: List[List[Edge]]) -> Dict[Edge,int]:
    d = {}
    for i,partition_edges in enumerate(edges):
        for e in partition_edges:
            if e.v == 0 or e.w == 0:
                d[e] = -1 # ignore starting edges from solar tower
            else:
                d[e] = i
    return d


def compute_incident_edges(edges: List[List[Edge]], coordinates: np.ndarray) -> Dict[int,List[Edge]]:
    incident = {i:[] for i in range(coordinates.shape[0])}
    for e in chain.from_iterable(edges):
        if e.v == 0 or e.w == 0:
            continue
        incident[e.v].append(e)
        incident[e.w].append(e)
    return incident


def compute_incident_vertices(edges: List[List[Edge]], coordinates: np.ndarray) -> Dict[int,List[Edge]]:
    incident = {i:[] for i in range(coordinates.shape[0])}
    for e in chain.from_iterable(edges):
        if e.v == 0 or e.w == 0:
            continue
        incident[e.v].append(e.w)
        incident[e.w].append(e.v)
    return incident


def partition_size(edge: Edge, d: Dict[int,int], coordinates) -> int:
    if edge in d.keys():
        return sum([d[edge] == j for j in d.values()])
    else:
        return coordinates.shape[0]+1


def compute_triangulation_edges(e: Edge, vertex: int, incident, d, T) -> TriangulationEdges:
    add = [Edge(e.v, vertex), Edge(e.w, vertex)]
    remove = [e]
    indices = [d[e]] * 2

    remove.extend(incident[vertex])
    if len(T[vertex]) == 2:
        add.append(Edge(T[vertex][0], T[vertex][1]))
        indices.append(d[incident[vertex][0]]) # always exists!
    return TriangulationEdges(add, remove, indices)


def cost_triangulation(te: TriangulationEdges, distances: np.ndarray):
    cost = sum([distances[a.v][a.w] for a in te.add])
    cost -= sum([distances[r.v][r.w] for r in te.remove])
    return cost


def get_candidates(edges: List[List[Edge]], d, incident, T, coordinates, distances, min_improvement, upper_cap) -> Tuple[float,TriangulationEdges]:
    result = []
    for e in chain.from_iterable(edges):
        if partition_size(e, d, coordinates) < upper_cap:
            for h in range(1,coordinates.shape[0]):
                if h != e.v and h != e.w and 0 != e.v and 0 != e.w:
                    te = compute_triangulation_edges(e, h, incident, d, T)
                    cost = cost_triangulation(te, distances)
                    if cost <= min_improvement: ## THRESHOLD of cost change (at least 1 euro)
                        result.append((cost, te))
    return sorted(result, key=lambda x: x[0])


def get_coords(e: Edge, coordinates):
    return np.array([[coordinates.iloc[e.v][0], coordinates.iloc[e.v][1],
                      coordinates.iloc[e.w][0], coordinates.iloc[e.w][1]]])


def add_edge(e: Edge, partition_index: int, edges, degrees, d, incident, edge_coords, T, coordinates):
    edges[partition_index].append(e)
    for v in e:
        degrees[v] += 1
    d[e] = partition_index
    incident[e.v].append(e)
    incident[e.w].append(e)
    T[e.v].append(e.w)
    T[e.w].append(e.v)

    edge_coords = np.vstack((edge_coords, get_coords(e, coordinates)))


def remove_edge(e: Edge, edges, degrees, d, incident, edge_coords, T, coordinates):
    partition_index = d[e]
    edges[partition_index].remove(e)
    for v in e:
        degrees[v] -= 1
    del d[e]
    incident[e.v].remove(e)
    incident[e.w].remove(e)
    T[e.v].remove(e.w)
    T[e.w].remove(e.v)

    index = np.where(np.all(get_coords(e, coordinates) == edge_coords, axis=1))
    edge_coords = np.delete(edge_coords, index, axis=0)

def is_edge_intersecting_single(edge_list: List[Edge], edge_coords: np.ndarray, coordinates):
    for e in edge_list:
        # add other edges
        tmp_coords = edge_coords.copy()
        for f in [g for g in edge_list if g != e]:
            tmp_coords = np.vstack((tmp_coords, get_coords(f, coordinates)))

        e_coords = get_coords(e, coordinates)
        if is_edge_intersecting(e_coords, tmp_coords):
            return True
    return False

def remove_coords(edge_list: List[Edge], edge_coords: np.ndarray, coordinates):
    res = edge_coords.copy()
    for e in edge_list:
        index = np.where(np.all(get_coords(e,coordinates) == res, axis=1))
        res = np.delete(res, index, axis=0)
    return res
