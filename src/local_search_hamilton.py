#!/usr/bin/env python3
# UTF-8 encoding
import copy
from itertools import chain
from pickle import dump, HIGHEST_PROTOCOL
from typing import Dict, List, Tuple, NamedTuple

import numpy as np
from scipy.spatial.distance import cdist

from initial_hamilton import Hamilton
from common import Edge, solution_value, cable_length
from intersection import is_edge_intersecting
from plotting import plot_solution
from solutions import DataCableSolution


TriangulationEdges = NamedTuple('TriangulationEdges', [('add', List[Edge]), ('remove', List[Edge]), ('indices', List[int])])


def run_local_search_hamilton(coordinates: np.ndarray,
                              edge_cost: float,
                              partitions: int,
                              distances: np.ndarray,
                              initial_graph_output: str,
                              initial_pkl_output: str,
                              graph_output: str,
                              pkl_output: str,
                              min_improvement: float,
                              upper_cap: float):
    # obtain initial Hamilton solution
    ham = Hamilton(coordinates=coordinates, cable_cost=edge_cost, partitions=partitions)
    ham.compute()

    # plot initial solution and save it via pickle
    init_sol_cost = solution_value(edge_cost=edge_cost, edges=ham.solution.edges, degrees=ham.solution.degrees, distances=distances)
    assert init_sol_cost == ham.solution.cost(distances)

    if initial_graph_output != '':
        plot_solution(coordinates=coordinates,
                      output=initial_graph_output,
                      edges=ham.solution.edges,
                      value=init_sol_cost,
                      edge_cost=edge_cost,
                      partitions=partitions)
    if initial_pkl_output != '':
        with open(initial_pkl_output, 'wb') as init_pkl:
            dump(obj=ham.solution, file=init_pkl, protocol=HIGHEST_PROTOCOL)
    print("Initial solution cost: {}".format(init_sol_cost))
    print("Cable length: {}".format(cable_length(ham.solution.edges, cdist(coordinates, coordinates))))

    # perform local search onto initial Hamilton solution
    solution_edges, degrees = perform_local_search_hamilton(ham.solution, coordinates, min_improvement, upper_cap)

    # plot local optimal solution and save it with pickle
    sol_cost = solution_value(edge_cost=edge_cost, edges=solution_edges, degrees=degrees, distances=distances)

    if graph_output != '':
        plot_solution(coordinates=coordinates,
                      output=graph_output,
                      edges=solution_edges,
                      value=sol_cost,
                      edge_cost=edge_cost,
                      partitions=partitions)
    if pkl_output != '':
        with open(pkl_output, 'wb') as pkl:
            dump(obj=[solution_edges, degrees], file=pkl, protocol=HIGHEST_PROTOCOL)
    print("After local search, solution cost: {}".format(sol_cost))
    print("After local search, cable length: {}".format(cable_length(solution_edges, cdist(coordinates, coordinates))))
    print("Local search improved cost by {}".format(init_sol_cost-sol_cost))


def perform_local_search_hamilton(initial_solution: DataCableSolution,
                                  coords: np.ndarray,
                                  min_improvement: float,
                                  upper_cap: float) -> Tuple[List[Edge], Dict[int, int]]:
    """

    :param initial_solution: initial hamilton solution
    :param coords: coordinates of heliostats and solar tower
    :param min_improvement: minimum threshold for performing an improvement step during local search.
    :param upper_cap: upper capacity of heliostats within one network
    :return: edges and their respective degrees that denote a local optimum with respect to the local search procedure
    """
    # perform local search onto initial Hamilton solution
    degrees = copy.deepcopy(initial_solution.degrees)
    edges = copy.deepcopy(initial_solution.edges)
    edge_coords = copy.deepcopy(np.vstack(initial_solution.edge_coordinates))
    d = obtain_initial_partition(edges)
    coordinates = coords[:]
    incident = compute_incident_edges(edges,coordinates)
    T = compute_incident_vertices(edges,coordinates)

    distances = cdist(coordinates, coordinates)

    solution_edges = []
    solution_edges, degrees = improve(edges, degrees, d, incident, edge_coords, T, coordinates, distances, min_improvement, upper_cap)

    return solution_edges, degrees


def improve(edges: List[List[Edge]],
            degrees,
            d,
            incident,
            edge_coords,
            T,
            coordinates,
            distance,
            min_improvement,
            upper_cap):
    """
    Performs local search onto an initial solution.

    :param edges: contains list of edges for each partition
    :param degrees: degrees of vertices
    :param d:
    :param incident:
    :param edge_coords: edge coordinates
    :param T:
    :param coordinates: coordinates of vertices
    :param distance: euclidean distance matrix
    :param min_improvement: minimum improvement in cost per step
    :param upper_cap: upper capacity of heliostats per partition
    :return:
    """
    while True:
        edges_copy = edges[:]
        a = False
        for e in chain.from_iterable(edges_copy):
            if e.v != 0:
                was_improved, edges, degrees, d, incident, edge_coords, T = improve_edge(e, edges, degrees, d,
                                                                                         incident, edge_coords,
                                                                                         T, coordinates, distance,
                                                                                         min_improvement, upper_cap)
                if was_improved:
                    a = True
                    break
        if not a:
            return edges, degrees


def improve_edge(e, edges, degrees, d, incident, edge_coords, T, coordinates, distance, min_improvement, upper_cap):
    for new_cost, t in get_candidates(edges, d, incident, T, coordinates, distance, min_improvement, upper_cap):
        tmp_coords = remove_coords(t.remove, edge_coords, coordinates)  # remove t.remove temporarily
        if not is_edge_intersecting_single(t.add, tmp_coords, coordinates):
            for f in t.remove:
                remove_edge(f, edges, degrees, d, incident, edge_coords, T, coordinates)
            for e, partition_index in zip(t.add, t.indices):
                add_edge(e, partition_index, edges, degrees, d, incident, edge_coords, T, coordinates)
            return True, edges, degrees, d, incident, edge_coords, T

    # could not improve e
    return False, edges, degrees, d, incident, edge_coords, T


def obtain_initial_partition(edges: List[List[Edge]]) -> Dict[Edge, int]:
    d = {}
    for i, partition_edges in enumerate(edges):
        for e in partition_edges:
            if e.v == 0 or e.w == 0:
                d[e] = -1  # ignore starting edges from solar tower
            else:
                d[e] = i
    return d


def compute_incident_edges(edges: List[List[Edge]], coordinates: np.ndarray) -> Dict[int, List[Edge]]:
    incident = {i: [] for i in range(coordinates.shape[0])}
    for e in chain.from_iterable(edges):
        if e.v == 0 or e.w == 0:
            continue
        incident[e.v].append(e)
        incident[e.w].append(e)
    return incident


def compute_incident_vertices(edges: List[List[Edge]], coordinates: np.ndarray) -> Dict[int, List[Edge]]:
    incident = {i: [] for i in range(coordinates.shape[0])}
    for e in chain.from_iterable(edges):
        if e.v == 0 or e.w == 0:
            continue
        incident[e.v].append(e.w)
        incident[e.w].append(e.v)
    return incident


def partition_size(edge: Edge, d: Dict[int, int], coordinates: np.ndarray) -> int:
    if edge in d.keys():
        return sum([d[edge] == j for j in d.values()])
    else:
        return coordinates.shape[0] + 1


def compute_triangulation_edges(e: Edge,
                                vertex: int,
                                incident: Dict[int, List[Edge]],
                                d: Dict[Edge, int],
                                T: Dict[int, List[Edge]]) -> TriangulationEdges:
    add = [Edge(e.v, vertex), Edge(e.w, vertex)]
    remove = [e]
    indices = [d[e]] * 2

    remove.extend(incident[vertex])
    if len(T[vertex]) == 2:
        add.append(Edge(T[vertex][0], T[vertex][1]))
        indices.append(d[incident[vertex][0]])  # always exists!
    return TriangulationEdges(add, remove, indices)


def cost_triangulation(te: TriangulationEdges, distances: np.ndarray):
    cost = sum([distances[a.v][a.w] for a in te.add])
    cost -= sum([distances[r.v][r.w] for r in te.remove])
    return cost


def get_candidates(edges: List[List[Edge]],
                   d: Dict[Edge, int],
                   incident: Dict[int, List[Edge]],
                   T: Dict[int, List[Edge]],
                   coordinates: np.ndarray,
                   distances: np.ndarray,
                   min_improvement: float,
                   upper_cap: int) -> List[Tuple[float, TriangulationEdges]]:
    result = []
    for e in chain.from_iterable(edges):
        if partition_size(e, d, coordinates) < upper_cap:
            for h in range(1,coordinates.shape[0]):
                if h != e.v and h != e.w and 0 != e.v and 0 != e.w:
                    te = compute_triangulation_edges(e, h, incident, d, T)
                    cost = cost_triangulation(te, distances)
                    if cost <= min_improvement:  # THRESHOLD of cost change
                        result.append((cost, te))
    return sorted(result, key=lambda x: x[0])


def get_coords(e: Edge, coordinates: np.ndarray) -> np.ndarray:
    return np.array([[coordinates[e.v][0], coordinates[e.v][1],
                      coordinates[e.w][0], coordinates[e.w][1]]])


def add_edge(e: Edge,
             partition_index: int,
             edges: List[List[Edge]],
             degrees: Dict[int, int],
             d: Dict[Edge, int],
             incident: Dict[int, List[Edge]],
             edge_coords: np.ndarray,
             T: Dict[int, List[Edge]],
             coordinates) -> None:
    edges[partition_index].append(e)
    for v in e:
        degrees[v] += 1
    d[e] = partition_index
    incident[e.v].append(e)
    incident[e.w].append(e)
    T[e.v].append(e.w)
    T[e.w].append(e.v)

    edge_coords = np.vstack((edge_coords, get_coords(e, coordinates)))


def remove_edge(e: Edge,
                edges: List[List[Edge]],
                degrees: Dict[int, int],
                d: Dict[Edge, int],
                incident: Dict[int, List[Edge]],
                edge_coords,
                T: Dict[int, List[Edge]],
                coordinates: np.ndarray):
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


def is_edge_intersecting_single(edge_list: List[Edge], edge_coords: np.ndarray, coordinates: np.ndarray) -> bool:
    for e in edge_list:
        # add other edges
        tmp_coords = edge_coords.copy()
        for f in [g for g in edge_list if g != e]:
            tmp_coords = np.vstack((tmp_coords, get_coords(f, coordinates)))

        e_coords = get_coords(e, coordinates)
        if is_edge_intersecting(e_coords, tmp_coords):
            return True
    return False


def remove_coords(edge_list: List[Edge], edge_coords: np.ndarray, coordinates) -> np.ndarray:
    res = edge_coords.copy()
    for e in edge_list:
        index = np.where(np.all(get_coords(e,coordinates) == res, axis=1))
        res = np.delete(res, index, axis=0)
    return res
