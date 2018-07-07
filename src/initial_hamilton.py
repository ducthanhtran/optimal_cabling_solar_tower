#!/usr/bin/env python3
# UTF-8 encoding
from typing import Dict, List, Set, NamedTuple

import numpy as np
from recordclass import RecordClass
from scipy.spatial.distance import cdist

from common import compute_partitions, Edge
from solutions import DataCableSolution


# TODO: document these NamedTuples
Candidate = RecordClass('Candidate', [('cost', float),
                                      ('edges', List[Edge]),
                                      ('vertex', int),
                                      ('index', int)])
HamiltonState = RecordClass('HamiltonState', [('permutation', List[int]),
                                              ('unvisited', Set[int])])


def shift_left(L: List[int]):
    """
    Works faster than np.roll(L, -1)
    """
    return L[1:] + L[:1]


class Hamilton:
    """
    We compute Hamiltonian paths for each partition that are initial solutions for the
    data cabling.
    """
    __slots__ = ('coordinates', 'cable_cost', 'edge_costs', 'partitions', 'solution', 'current_state')

    def __init__(self, coordinates: np.ndarray, cable_cost: float, partitions: int) -> None:
        """
        :param coordinates: coordinates vertices
        :param cable: cost for hamiltonian path
        :param partitions: number of partitions
        """
        self.coordinates = coordinates
        self.cable_cost = cable_cost
        self.edge_costs = cdist(self.coordinates, self.coordinates) * cable_cost # only consider glass fiber cables!
        self.partitions = partitions
        self.solution = DataCableSolution(self.coordinates.shape[0], partitions)
        self.current_state = HamiltonState(None, None)

    def compute(self) -> None:
        """
        Compute Hamiltonian paths for each partition.
        """
        for i, p in enumerate(compute_partitions(coordinates=self.coordinates, partitions=self.partitions)):
            self._compute_hamilton(i, p)

    def _compute_hamilton(self, partition: int, partition_indices: np.ndarray) -> None:
        """
        Computes for a partition of indices a valid solution regarding planarity.

        :param partition_indices: indices of heliostats that belong together in a partition
        :return: edges and edge_coords that form a Hamiltonian path
        """
        self.current_state._replace(permutation=[0])
        self.current_state._replace(unvisited=set(partition_indices.astype(int).flat))

        while len(self.solution.edges[partition]) < partition_indices.shape[0]:
            best_candidate = min(self._compute_candidates())
            if len(best_candidate.edges) == 2:
                self._insert_between(best_candidate, partition)
            else:
                self._insert_last(best_candidate, partition)

    def _compute_candidates(self) -> Candidate:
        """
        Compute cost and edges for a vertex and index position of permutation. In other words
        for a vertex v we compute its minimum cost of insertion into the current permutation
        of our Hamiltonian path. This can either be an insertion and appending to the permutation.

        :param vertex: vertex to be added to permutation
        :param index: index of insertion/appending
        """
        for vertex in self.current_state.unvisited:
            shifted_perm = shift_left(self.current_state.permutation)
            unvisited = [vertex]*len(self.current_state.permutation)

            costs_edge_two = self.edge_costs[ unvisited, shifted_perm]
            costs_edge_two[-1] = 0
            adding_costs = self.edge_costs[ unvisited, self.current_state.permutation] + costs_edge_two

            cost_permutation_edges = self.edge_costs[self.current_state.permutation, shifted_perm]
            cost_permutation_edges[-1] = 0
            index = np.argmin(adding_costs - cost_permutation_edges)

            if index < len(self.current_state.permutation)-1:
                e1 = Edge(self.current_state.permutation[index], vertex)
                e2 = Edge(vertex, self.current_state.permutation[index+1])
                d = Edge(self.current_state.permutation[index], self.current_state.permutation[index+1])

                cost = self.edge_costs[e1.v, e1.w] + self.edge_costs[e2.v, e2.w] - self.edge_costs[d.v, d.w]
                yield Candidate(cost=cost, edges=[e1,e2], vertex=vertex, index=index)
            else:
                e = Edge(self.current_state.permutation[index], vertex)
                yield Candidate(cost=self.edge_costs[e.v, e.w], edges=[e], vertex=vertex, index=index)

    def _insert_edge(self, edge: Edge, partition: int) -> None:
        """
        Adds edge to solution partition
        """
        self.solution.add_edge(edge=edge,
                               edge_cost=self.cable_cost,
                               partition=partition,
                               coordinates=self.coordinates)
        # update current unvisited heliostats set
        for vertex in edge:
            if vertex in self.current_state.unvisited:
                self.current_state.unvisited.remove(vertex)

    def _insert_between(self, candidate: Candidate, partition: int) -> None:
        # remove old edge
        index = candidate.index
        old_edge = Edge(self.current_state.permutation[index], self.current_state.permutation[index+1])
        self.solution.remove_edge(old_edge, partition, self.coordinates)
        # add new edges
        self._insert_edge(candidate.edges[0], partition)
        self._insert_edge(candidate.edges[1], partition)
        # update current permutation
        self.current_state.permutation.insert(candidate.index+1, candidate.vertex)

    def _insert_last(self, candidate: Candidate, partition: int) -> None:
        self._insert_edge(candidate.edges[0], partition)
        self.current_state.permutation.append(candidate.vertex)
