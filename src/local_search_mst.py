import sys
from itertools import product
from typing import List, Tuple, Set

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

from common import Edge, compute_partitions, CABLE_COSTS, CABLE_LENGTHS, CABLE_CAPACITIES, total_cost
from plotting import plot_solution
from local_search_upgrade_downgrade import LocalSearchUpgradeDowngrade


def run_local_search_mst(coordinates, cable_cost, partitions):
    mst = MSTPrim(coordinates, cable_cost)
    mst.compute(partitions)

    ls = LocalSearchUpgradeDowngrade()
    print('Before local search. Total cost:'.format())
    print('Degree statistic:'.format())

    ls.perform_local_search()

    print('After local search. Total cost: {}'.format(total_cost(ls.edges, ls.degrees, cable_cost, ls.cable_assignment)))



class MSTPrim:
    def __init__(self, coordinates: np.ndarray, cable_cost: int) -> None:
        self.coordinates = coordinates
        self.cable_cost = cable_cost
        self.edge_costs = cdist(self.coordinates, self.coordinates) * self.cable_cost + 100 # cable and conductor costs
        self.degrees = {i:0 for i in range(self.coordinates.shape[0])}
        self.successor = {i:[] for i in range(self.coordinates.shape[0])}
        self.predecessor = {i:[] for i in range(self.coordinates.shape[0])}

        self.edges = [] # type: List[List[Edge]]
        self.edge_coords = []

    def compute(self, partitions: int) -> None:
        self.partitions = compute_partitions(coordinates=self.coordinates, partitions=partitions)
        for p in self.partitions:
            self.edges.append(self._compute_mst(p))

    def _compute_mst(self, partition_indices: np.ndarray) -> List[Edge]:
        """
        Computes for a partition of indices a valid solution regarding planarity.
        TODO: extensive documentation
        """
        # TODO: introduce Current namedTuple for these self-variables
        # initialization
        self.visited_heliostats = set([0])
        self.unvisited_heliostats = {int(i) for i in partition_indices}

        partition_edges = [] # forming a MST
        partition_edges_coords = np.empty((0,4))

        while len(partition_edges) < partition_indices.shape[0]:
            # select min cost edge
            edges = self._next_cut_edges()
            if not edges:
                raise RuntimeError('empty edge candidates list. WEIRD ERROR')

            # TODO: looks ugly - refactoring needed
            updated = False
            for e in edges:
                e_coords = np.array([[self.coordinates[e.v][0],
                                     self.coordinates[e.v][1],
                                     self.coordinates[e.w][0],
                                     self.coordinates[e.w][1]]])
                if not is_edge_intersecting(e_coords, [partition_edges_coords]):
                    partition_edges.append(e)
                    partition_edges_coords = np.vstack((partition_edges_coords, e_coords))
                    self.successor[e.v].append(e.w)
                    self.predecessor[e.w].append(e.v)

                    self._update(e)
                    updated = True
                    break
            if not updated:
                raise RuntimeError('SHOULD NOT HAPPEN')
        self.edge_coords.append(partition_edges_coords)
        return partition_edges


    def _next_cut_edges(self) -> List[Edge]:
        edges = [Edge(*t) for t in product(self.visited_heliostats, self.unvisited_heliostats)]
        costs = [self.edge_costs[e.v][e.w] for e in edges]
        return [x for _,x in sorted(zip(costs,edges))]

    def _update(self, edge: Edge) -> None:
        for vertex in edge:
            self.degrees[vertex] += 1

            for h in self.unvisited_heliostats:
                if self.degrees[vertex] == 2 or self.degrees[vertex] == 9:
                    self.edge_costs[vertex][h] += 700
                    self.edge_costs[h][vertex] += 700

                if self.degrees[vertex] == 3 or self.degrees[vertex] == 10:
                    self.edge_costs[vertex][h] -= 700
                    self.edge_costs[h][vertex] -= 700

            self.visited_heliostats.add(vertex)
            if vertex in self.unvisited_heliostats:
                self.unvisited_heliostats.remove(vertex)
