#!/usr/bin/env python3
# UTF-8 encoding
from itertools import chain
from typing import Dict, Generator, List

from common import Edge, switch_cost
from intersection import is_edge_intersecting

import numpy as np


class EdgeSolution:
    def __init__(self, partitions: int) -> None:
        self.edges = [[] for _ in range(partitions)]  # type: List[List[Edge]]
        self.edge_coordinates = [np.empty((0, 4)) for _ in range(partitions)]
        self.edge_costs = {}  # type: Dict[Edge,float]

    def add_edge(self, edge: Edge, edge_cost: float, partition: int, coordinates: np.ndarray) -> None:
        self.edges[partition].append(edge)
        self.edge_costs[edge] = edge_cost

        edges_coordinates = np.array([[coordinates[edge.v][0], coordinates[edge.v][1],
                                       coordinates[edge.w][0], coordinates[edge.w][1]]])
        self.edge_coordinates[partition] = np.vstack((self.edge_coordinates[partition], edges_coordinates))

    def remove_edge(self, edge: Edge, partition: int, coordinates: np.ndarray) -> None:
        self.edges[partition].remove(edge)
        del self.edge_costs[edge]

        edge_coordinates = np.array([[coordinates[edge.v][0], coordinates[edge.v][1],
                                      coordinates[edge.w][0], coordinates[edge.w][1]]])
        delete_index = np.where(np.all(edge_coordinates == self.edge_coordinates[partition], axis=1))
        self.edge_coordinates[partition] = np.delete(self.edge_coordinates[partition], delete_index, axis=0)

    def intersects(self, edge_coordinates: np.ndarray) -> bool:
        """
        Checks whether a given edge that is represented by its coordinates is intersecting
        with our solution edges.

        :param edge_coordinates: a numpy array that holds both x- and y-coordinates
        :returns: True if edge_coordinates intersects with solution edges
        """
        return is_edge_intersecting(edge_coordinates, self.edge_coordinates)

    def cost(self, distances: np.ndarray) -> float:
        return sum(distances[e.v, e.w]*self.edge_costs[e] for e in chain.from_iterable(self.edges))


class DataCableSolution(EdgeSolution):
    def __init__(self, n: int, partitions: int) -> None:
        super().__init__(partitions)
        self.degrees = {i: 0 for i in range(n)}  # type: Dict[int,int]

    def add_edge(self, edge: Edge, edge_cost: float, partition: int, coordinates: np.ndarray) -> None:
        super().add_edge(edge, edge_cost, partition, coordinates)
        self.degrees[edge.v] += 1
        self.degrees[edge.w] += 1

    def remove_edge(self, edge: Edge, partition: int, coordinates: np.ndarray) -> None:
        super().remove_edge(edge, partition, coordinates)
        self.degrees[edge.v] -= 1
        self.degrees[edge.w] -= 1

    def cost(self, distances: np.ndarray) -> float:
        cost = super().cost(distances)
        return cost + sum(switch_cost(d) for d in self.degrees.values()) - switch_cost(self.degrees[0])


class PowerCableSolution(EdgeSolution):
    def __init__(self, n: int,
                 partitions: int,
                 highest_cable_cost: float,
                 highest_cable_length: float,
                 highest_cable_apacity: float) -> None:
        """
        :param n: number of vertices
        :param partitions: number of partitions
        """
        super().__init__(partitions)
        self.predecessor = np.zeros(n, dtype=int)
        self.predecessor[0] = -1

        self.successors = {i: [] for i in range(n)}  # type: Dict[int, List[int]]
        self.number_children_heliosats = {i: 0 for i in range(n)}  # type: Dict[int, int]
        self.parents = {i: [0] for i in range(n)}  # type: Dict[int, List[int]]

        self.current_cap = np.ones(n)
        self.current_cap[0] = n - 1

        self.current_max_cap = np.ones(n) * highest_cable_apacity

        # NOTE: needs refactoring
        self.current_cable_cost = np.ones(n) * highest_cable_cost
        self.current_cable_cost[0] = 0

        self.current_max_cable_length = np.ones(n) * highest_cable_length

    def add_edge(self, edge: Edge, edge_cost: float, partition: int, coordinates: np.ndarray) -> None:
        super().add_edge(edge, edge_cost, partition, coordinates)
        self.predecessor[edge.w] = edge.v
        self.successors[edge.v].append(edge.w)

        for parent in self.heliostat_parents(edge.w):
            self.number_children_heliosats[parent] += 1

    def remove_edge(self, edge: Edge, partition: int, coordinates: np.ndarray) -> None:
        super().remove_edge(edge, partition, coordinates)
        self.predecessor[edge.w] = -1
        self.successors[edge.v].remove(edge.w)

        for parent in self.heliostat_parents[edge.w]:
            self.number_children_heliosats[parent] -= 1

    def cost(self, distances: np.ndarray) -> float:
        return super().cost(distances)

    def heliostat_parents(self, vertex: int) -> Generator[int, None, None]:
        """
        Computes parent vertices up to the solar tower.

        :param vertex: starting vertex from where we obtain predecessor vertices.
        """
        predecessor = vertex
        while predecessor in self.predecessor:
            yield self.predecessor[predecessor]
            predecessor = self.predecessor[predecessor]
