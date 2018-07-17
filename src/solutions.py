#!/usr/bin/env python3
# UTF-8 encoding
from itertools import chain
from typing import Dict, List

from .common import Edge, switch_cost
from .intersection import is_edge_intersecting

import numpy as np


class EdgeSolution:
    def __init__(self, partitions: int) -> None:
        self.edges = [[] for _ in range(partitions)]  # type: List[List[Edge]]
        self.edge_coordinates = [np.empty((0, 4)) for _ in range(partitions)]
        self.edge_costs = {}  # type: Dict[Edge,float]

    def add_edge(self, edge: Edge, edge_cost: float, partition: int, coordinates: np.ndarray):
        self.edges[partition].append(edge)
        self.edge_costs[edge] = edge_cost

        edges_coordinates = np.array([[coordinates[edge.v][0], coordinates[edge.v][1],
                                       coordinates[edge.w][0], coordinates[edge.w][1]]])
        self.edge_coordinates[partition] = np.vstack((self.edge_coordinates[partition], edges_coordinates))

    def remove_edge(self, edge: Edge, partition: int, coordinates: np.ndarray):
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

    def cost(self, distances: np.ndarray):
        return sum(distances[e.v, e.w]*self.edge_costs[e] for e in chain.from_iterable(self.edges))


class DataCableSolution(EdgeSolution):
    def __init__(self, n: int, partitions: int):
        super().__init__(partitions)
        self.degrees = {i:0 for i in range(n)}  # type: Dict[int,int]

    def add_edge(self, edge: Edge, edge_cost: float, partition: int, coordinates: np.ndarray):
        super().add_edge(edge, edge_cost, partition, coordinates)
        self.degrees[edge.v] += 1
        self.degrees[edge.w] += 1

    def remove_edge(self, edge: Edge, partition: int, coordinates: np.ndarray):
        super().remove_edge(edge, partition, coordinates)
        self.degrees[edge.v] -= 1
        self.degrees[edge.w] -= 1

    def cost(self, distances: np.ndarray):
        cost = super().cost(distances)
        return cost + sum(switch_cost(d) for d in self.degrees.values()) - switch_cost(self.degrees[0])
