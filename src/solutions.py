#!/usr/bin/env python3
# UTF-8 encoding
from typing import Generator

from .common import Edge
from .intersection import is_edge_intersecting

import numpy as np


class EdgeSolution:
    def __init__(self, partitions: int) -> None:
        """
        :param partitions: number of edge partitions
        """
        self.edges= [[] for _ in range(partitions)] # type: List[List[Edge]]
        self.edges_coords = [np.empty((0,4)) for _ in range(partitions)]
        self.edge_costs = {} # type: Dict[Edge,int]

    def add_edge(self, edge: Edge, edge_cost: float, partition: int, coordinates: np.ndarray):
        self.edges[partition].append(edge)
        self.edge_costs[edge] = edge_cost

        edges_coords = np.array([[coordinates[edge.v][0], coordinates[edge.v][1],
                                 coordinates[edge.w][0], coordinates[edge.w][1]]])
        self.edges_coords[partition] = np.vstack((self.edges_coords[partition], edges_coords))

    def remove_edge(self, edge: Edge, partition: int, coordinates: np.ndarray):
        self.edges.remove(edge)
        del self.edge_costs[edge]

        edges_coords = np.array([[coordinates[edge.v][0], coordinates[edge.v][1],
                                 coordinates[edge.w][0], coordinates[edge.w][1]]])
        delete_index = np.where(np.all(edges_coords == self.edges_coords[partition], axis=1))
        self.edges_coords[partition] = np.delete(self.edges_coords[partition], delete_index, axis=0)

    def intersects(self, edges_coords: np.ndarray) -> bool:
        """
        Checks whether a given edge that is represented by its coordinates is intersecting
        with our solution edges.

        :param edges_coords: a numpy array that holds both x- and y-coordinates
        :returns: True if edges_coords intersects with solution edges
        """
        return is_edge_intersecting(edges_coords, self.edges_coords)

    def cost(self, distances: np.ndarray):
        return sum(distances[e.v,e.w]*self.edge_costs[e] for e in chain.from_iterable(self.edges))


class DataCableSolution(EdgeSolution):
    def __init__(self, n: int, partitions: int):
        super().__init__(partitions)
        self.degrees = {i:0 for i in range(n)} # type: Dict[int,int]

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
        # subtract switch costs of solar tower at the end
        # NOTE: make it more readable or omit switch cost of solar tower immediately
        return cost + sum(self._switch_cost(d) for d in self.degrees.values()) - self._switch_cost(self.degrees[0])

    def _switch_cost(self, degree: int) -> float:
        """Note: Hard-coded values for quick and dirty development"""
        if degree <= 2:
            return 100
        elif degree >= 3 and degree <= 9:
            return 800
        elif degree >= 10 and degree <= 17:
            return 1500
        else:
            raise ValueError('Degree is too high. No cost found.')


class PowerCableSolution(EdgeSolution):
    def __init__(self, n: int, partitions: int):
        super().__init__(partitions)
        self.predecessor = {} # type: Dict[int,int]
        self.successors = {i:[] for i in range(n)} # type: Dict[int,List[int]]
        self.connected_children = {i:0 for i in range(n)} # type: Dict[int,int]
        self.heliostat_parents = {i:[] for i in range(n)} # type: Dict[int,List[int]]

    def add_edge(self, edge: Edge, edge_cost: float, partition: int, coordinates: np.ndarray):
        super().add_edge(edge, edge_cost, partition, coordinates)
        if not self.predecessor[edge.w]:
            self.predecessor[edge.w] = edge.v
        else:
            raise ValueError('{} has already a predecessor connection to {}'.format(edge.v, self.predecessors[edge.v]))
        self.successors[edge.v].append(edge.w)

    def remove_edge(self, edge: Edge, partition: int, coordinates: np.ndarray):
        super().remove_edge(edge, partition, coordinates)
        del self.predecessor[edge.w]
        self.successors[edge.v].remove(edge.w)

    def cost(self, distances: np.ndarray):
        cost = super().cost(distances)

    def heliostat_parents(self, vertex: int) -> Generator[int,None,None]:
        pred = self.predecessor[vertex]
        while pred != 0:
            yield pred
            pred = self.predecessors[pred]
