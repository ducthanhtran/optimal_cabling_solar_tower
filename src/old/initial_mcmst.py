from typing import List
from scipy.spatial.distance import cdist

import numpy as np

from common import Edge, compute_partitionshttps://scholar.google.com/schhp?hl=de&as_sdt=0,5
from plotting import plot_solution_data
from solutions import EdgeSolution
from mcmst_state import MCMSTState


CABLE_COSTS = np.array([0.58,0.87,1.24,1.95,3.13,5.19,6.9])
CABLE_LENGTHS = np.array([38.36,47.08,56.04,69.3,84.87,102.79,120.31])
CABLE_CAPACITIES = np.array([56,73,92,124,162,209,250])


class MCMST:
    # __slots__ = ('coordinates', 'edge_costs', 'partitions', 'solution', 'current_state')

    def __init__(self,
                 coordinates: np.ndarray,
                 partitions: int,
                 working_cost: float,
                 data_cables: List[List[Edge]],
                 cable_costs: np.ndarray,
                 cable_lengths: np.ndarray,
                 cable_capacities: np.ndarray):
        """
        todoc
        !assume cable_cost is sorted in ascending order!
        """
        self.coordinates = coordinates
        self.partitions = partitions
        self.working_cost = working_cost
        self.data_cables = data_cables
        # NOTE: hard-coded for fast and simple development
        self.cable_costs = CABLE_COSTS
        self.cable_lengths = CABLE_LENGTHS
        self.cable_capacities = CABLE_CAPACITIES

        self.solution = EdgeSolution(partitions)

        self.distance = cdist(coordinates,coordinates)

    def compute(self) -> None:
        partitions = compute_partitions(coordinates=self.coordinates, partitions=self.partitions)
        for index, partition in enumerate(partitions):
            partitions_int = partition.astype(int)
            for i in partitions_int:
                self.solution.add_edge(Edge(0,i), self.cable_costs[-1], index, self.coordinates)
            self.mcmst_state = MCMSTState(self.cable_costs,
                                          self.cable_lengths,
                                          self.cable_capacities,
                                          self.coordinates,
                                          self.working_cost,
                                          self.data_cables,
                                          partitions_int)
            self._mcmst(index, partition)

    def _mcmst(self, partition: int, partition_indices) -> None:
        # i = 0
        for cable_index in reversed(range(7)):
            # print("Cable: {}".format(cable_index))
            while(True):
                next_candidate = self.mcmst_state.next_candidate(cable_index, self.solution)

                # if no candidate is profitable, go to next cable
                if next_candidate is None:
                    break

                # can improve with current cable index
                self._update(next_candidate, partition)
                # plot_solution_data(self.coordinates,
                                   # '/home/duc/a_{}'.format(i),
                                   # self.solution.edges,
                                   # 0,0,0)
                # i += 1

    def _update(self, candidate, partition):
        # print('\nUPDATE')
        # update solution state
        for c in candidate.new_children:
            # print("ADD {}->{}".format(candidate.parent, c))
            self.solution.remove_edge(Edge(self.mcmst_state.predecessor[c], c), partition, self.coordinates)
            self.solution.add_edge(Edge(candidate.parent, c), self.mcmst_state.current.cable_cost[c], partition, self.coordinates)

        self.solution.edge_costs[candidate.parent] = self.cable_costs[candidate.new_cable_index]
        self.mcmst_state.update(candidate) # update internal mcmst-state
