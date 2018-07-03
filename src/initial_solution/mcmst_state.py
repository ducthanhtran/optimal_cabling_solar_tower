from typing import Dict, List, NamedTuple

import numpy as np
from scipy.spatial.distance import cdist


Candiate = NamedTuple('Candidate', [('parent',int),
                                     ('new_children', List[int]),
                                     ('new_cable_index', int)])

Current = NamedTuple('Current', [('capacity', int),
                                 ('max_capacity', np.ndarray),
                                 ('max_length', np.ndarray),
                                 ('cable_index_to_parent', int)])


class MCMSTState:
    def __init__(self, cable_costs: np.ndarray,
                       cable_lengths: np.ndarray,
                       cable_capacities: np.ndarray,
                       coordinates: np.ndarray,
                       indices: np.ndarray):
        """
        Indices specifies for which heliostats we want to compute the multi-level capacitated minimum spanning tree.

        :param cable_costs:
        :param cable_lengths:
        :param cable_capacities:
        :param coordinates:
        :param partition_indices:
        """
        self.cable_costs = cable_costs
        self.cable_lengths = cable_lengths
        self.cable_capacities = cable_capacities
        self.coordinates = coordinates
        self.indices = indices # only these heliostats are considered from self.coordinates

        self.distance = cdist(coordinates,coordinates)

        N = coordinates.shape[0]
        self.predecessor = np.zeros(N)
        self.successors = {i:[] for i in range(N)}

        self.number_connected_heliostats = np.zeros(N)
        self.heliostat_parents = {i:[0] for i in range(N)} # all parents on path up to the solar tower

        self.upgraded_vertices = [0]

        # initialize current-data-structure
        largest_cable_index = np.argmax(cable_capacities)
        max_capacity = np.ones(N) * cable_capacities[largest_cable_index]
        max_length = np.ones(N) * cable_lengths[largest_cable_index]
        self.current = Current(capacity=np.ones(N),
                               max_capacity=max_capacity,
                               max_length=max_length,
                               cable_index_to_parent=largest_cable_index)


    def __iter__(self):
        return self

    def __next__(self) -> Candiate:
        pass


    def _candidate_indices(self, parent: int, forbidden_candidates: List[int]) -> np.ndarray:
        """
        Computes for a given parent vertex possible children candidates, that is, these have benefits when
        reconnecting to the new parent vertex as children.

        :param parent: new parent for candidates
        :param forbidden_candidates: these should not be considered as candidates
        """
        candidates = np.arange(self.coordinates.shape[0])
        # candidates = np.setdiff1d(candidates, forbidden_candidates)

        # capacity constraint
        mask_capacity = self.current.capacity[candidates] < self._max_number_reconnections(parent)

        # cable length constraint
        mask_cable_length = self.distance[parent,candidates] <= self.current.max_length[candidates]

        final_mask = np.logical_and(mask_capacity, mask_cable_length)
        final_mask[0] = False # solar tower cannot be 'reconnected'
        final_mask[parent] = False # disallow loops
        final_mask[forbidden_candidates] = False

        return candidates[final_mask]

    def _max_number_reconnections(self, parent: int) -> int:
        """
        Returns maximum number of new children-connections to parent vertex.

        :param parent: parent vertex index
        """
        parents_of_parent = self.heliostat_parents[parent]
        return np.min(self.current.max_capacity[parents_of_parent] - self.current.capacity[parents_of_parent])

    # def _reconnecting_profit(self, candidates: np.ndarray, parent: int):
    #     j = candidates
    #     old_cost = self.distance[self.solution.predecessor[j],j]*self.solution.current_cable_cost[j]
    #     profit =  old_cost - self.distance[parent,j]*self.solution.current_cable_cost[j]
    #     return profit
    #
    # def _max_number_reconnections(self, parent: int) -> int:
    #     indices = self.solution.parents[parent]
    #     return np.min(self.solution.current_max_cap[indices] - self.solution.current_cap[indices])






# class PowerCableSolution(EdgeSolution):
#     def __init__(self, n: int, partitions: int, highest_cable_cost: float, highest_cable_length: float, highest_cable_apacity: float):
#         """
#         :param n: number of vertices
#         :param partitions: number of partitions
#         """
#         super().__init__(partitions)
#         self.predecessor = np.zeros(n, dtype=int)
#         self.predecessor[0] = -1
#
#         self.successors = {i:[] for i in range(n)} # type: Dict[int,List[int]]
#         self.number_children_heliosats = {i:0 for i in range(n)} # type: Dict[int,int]
#         self.parents = {i:[0] for i in range(n)} # type: Dict[int,List[int]]
#
#         self.current_cap = np.ones(n)
#         self.current_cap[0] = n - 1
#
#         self.current_max_cap = np.ones(n) * highest_cable_apacity
#
#         # NOTE: needs refactoring
#         self.current_cable_cost = np.ones(n) * highest_cable_cost
#         self.current_cable_cost[0] = 0
#
#         self.current_max_cable_length = np.ones(n) * highest_cable_length
#
#     def add_edge(self, edge: Edge, edge_cost: float, partition: int, coordinates: np.ndarray):
#         super().add_edge(edge, edge_cost, partition, coordinates)
#         self.predecessor[edge.w] = edge.v
#         self.successors[edge.v].append(edge.w)
#
#         for parent in self.heliostat_parents(edge.w):
#             self.number_children_heliosats[parent] += 1
#
#     def remove_edge(self, edge: Edge, partition: int, coordinates: np.ndarray):
#         super().remove_edge(edge, partition, coordinates)
#         self.predecessor[edge.w] = -1
#         self.successors[edge.v].remove(edge.w)
#
#         for parent in self.heliostat_parents[edge.w]:
#             self.number_children_heliosats[parent] -= 1
#
#     def cost(self, distances: np.ndarray):
#         return super().cost(distances)
#
#     def heliostat_parents(self, vertex: int) -> Generator[int,None,None]:
#         """
#         Generator that computes the parent vertices up to the
#         solar tower.
#
#         :param vertex: starting vertex from where we start obtaining predecessor vertices
#         """
#         pred = vertex
#         while pred in self.predecessor:
#             yield self.predecessor[pred]
#             pred = self.predecessor[pred]








        #
        # def next_candidate(self, partition_indices: np.ndarray, cable_index: int) -> Candidate:
        #     """
        #     :param partition_indices:
        #     :param i: index to self.cable_cost
        #     """
        #     all_profit = {i:0.0 for i in partition_indices if i not in self.forbidden}
        #     all_candidates = {i:[] for i in partition_indices if i not in self.forbidden}
        #     if not all_candidates:
        #         return None
        #
        #     while(True):
        #         updated=False
        #         for parent in all_candidates:
        #             candidate, profit = self._best_candidates(parent, cable_index, np.array(all_candidates[parent]))
        #             if candidate != -1:
        #                 all_candidates[parent].append(candidate)
        #                 all_profit[parent] += profit
        #                 updated=True
        #         else:
        #             if not updated:
        #                 break
        #
        #     # return best
        #     best_parent = max(all_candidates, key=all_candidates.get)
        #     best_candidates = all_candidates[best_parent]
        #
        #     return Candidate(parent=best_parent,
        #                      cable_index=cable_index,
        #                      new_children=best_candidates)
        #
        # def _upgrade_cost(self, parent: np.ndarray, cable_index: int):
        #     pred = self.solution.predecessor[parent]
        #     current_cost = self.distance[pred,parent] * self.solution.current_cable_cost[parent]
        #     after_reconnection_cost = self.distance[pred,parent] * self.cable_cost[cable_index]
        #     cost = current_cost - after_reconnection_cost
        #
        #     # mask away upgrades that
        #     if self.distance[pred,parent] > cable_length[cable_index]:
        #         return np.inf
        #     else:
        #         return cost
        #
        # def _reconnecting_profit(self, candidates: np.ndarray, new_parent: int):
        #     j = candidates
        #     old_cost = self.distance[self.solution.predecessor[j],j]*self.solution.current_cable_cost[j]
        #     profit =  old_cost - self.distance[new_parent,j]*self.solution.current_cable_cost[j]
        #     return profit
        #
        # def _max_number_reconnections(self, new_parent: int) -> int:
        #     indices = self.solution.parents[new_parent]
        #     return np.min(self.solution.current_max_cap[indices] - self.solution.current_cap[indices])
        #
        # def _candidate_indices(self, new_parent: int, forbidden_candidates):
        #     """
        #     :param new_parent: new parent for candidates
        #     """
        #     candidates = np.arange(self.coordinates.shape[0])
        #     candidates = np.setdiff1d(candidates, forbidden_candidates)
        #
        #     # capacity constraint
        #     mask_capacity = self.solution.current_cap[candidates] < self._max_number_reconnections(new_parent)
        #     mask_capacity[0] = False
        #     mask_capacity[new_parent] = False
        #
        #     # cable length constraint
        #     mask_cable_length = self.distance[new_parent,candidates] < self.solution.current_max_cable_length[candidates]
        #
        #     return candidates[np.logical_and(mask_capacity, mask_cable_length)]
        #
        # def _best_candidates(self, new_parent: int, cable_index: int, forbidden_candidates: np.ndarray):
        #     candidates = self._candidate_indices(new_parent, forbidden_candidates)
        #     profit = self._reconnecting_profit(candidates, new_parent) - self._upgrade_cost(new_parent, cable_index)
        #     i = np.argmax(profit)
        #     if profit[i] > 0:
        #         return candidates[i], profit[i]
        #     else:
        #         return -1, 0
        #
        # #def _perform_update(self, parent: int, new_cable_cost: float, new_children: List[int], partition: int) -> None:
        # def _perform_update(self, candidate: Candidate, partition: int) -> None:
        #     self.forbidden.add(candidate.parent)
        #
        #     for child in candidate.new_children:
        #         old_parent = self.solution.predecessor[child]
        #         cable_cost = self.solution.edge_costs[child]
        #         self.solution.remove_edge(Edge(old_parent,child), partition, self.coordinates)
        #         self.solution.add_edge(Edge(candidate.parent,child), cable_cost, partition, self.coordinates)
        #     # update parent cable
        #     self.solution.current_max_cable_length[candidate.parent] = self.cable_length[candidate.cable_index]
        #     self.solution.current_max_cap[candidate.parent] = self.cable_capacity[candidate.cable_index]
        #     self.solution.edge_costs[self.solution.predecessor[candidate.parent], candidate.parent] = self.cable_cost[candidate.cable_index]
