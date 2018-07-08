from itertools import chain
from typing import Generator, Dict, List, NamedTuple

import numpy as np
from scipy.spatial.distance import cdist

from common import Edge


Current = NamedTuple('Current', [('capacity', np.ndarray),
                                 ('max_capacity', np.ndarray),
                                 ('max_length', np.ndarray),
                                 ('cable_cost', np.ndarray)])


Candidate = NamedTuple('Candidate', [('parent',int),
                                     ('value', float),
                                     ('added_capacity', int),
                                     ('new_children', List[int]),
                                     ('new_cable_index', int)])

ChildrenCandidates = NamedTuple('ChildrenCandidates', [('value', float),
                                                       ('children', List[int])])


class MCMSTState:
    def __init__(self, cable_costs: np.ndarray,
                       cable_lengths: np.ndarray,
                       cable_capacities: np.ndarray,
                       coordinates: np.ndarray,
                       working_cost: float,
                       edges_data_cable: List[List[Edge]],
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

        self.working_cost = working_cost
        self.trench_costs = self.distance * working_cost
        self.data_cables = set(chain.from_iterable(edges_data_cable))

        # compute trench costs
        for e in self.data_cables:
            self.trench_costs[e.v, e.w] = 0
            self.trench_costs[e.w, e.v] = 0

        N = coordinates.shape[0]
        self.predecessor = np.zeros(N, dtype=int)
        self.successors = {i:[] for i in range(N)}
        self.heliostat_parents = {i:[0] for i in range(N)} # all parents on path up to the solar tower
        self.unupgraded = [i for i in indices]

        self.upgraded = []

        # initialize current-data-structure
        largest_cable_index = np.argmax(cable_capacities)
        max_capacity = np.ones(N) * cable_capacities[largest_cable_index]
        max_length = np.ones(N) * cable_lengths[largest_cable_index]
        cable_cost = np.ones(N) * cable_costs[largest_cable_index]
        self.current = Current(capacity=np.ones(N),
                               max_capacity=max_capacity,
                               max_length=max_length,
                               cable_cost=cable_cost)

    def update(self, candidate: Candidate) -> None:
        # CAPACITY
        # add capacity to parents of parent
        for p in self._parent_of_parents(candidate.parent):
            self.current.capacity[p] += candidate.added_capacity
        # reduce capacity of parents of new children
        for child in candidate.new_children:
            for p in self._parent_of_parents(child):
                self.current.capacity[p] -= self.current.capacity[child]

        # PREDECESSOR RELATION and TRENCH COSTS
        for child in candidate.new_children:
            # new cable connection -> trench installed
            self._update_trench_costs_child(candidate.parent, child)
            self.predecessor[child] = candidate.parent # NOTE: should be executed after _update_trench_costs_child!

        ############ parent vertex
        self.current.capacity[candidate.parent] += candidate.added_capacity
        self.current.max_capacity[candidate.parent] = self.cable_capacities[candidate.new_cable_index]
        self.current.max_length[candidate.parent] = self.cable_lengths[candidate.new_cable_index]
        self.current.cable_cost[candidate.parent] = self.cable_costs[candidate.new_cable_index]

        self.unupgraded.remove(candidate.parent)
        self.upgraded.append(candidate.parent)

    def _update_trench_costs_child(self, parent: int, child: int) -> None:
        """
        Only reset trench costs if does not have any data cable in it. Moreover, we add substract trench costs of
        v-w (newly installed).
        :param v: 1st vertex endpoint of edge
        :param w: 2nd vertex endpoint of edge
        """
        old_parent = self.predecessor[child]
        if Edge(old_parent,child) not in self.data_cables and Edge(child,old_parent) not in self.data_cables:
            self.trench_costs[old_parent,child] = self.distance[old_parent,child] * self.working_cost
            self.trench_costs[child,old_parent] = self.distance[child,old_parent] * self.working_cost
        self.trench_costs[parent, child] = 0
        self.trench_costs[child, parent] = 0

    def _parent_of_parents(self, parent: int) -> Generator[int,None,None]:
        """
        Returns indices to parent-vertices of parent, excluding the solar tower.
        """
        curr = parent
        while(self.predecessor[curr] != 0):
            yield self.predecessor[curr]
            curr = self.predecessor[curr]

    def next_candidate(self, cable_index: int, current_solution) -> Candidate:
        candidates = []
        for parent in self.unupgraded:
            candidates.append(self._best_candidate(parent, cable_index))

        # parent, best_candidate = max(candidates.items(), key=lambda x: x[1].value)
        candidates.sort(key=lambda x: -x.value)
        for best_candidate in candidates:
            # if best_candidate.value > 0 and self._not_intersecting(best_candidate, current_solution):
            if best_candidate.value > 0:
                return best_candidate
        return None

    def _not_intersecting(self, candidate: Candidate, current_solution) -> bool:
        for c in candidate.new_children:
            edge = Edge(candidate.parent, c)
            e_coords = np.array([[self.coordinates[edge.v][0], self.coordinates[edge.v][1],
                                 self.coordinates[edge.w][0], self.coordinates[edge.w][1]]])
            if current_solution.intersects(e_coords):
                return False
        return True

    def _best_candidate(self, parent: int, new_cable_index: int) -> Candidate:
        candidates = self._candidate_indices(parent)
        profit = self._reconnecting_profit(parent, candidates) - self._upgrade_cost(parent, new_cable_index)

        assert len(profit) == len(candidates)

        children_candidates = self._obtain_new_children_list(parent, candidates, profit)
        children_capacity = sum(self.current.capacity[c] for c in children_candidates.children)
        return Candidate(parent, children_candidates.value, children_capacity, children_candidates.children, new_cable_index)

    def _obtain_new_children_list(self, parent: int, candidates: np.ndarray, profit: np.ndarray) -> ChildrenCandidates:
        """
        Computes for a given parent vertex a list of children candidates that can be reconnected to
        the parent vertex that have positive profit values. If a profit value is negative for a certain
        candidate, we do not perform a reconnection to the parent vertex. We greedily take candidates that have the
        highest profit value in descending order until either all candidates are taken, we encounter
        negative profit values or the capacity of a parent-of-parent vertex (parent including) is
        not sufficient anymore.

        :param parent: parent vertex
        :param candidates: all candidate vertices that can be reconnected due to length and capacity constraints
        :param
        """
        if np.all(profit <= 0):
            return ChildrenCandidates(0.0, [])

        sorted_profit_indices = (-profit).argsort() # we sort in descending profit order
        sorted_candidates = candidates[sorted_profit_indices]

        max_recon_number = self._max_number_reconnections(parent)
        for i in range(len(candidates)):
            if profit[sorted_profit_indices][i] <= 0 or \
               np.sum(self.current.capacity[sorted_candidates][:i] > max_recon_number):
                return ChildrenCandidates(np.sum(profit[sorted_profit_indices][:i]), sorted_candidates[:i])
        return ChildrenCandidates(np.sum(profit), sorted_candidates.tolist())

    def _upgrade_cost(self, parent: np.ndarray, new_cable_index: int):
        pred = self.predecessor[parent]

        if self.distance[pred,parent] > self.cable_lengths[new_cable_index]:
            return np.inf

        current_cost = self.distance[pred,parent] * self.current.cable_cost[parent]
        new_cable_cost = self.distance[pred,parent] * self.cable_costs[new_cable_index]
        return current_cost - new_cable_cost

    def _reconnecting_profit(self, parent: int, candidates: np.ndarray) -> float:
        old_costs = self.distance[self.predecessor[candidates], candidates] * self.current.cable_cost[candidates]
        old_trench_costs = self.trench_costs[self.predecessor[candidates],candidates]
        new_costs = self.distance[parent, candidates] * self.current.cable_cost[candidates]
        new_trench_costs = self.trench_costs[parent, candidates]
        # return old_costs + old_trench_costs - new_costs - new_trench_costs
        return old_costs - new_costs

    def _candidate_indices(self, parent: int) -> np.ndarray:
        """
        Computes for a given parent vertex possible children candidates, that is, these have benefits when
        reconnecting to the new parent vertex as children.

        :param parent: new parent for candidates
        """
        candidates = np.arange(self.coordinates.shape[0])

        # capacity constraint
        mask_capacity = self.current.capacity[candidates] < self._max_number_reconnections(parent)

        # cable length constraint
        mask_cable_length = self.distance[parent,candidates] <= self.current.max_length[candidates]

        final_mask = np.logical_and(mask_capacity, mask_cable_length)
        final_mask[0] = False # solar tower cannot be 'reconnected'
        final_mask[parent] = False # disallow loops
        final_mask[self.upgraded] = False

        return candidates[final_mask]

    def _max_number_reconnections(self, parent: int) -> int:
        """
        Returns maximum number of new children-connections to parent vertex.

        :param parent: parent vertex index
        """
        parents_of_parent = self.heliostat_parents[parent]
        return np.min(self.current.max_capacity[parents_of_parent] - self.current.capacity[parents_of_parent])
