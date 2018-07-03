from typing import List
from recordclass import RecordClass
from scipy.spatial.distance import cdist

MCMSTState = RecordClass('MCMSTState', [('unvisited', List[int])])
Candidate = RecordClass('Candidate', [('parent', int),
                                      ('cable_index', float),
                                      ('new_children', List[int])])

class MCMST:
    # __slots__ = ('coordinates', 'edge_costs', 'partitions', 'solution', 'current_state')

    def __init__(self,
                 coordinates: np.ndarray,
                 partitions: int,
                 work_cost: float,
                 open_trenches: List[List[Edge]],
                 cable_cost: np.ndarray,
                 cable_length: np.ndarray,
                 cable_capacity: np.ndarray):
        """
        todoc
        !assume cable_cost is sorted in ascending order!
        """
        self.cable_cost = cable_cost
        self.cable_length = cable_length
        self.cable_capacity = cable_capacity
        self.forbidden = set() # type: int

        self.coordinates = coordinates
        self.partitions = partitions
        self.solution = PowerCableSolution(self.coordinates.shape[0], partitions,
                                          np.max(cable_cost), np.max(cable_length), np.max(cable_capacity))

        self.distance = cdist(coordinates,coordinates)
        #indices = np.vstack(open_trenches)
        #self.trench_cost = np.self.distance[indices[:,0], indices[:,1]]*trench_cost

    def compute(self) -> None:
        for i, p in enumerate(compute_partitions(coordinates=self.coordinates, partitions=self.partitions)):
            # connect all heliostats from partition with solar tower
            print(p)
            for heliostat in p:
                self.solution.add_edge(Edge(0,heliostat), self.cable_cost[6], i, self.coordinates)

            self.current_state = MCMSTState(unvisited=p)
            self._mcmst(i, p)

    def _mcmst(self, partition, partition_indices) -> None:
        """
        """
        for i in reversed(range(7)):
            print("Cable: {}".format(i))
            while(True):
                candidate = self.next_candidate(partition_indices, i)
                if candidate is None:
                    break # no suitable candidate for upgrade + reconnections
                self._perform_update(candidate, partition=i)

    def next_candidate(self, partition_indices: np.ndarray, cable_index: int) -> Candidate:
        """
        :param partition_indices:
        :param i: index to self.cable_cost
        """
        all_profit = {i:0.0 for i in partition_indices if i not in self.forbidden}
        all_candidates = {i:[] for i in partition_indices if i not in self.forbidden}
        if not all_candidates:
            return None

        while(True):
            updated=False
            for parent in all_candidates:
                candidate, profit = self._best_candidates(parent, cable_index, np.array(all_candidates[parent]))
                if candidate != -1:
                    all_candidates[parent].append(candidate)
                    all_profit[parent] += profit
                    updated=True
            else:
                if not updated:
                    break

        # return best
        best_parent = max(all_candidates, key=all_candidates.get)
        best_candidates = all_candidates[best_parent]

        return Candidate(parent=best_parent,
                         cable_index=cable_index,
                         new_children=best_candidates)

    def _upgrade_cost(self, parent: np.ndarray, cable_index: int):
        pred = self.solution.predecessor[parent]
        current_cost = self.distance[pred,parent] * self.solution.current_cable_cost[parent]
        after_reconnection_cost = self.distance[pred,parent] * self.cable_cost[cable_index]
        cost = current_cost - after_reconnection_cost

        # mask away upgrades that
        if self.distance[pred,parent] > cable_length[cable_index]:
            return np.inf
        else:
            return cost

    def _reconnecting_profit(self, candidates: np.ndarray, new_parent: int):
        j = candidates
        old_cost = self.distance[self.solution.predecessor[j],j]*self.solution.current_cable_cost[j]
        profit =  old_cost - self.distance[new_parent,j]*self.solution.current_cable_cost[j]
        return profit

    def _max_number_reconnections(self, new_parent: int) -> int:
        indices = self.solution.parents[new_parent]
        return np.min(self.solution.current_max_cap[indices] - self.solution.current_cap[indices])

    def _candidate_indices(self, new_parent: int, forbidden_candidates):
        """
        :param new_parent: new parent for candidates
        """
        candidates = np.arange(self.coordinates.shape[0])
        candidates = np.setdiff1d(candidates, forbidden_candidates)

        # capacity constraint
        mask_capacity = self.solution.current_cap[candidates] < self._max_number_reconnections(new_parent)
        mask_capacity[0] = False
        mask_capacity[new_parent] = False

        # cable length constraint
        mask_cable_length = self.distance[new_parent,candidates] < self.solution.current_max_cable_length[candidates]

        return candidates[np.logical_and(mask_capacity, mask_cable_length)]

    def _best_candidates(self, new_parent: int, cable_index: int, forbidden_candidates: np.ndarray):
        candidates = self._candidate_indices(new_parent, forbidden_candidates)
        profit = self._reconnecting_profit(candidates, new_parent) - self._upgrade_cost(new_parent, cable_index)
        i = np.argmax(profit)
        if profit[i] > 0:
            return candidates[i], profit[i]
        else:
            return -1, 0

    #def _perform_update(self, parent: int, new_cable_cost: float, new_children: List[int], partition: int) -> None:
    def _perform_update(self, candidate: Candidate, partition: int) -> None:
        self.forbidden.add(candidate.parent)

        for child in candidate.new_children:
            old_parent = self.solution.predecessor[child]
            cable_cost = self.solution.edge_costs[child]
            self.solution.remove_edge(Edge(old_parent,child), partition, self.coordinates)
            self.solution.add_edge(Edge(candidate.parent,child), cable_cost, partition, self.coordinates)
        # update parent cable
        self.solution.current_max_cable_length[candidate.parent] = self.cable_length[candidate.cable_index]
        self.solution.current_max_cap[candidate.parent] = self.cable_capacity[candidate.cable_index]
        self.solution.edge_costs[self.solution.predecessor[candidate.parent], candidate.parent] = self.cable_cost[candidate.cable_index]
