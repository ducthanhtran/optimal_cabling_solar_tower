from itertools import chain, product
from typing import Dict, List,Set

from assign_power_cables import *
from common import Cables
from intersection import is_edge_intersecting


class LocalSearchUpgradeDowngrade:

    def __init__(self, edges: List[List[Edge]],
                 edges_coordinates: np.ndarray,
                 degrees: Dict[int,int],
                 cables: Cables,
                 data_cable_cost: float,
                 coordinates: np.ndarray):
        self.edges = edges
        self.edge_set = set(chain.from_iterable(self.edges))
        self.edge_partition = {}
        for i in range(len(self.edges)):
            for e in self.edges[i]:
                self.edge_partition[e] = i

        self.edges_coordinates = edges_coordinates
        self.degrees = degrees

        self.power_cable_assignment = assign_power_cables(edges, coordinates, cables.lengths, cables.capacities)
        self.capacities = compute_capacities(edges, coordinates.shape[0]) # type: np.ndarray
        self.max_capacities = compute_max_capacities(self.power_cable_assignment, coordinates.shape[0])
        self.cable_assignment = assign_cable(power_cable_assignment, coordinates.shape[0])

        self.parents = obtain_parents(edges, coordinates.shape[0])
        self.predecessor = get_predecessor(edges)
        self.all_successors = get_successor_list(edges, coordinates.shape[0])
        for i in range(coordinates.shape[0]):
            self.all_successors[i].remove(i) # remove oneself

        self.data_cable_cost = data_cable_cost
        self.cables = cables # assume all are ordered ascendingly
        self.coordinates = coordinates

        self.distance = cdist(coordinates, coordinates)
        self.heliostats = range(1,coordinates.shape[0])

    def perform_local_search(self):
        i = 0
        improv = 0
        while True:
            candidate = self._next_candidate()
            if candidate is None:
                break
            improv += candidate[0]
            self._update(candidate)

            plot_solution(coordinates, '/home/duc/mst_{}.png'.format(i), self.edges, 0, 0, 3)
            i += 1
        print("Total improvement: {}".format(improv))

    def _update(self,candidate):
        total, edge, new_edge_cable, parents_upgrade, cable_indices_upgrade, parents_downgrade, cable_indices_downgrade = candidate

        print('Improve by {}'.format(total))

        # remove old edge
        old_edge = Edge(self.predecessor[edge.w], edge.w)

        p = self.edge_partition[old_edge]
        self.edges[p].remove(old_edge)
        self.degrees[old_edge.v] -= 1
        del self.edge_partition[old_edge]
        self.edge_set.remove(old_edge)

        index = np.where(np.all(self._coordinates(old_edge) == self.edges_coordinates[p], axis=1))
        self.edges_coordinates[p] = np.delete(self.edges_coordinates[p], index, axis=0)
        self.predecessor[edge.w] = edge.v # new predecessor relation


        for q in self.parents[edge.w]:
            self.all_successors[q] = list(set(self.all_successors[q]) - set(self.all_successors[edge.w]))
            self.all_successors[q].remove(edge.w)

        # add edge
        for s in self.all_successors[edge.w]:
            for old_parent in self.parents[edge.w]:
                self.parents[s].remove(old_parent)
            self.parents[s] = self.parents[s] + self.parents[edge.v] + [edge.v]

        self.parents[edge.w] = self.parents[edge.v] + [edge.v]

        for parent in self.parents[edge.v]:
            self.all_successors[parent] = self.all_successors[parent] + self.all_successors[edge.w] + [edge.w]

        self.all_successors[edge.v] += self.all_successors[edge.w] + [edge.w]

        partition = self.edge_partition[self.predecessor[edge.v], edge.v]
        self.edges[partition].append(edge)
        self.edge_set.add(edge)

        self.cable_assignment[edge.w] = new_edge_cable
        self.capacities[edge.v] += self.capacities[edge.w]
        self.degrees[edge.v] += 1
        self.edge_partition[edge] = partition

        self.edges_coordinates[partition] = np.vstack((self.edges_coordinates[partition], self._coordinates(edge)))

        # upgrade parents of edge.v
        for parent,new_parent_cable in zip(parents_upgrade, cable_indices_upgrade):
            self.cable_assignment[parent] = new_parent_cable
            self.capacities[parent] += self.capacities[edge.w]
            self.max_capacities[parent] = self.cables.capacities[new_parent_cable]

        # downgrade parents of edge.w
        for parent,new_parent_cable in zip(parents_downgrade, cable_indices_downgrade):
            self.cable_assignment[parent] = new_parent_cable
            self.capacities[parent] -= self.capacities[edge.w]
            self.max_capacities[parent] = self.cables.capacities[new_parent_cable]

        for v in self.heliostats:
            assert len(self.parents[v]) < self.coordinates.shape[0]
            assert len(self.all_successors[v]) < self.coordinates.shape[0]


    def _next_candidate(self):
        for v,w in product(self.heliostats, self.heliostats):
            if v != w and self._valid_edge(v,w):
                # upgrade costs
                upgrade_cost, parents_upgrade, cable_indices_upgrade = self._upgrade_power_cables(v,w)
                switch_upgrade = self._upgrade_switch(v)

                # downgrade costs
                downgrade_profit, parents_downgrade, cable_indices_downgrade = self._downgrade_power_cables(w)

                # new edge
                cost_new_edge, new_edge_cable = self._new_edge(v,w)

                # deleted edge
                profit_deleted_edge = self._delete_edge(w)

                total = upgrade_cost + switch_upgrade + cost_new_edge - downgrade_profit - profit_deleted_edge
                if total < 0: # first improvement
                    candidate = (total,
                                  Edge(v,w),
                                  new_edge_cable,
                                  parents_upgrade,
                                  cable_indices_upgrade,
                                  parents_downgrade,
                                  cable_indices_downgrade)
                    return candidate
        return None

    def _valid_edge(self, v, w):
        return self.distance[v,w] <= self.cables.lengths[-1] and \
               not is_edge_intersecting(self._coordinates(Edge(v,w)), self.edges_coordinates) and \
               Edge(v,w) not in self.edge_set and \
               self.predecessor[w] != 0 and \
                Edge(w,v) not in self.edge_set and \
                w not in self.parents[v]

    ### NEW EDGE #########################
    def _new_edge(self, v, w):
        # get cable for v-> w s.t. capacity of w can be covered + length restriction
        cap = self.capacities[w] < self.cables.capacities
        length = self.distance[v,w] < self.cables.lengths

        res = np.where(np.logical_and(cap,length) == True)[0]
        if res.size == 0:
            return RuntimeError('New Edge error!')

        power_cable_cost = self.cables.costs[res[0]] * self.distance[v,w] * 2
        data_dable_cost = self.distance[v,w] * self.data_cable_cost
        return power_cable_cost+data_dable_cost, res[0]

    ### DELETED EDGE #########################
    def _delete_edge(self, w):
        power_cable_cost = self.distance[self.predecessor[w],w] * self.cables.costs[self.cable_assignment[w]] * 2
        data_cable_cost = self.distance[self.predecessor[w],w] * self.data_cable_cost
        if self.degrees[self.predecessor[w]] == 3 or self.degrees[self.predecessor[w]] == 10:
            data_cable_cost += 700

        return power_cable_cost + data_cable_cost

    ### UPGRADE COST #########################
    def _upgrade_switch(self, v):
        if self.degrees[v] == 2 or self.degrees[v] == 9:
            return 700
        return 0

    def _upgrade_power_cables(self, v, w):
        cost = 0
        cables_upgrade = []
        for parent in self.parents[v]:
            new_cable_index = self._upgrade_cable_index(parent, w)
            if new_cable_index == -1:
                return np.inf, None, None
            cables_upgrade.append(new_cable_index)
            cost += self._cable_cost_difference_upgrade(parent, new_cable_index)
        return cost, self.parents[v], cables_upgrade

    def _upgrade_cable_index(self, parent, w):
        new_cap = self.capacities[parent] + self.capacities[w]
        if new_cap > self.cables.capacities[-1]:
            return -1

        valid_cables = new_cap <= self.cables.capacities
        valid_lengths = self.distance[self.predecessor[parent],parent] <= self.cables.lengths

        res = np.where(np.logical_and(valid_cables, valid_lengths) == True)[0]
        if res.size == 0:
            raise RuntimeError('SHOULD NOT BE REACHABLE ANYMORE. No suitable cable found for {}. Additional capacity:'.format(parent,self.capacities[w]))
        return res[0]

    def _cable_cost_difference_upgrade(self, parent, new_cable):
        cost_difference = self.cables.costs[new_cable] - self.cables.costs[self.cable_assignment[parent]]
        return self.distance[self.predecessor[parent], parent] * cost_difference * 2
    #########################

    ### DOWNGRADE PROFIT #########################
    def _downgrade_power_cables(self, w):
        profit = 0
        cables_downgrade = []
        for parent in self.parents[w]:
            new_cable_index = self._downgrade_cable_index(parent, w)
            cables_downgrade.append(new_cable_index)
            profit += self._cable_cost_difference_downgrade(parent, new_cable_index)
        return profit, self.parents[w], cables_downgrade

    def _downgrade_cable_index(self, parent, w):
        new_cap = self.capacities[parent] - self.capacities[w]
        valid_cables = new_cap <= self.cables.capacities
        valid_lengths = self.distance[self.predecessor[parent],parent] <= self.cables.lengths

        res = np.where(np.logical_and(valid_cables, valid_lengths) == True)[0]
        if res.size == 0:
            raise RuntimeError('No suitable cable found')
        return res[0]

    def _cable_cost_difference_downgrade(self, parent, new_cable):
        cost_difference = self.cables.costs[self.cable_assignment[parent]] - self.cables.costs[new_cable]
        return self.distance[self.predecessor[parent], parent] * cost_difference * 2
    #########################

    def _coordinates(self, edge: Edge):
        return np.array([[self.coordinates[edge.v][0],
                          self.coordinates[edge.v][1],
                          self.coordinates[edge.w][0],
                          self.coordinates[edge.w][1]]])
