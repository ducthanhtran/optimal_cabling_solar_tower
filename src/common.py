from typing import Dict, NamedTuple


Edge = NamedTuple('Edge', [('v', int), ('w', int)]) # edge contains two indices to coordinates-array


class EdgeSolution:
    def __init__(self, n: int, partitions: int) -> None:
        self.edges= [[] for _ in range(partitions)] # type: List[List[Edge]]
        self.edge_type_cost = {} # type: Dict[Edge,int]

    def add_edge(self, edge: Edge, edge_type_cost: float, partition: int):
        self.edges[partition].append(edge)
        self.edge_type_cost[edge] = edge_type_cost

    def cost(self, distances: np.ndarray):
        return sum(distances[e.v,e.w]*self.edge_type_cost[e] for e in chain.from_iterable(self.edges))


class EdgeVertexSolution(EdgeSolution):
    def __init__(self, n: int, partitions: int):
        super().__init__(n, partitions)
        self.degrees = {i:0 for i in range(n)} # type: Dict[int,int]

    def add_edge(self, edge: Edge, edge_type_cost: float, partition: int):
        self.edges[partition].append(edge)
        self.edge_type_cost[edge] = edge_type_cost
        self.degrees[edge.v] += 1
        self.degrees[edge.w] += 1

    def cost(self, distances: np.ndarray):
        cost = super().cost(distances)
        return cost + sum(self._switch_cost(d) for d in self.degrees.values())

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


def compute_partitions(coordinates: np.ndarray, partitions: int) -> List[np.ndarray]:
    """
    Partitions the coordinates into an array of indices. For this matter we use the angles
    between the reference vector of the x-axis.

    :param coordinates: a 2D real array of x- and y-coordinates
    :param partitions: number of partitions
    :return: list of arrays that contain indices of heliostats with regards to the coordinates array
    """
    degrees = np.degrees(np.arctan2(coordinates[1:,1], self.coordinates.iloc[1:,0]))
    indices = degrees.argsort()
    padding = (-len(indices))%partitions
    L = np.split(np.concatenate((indices,np.ones(padding)*-1)),partitions) # padding value: -1
    # remove padding from last partition and increase index
    return [np.delete(a, np.where(a == -1)) + 1 for a in L]
