from common import Edge
from intersection import is_edge_intersecting


class EdgeSolution:
    def __init__(self, n: int, partitions: int) -> None:
        self.edges= [[] for _ in range(partitions)] # type: List[List[Edge]]
        self.edges_coords = [np.empty((0,4)) for _ in range(partitions)]
        self.edge_type_cost = {} # type: Dict[Edge,int]

    def add_edge(self, edge: Edge, edge_type_cost: float, partition: int):
        self.edges[partition].append(edge)
        self.edge_type_cost[edge] = edge_type_cost

    def intersects(self, edge_coords: np.ndarray) -> bool:
        """
        Checks whether a given edge that is represented by its coordinates is intersecting
        with our solution edges.

        :param edge_coords: a numpy array that holds both x- and y-coordinates
        :returns: True if edge_coords intersects with solution edges
        """
        return is_edge_intersecting(edge_coords, self.edges_coords)

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
