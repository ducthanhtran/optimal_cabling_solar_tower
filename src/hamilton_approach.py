import copy
from itertools import chain, product
from pickle import dump, HIGHEST_PROTOCOL
from typing import Dict, List, Set, Tuple, NamedTuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from initial_hamilton import Hamilton
from ls_hamilton import perform_local_search_hamilton
from common import Edge, solution_value
from intersection import is_edge_intersecting

TriangulationEdges = NamedTuple('TriangulationEdges', [('add', List[Edge]), ('remove', List[Edge]), ('indices', List[int])])

# TODO: Utilize a config class/tuple for readability
# TODO: Improve structure by introducing a class for local search procedure
def run_local_search_hamilton(coordinates: pd.DataFrame,
                              edge_cost: float,
                              partitions: int,
                              distances: np.ndarray,
                              initial_graph_output: str,
                              initial_pkl_output: str,
                              graph_output: str,
                              pkl_output: str,
                              min_improvement: float,
                              upper_cap: float):
    # obtain initial Hamilton solution
    ham = Hamilton(coordinates=coordinates, cable_cost=edge_cost, partitions=partitions)
    ham.compute()

    # plot initial solution and save it via pickle
    init_sol_cost = solution_value(edge_cost=edge_cost, edges=ham.solution.edges, degrees=ham.solution.degrees, distances=distances)
    assert init_sol_cost == ham.solution.cost(distances)

    plot_solution(coordinates=coordinates,
                  output=initial_graph_output,
                  edges=ham.edges,
                  value=init_sol_cost,
                  edge_cost=edge_cost,
                  partitions=partitions)
    with open(initial_pkl_output, 'wb') as init_pkl:
        dump(obj=[ham.edges, ham.degrees], file=init_pkl, protocol=HIGHEST_PROTOCOL)
    print("Initial solution cost: {}".format(init_sol_cost))
    print("Cable length: {}".format(cable_length(ham.edges, cdist(coordinates, coordinates))))

    # perform local search onto initial Hamilton solution
    solution_edges, degrees = perform_local_search_hamilton(ham.solution, min_improvement, upper_cap)

    # plot local optimal solution and save it with pickle
    sol_cost = solution_value(edge_cost=edge_cost, edges=solution_edges, degrees=degrees, distances=distances)
    plot_solution(coordinates=coordinates,
                  output=graph_output,
                  edges=solution_edges,
                  value=sol_cost,
                  edge_cost=edge_cost,
                  partitions=partitions)
    with open(pkl_output, 'wb') as pkl:
        dump(obj=[solution_edges,degrees], file=pkl, protocol=HIGHEST_PROTOCOL)
    print("After local search, solution cost: {}".format(sol_cost))
    print("After local search, cable length: {}".format(cable_length(solution_edges, cdist(coordinates, coordinates))))
    print()
    print("Local search improved cost by {}".format(init_sol_cost-sol_cost))
