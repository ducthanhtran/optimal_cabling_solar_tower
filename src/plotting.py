#!/usr/bin/env python3
# UTF-8 encoding
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from common import Edge


def plot_solution(coordinates: np.ndarray,
                  output: str,
                  edges: List[List[Edge]],
                  value: float,
                  edge_cost: float,
                  partitions: int) -> None:
    plt.figure(figsize=(16,16), dpi=350)
    plt.axis('equal')

    coordinates_arr = np.array([tuple(x) for x in coordinates])
    for i, all_edges in enumerate(edges):
        edges_arr = np.array([tuple(list(x)) for x in all_edges])

        x = coordinates_arr[:,0].flatten()
        y = coordinates_arr[:,1].flatten()

        norm = matplotlib.colors.Normalize(vmin=0, vmax=len(edges), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.cool)

        plt.plot(x[edges_arr.T], y[edges_arr.T], linestyle='-', color=mapper.to_rgba(i),
                 markerfacecolor='red', marker='o')
    plt.title("Cable cost/m: {}€ \nPartitions: {}\nTotal costs: {:0,.2f}€".format(edge_cost, partitions, value))
    plt.scatter(coordinates_arr[:, 0], coordinates_arr[:, 1], color='red', marker='o')
    plt.savefig(output, dpi=300)
    plt.close()
