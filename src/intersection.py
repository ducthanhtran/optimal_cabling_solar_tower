#!/usr/bin/env python3
# UTF-8 encoding
from typing import List

import numpy as np


def is_edge_intersecting(edge: np.ndarray, other_edges: List[np.ndarray]) -> bool:
    """
    Checks whether an edge intersects with a set of other_edges.

    :param edge: array of size (1,4)
    :param other_edges: array of shape (N,4) denoting N other_edges
    """

    other_edges = np.vstack(other_edges)

    # NOTE: ugly hack
    if len(other_edges) == 0:
        return False

    matrix = np.concatenate((np.repeat(edge, len(other_edges), axis=0), other_edges), axis=1)

    # obtain masking of incident other_edges
    incident_mask = np.logical_or.reduce((equal_coords(matrix, 0, 2, 4, 6),
                                          equal_coords(matrix, 0, 2, 6, 8),
                                          equal_coords(matrix, 2, 4, 4, 6),
                                          equal_coords(matrix, 2, 4, 6, 8)))
    matrix = matrix[np.logical_not(incident_mask)]
    if matrix.size == 0:
        return False

    o1 = orientation(matrix, 0, 1, 2, 3, 4, 5)
    o2 = orientation(matrix, 0, 1, 2, 3, 6, 7)
    o3 = orientation(matrix, 4, 5, 6, 7, 0, 1)
    o4 = orientation(matrix, 4, 5, 6, 7, 2, 3)

    if np.logical_and(o1 != o2, o3 != o4).any():
        return True

    if np.logical_and(o1 == 0, on_segment(matrix, 0, 1, 4, 5, 2, 3)).any():
        return True
    if np.logical_and(o2 == 0, on_segment(matrix, 0, 1, 6, 7, 2, 3)).any():
        return True
    if np.logical_and(o3 == 0, on_segment(matrix, 4, 5, 0, 1, 6, 7)).any():
        return True
    if np.logical_and(o4 == 0, on_segment(matrix, 4, 5, 2, 3, 6,7 )).any():
        return True
    return False


def equal_coords(matrix: np.ndarray,
                 start_l: int,
                 end_l: int,
                 start_r: int,
                 end_r: int) -> np.ndarray:
    return np.all(matrix[:, start_l:end_l] == matrix[:, start_r:end_r], axis=1)


def disjoint_region(matrix: np.ndarray,
                    i: int,
                    j: int,
                    k: int,
                    l: int) -> bool:
    return np.all(np.maximum(matrix[:, i], matrix[:, j]) < np.minimum(matrix[:, k], matrix[:, l]))


def orientation(matrix: np.ndarray,
                px: int,
                py: int,
                qx: int,
                qy: int,
                rx: int,
                ry: int) -> np.ndarray:
    val = (matrix[:, qy] - matrix[:, py]) * (matrix[:, rx] - matrix[:, qx]) - \
          (matrix[:, qx] - matrix[:, px]) * (matrix[:, ry] - matrix[:, qy])
    np.place(val, val > 0, 1)
    np.place(val, val < 0, 2)
    return val


def on_segment(M, px, py, qx, qy, rx, ry) -> np.ndarray:
    return np.logical_and.reduce((M[:,qx] <= np.maximum(M[:,px], M[:,rx]),
                                  M[:,qx] >= np.minimum(M[:,px], M[:,rx]),
                                  M[:,qy] <= np.maximum(M[:,py], M[:,ry]),
                                  M[:,qy] >= np.minimum(M[:,py], M[:,ry])))
