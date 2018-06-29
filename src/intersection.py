import numpy as np

def equal_coords(M, start_l, end_l, start_r, end_r):
    return np.all(M[:, start_l:end_l] == M[:,start_r:end_r], axis=1)


def disjoint_region(M, i, j, k, l):
    return np.all(np.maximum(M[:,i], M[:,j]) < np.minimum(M[:,k], M[:,l]))


def orientation(M, px, py, qx, qy, rx, ry) -> np.ndarray:
    val = (M[:,qy]-M[:,py]) * (M[:,rx]-M[:,qx]) - (M[:,qx]-M[:,px]) * (M[:,ry]-M[:,qy])
    np.place(val, val > 0, 1)
    np.place(val, val < 0, 2)
    return val


def on_segment(M, px, py, qx, qy, rx, ry) -> np.ndarray:
    return np.logical_and.reduce((M[:,qx] <= np.maximum(M[:,px], M[:,rx]),
                                  M[:,qx] >= np.minimum(M[:,px], M[:,rx]),
                                  M[:,qy] <= np.maximum(M[:,py], M[:,ry]),
                                  M[:,qy] >= np.minimum(M[:,py], M[:,ry])))


def do_intersect(edge: np.ndarray, edge_set: np.ndarray):
    """
    :param edge: array of size (1,4)
    :param edge_set: array of shape (N,4)
    """
    # NOTE: ugly hack
    if len(edge_set) == 0:
        return False

    M = np.concatenate((np.repeat(edge, len(edge_set), axis=0), edge_set), axis=1)

    # obtain masking of incident edges
    incident_mask = np.logical_or.reduce((equal_coords(M,0,2,4,6),
                                          equal_coords(M,0,2,6,8),
                                          equal_coords(M,2,4,4,6),
                                          equal_coords(M,2,4,6,8)))
    M = M[np.logical_not(incident_mask)]
    if M.size == 0:
        return False

    o1 = orientation(M,0,1,2,3,4,5)
    o2 = orientation(M,0,1,2,3,6,7)
    o3 = orientation(M,4,5,6,7,0,1)
    o4 = orientation(M,4,5,6,7,2,3)

    if np.logical_and(o1 != o2, o3 != o4).any():
        return True

    if np.logical_and(o1 == 0, on_segment(M,0,1,4,5,2,3)).any():
        return True
    if np.logical_and(o2 == 0, on_segment(M,0,1,6,7,2,3)).any():
        return True
    if np.logical_and(o3 == 0, on_segment(M,4,5,0,1,6,7)).any():
        return True
    if np.logical_and(o4 == 0, on_segment(M,4,5,2,3,6,7)).any():
        return True
    return False
