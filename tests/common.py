import numpy as np

def random_list(size: int, max_length: int):
    a = np.zeros(max_length, dtype=int)
    a[:size] = 1
    np.random.shuffle(a)
    mask = a.astype(bool)
    return np.arange(max_length)[mask]
