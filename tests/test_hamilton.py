#!/usr/bin/env python3
# UTF-8 encoding
from itertools import chain

import pytest
from src.hamilton import Hamilton

COORDINATES = np.array([[  0.  ,   0.  ],
                        [ 73.33,  43.86],
                        [ 59.72,  61.12],
                        [ 42.15,  74.33],
                        [ 21.8 ,  82.62],
                        [  0.  ,  85.45],
                        [-21.8 ,  82.62],
                        [-42.15,  74.33],
                        [-59.72,  61.12],
                        [-73.33,  43.86]])

def test_hamilton():
    ham = Hamilton(COORDIANTES, 54, 1)
    ham.compute()

    assert len(list(chain.from_iterable(ham.solution.edges))) == COORDINATES.shape[0]
