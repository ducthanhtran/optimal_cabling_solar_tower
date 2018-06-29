#!/usr/bin/env python3
# UTF-8 encoding
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def int_greater_or_equal(i: int):
    def _check_valid(input: str):
        value = int(input)
        if value < i:
            raise argparse.ArgumentTypeError('Parameter below {}'.format(i))
        return value
    return _check_valid


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, required=True,
                        help='Coordinates of heliostats. We assume that the solar tower is at coordinate (0,0).')
    parser.add_argument('--alg', choices=['mst', 'hamilton'], required=True, default='hamilton',
                        help='Local search algorithm.')

    inputs = parser.add_argument_group('Input files')
    inputs.add_argument('--data-cables', type=str, required=True)
    inputs.add_argument('--labor-costs', type=str, required=True)
    inputs.add_argument('--switches', type=str, required=True)

    outputs = parser.add_argument_group('Output file')
    outputs.add_argument('--output-graph', type=str, required=True)
    outputs.add_argument('--output-pkl', type=str, required=True)
    outputs.add_argument('--output-init-graph', type=str, required=True)
    outputs.add_argument('--output-init-pkl', type=str, required=True)

    restrictions = parser.add_argument_group('Size Restrictions')
    restrictions.add_argument('--max-connections-data-subnetwork', type=int_greater_or_equal(1), default=128,
                              help='Maximum number of heliostats within one subnetwork.')
    restrictions.add_argument('--k', type=int_greater_or_equal(1), required=True,
                              help='Number of partitions for data cable.')
    restrictions.add_argument('--hamilton-min-improvement', type=float, default=-1.0,
                              help='Minimum cost change in Euros.')
    return parser


def check_file_existence(file: str) -> bool:
    return Path(file).is_file()


if __name__ == '__main__':
    args = create_parser().parse_args()

    if not check_file_existence(args.input):
        raise argparse.ArgumentTypeError("Cannot open file {}".format(args.input))

    coordinates = np.loadtxt(args.input, delimiter=';')
    coordinates = np.vstack((np.array([0,0]), coordinates))
