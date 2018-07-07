#!/usr/bin/env python3
# UTF-8 encoding
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

from hamilton_approach import run_local_search_hamilton
from initial_mcmst import MCMST


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
    parser.add_argument('--alg', choices=['mst', 'hamilton'], default='hamilton',
                        help='Local search algorithm.')

    inputs = parser.add_argument_group('Input files')
    inputs.add_argument('--edge-cost-data', type=float, help='trench cost+cable_cost_data')
    inputs.add_argument('--trench-cost', type=float)

    outputs = parser.add_argument_group('Output file')
    outputs.add_argument('--output-graph', type=str, required=True)
    outputs.add_argument('--output-pkl', type=str, required=True)
    outputs.add_argument('--output-init-graph', type=str, required=True)
    outputs.add_argument('--output-init-pkl', type=str, required=True)

    restrictions = parser.add_argument_group('Size Restrictions')
    restrictions.add_argument('--max-connections-data-subnetwork', type=int_greater_or_equal(1), default=999999,
                              help='Maximum number of heliostats within one subnetwork.')
    restrictions.add_argument('--partitions', type=int_greater_or_equal(1),
                              help='Number of partitions for data cable.', default=1)
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

    coordiantes = coordinates[:50]

    distances = cdist(coordinates,coordinates)

    if args.alg == 'hamilton':
        run_local_search_hamilton(coordinates=coordinates,
                                  edge_cost=args.edge_cost_data,
                                  partitions=args.partitions,
                                  distances=distances,
                                  initial_graph_output=args.output_init_graph,
                                  initial_pkl_output=args.output_init_pkl,
                                  graph_output=args.output_graph,
                                  pkl_output=args.output_pkl,
                                  min_improvement=args.hamilton_min_improvement,
                                  upper_cap=args.max_connections_data_subnetwork)
