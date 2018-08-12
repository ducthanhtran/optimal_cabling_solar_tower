#!/usr/bin/env python3
# UTF-8 encoding
import argparse

import numpy as np
from scipy.spatial.distance import cdist

from local_search_hamilton import run_local_search_hamilton
from local_search_upgrade_downgrade import run_local_search_mst


def int_greater_or_equal(i: int):
    def _check_valid(arg: str):
        value = int(arg)
        if value < i:
            raise argparse.ArgumentTypeError('Parameter below {}'.format(i))
        return value
    return _check_valid


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, required=True,
                        help='Coordinates of heliostats.')
    parser.add_argument('--alg', choices=['mst', 'hamilton'], default='hamilton',
                        help='Local search algorithm.')

    costs = parser.add_argument_group('Costs')
    costs.add_argument('--edge-cost-data', type=float, required=True,
                       help='Trench cost + glass fiber cable cost + protective foil cost.')
    costs.add_argument('--trench-cost', type=float, required=True, help='Trench costs only.')

    outputs = parser.add_argument_group('Output files')
    outputs.add_argument('--output-graph', type=str, default='')
    outputs.add_argument('--output-pkl', type=str, default='')
    outputs.add_argument('--output-init-graph', type=str, default='')
    outputs.add_argument('--output-init-pkl', type=str, default='')

    restrictions = parser.add_argument_group('Size Restrictions')
    restrictions.add_argument('--max-connections-data-subnetwork', type=int_greater_or_equal(1), default=999999,
                              help='Maximum number of heliostats within one subnetwork with regards to the data cable.')
    restrictions.add_argument('--partitions', type=int_greater_or_equal(1),
                              help='Number of partitions for data cable.', default=1)
    restrictions.add_argument('--hamilton-min-improvement', type=float, default=-1.0,
                              help='Minimum cost change in Euros.')
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()

    coordinates = np.loadtxt(args.input, delimiter=';')
    coordinates = np.vstack((np.array([0, 0]), coordinates))  # add solar tower

    distances = cdist(coordinates, coordinates)

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
    elif args.alg == 'mst':
        pass
