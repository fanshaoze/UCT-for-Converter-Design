import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # used by querying
    parser.add_argument(
        '--train-size', type=int, default=2000, help='training data size')
    parser.add_argument(
        '--test-size', type=int, default=2000, help='test data size')
    parser.add_argument(
        '--valid-size', type=int, default=200, help='validation data size')

    parser.add_argument(
        '--no_cuda', action='store_true', default=False, help='do not use cuda')

    parser.add_argument(
        '--query-times', type=int, default=1, help='the number of queries')
    parser.add_argument(
        '--sigma', type=float, default=1e-4, help='likelihood noise')

    parser.add_argument(
        '--num-runs', type=int, help='number of runs for UCT')

    parser.add_argument(
        '--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument(
        '--seed-range', nargs='+', type=int, default=[0, 10], help='random seed range')

    parser.add_argument(
        '--dry', action='store_true', default=False, help='dry run')

    parser.add_argument(
        '--debug', action='store_true', default=False, help='debug mode')

    parser.add_argument(
        '--skip-sim', action='store_true', default=True, help='skip the the topologies which is not in the '
                                                              'simulation hash')

    parser.add_argument(
        '--sweep', action='store_true', default=True, help='sweep duty cycles')

    parser.add_argument(
        '--k-list', nargs='+', type=int, default=[1, 10, 20, 50, 100, 250, 500, 1500, 2500], help='evaluate top k topos'
    )
    parser.add_argument(
        '--output', type=str, default='result', help='output json file name'
    )

    parser.add_argument(
        '--model', type=str, default='gnn', choices=['simulator', 'transformer', 'gp', 'analytics', 'gnn'], help='surrogate model'
    )
    parser.add_argument(
        '--traj', nargs='+', type=int, default=[100], help='trajectory numbers'
    )

    # parser.add_argument(
    #     '--eff-model', type=str, default='5_comp_transformer_eff.pt', required=True, help='eff pt model file name'
    # )
    # parser.add_argument(
    #     '--vout-model', type=str, default='5_comp_transformer_vout.pt', required=True, help='vout pt model file name'
    # )
    parser.add_argument(
        '--eff-model', type=str, default='5_comp_transformer_eff.pt', help='eff pt model file name'
    )
    parser.add_argument(
        '--vout-model', type=str, default='5_comp_transformer_vout.pt', help='vout pt model file name'
    )
    parser.add_argument(
        '--vocab', type=str, default='dataset_5_vocab.json', help='transformer vocab file'
    )
    parser.add_argument(
        '--round', type=str, default='vout', help='transformer vocab file'
    )

    args = parser.parse_args()

    return args
