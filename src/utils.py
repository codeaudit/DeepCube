import argparse
import logging

import numpy as np

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Rubik's Cube experiments")

    # Game options
    # Number of cubies on each edge of the big cube (standard is 3)
    parser.add_argument('--cube_size', type=int,
                        default=3)

    # Qlearning options
    # The number of permutations from a finished cube to get the initial cube
    parser.add_argument('--rand_nb', type=int,
                        default=1)
    parser.add_argument('--epsilon', type=float,
                        default=0.1)    # The probability of taking a random moves
    parser.add_argument('--gamma', type=float,
                        default=0.9)    # The discount
    parser.add_argument('--nb_episode', type=int,
                        default=50)     # Number of considered episodes
    parser.add_argument('--steps_in_episode', type=int,
                        default=40)     # Number of steps in an episode

    # Model options
    parser.add_argument('--layers', type=int,
                        default=4)
    parser.add_argument('--units_per_layer', type=int,
                        default=100)

    # Serialization options
    parser.add_argument('--load_path', type=str,
                        default=None)
    parser.add_argument('--save_path', type=str,
                        default="trained_models/")

    # Training options
    parser.add_argument('--algorithm', choices=['rms_prop', 'adam', 'sgd'],
                        default='sgd')
    parser.add_argument('--learning_rate', type=float,
                        default=1e-3)
    parser.add_argument('--momentum', type=float,
                        default=0.9)
    parser.add_argument('--clipping', type=float,
                        default=5)
    parser.add_argument('--mini_batch_size', type=float,
                        default=8)

    # Monitoring options
    parser.add_argument('--patience', type=int,
                        default=20)
    parser.add_argument('--monitoring_freq', type=int,
                        default=100)

    args = parser.parse_args()
    # Print all the arguments
    logger.info("\n" + "#" * 50)
    for arg in vars(args):
        logger.info('\"' + str(arg) + '\" \t: ' + str(vars(args)[arg]))
    logger.info("#" * 40 + "\n")

    return args


def _permutation(n):
    r = np.arange(n, dtype=np.uint32)
    np.random.shuffle(r)
    return r
