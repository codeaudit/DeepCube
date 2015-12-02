import argparse
import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Rubik's Cube experiments")

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

    # Monitoring options
    parser.add_argument('--patience', type=int,
                        default=20)
    parser.add_argument('--monitoring_freq', type=int,
                        default=100)

    # Print all the arguments
    logger.info("\n" + "#" * 50)
    for arg in vars(args):
        logger.info('\"' + str(arg) + '\" \t: ' + str(vars(args)[arg]))
    logger.info("#" * 40 + "\n")

    return args
