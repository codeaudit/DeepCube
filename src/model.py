import logging

import numpy as np
import theano
from theano import tensor

from blocks import initialization
from blocks.bricks import MLP, Rectifier
from blocks.bricks.cost import SquaredError

from blocks.filter import VariableFilter, get_brick
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.serialization import load_parameter_values

floatX = theano.config.floatX
logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def build_model_mlp(args, dtype=floatX):
    logger.info('Building model ...')

    # Variables of the model
    # the rubik's cube stickers
    x = tensor.fmatrix("x")

    # the action taken
    action = tensor.fvector("action")

    # y is the reward (Batch,)
    y = tensor.fvector("y")

    #####
    # MLP
    #####
    # Concatenate inputs :
    action = action.dimshuffle(0, "x")
    mlp_input = tensor.concatenate((x, action), axis=1)

    # Bricks
    l = args.layers
    activations = []
    # first layer dimension
    dims = [6 * 9 + 1]

    # every hidden layer dimension and activation function
    for _ in range(l):
        activations.append(Rectifier())
        dims.append(args.units_per_layer)
    # last layer dimension
    dims[-1] = 1

    mlp = MLP(activations=activations, dims=dims)

    y_hat = mlp.apply(mlp_input)
    y_hat.name = "y_hat"
    cost = SquaredError().apply(y, y_hat)

    cost.name = "mean_squared_error"

    mlp.weights_init = initialization.IsotropicGaussian(0.1)
    mlp.biases_init = initialization.Constant(0)
    mlp.initialize()

    return cost
