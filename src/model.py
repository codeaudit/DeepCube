import logging
from collections import OrderedDict

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


def build_model(args, dtype=floatX):
    logger.info('Building model ...')

    # Variables of the model
    # the rubik's cube stickers
    x = tensor.fmatrix("x")

    # the action taken
    action = tensor.fmatrix("action")

    # y is the reward (Batch,)
    y = tensor.fvector("y")

    #####
    # MLP
    #####
    # Concatenate inputs :
    mlp_input = tensor.concatenate((x, action), axis=1)

    # Bricks
    l = args.layers
    activations = []
    # first layer dimension
    dims = [6 * (args.cube_size ** 2) + 3]

    # every hidden layer dimension and activation function
    for _ in range(l):
        activations.append(Rectifier())
        dims.append(args.units_per_layer)
    # last layer dimension
    dims[-1] = 1

    mlp = MLP(activations=activations, dims=dims)

    y_hat = mlp.apply(mlp_input)

    cost = SquaredError().apply(y.dimshuffle(0, "x"), y_hat)
    cost.name = "mean_squared_error"

    # Initialization
    mlp.weights_init = initialization.IsotropicGaussian(0.1)
    mlp.biases_init = initialization.Constant(0)
    mlp.initialize()

    # Q function
    # Check if the parameters in this function will change through
    # the updates of the gradient descent
    Q = theano.function(inputs=[x, action],
                        outputs=y_hat, allow_input_downcast=True)

    # Cost, gradient and learning rate
    lr = tensor.scalar('lr')
    params = ComputationGraph(cost).parameters
    gradients = tensor.grad(cost, params)
    updates = OrderedDict((p, p - lr * g) for p, g in zip(params, gradients))

    # Function to call to perfom a gradient descent on (y - Q)^2
    gradient_descent_step = theano.function(
        [x, action, y, lr], cost, updates=updates, allow_input_downcast=True)

    return Q, gradient_descent_step, params
