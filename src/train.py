import logging
import os

import numpy as np

import theano

from blocks.algorithms import (Adam, CompositeRule, GradientDescent,
                               Momentum, RMSProp, StepClipping,
                               RemoveNotFinite)
from blocks.extensions import Printing, ProgressBar, SimpleExtension, FinishAfter
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.saveload import Load
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import WEIGHT
from blocks.serialization import secure_dump

from generate_predictions import predictions_to_file

floatX = theano.config.floatX
logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def learning_algorithm(args):
    name = args.algorithm
    learning_rate = float(args.learning_rate)
    momentum = args.momentum
    clipping_threshold = args.clipping
    if name == 'adam':
        clipping = StepClipping(threshold=np.cast[floatX](clipping_threshold))
        adam = Adam(learning_rate=learning_rate)
        step_rule = CompositeRule([adam, clipping])
    elif name == 'rms_prop':
        clipping = StepClipping(threshold=np.cast[floatX](clipping_threshold))
        rms_prop = RMSProp(learning_rate=learning_rate)
        rm_non_finite = RemoveNotFinite()
        step_rule = CompositeRule([clipping, rms_prop, rm_non_finite])
    else:
        clipping = StepClipping(threshold=np.cast[floatX](clipping_threshold))
        sgd_momentum = Momentum(learning_rate=learning_rate, momentum=momentum)
        rm_non_finite = RemoveNotFinite()
        step_rule = CompositeRule([clipping, sgd_momentum, rm_non_finite])
    return step_rule


def train_model(cost, stream, args):
    step_rule = learning_algorithm(args)
    cg = ComputationGraph(cost)

    # Dropout
    weights = VariableFilter(roles=[WEIGHT])(cg.parameters)
    cg = apply_dropout(cg, weights, 0.5)

    logger.info(cg.parameters)

    # Define algorithm
    algorithm = GradientDescent(cost=cost, step_rule=step_rule,
                                parameters=cg.parameters)

    # Extensions to be added
    extensions = []

    # # Load from a dumped model
    if args.load_path is not None:
        extensions.append(Load(args.load_path))

    if args.generate is not None:
        extensions.extend([DataStreamMonitoring([y_hat],
                                                streams[args.generate],
                                                prefix=args.generate,
                                                before_first_epoch=True),
                           FinishAfter(after_n_batches=1)])
        # if args.generate != "test":
        #     extensions.append(DataStreamMonitoring([true_values],
        #                                            streams[args.genreate],
        #                                            prefix=args.generate,
        #                                            before_first_epoch=True))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Printing
    extensions.append(ProgressBar())
    extensions.append(Printing())

    main_loop = MainLoop(
        model=Model(cost),
        data_stream=stream,
        algorithm=algorithm,
        extensions=extensions
    )

    # This is where the magic happens!
    main_loop.run()
