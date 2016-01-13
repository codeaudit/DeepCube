#!/usr/bin/env python2
import sys
import numpy as np
import cPickle

from cube import Cube, reverse_action
from environment import Environment
from model import build_model
from serialization import secure_dump
from utils import parse_args, _permutation


def sample_minibatch(replay_memory, minibatch_size, N):
    """
    returns [x_t, a, r, x_tp1] where
    x_t and x_tp1 are 4D (batch, 3D of the cube)
    a is 2D (batch, actions)
    r is 1D (batch,)
    """

    l = len(replay_memory)
    perm = _permutation(l)
    minibatch_indices = perm[:minibatch_size]

    x_t = np.zeros((minibatch_size, 6, N, N)).astype(np.int8)
    a = np.zeros((minibatch_size, 3)).astype(np.int8)
    r = np.zeros((minibatch_size,))
    x_tp1 = np.zeros((minibatch_size, 6, N, N)).astype(np.int8)

    for i, minibatch_index in enumerate(minibatch_indices):
        example = replay_memory[minibatch_index]
        x_t[i, :, :, :] = example[0]
        a[i, :] = example[1]
        r[i] = example[2]
        x_tp1[i, :, :, :] = example[3]

    return [x_t, a, r, x_tp1]


# Given an x, returns the action a such that Q(x, a) is maximal
class max_action_Q(object):

    def __init__(self, N, Q):
        possible_actions = []
        for i in range(6):
            for l in range(N):
                for d in range(1, 4):
                    # Carefull: the action value are shifted
                    # for the lookupTable
                    possible_actions.append(np.array([i,
                                                      l + 6,
                                                      d + 6 + N - 1])[None, :])
        self.possible_actions = possible_actions
        self.Q = Q

    def __call__(self, x):
        flag_first = True
        for a in self.possible_actions:
            temp = self.Q(x, a)
            print temp
            if flag_first:
                flag_first = False
                max_q = temp
                max_a = a
            if temp > max_q:
                max_a = a
                max_q = temp
        return max_a, max_q


def fill_replay_memory(replay_memory, max_replay_memory, episode):
    if max_replay_memory > len(replay_memory):
        replay_memory.append(episode)
    else:
        i = np.random.randint(0, max_replay_memory)
        replay_memory[i] = episode
    return replay_memory


# The Deep Q learning algorithm is the following
if __name__ == "__main__":
    args = parse_args()
    M = args.nb_episode  # Number of considered episodes
    T = args.steps_in_episode  # Number of steps in an episode

    # Number of cubies on each edge of the big cube (standard is 3)
    N = args.cube_size

    # The number of permutations from a finished cube to get the initial cube
    rand_nb = args.rand_nb
    eps = args.epsilon  # The probability of taking a random moves
    gamma = args.gamma  # The discount

    lr = args.learning_rate

    mb_size = args.mini_batch_size  # The minibatch size

    # Initialize the Replay_Memory:
    replay_memory = []

    # Initialize Q: function of the Neural Network
    Q, gradient_descent_step, params = build_model(args)
    max_action = max_action_Q(N, Q)

    for episode in range(args.max_replay_memory):
        # Initialize a random cube
        env = Environment(N)

        moves = env.suffle(rand_nb=rand_nb)

        for i, move in enumerate(reversed(moves)):
            r_move = reverse_action(move)

            # Make the move
            if i == 0:
                x_t = np.copy(env.get_state())
                y_t = np.copy(env.get_state())
            else:
                x_t = np.copy(x_tp1)
                y_t = np.copy(x_tp1)
            cube = Cube(stickers=y_t)
            cube.move(r_move[0], r_move[1], r_move[2])
            x_tp1 = np.copy(cube.stickers)

            # Compute reward
            reward = args.gamma ** (len(moves) - i - 1)

            # Shift action
            r_move[1] += 6
            r_move[2] += 6 + N -1

            fill_replay_memory(replay_memory,
                               args.max_replay_memory,
                               [x_t, r_move, reward, x_tp1])

    # Printing
    current_episode_century = 0
    count = 0
    for episode in range(M):
        # Initialize a random cube
        env = Environment(N)

        moves = env.suffle(rand_nb=rand_nb)

        finish_episode = False
        t = 0
        while (t < T) and not finish_episode:
            x_t = np.copy(env.get_state())

            stickers = env.get_state()
            stickers = stickers.flatten()
            stickers = stickers[None, :]
            action, max_q = max_action(stickers)
            action1 = action[0, :]

            # Shift back the action
            action1[1] += -6
            action1[2] += -6 - N + 1

            reward = env.perform_action(action1)
            if reward:
                finish_episode = True

            # Printing
            if reward:
                if current_episode_century == (episode / 100):
                    count += 1
                else:
                    count = 1
                    current_episode_century = (episode / 100)
                    print ""
                sys.stdout.write("%d : score = %d\r" % (episode / 100, count))
                sys.stdout.flush()

            # If the replay_memory is not big enough to take a minibatch
            if len(replay_memory) < mb_size:
                continue

            # Sample some mini-batches of the Replay_Memory
            # x_t and x_tp1 are 4D matrices (batch, 3D of the cube)
            # a is 2 D (batch, actions)
            # r is 1 D (batch,)
            [x_t, a, r, x_tp1] = sample_minibatch(replay_memory, mb_size, N)
            # Make x 2d (batch, flattened cube)
            x_t = x_t.reshape(x_t.shape[0], 6 * (N ** 2))
            x_tp1 = x_tp1.reshape(x_tp1.shape[0], 6 * (N ** 2))

            # Compute y_j for all j in the minibatch
            # y_j = r_j if the state x_t+1 is terminal (r_j = 1)
            # y_j = r_j + gamma * max_actions_Q(Q(x_t_j+1, actions))
            y = np.zeros(mb_size,)
            for j in range(mb_size):
                if r[j] == 1.:
                    y[j] = r[j]
                else:
                    _, max_q = max_action(x_tp1[j, :][None, :])
                    y[j] = r[j] + gamma * max_q

            # Compute a gradient step on (y - Q(x, a))^2
            cost = gradient_descent_step(x_t, a, y, lr)

            t += 1

    if args.save_path is not None:
        secure_dump(params, args.save_path)
