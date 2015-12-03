from model import build_model
from utils import parse_args
from environment import Environment


def sample_minibatch(replay_memory):
    # TODO
    pass


# Given an x, returns the action a such that Q(x, a) is maximal
class max_action_Q:

    def __init__(N):
        possible_actions = []
        for i in range(6):
            for l in range(N):
                for d in range(1, 4):
                    possbile_actions.append([i, l, , d])
        self.possible_actions = possible_actions

    def __call__(x):
        max_q = 0
        for a in possible_actions:
            temp = Q(x, a)
            if temp > max_q:
                max_a = a
                max_q = temp
        return max_a

# The Deep Q learning algorithm is the following
if __name__ == "__main__":
    args = parse_args
    M = args.nb_episode  # Number of considered episodes
    T = args.steps_in_episode  # Number of steps in an episode

    # Number of cubies on each edge of the big cube (standard is 3)
    N = args.cube_size

    # The number of permutations from a finished cube to get the initial cube
    rand_nb = args.rand_nb
    eps = args.epsilon  # The probability of taking a random moves
    gamma = args.gamma  # The discount

    lr = args.learning_rate

    # Initialize the Replay_Memory:
    replay_memory = []

    # Initialize Q: function of the Neural Network
    Q, gradient_descent_step = build_model(args)
    max_action_Q = max_action_Q(N)

    for episode in range(M):
        # Initialize a random cube
        env = Environment(N, rand_nb)

        finish = False
        t = 0
        while (t < T) and not finish:
            x_t = env.get_state()
            # Select random action with probability eps
            # Or select action that maximize Q(x_t, a)
            # Execute the action and get reward
            r = numpy.random.uniform(0., 1., 1)
            if r < eps:
                action = env.random_action()
            else:
                action = max_actions_Q(env.get_state())

            reward = env.perform_action(action)

            x_tp1 = env.get_state()

            # Store transition s_t, a_t, r_t, s_t+1 in the Replay_Memory
            replay_memory.append([x_t, action, reward, x_tp1])

            # Sample some mini-batches of the Replay_Memory
            # x_t and x_tp1 are (N+1)D matrices (batch, N dimensions of cube)
            # a is 2 D (batch, actions)
            # r is 1 D (batch,)
            x_t, a, r, x_tp1 = sample_minibatch(replay_memory)

            # Make x 2d (batch, feature)
            x_t = np.reshape(x_t.shape[0], -1)
            x_tp1 = np.reshape(x_tp1.shape[0], -1)

            # Compute y_j for all j in the minibatch
            # y_j = r_j if the state x_t+1 is terminal (r_j = 1)
            # y_j = r_j + gamma * max_actions_Q(Q(x_t_j+1, actions))
            y = np.zeros(x_t.shape[0],)
            for j in range(x_t.shape[0]):
                if r_j == 1.:
                    y[j] = r_j
                else:
                    y[j] = r_j + gamma * max_actions_Q(x_tp1[j])

            # Compute a gradient step on (y - Q(x, a))^2
            cost = gradient_descent_step(x_t, a, y, lr)

            t += 1
