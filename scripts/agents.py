import numpy as np
from collections import defaultdict


def argmax(values):
    """
    Takes in a list of values and returns the index of the item
    with the highest value. Breaks ties randomly.
    (Because np.argmax always return the first index of highest value)
    returns: int - the index of the highest value in values
    """
    top_value = float("-inf")
    ties = []

    for i in range(len(values)):
        # if a value in values is greater than
        # ~the highest value update top and reset ties to zero
        if values[i] > top_value:
            top_value = values[i]
            ties = []
        # if a value is equal to top value add the index to ties
        if values[i] == top_value:
            ties.append(i)
    # print(f"q_values={values}")
    # print(f"ties={ties}")

    # return a random selection from ties.
    return np.random.choice(ties)


class BaseAgent:
    """ Base class for a RL agent.
    Different state-value / state-action value algorithms overwrite run_episode and update functions
    Approximation agents overwrite the q_value function representation from dictionary enumeration to feature vec approximation using
    get_q_value and reset
    """

    def __init__(self, mdp, discount=1, epsilon=0.1, alpha=0.5):
        """
        Args
            mdp             -- class with markov decision process providing the following function calls:
                                - get_possible_actions
                                - get_state_reward_transition
            run_episode_fn  -- function specifying the sequence of agent-environment interactions and updates
                                for the specific algorithm (e.g. Sarsa, Q-learning). This will be run during training by
                                calling agent.run_episode()
            discount        -- float in [0, 1]; discount for state / state-action value calculation (gamma in Sutton&Barto)
            epsilon         -- float in [0, 1]; spec for epsilon-greedy algorithms % exploration
            alpha           -- float in [0, 1]; learning step size parameter
        """
        self.mdp = mdp
        self.discount = discount
        self.epsilon = epsilon
        self.alpha = alpha

        # initialize q_values
        self.reset()

    def get_action(self, state, verbose=False):
        """ e-greedy policy """
        rand = np.random.rand()
        actions = self.mdp.get_possible_actions()
        if rand < self.epsilon:
            random_action = actions[np.random.choice(len(actions))]
            if verbose:
                print(f"pick a random action: action{random_action}")
            return random_action
        else:
            best_action = self.compute_best_action(state, verbose=verbose)
            if verbose:
                print(f'choose the "best action": action{best_action}')
            return best_action

    def get_q_value(self, state, action):
        return self.q_values[(state, action)]

    def get_value(self, state):  # get overwritten
        return self.compute_value(state)

    def compute_best_action(self, state, verbose=False):
        # several actions may have the 'best' q_value; choose among them randomly
        legal_actions = self.mdp.get_possible_actions()
        if legal_actions[0] is None:
            return None
        q_values = [self.get_q_value(state, a) for a in legal_actions]
        if verbose:
            print(f"list of q values: {q_values}")
        best_action = argmax(q_values)
        return best_action

    def compute_q_value(self, state, action):
        next_state, reward = self.mdp.get_state_reward_transition(state, action)
        return reward + self.discount * self.get_value(next_state)

    def compute_value(self, state):
        best_action = self.compute_best_action(state)
        if best_action is None:
            return 0
        else:
            return self.get_q_value(state, best_action)

    def update(self, state, action, reward, next_state, next_action):
        """ Update to the q_values to be overwriten per the specific algorithm in sync with the run_episode function """
        raise NotImplementedError

    def reset(self):
        self.q_values = defaultdict(float)
        self.num_updates = 0


# --------------------
# Q-learning agent
# --------------------

def run_qlearning_episode(mdp, agent):
    """ Execute the Q-learning off-policy algorithm per Section 6.5.
    This is paired to an agent for the agent.run_episode() call.
    """
    # record episode path and actions
    states_visited = []
    actions_performed = []
    episode_rewards = 0

    # initialize S
    state = mdp.reset_state()
    states_visited.append(state)

    # loop for each step
    while not mdp.is_goal(state):
        # choose A from S using policy derived from Q
        action = agent.get_action(state)

        # take action A, observe R, S'
        next_state, reward = mdp.get_state_reward_transition(state, action)

        # update agent
        agent.update(state, action, reward, next_state)
        # update state
        state = next_state

        # record path
        states_visited.append(state)
        actions_performed.append(action)
        episode_rewards += reward

    return states_visited, actions_performed, episode_rewards


class QLearningAgent(BaseAgent):
    def __init__(self, run_episode_fn=run_qlearning_episode, **kwargs):
        super().__init__(run_episode_fn=run_qlearning_episode, **kwargs)

    def update(self, state, action, reward, next_state):
        """ Q learning update to the policy -- eq 6.8 """

        q_t0 = self.get_q_value(state, action)
        q_t1 = self.get_value(next_state)

        # q learning update per eq 6.8 -- greedy policy after the current step
        new_value = q_t0 + self.alpha * (reward + self.discount * q_t1 - q_t0)

        # perform update
        self.q_values[(state, action)] = new_value

        self.num_updates += 1

        return new_value
