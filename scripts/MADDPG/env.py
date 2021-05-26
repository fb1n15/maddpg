from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import copy
from abc import ABC

from tf_agents.specs import BoundedArraySpec
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_spec as tspec
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()


class NetworkEnvironment(py_environment.PyEnvironment, ABC):

    def __init__(self, df_tasks, df_nodes, nr_timestamps,
            forgiveness_factor=30,
            allow_negative_reward=False,
            alpha=1.0, lam=1e2,
            not_verbose=-1):
        """
        Initialization function for the environment.
        Args:

            df_tasks: A dataframe with task information.
            df_nodes: A dataframe with node information.
            nr_timestamps: The number of timestamps.
            allow_negative_reward: Flag for allowing negative rewards for the bad allocation of tasks.
            forgiveness_factor: Tolerance to sequential bad allocation of tasks.
            alpha: Percentage of the total rewards influenced by the prioritisation of high valuation tasks.
            lam: Speed of increase in the rewards generated from the prioritisation of high valuation tasks.
            not_verbose: A boolean as a flag to print information about the node allocation.

        """

        # Set the class variables
        super().__init__()

        self.allocated_tasks = []  # [(task_info, action)]

        self.lam = lam
        self.alpha = alpha
        self.not_verbose = not_verbose
        self.allow_negative_reward = allow_negative_reward
        self.nr_nodes = len(df_nodes)
        self.nr_timestamps = nr_timestamps
        self.nr_tasks = len(df_tasks)

        # initialise the ndarray of idle resources
        self.default_resource_map = np.empty(
            [self.nr_nodes, self.nr_timestamps, 3])

        for node in df_nodes.iterrows():
            self.default_resource_map[node[0]] = [
                [df_nodes.loc[node[0], 'CPU'],
                    df_nodes.loc[node[0], 'RAM'],
                    df_nodes.loc[node[0], 'DISK']]
                for _ in range(nr_timestamps)]

        self.df_tasks = df_tasks
        self.df_nodes = df_nodes

        self.current_task = 0
        self.failed = 0

        self.forgiveness_factor = forgiveness_factor

        self.total_welfare_seen = 0

        self.processed_tasks = 0
        self.action_history = []

        self.allocation_map = dict((node, []) for node in range(
            self.nr_nodes))  # A dict with the shape {node: [(task, start_index, stop_index),]}

        self.resources_map = copy.deepcopy(self.default_resource_map)

        # Define the boundaries of the action space: (task, node, nr timestamps allocated)
        # self._action_spec = array_spec.BoundedArraySpec(
        #     shape=(4,), dtype=np.int32, minimum=-1,
        #     maximum=[self.nr_tasks - 1, self.n_nodes - 1, self.n_timesteps - 1, self.n_timesteps - 1],
        #     name='action')

        # Action shape ( node, start timestamp,  number of timestamps)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.int32, minimum=[0, 0, 1],
            maximum=[self.nr_nodes - 1, self.nr_timestamps - 1,
                self.nr_timestamps - 1],
            name='action')

        # self._action_spec = array_spec.ArraySpec(
        #     shape=(3,), dtype=np.int32,
        #     name='action')

        # Define the boundaries of the observation space
        # observation is the idle resources now?
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.nr_nodes, self.nr_timestamps, 3,), dtype=np.int32,
            minimum=0, name='observation')

        # self._observation_spec = array_spec.BoundedArraySpec(
        #     shape=(1, self.n_nodes, self.n_timesteps), dtype=np.int32, minimum=0, name='observation')

        # also the idle resources?
        self._state = self.generate_initial_state()

        self._episode_ended = False
        # an upper bound of the social welfare
        self.total_possible_reward = sum(
            df_tasks.valuation_coefficient * df_tasks.usage_time)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = self.generate_initial_state()
        self._episode_ended = False
        self.current_task = 0
        self.processed_tasks = 0
        self.failed = 0
        self.total_welfare_seen = 0

        self.allocated_tasks = []

        # Reset the allocation map too
        self.allocation_map = dict((node, []) for node in range(self.nr_nodes))

        # Reset the resource allocation matrix
        self.resources_map = copy.deepcopy(self.default_resource_map)

        return np.array([
            self._state])  # ts.restart(observation=np.array([self._state], dtype=np.int32))

    # Shape of the observation is node x time_step
    def generate_initial_state(self):
        """
        Generate the initial state map.

        Returns:
            An array as the state with default values in shape (nr nodes, nr timestamps, 3). i.e., all the resources are idle
        """
        state = []
        for node in self.df_nodes.iterrows():  # iterate rows as (index, Series)s
            default_value = [node[1]['CPU'], node[1]['RAM'], node[1]['DISK']]

            state.append([copy.deepcopy(default_value) for t in
                range(self.nr_timestamps)])

        return state

    def timestamp_has_enough_resources(self, task, node, timestamp):
        """
        Check if the current timestamp has enough resources to compute a timestamp of task 'task'. (not needed for me)
        Args:
            task: The task to be processed as an integer.
            timestamp: The timestamp for the processing as an integer.
        """

        return self.df_tasks.loc[task, 'CPU'] <= self.resources_map[
            node, timestamp, 0] and \
               self.df_tasks.loc[task, 'RAM'] <= self.resources_map[
                   node, timestamp, 1] and \
               self.df_tasks.loc[task, 'DISK'] <= self.resources_map[
                   node, timestamp, 2]

    def get_available_nodes(self, task=-1):
        """
        Return a list of available nodes for the current task and timestamp.
        Args:
            task: The current task as an integer.

        Returns:
             [(node, start_index, stop_index)]: Returns a dict with available allocation windows for each node.
        """

        available_allocation_windows = []

        if task == -1:
            task = self.current_task

        # Check the available nodes in the start time - deadline range
        start_time = self.df_tasks.loc[task, 'start_time']
        deadline = self.df_tasks.loc[task, 'deadline']

        for node in range(self.nr_nodes):
            start_index = -1

            for timestamp in range(start_time, deadline + 1):

                # If encountered a new available window, set it
                if self.timestamp_has_enough_resources(task, node, timestamp):

                    if start_index == -1:
                        start_index = timestamp
                    continue

                if start_index != -1:
                    available_allocation_windows.append(
                        (node, start_index, timestamp - start_index))
                    start_index = -1

            if start_index != -1:
                available_allocation_windows.append(
                    (node, start_index, deadline - start_index))

        return available_allocation_windows

    def calculate_reward_value(self, action, task):
        """Calculate the social welfare increase after the allocation of the task

        Args:
            action: action taken
            task: the index of the allocated task

        Returns:

        """
        node = action[0]
        length = action[2]

        return length * (self.df_tasks.loc[task, 'valuation_coefficient'] -
                         self.df_tasks.loc[task, 'CPU'] * self.df_nodes.loc[
                             node, 'CPU_cost'] -
                         self.df_tasks.loc[task, 'DISK'] * self.df_nodes.loc[
                             node, 'DISK_cost'] -
                         self.df_tasks.loc[task, 'RAM'] * self.df_nodes.loc[
                             node, 'RAM_cost'])

    def calculate_total_welfare_value(self):
        """
        Calculate the total rewards in the network.

        Returns:
            lsum (float): The total social welfare in the network.
        """
        lsum = 0

        for node, v in self.allocation_map.items():

            for a in v:
                task = a[0]
                lsum += (a[2] - a[1]) * (
                        self.df_tasks.loc[task, 'valuation_coefficient'] -
                        self.df_tasks.loc[task, 'CPU'] * self.df_nodes.loc[
                            node, 'CPU_cost'] -
                        self.df_tasks.loc[task, 'DISK'] * self.df_nodes.loc[
                            node, 'DISK_cost'] -
                        self.df_tasks.loc[task, 'RAM'] * self.df_nodes.loc[
                            node, 'RAM_cost'])
        return lsum

    def process_alloc_map(self, alloc_map):
        """
        Transform the allocation map into an array form.

        Args:
            alloc_map: The current allocation map.

        Returns:
            The current allocation map in an array form.
        """

        state = self._state

        # For every node
        for node_allocation in alloc_map.items():

            # If the node has tasks allocated to him
            if len(node_allocation[1]) > 0:

                # For every task allocated to this node
                for (task, start, allocated) in node_allocation[1]:
                    node = node_allocation[0]

                    task_details = self.df_tasks.iloc[task]

                    # Adjust the current state with new
                    for j in range(start,
                            min(start + allocated - 1, self.nr_timestamps)):
                        state[node][j][0] -= task_details['CPU']
                        state[node][j][1] -= task_details['RAM']
                        state[node][j][2] -= task_details['DISK']

        return state

    def annealing_reward(self, tr, xc):
        """
        Function for computing the joint training and valuation rewards.
        I don't need?

        Args:
            tr: The current training rewards.
            xc: The current task valuation.
        Returns:
            The joint rewards value.
        """
        return tr + 8 * self.alpha * (
                xc * self.processed_tasks / self.total_welfare_seen) * (
                       xc / self.total_possible_reward) ** (
                   np.log(self.lam))

    def get_reward_for_action(self, action):
        """
        Validate the action and return the rewards of the specific action.
        Args:
            action: A tuple with shape [node, timestamp, time length].

        Returns:
            A tuple of shape [rewards, action]. Returns [0, None] if the action is invalid.

        """
        # Generate random allocation
        task = self.current_task
        task_details = self.df_tasks.iloc[self.current_task]

        # Get the dict of valid allocation windows and chose an action
        allocation_windows = self.get_available_nodes(task=task)

        current_no_cost_reward = (self.df_tasks.iloc[self.current_task][
                                      'valuation_coefficient'] *
                                  self.df_tasks.iloc[self.current_task][
                                      'usage_time'])

        if self.not_verbose > 1: print("Alloc window: ", allocation_windows)

        node = action[0]
        time = action[1]
        length = action[2]

        # Accumulate partial rewards for each part of the assignemnt done correctly
        partial_reward = 0
        reward = 0

        # CHECK BOUNDERIES
        if not (0 <= node < self.nr_nodes):  # Nodes
            partial_reward -= 1
        # return 0, None

        partial_reward += 1

        if not (0 <= time < self.nr_timestamps):  # Timestamps
            partial_reward -= 1
            # return 0, None

        partial_reward += 1

        if partial_reward == 2:
            # CHECK RESOURCE USAGE
            up_bound = min(self.nr_timestamps, time + length - 1) - 1
            for i in range(time, up_bound):

                if not self.timestamp_has_enough_resources(task, node, i):
                    partial_reward -= 1
                    break
                    # return 0, None

            partial_reward += 1

            # If node and timestamp values are correct, calculate the real rewards value
            reward = self.calculate_reward_value(action, task)
        else:

            # If the node and timestamp values are not correct, only use the training rewards
            reward = task_details['valuation_coefficient'] / 100 + 1

        if not (0 <= length < self.nr_timestamps):  # Time length
            partial_reward -= 1
            # return 0, None

        partial_reward += 1

        if not (0 <= length + time - 1 < self.nr_timestamps):
            partial_reward -= 1
            # return 0, None

        partial_reward += 1

        # CHECK START TIME AND DEADLINE
        if not (int(task_details['start_time']) <= time <= int(
                task_details['deadline'])):
            partial_reward -= 1
            # return 0, None

        partial_reward += 1

        # Check if the time is in the past
        if time < self.processed_tasks:
            partial_reward -= 1

            # return 0, None

        partial_reward += 1

        # Check if the number of allocated timestamps is correct
        if not length <= task_details['usage_time']:
            partial_reward -= 1
            # return 0, None

        partial_reward += 1

        # If the partial rewards is 0, the return none, otherwise return a fraction of the total possible rewards
        if partial_reward == 0:
            return 0, None

        reward *= (partial_reward / 8)

        # Choose a random action from all possible valid actions
        # action = random.choice(allocation_windows)

        # If the action is not correct return the rewards and no action
        if partial_reward != 8:
            return reward, None

        # Calculate the joint rewards value
        # partial_reward = self.aneal(partial_reward, current_no_cost_reward)

        return partial_reward, action  # rewards, action

    def add_time_constraints(self, state, task):
        """I don't need?

        Args:
            state: The current state to add constraints too. ([n_nodes, n_timesteps, 3])
            task: The index of the current task being processed.

        Returns: The modified state and the old state.

        """

        task_info = self.df_tasks.loc[task]

        start_time = int(task_info.loc['start_time'])
        end_time = int(task_info.loc['deadline'])

        # Replace the resource values in the unavailable timestamps with inf
        for i in range(self.nr_nodes):

            # Make the timestamps previous to the start time invalid
            for j in range(start_time):
                state[i][j] = [-1] * 3

            # Make the timestamps past the deadline invalid
            for j in range(end_time + 1, self.nr_timestamps):
                state[i][j] = [-1] * 3

        return state

    def get_allocated_tasks(self):
        """

        Get the list of allocated tasks so far and their allocation.

        Returns: A list of tasks and their allocation. (shape [(task_info, action)]

        """

        return self.allocated_tasks

    def _step(self, action):
        """
        Step function for the environment.
        Args:
            action: Array of shape Nodes x Timestamp x Timestamp.

        Returns:
            observation (object): next observation?
            rewards (float): rewards of previous actions
            done (boolean): whether to reset the environment
            info (dict): diagnostic information for debugging.

        """

        reward = 0
        action = [int(x) for x in action]

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Check if the current task is the last one received and end the episode if true
        if self.processed_tasks >= self.nr_tasks or self.current_task >= self.nr_tasks:
            self._episode_ended = True

        # If action has a bad shape, do not allocate
        elif len(action) != 3 or not isinstance(action[0],
                int) or not isinstance(action[1], int) or not isinstance(
            action[2], int):
            pass

        else:

            # Add the task total welfare to the total welfare seen
            self.total_welfare_seen += (self.df_tasks.iloc[self.current_task][
                                            'valuation_coefficient'] *
                                        self.df_tasks.iloc[self.current_task][
                                            'usage_time'])

            # Compute the rewards of the current action
            reward, chosen_action = self.get_reward_for_action(action)

            # Add the time evolution
            if self.not_verbose > -1: print("Action:", action,
                "\nChosen action: ", chosen_action, "\nAction rewards:",
                reward)

            if chosen_action is None:
                self.failed += 1
                pass
            else:
                node = chosen_action[0]

                timestamp = chosen_action[1]
                allocated_usage = min(chosen_action[2],
                    self.df_tasks.loc[self.current_task, 'usage_time'])

                allocation = [self.current_task, node, timestamp,
                    allocated_usage]

                # Update the allocation matrix
                self.allocation_map[allocation[1]].append((
                    allocation[0], allocation[2],
                    allocation[2] + allocation[3]))

                # Update the resource matrix
                for i in range(timestamp,
                        min(timestamp + allocated_usage, self.nr_timestamps)):
                    self.resources_map[node, i, 0] -= self.df_tasks.loc[
                        self.current_task, 'CPU']
                    self.resources_map[node, i, 1] -= self.df_tasks.loc[
                        self.current_task, 'RAM']
                    self.resources_map[node, i, 2] -= self.df_tasks.loc[
                        self.current_task, 'DISK']

                self._state = self.process_alloc_map(self.allocation_map)

        # If the episode has ended or all the task have been processed, calculate the rewards and return it
        if self._episode_ended or self.processed_tasks == self.nr_tasks:
            if self.not_verbose > 1:
                print("stop")

            if self.not_verbose > 1: print("Episode ended! Final rewards is: ",
                self._state)

            print(reward)

            return np.array([self._state], dtype=np.int32), reward, True, None

            # return ts.termination(np.array([self._state], dtype=np.int32), rewards=rewards)
        else:
            # Add time constraints
            state_copy = copy.deepcopy(self._state)
            return_state = self.add_time_constraints(state_copy,
                self.current_task)
            # print(self._state, "\n\n", return_state)

            if chosen_action is not None:
                print("advance", action, '->', reward)
                self.allocated_tasks.append(
                    (self.df_tasks.loc[self.current_task], action))

            if self.failed > self.forgiveness_factor or chosen_action is not None:
                self.processed_tasks += 1
                self.current_task += 1
                self.failed = 0

            # Add negative rewards for poor allocations
            if self.allow_negative_reward:
                reward = reward - (self.failed / 100)

            if self.not_verbose > -1:
                print("continue...")
            # return ts.transition(
            #     np.array([self._state], dtype=np.int32), rewards=rewards, discount=1.0)
            # print(action, rewards, self.df_tasks.loc[self.current_task_id].tolist())

            return np.array([return_state], dtype=np.int32), reward, False, None
