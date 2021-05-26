import itertools
import random

import numpy as np


# find the longest sequence of ones (for find the maximum possible usage time)
def largest_row_of_ones(iterable):
    return max(
        (sum(1 for _ in group) for value, group in itertools.groupby(iterable) if value == 1),
        default=0)


class ValueIteration:

    def __init__(self, df_tasks, df_nodes, actions, k_resource, number_of_tasks):
        self.df_tasks = df_tasks  # the dataframe of the tasks
        self.df_nodes = df_nodes  # the dataframe of the fog nodes
        self.actions = actions
        self.k_resource = k_resource
        self.number_of_tasks = number_of_tasks

    def value_iteration(self, actions, states, gamma, theta, N):
        V = {}
        # # initialise the value function of each state
        # non_terminal_states = []
        for s in states:
            V[s] = 0
            # if s[38] != n_tasks-1:
            #     non_terminal_states.append(s)  # find the non-terminal states
        #
        # # # print all the possible states
        # # print(f"all non terminal states: ")
        # #
        # # column_width = 5
        # # for row in non_terminal_states:
        # #     row = "".join(str(np.round(element, 2)).ljust(column_width + 2) for element in row)
        # #     print(row)

        while True:
            delta = 0
            for s in states:
                v = V[s]
                self.bellman_optimality_update(actions, V, s, gamma)
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
            else:
                print(f"delta = {delta}")
        #     for s in states:
        #         q_greedify_policy(actions, V, s, gamma)
        #     return V, pi
        return V

    # update rule for value iteration
    def bellman_optimality_update(self, actions, V, s, gamma):
        """Mutate ``V`` according to the Bellman optimality update equation."""
        # YOUR CODE HERE
        max_value = 0
        for a in actions:
            value = 0
            transitions = self.transitions(s, a, 2)
            for sp, r, p in transitions:
                value += p * (r + gamma * V[tuple(sp)])
            if value > max_value:
                max_value = value

        # update value based on the best_action
        V[tuple(s)] = max_value

    def q_greedify_policy(self, actions, V, pi, s, gamma):

        # find the best action
        max_value = 0
        best_action = 0  # initialise the best action
        for a in actions:
            value = 0
            transitions = self.transitions(s, a, self.number_of_tasks)
            for sp, r, p in transitions:
                value += p * (r + V[tuple(sp)] * gamma)
            if value > max_value:
                max_value = value
                best_action = a
        #     print(f"max value: {max_value}")
        pi[tuple(s)] = best_action

    # get the maximum possible usage time from a state
    def max_usage_time(self, s):
        task_no = s[38]
        start_time = s[1]
        deadline = s[2]
        usage_time = s[3]
        resource_demand = s[4:7]
        remaining_resource = [self.k_resource * (1 - float(x)) for x in s[7:37]]

        l = []  # a list indicating whether this time slot has enough resource
        for t in range(int(start_time), int(deadline) + 1):
            if remaining_resource[(t) * 3] >= resource_demand[0] and \
                    remaining_resource[(t) * 3 + 1] >= resource_demand[1] and \
                    remaining_resource[(t) * 3 + 2] >= resource_demand[2]:
                l.append(1)
            else:
                l.append(0)
        max_usage_time = largest_row_of_ones(l)

        # maximum usage time cannot greater than the required usage time
        if max_usage_time > usage_time:
            max_usage_time = usage_time
        return max_usage_time

    # find the rewards and the next step for action a under state s
    def transitions(self, s, a, number_of_tasks):
        list_outcomes = []  # the lists of possible next states
        s = list(s)
        resource_demand = s[4:7]
        vc = s[0]
        remaining_resource = [self.k_resource * (1 - float(x)) for x in s[7:37]]
        task_no = s[38]
        start_time = s[1]
        deadline = s[2]
        usage_time = s[3]
        # find the maximum usage time can be allocated
        max_usage_time = self.max_usage_time(s)

        for next_task_no in range(number_of_tasks):
            # update the state
            can_allocate = False  # change to true if can allocate this task
            next_cpu_demand = self.df_tasks.loc[next_task_no, 'CPU']
            next_ram_demand = self.df_tasks.loc[next_task_no, 'RAM']
            next_storage_demand = self.df_tasks.loc[next_task_no, 'storage']
            next_vc_next = self.df_tasks.loc[next_task_no, 'valuation_coefficient']
            next_start_time = self.df_tasks.loc[next_task_no, 'start_time']
            next_deadline = self.df_tasks.loc[next_task_no, 'deadline']
            next_usage_time = self.df_tasks.loc[next_task_no, 'usage_time']
            # how many time step has passed until the next task arrives
            time_pass = int(self.df_tasks.loc[next_task_no, 'arrive_time']+1)
            # update next state
            s[4] = next_cpu_demand
            s[5] = next_ram_demand
            s[6] = next_storage_demand
            s[0] = next_vc_next
            s[1] = next_start_time
            s[2] = next_deadline
            s[3] = next_usage_time
            s[37] = a  # change the action element in the state
            s[38] = next_task_no
            # if the action is rejecting the task
            if a == 0:
                r = 0
                sp = s
                can_allocate = True

            # if the task is accepted
            else:
                for t in range(int(start_time), int(deadline - max_usage_time + 2)):
                    usage_time_available = 0
                    for i in range(0, int(max_usage_time)):
                        if remaining_resource[(t + i - 1) * 3] >= resource_demand[0] and \
                                remaining_resource[(t + i - 1) * 3 + 1] >= resource_demand[1] and \
                                remaining_resource[(t + i - 1) * 3 + 2] >= resource_demand[2]:
                            usage_time_available += 1
                    if usage_time_available == int(max_usage_time):
                        can_allocate = True
                        # print(f"can allocate {usage_time_available} time steps")
                        # update the state
                        for i in range(0, int(max_usage_time)):
                            remaining_resource[(t + i - 1) * 3] -= resource_demand[0]
                            s[7 + (t + i - 1) * 3] = "{:.6f}".format(1 - remaining_resource[(t + i - 1) * 3] / self.k_resource)
                            remaining_resource[(t + i - 1) * 3 + 1] -= resource_demand[1]
                            s[7 + (t + i - 1) * 3 + 1] = "{:.6f}".format(1 - remaining_resource[
                                (t + i - 1) * 3 + 1] / self.k_resource)
                            remaining_resource[(t + i - 1) * 3 + 2] -= resource_demand[2]
                            s[7 + (t + i - 1) * 3 + 2] = "{:.6f}".format(1 - remaining_resource[
                                (t + i - 1) * 3 + 2] / self.k_resource)
                        sp = s
                        # update the rewards
                        r = a * vc * max_usage_time / 3
                        break
            # rewards is -1 million if cannot allocate the task
            if not can_allocate:
                sp = s
                r = -1e6

            # move the resource occupation based on the time_pass
            for i in range(3*time_pass):
                sp.pop(7)
                sp.insert(36, "{:.6f}".format(0))

            p = 0.5  # the probability for the arrival of this task
            list_outcomes.append((sp.copy(), r, p))

        return list_outcomes
