"""
Plot the average SW when trainning
"""

import pickle
from multi_agent_sarsa import train_multi_agent_sarsa, \
    execute_multi_agent_sarsa
import sys

resource_coefficient = float(sys.argv[1])
seed = int(sys.argv[2])
resource_coefficient
auction_type = "second-price"
# seed = 0
# resource_coefficient_original = 0.3
verbose = False  # do not print the details of training
# verbose = True  # print the details of training
number_of_steps = 10000
time_length = int(number_of_steps / 10)
num_actions = 10
# stop exploration after 5000 steps
epsilons_tuple = (0.3, 0.2, 0.1)
epsilon_steps_tuple = (
    int(number_of_steps / 3), int(number_of_steps / 3),
    int(number_of_steps / 4))
valuation_coefficient_ratio = 10
high_value_proportion = 0.1
resource_ratio = 3
dict_of_agents_list_v1 = {}
(sw_list, total_value, df_tasks, df_nodes, agents_list,
allocation_scheme) = train_multi_agent_sarsa(alpha=0.02,
    beta=0.02,
    epsilon_tuple=epsilons_tuple,
    epsilon_steps_tuple=epsilon_steps_tuple,
    high_value_proportion=high_value_proportion,
    num_actions=num_actions,
    time_length=time_length,
    total_number_of_steps=number_of_steps,
    num_fog_nodes=6,
    resource_coefficient_original=resource_coefficient,
    valuation_coefficient_ratio=valuation_coefficient_ratio,
    resource_ratio=resource_ratio,
    seed=seed, verbose=verbose,
    plot_bool=True, auction_type=auction_type)

# filehandler = open(
#     f"../trained_agents/reverse_auction_v1_seed={seed}_rc={resource_coefficient}_agents",
#     'wb')
# pickle.dump(agents_list, filehandler)
# print(allocation_scheme)

print(f"total value of tasks = {total_value}")
social_welfare = sw_list[-1]
print(f"total social welfare = {social_welfare}")
