"""
Plot the average SW when trainning
"""
import csv
import json
import pickle
from multi_agent_sarsa import train_multi_agent_sarsa, \
    execute_multi_agent_sarsa
import sys

from other_functions import set_parameters

seed = int(sys.argv[1])
resource_coefficient = float(sys.argv[2])
n_tasks = int(sys.argv[3])
alpha = float(sys.argv[4])
beta = float(sys.argv[5])
auction_type = sys.argv[6]
num_actions = 10
num_trials = 1
verbose = False  # do not print the details of training
# number_of_steps = 10000
# alpha = 0.01
# beta = 0.02
# auction_type = "second-price"
# seed = 0
# resource_coefficient_original = 0.3
# verbose = True  # print the details of training
# time_length = int(number_of_steps / 10)
# # stop exploration after 5000 steps
# epsilons_tuple = (0.3, 0.2, 0.1)
# epsilon_steps_tuple = (
#     int(number_of_steps / 3), int(number_of_steps / 3),
#     int(number_of_steps / 4))
# valuation_coefficient_ratio = 10
# p_high_value_tasks = 0.1
# resource_ratio = 3
# dict_of_agents_list_v1 = {}

# set parameters
(seed, mipgap, n_tasks, n_time, n_nodes, resource_coefficient,
high_value_slackness,
low_value_slackness, valuation_ratio, resource_ratio, p_high_value_task,
avg_resource_capacity,
avg_unit_cost, epsilons_tuple, epsilon_steps_tuple,
auction_type) = set_parameters(seed=seed, n_tasks=n_tasks,
    auction_type=auction_type)

(_, _, _, _, agents_list, _) = train_multi_agent_sarsa(avg_resource_capacity,
    avg_unit_cost, alpha=alpha, beta=beta,
    epsilon_tuple=epsilons_tuple,
    epsilon_steps_tuple=epsilon_steps_tuple,
    high_value_proportion=p_high_value_task,
    num_actions=num_actions,
    time_length=n_time,
    total_number_of_steps=n_tasks,
    num_fog_nodes=6,
    resource_coefficient_original=resource_coefficient,
    valuation_coefficient_ratio=valuation_ratio,
    resource_ratio=resource_ratio,
    seed=seed, verbose=verbose,
    plot_bool=True, auction_type=auction_type)

sw_list, total_value, df_tasks_2, df_nodes, agents_list, allocation_scheme = \
    execute_multi_agent_sarsa(avg_resource_capacity, avg_unit_cost,
        num_actions=num_actions,
        time_length=n_time,
        high_value_proportion=p_high_value_task,
        total_number_of_steps=10000,
        num_fog_nodes=6,
        valuation_coefficient_ratio=valuation_ratio,
        number_of_runs=num_trials, plot_bool=True,
        bool_decay=True,
        resource_ratio=resource_ratio,
        agents_list=agents_list, training_seed=seed,
        verbose=False, auction_type=auction_type)
# filehandler = open(
#     f"../trained_agents/reverse_auction_v1_seed={seed}_rc={resource_coefficient}_agents",
#     'wb')
# pickle.dump(agents_list, filehandler)


# for grid search
print(f'seed={seed}')
print(f"number_of_tasks={n_tasks}")
print(f"alpha={alpha}")
print(f"beta={beta}")
print(f"auction_type={auction_type}")
print(f"total value of tasks = {total_value}")
social_welfare = sw_list[-1]
print(f"total social welfare = {social_welfare}")
result = [seed, n_tasks, alpha, beta, auction_type, social_welfare,
    total_value]

# save the result to a file
with open(f'../simulation_results/auction_v1_grid_search.csv', 'a') as f:
    wr = csv.writer(f, dialect='excel')
    wr.writerow(result)
