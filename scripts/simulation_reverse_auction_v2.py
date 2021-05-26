import sys
import pickle
import timeit
import numpy as np
import unittest
import pandas as pd

from numpy.core.multiarray import ndarray
from optimal_pricing_m import optimal_pricing
from multi_agent_sarsa_menu import train_multi_agent_sarsa, execute_multi_agent_sarsa
from generate_simulation_data import generate_synthetic_data_edge_cloud

pd.set_option("display.max_rows", None, "display.max_columns", None)

# set the parameters
n_nodes = 6

index = 1
seed = int(sys.argv[index])
index += 1
number_of_tasks = int(sys.argv[index])
index += 1
resource_coefficient_original = float(sys.argv[index])
index += 1
high_value_slackness = int(sys.argv[index])
index += 1
low_value_slackness = int(sys.argv[index])
index += 1
high_value_proportion = float(sys.argv[index])
index += 1
valuation_coefficient_ratio = float(sys.argv[index])
index += 1
resource_ratio = float(sys.argv[index])
index += 1
time_length = int(sys.argv[index])
index += 1
num_trials = int(sys.argv[index])
num_actions = 4

filehandler = open(f"../trained_agents/reverse_auction_v2_seed={seed}_rc={resource_coefficient_original}_agents",
                   'rb')
agents_list = pickle.load(filehandler)

start_time = timeit.default_timer()
execute_multi_agent_sarsa(num_actions=num_actions,
                          time_length=time_length,
                          total_number_of_steps=number_of_tasks, num_fog_nodes=6,
                          resource_coefficient_original=resource_coefficient_original,
                          valuation_coefficient_ratio=valuation_coefficient_ratio,
                          number_of_runs=num_trials, plot_bool=False, bool_decay=True,
                          agents_list=agents_list, training_seed=seed)

solve_time = timeit.default_timer() - start_time
print("Online Optimal is solved in {}s".format(solve_time))
# print("social welfare:", social_welfare)
# print("number of allocated tasks:", number_of_allocated_tasks)
# print the total value of tasks
# total_value = 0
# for i in range(number_of_tasks):
#     total_value += (df_tasks.loc[i, "valuation_coefficient"] *
#                     df_tasks.loc[i, "usage_time"])
# print(f"total value of tasks = {total_value}")
# print("allocation scheme")
# print(mat_time_allocated)
# print("start time of tasks")
# print(mat_start_time)


# # save the result to a file
# # open a file to append
# simulation_result = np.array([seed, resource_coefficient_original, social_welfare,
#                               number_of_allocated_tasks, solve_time])  # type: ndarray
# with open("../simulation_results/optimalPricingResults.csv", "ab") as f:
#     np.savetxt(f, [simulation_result], delimiter=',', fmt='%.6e')
