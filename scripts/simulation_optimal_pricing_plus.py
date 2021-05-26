"""Different prices for each type of resource at different fog nodes"""

import sys
import timeit

import numpy as np
import pandas as pd
from numpy.core.multiarray import ndarray

from generate_simulation_data import generate_synthetic_data_edge_cloud
from optimal_pricing_hill_climbing import optimal_pricing

pd.set_option("display.max_rows", 50, "display.max_columns", 50)

# set the parameters
n_nodes = 6

index = 1
i = int(sys.argv[index])
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
n_iterations = int(sys.argv[index])
index += 1
step_size_para = float(sys.argv[index])
index += 1

print("seed: ", i)
seed = i

resource_coefficient = resource_coefficient_original * number_of_tasks / time_length
print("resource coefficient: ", resource_coefficient)

# generate synthetic data for the simulation
df_tasks, df_nodes, n_time, n_tasks, n_nodes = \
    generate_synthetic_data_edge_cloud(n_tasks=number_of_tasks,
                                       n_time=time_length, seed=seed,
                                       n_nodes=n_nodes,
                                       p_high_value_tasks=high_value_proportion,
                                       high_value_slackness_lower_limit=high_value_slackness,
                                       high_value_slackness_upper_limit=high_value_slackness + 2,
                                       low_value_slackness_lower_limit=low_value_slackness,
                                       low_value_slackness_upper_limit=low_value_slackness + 2,
                                       resource_demand_high=resource_ratio,
                                       vc_ratio=valuation_coefficient_ratio,
                                       k_resource=resource_coefficient)
df_tasks = df_tasks.rename(columns={"storage": "DISK"})
df_nodes = df_nodes.rename(
    columns={"storage": "DISK", "storage_cost": "DISK_cost"})

print(f"low value slackness = {low_value_slackness}")
print(f"high value slackness = {high_value_slackness}")
print("df_tasks:")
print(df_tasks.head())
print("df_nodes:")
print(df_nodes.head())

# set price range
column = df_tasks['valuation_coefficient']
price_upper_value = column.max()
price_lower_value = 0

start_time = timeit.default_timer()
social_welfare, number_of_allocated_tasks, optimal_phi, allocation_scheme = \
    optimal_pricing(df_tasks, df_nodes, n_time, n_tasks, n_nodes,
                    n_iterations=n_iterations,
                    price_upper_value=price_upper_value,
                    price_lower_value=price_lower_value,
                    step_size_para=step_size_para)
solve_time = timeit.default_timer() - start_time
print("Optimal pricing is solved in {}s".format(solve_time))
print("social welfare:", social_welfare)
print("number of allocated tasks:", number_of_allocated_tasks)
print(f"optimal_phi = {optimal_phi}")
# print the total value of tasks
total_value = 0
for i in range(number_of_tasks):
    total_value += (df_tasks.loc[i, "valuation_coefficient"] *
                    df_tasks.loc[i, "usage_time"])
print(f"total value of tasks = {total_value}")
# print("allocation scheme")
# print(mat_time_allocated)
# print("start time of tasks")
# print(mat_start_time)


# save the result to a file
# open a file to append
simulation_result = np.array(
    [seed, resource_coefficient_original, social_welfare,
     number_of_allocated_tasks, step_size_para, n_iterations, solve_time])  # type: ndarray
with open("../simulation_results/optimalPricingPlusResults.csv", "ab") as f:
    np.savetxt(f, [simulation_result], delimiter=',', fmt='%.6e')
