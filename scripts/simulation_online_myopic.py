import sys
import timeit

import numpy as np
import pandas as pd
from numpy.core.multiarray import ndarray

from generate_simulation_data import generate_synthetic_data_edge_cloud
from online_myopic_m import online_myopic

pd.set_option("display.max_rows", None, "display.max_columns", None)

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

print("seed: ", i)
seed = i

resource_coefficient = resource_coefficient_original * number_of_tasks / time_length
print("resource coefficient: ", resource_coefficient)

# generate synthetic data for the simulation
df_tasks, df_nodes, n_time, n_tasks, n_nodes = \
    generate_synthetic_data_edge_cloud(n_tasks=number_of_tasks, n_time=time_length, seed=seed,
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
df_nodes = df_nodes.rename(columns={"storage": "DISK", "storage_cost": "DISK_cost"})

print(f"low value slackness = {low_value_slackness}")
print(f"high value slackness = {high_value_slackness}")
print("df_tasks:")
print(df_tasks.head())
print("df_nodes:")
print(df_nodes.head())

start_time = timeit.default_timer()
social_welfare, number_of_allocated_tasks, allocation_scheme = \
    online_myopic(df_tasks, df_nodes, n_time, n_tasks, n_nodes)
solve_time = timeit.default_timer() - start_time
print("Online Myopic is solved in {}s".format(solve_time))
print("social welfare:", social_welfare)
print("number of allocated tasks:", number_of_allocated_tasks)
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
simulation_result = np.array([seed, resource_coefficient_original, social_welfare,
                              number_of_allocated_tasks, solve_time])  # type: ndarray
with open("../simulation_results/onlineMyopicResults.csv", "ab") as f:
    np.savetxt(f, [simulation_result], delimiter=',', fmt='%.6e')


# class TestSimulationResults(unittest.TestCase):
#
#     def test_social_welfare(self):
#         social_welfare_expected = 0
#         for n in range(n_tasks):
#             number_of_time_steps = np.amax(mat_time_allocated[n])
#             fn = np.where(mat_time_allocated[n] == number_of_time_steps)[0][0]
#             # value of tasks
#             social_welfare_expected += df_tasks.loc[n, 'valuation_coefficient'] \
#                                        * number_of_time_steps
#             # operational cost of tasks
#             operational_cost = (df_nodes.loc[fn, 'CPU_cost'] * df_tasks.loc[n, 'CPU'] +
#                                 df_nodes.loc[fn, 'RAM_cost'] * df_tasks.loc[n, 'RAM'] +
#                                 df_nodes.loc[fn, 'storage_cost'] * df_tasks.loc[n, 'storage']) * \
#                                number_of_time_steps
#             social_welfare_expected -= operational_cost
#         self.assertAlmostEqual(social_welfare, social_welfare_expected, places=4,
#                                msg='social welfare is inconsistent')
#
#     def test_resource_capacity(self):
#         array_resource = np.array(['CPU', 'RAM', 'storage'])  # the array of resource types
#         for resource in array_resource:
#             for fn in range(n_nodes):
#                 for t in range(n_time):
#                     resource_allocated = 0
#                     for n in range(n_tasks):
#                         if t in range(mat_start_time[n, fn],
#                                       mat_start_time[n, fn] + mat_time_allocated[n, fn]):
#                             resource_allocated += df_tasks.loc[n, resource]
#                     # print("t:", t, "fn:", fn, "resource:", resource)
#                     # print(resource_allocated)
#                     self.assertLessEqual(resource_allocated, df_nodes.loc[fn, resource],
#                                          'resource capacity is violated')
#
#     # allocated time slots should satisfy task's time constraints
#     def test_time_constraints(self):
#         for n in range(n_tasks):
#             for fn in range(n_nodes):
#                 if mat_time_allocated[n, fn] > 0:  # if task n is allocated to FN fn
#                     start_time = mat_start_time[n, fn]
#                     finish_time = start_time + mat_time_allocated[n, fn] - 1
#                     self.assertGreaterEqual(start_time, df_tasks.loc[n, 'start_time'],
#                                             'start time out of bound')
#                     self.assertLessEqual(finish_time, df_tasks.loc[n, 'deadline'],
#                                          'finish time out of bound')
#
#     # one task is at most allocated to one fog node
#     def test_one_fog_node(self):
#         for n in range(n_tasks):
#             count = 0
#             for fn in range(n_nodes):
#                 if mat_time_allocated[n, fn] > 0:
#                     count += 1
#             self.assertLessEqual(count, 1, 'one task is allocated to more than one fog node')
#
#     # # only one VM for every task
#     # def test_one_VM(self):
#     #     for n in n_tasks:
#     #         for t in n_time:
#     #             result = 0
#     #             expected = 1
#     #             for p in P:
#     #                 result += z[n, p, t]
#     #             self.assertLessEqual(result, expected, 'more than one VM is allocated for a task!')
#
#
# if __name__ == '__main__':
#     unittest.main(argv=[sys.argv[0]])
