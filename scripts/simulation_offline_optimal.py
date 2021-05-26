import sys
import timeit

import numpy as np

from generate_simulation_data import generate_synthetic_data_edge_cloud
from offline_optimal_m import offline_optimal

# set the parameters
n_nodes = 6
mipgap = 0.15
# get the parameters
index = 1
i = int(sys.argv[index]); index += 1
number_of_tasks = int(sys.argv[index]); index += 1
resource_coefficient_original = float(sys.argv[index]); index += 1
high_value_slackness = int(sys.argv[index]); index += 1
low_value_slackness = int(sys.argv[index]); index += 1
high_value_proportion = float(sys.argv[index]); index += 1
valuation_coefficient_ratio = float(sys.argv[index]); index += 1
resource_ratio = float(sys.argv[index]); index += 1
time_length = int(sys.argv[index]); index += 1

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
                                       high_value_slackness_upper_limit=high_value_slackness+2,
                                       low_value_slackness_lower_limit=low_value_slackness,
                                       low_value_slackness_upper_limit=low_value_slackness+2,
                                       resource_demand_high=resource_ratio,
                                       vc_ratio=valuation_coefficient_ratio,
                                       k_resource=resource_coefficient)
df_tasks = df_tasks.rename(columns={"storage": "DISK"})
df_nodes = df_nodes.rename(columns={"storage": "DISK", "storage_cost": "DISK_cost"})
# print("task types:")
# print(df_tasks)

start_time = timeit.default_timer()
social_welfare, social_welfare_solution, number_of_allocated_tasks, allocation_scheme = \
    offline_optimal(df_tasks, df_nodes, n_time, n_tasks, n_nodes, mipgap=mipgap)
solve_time = timeit.default_timer() - start_time
print("Offline optimal is solved in {}s".format(solve_time))
print("social welfare:", social_welfare)
print("number of allocated tasks:", number_of_allocated_tasks)

# save the result to a file
# open a file to append
simulation_result = np.array([seed, resource_coefficient_original, social_welfare,
                              number_of_allocated_tasks, solve_time])
with open("../simulation_results/offlineOptimalResults.csv", "ab") as f:
    np.savetxt(f, [simulation_result], delimiter=',', fmt='%.6e')

# # transfer the allocation scheme to a matrix
# mat_scheme = np.zeros([n_tasks, n_time], dtype=int)
# for n in range(n_tasks):
#     for t in range(n_time):
#         mat_scheme[n, t] = sum([allocation_scheme[n, p, t].solution_value for p in range(n_nodes)])
#
# print('Allocation scheme:')
# print(mat_scheme)


# class TestSimulationResults(unittest.TestCase):
#
#     def test_social_welfare1(self):
#         social_welfare_expected = 0.
#         for n in range(n_tasks):
#             for p in range(n_nodes):
#                 for t in range(n_time):
#                     social_welfare_expected += df_tasks.loc[n, "valuation_coefficient"] \
#                                                * z[n, p, t].solution_value \
#                                                - z[n, p, t].solution_value \
#                                                * (df_tasks.loc[n, 'CPU'] * df_nodes.loc[
#                         p, 'CPU_cost']
#                                                   + df_tasks.loc[n, 'RAM'] * df_nodes.loc[
#                                                       p, 'RAM_cost']
#                                                   + df_tasks.loc[n, 'storage'] * df_nodes.loc[
#                                                       p, 'storage_cost'])
#         self.assertAlmostEqual(social_welfare, social_welfare_expected, places=4,
#                                msg='social welfare is inconsistent')
#
#     def test_computational_resource_constraints(self):
#         for p in range(n_nodes):
#             for t in range(n_time):
#                 cpu_total = 0
#                 ram_total = 0
#                 storage_total = 0
#                 for n in range(n_tasks):
#                     cpu_total += z[n, p, t].solution_value * df_tasks.loc[n, 'CPU']
#                     ram_total += z[n, p, t].solution_value * df_tasks.loc[n, 'RAM']
#                     storage_total += z[n, p, t].solution_value * df_tasks.loc[n, 'storage']
#                 # print('cpu resource allocated:', cpu_total)
#                 self.assertLessEqual(cpu_total, df_nodes.loc[p, "CPU"],
#                                      msg='exceed CPU capacity!')
#                 self.assertLessEqual(ram_total, df_nodes.loc[p, 'RAM'],
#                                      msg='exceed RAM capacity!')
#                 self.assertLessEqual(storage_total, df_nodes.loc[p, 'storage'],
#                                      msg='exceed storage capacity!')
#
#     # VM is created between the time window of task n
#     def test_time_interval(self):
#         for n in range(n_tasks):
#             for p in range(n_nodes):
#                 for t in range(n_time):
#                     if t < df_tasks.loc[n, 'start_time'] or t > df_tasks.loc[n, 'deadline']:
#                         self.assertAlmostEqual(z[n, p, t].solution_value, 0., places=4,
#                                                msg='allocation is out of the time interval!')
#
#     # one task is at most allocated to one fog node
#     def test_one_fog_node(self):
#         for n in range(n_tasks):
#             for t in range(n_time):
#                 result = 0
#                 expected = 1
#                 for p in range(n_nodes):
#                     result += z[n, p, t].solution_value
#                 self.assertLessEqual(result, expected,
#                                      msg='one task is allocated to more than one fog node')
#
#     # tasks are non-preemptive
#     def test_non_preemptive(self):
#         for n in range(n_tasks):
#             zero_to_one = 0
#             for t in range(n_time-1):
#                 if mat_scheme[n, t+1] - mat_scheme[n, t] ==1:
#                     zero_to_one += 1
#             # print('times that change from zero to one', zero_to_one)
#             self.assertLessEqual(zero_to_one, 1, msg="the task should be non-preemptive")
#
#
# if __name__ == '__main__':
#     unittest.main(argv=[sys.argv[0]])
