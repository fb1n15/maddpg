import pickle
from multi_agent_sarsa_menu import train_multi_agent_sarsa, execute_multi_agent_sarsa
import sys

# first train the agents
# number_of_steps = 20000
resource_coefficient = float(sys.argv[1])
seed = int(sys.argv[2])

number_of_steps = 20000
time_length = number_of_steps / 40
num_actions = 4
# stop exploration after 5000 steps
epsilons_tuple = (0.2, 0.1, 0.05)
epsilon_steps_tuple = (5000, 4000)
valuation_coefficient_ratio = 10
dict_of_agents_list_v1 = {}
resource_coefficient_original = resource_coefficient
(sw_list, total_value, df_tasks, df_nodes, agents_list
 ) = train_multi_agent_sarsa(alpha=0.02,
                             beta=0.01,
                             epsilon_tuple=epsilons_tuple,
                             epsilon_steps_tuple=epsilon_steps_tuple,
                             num_actions=num_actions,
                             time_length=time_length,
                             total_number_of_steps=number_of_steps,
                             num_fog_nodes=6,
                             resource_coefficient_original=resource_coefficient_original,
                             valuation_coefficient_ratio=valuation_coefficient_ratio,
                             seed=seed,
                             plot_bool=False)

filehandler = open(f"../trained_agents/reverse_auction_v2_seed={seed}_rc={resource_coefficient_original}_agents",
                   'wb')
pickle.dump(agents_list, filehandler)
