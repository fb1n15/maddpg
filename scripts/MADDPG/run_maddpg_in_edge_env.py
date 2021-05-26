import os
import time

import numpy as np
import pandas
import pandas as pd

from scripts.MADDPG.maddpg import MADDPG
from scripts.MADDPG.buffer import MultiAgentReplayBuffer

# from scripts.MADDPG_original.maddpg import MADDPG
# from scripts.MADDPG_original.buffer import MultiAgentReplayBuffer
from make_env import make_env
from scripts.MADDPG.edge_env import EdgeEnv

pandas.set_option('display.max_columns', None)  # display all columns


def obs_list_to_state_vector(observation):
    """convert several ndarrays to one ndarray"""
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


if __name__ == '__main__':
    debug = False
    # debug = True
    evaluate = False
    # evaluate = True
    # scenario = 'simple'
    scenario = 'edge_cloud'
    # print something every 500 games
    PRINT_INTERVAL = 500
    N_GAMES = 80000
    print(f"Run {N_GAMES} episodes in total")
    # the game do not have a terminal state so we set a max steps for each episode
    MAX_STEPS = 20
    total_steps_cntr = 0
    score_history = []
    sw_history_om = []
    best_score = 10000  # save the model if the score > -10
    # parameters of fog nodes
    h_r = 5
    l_r = 3
    l_c = 2
    n_c = 2.5
    h_c = 3
    avg_resource_capacity = {0: [h_r, h_r, h_r]}
    avg_unit_cost = {0: [l_c, l_c, l_c]}

    env = EdgeEnv(avg_resource_capacity, avg_unit_cost, n_nodes=1,
        n_timesteps=10, n_tasks=500, max_steps=MAX_STEPS,
        n_actions=2, p_high_value_tasks=0.2)
    n_agents = env.n_nodes
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
        # print(env.observation_space[i])
        # exit()
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = env.n_actions
    # print(env.action_space[0])
    # exit()
    # print(env.action_space[0].shape[0])
    # exit()

    # print(f"actor_dims = {actor_dims}")
    # print(f"critic_dims = {critic_dims}")
    print(f"number of agents = {n_agents}")
    print(f"number of actions = {n_actions}")
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
        fc1=64, fc2=64, alpha=0.01, beta=0.01, scenario=scenario,
        chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(int(1e6), critic_dims, actor_dims,
        n_actions, n_agents, batch_size=1024)

    if evaluate:
        maddpg_agents.load_checkpoint()

    avg_sw_df = pd.DataFrame(columns=['episode_ID', 'avg_sw'])

    if debug:
        env.verbose = True  # print the details of process
        N_GAMES = 3
    else:
        env.verbose = False

    for i in range(N_GAMES):
        # for i in range(1):

        obs, om_sw = env.reset()
        if env.verbose:
            print("df_tasks:")
            print(env.df_tasks.head(20))
            print(env.df_nodes)
            # exit()

        score = 0
        done = False
        episode_step_cntr = 0
        node_0_actions = []
        while not done:
            if evaluate:
                env.render()
                # time.sleep(0.1)  # to slow down the action for the video
            actions_probs = maddpg_agents.choose_action(obs)
            # choose the action according to the probabilities
            actions = []
            for actions_prob in actions_probs:
                s = sum(actions_prob)
                p = [i / s for i in actions_prob]
                # a = np.random.choice(n_actions, 1, p=action)
                action = np.random.choice(n_actions, 1, p=p)
                actions.append(action[0])  # action in {1,2,...,10}

            node_0_actions.append(actions[0])
            # the actions are greater than one because of noises
            # actions = np.concatenate(actions)

            # print(f"actions_probs = {actions_probs}")
            # print(f"actions = {actions}")
            # exit()
            obs_, reward, done, sw_increase = env.step(actions)
            reward = reward * n_agents
            # print(total_steps_cntr)
            # print(f"sw_increase = {reward}")

            if episode_step_cntr >= MAX_STEPS - 1:
                done = True
            else:
                state = obs_list_to_state_vector(obs)
                state_ = obs_list_to_state_vector(obs_)

                memory.store_transition(obs, state, actions_probs, reward, obs_,
                    state_, done)

                # print(f"store transition:")
                # print(f"current observation = {obs}")
                # print(f"next observation = {obs_}")
                # print(f"current state = {state}")
                # print(f"next state = {state_}")
                # print(f"actions = {actions_probs}")
                # print(f"reward = {reward}")
                # exit()

                # do not learn when evaluate, learn every 100 steps
                if total_steps_cntr % 100 == 0 and not evaluate:
                    maddpg_agents.learn(memory)

                # set the current state to new state
                obs = obs_

            score += sw_increase
            total_steps_cntr += 1
            episode_step_cntr += 1

        node_0_actions_df = pandas.DataFrame(node_0_actions,
            columns=['action_of_node_0'])
        # print(f"social welfare of episode {i} = {score}")
        # print(f"social welfare (achieved by OM) = {om_sw}")
        sw_history_om.append(om_sw)
        score_history.append(score)
        # average score of previous 100 games
        avg_score = np.mean(score_history[-100:])
        avg_sw_om = np.mean(sw_history_om[-100:])

        # print("score_history")
        # print(score_history)
        # print("avg_score")
        # print(avg_score)
        if env.verbose:
            print('episode', i,
                'social welfare by RL {:.1f}'.format(score))
            print('episode', i,
                'social welfare by OM {:.1f}'.format(om_sw))

        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i,
                'average social welfare by RL {:.1f}'.format(avg_score))
            print('episode', i,
                'average social welfare by OM {:.1f}'.format(avg_sw_om))
            # print actions every * episodes
            # print("actions:")
            # print(actions)
            part_tasks = env.df_tasks['valuation_coefficient']
            part_tasks = part_tasks[0:MAX_STEPS + 1]
            # print("part_tasks:")
            # print(part_tasks)
            # print("actions of node 0:")
            # print(node_0_actions_df)

            # # print actions of node 0 (the high-capacity node)
            # watch_actions_df = pd.DataFrame(
            #     columns=['valuation_coefficient', 'node_0_action'])
            # watch_actions_df['valuation_coefficient'] = part_tasks
            # watch_actions_df['node_0_action'] = node_0_actions_df
            # print(watch_actions_df)
            # # exit()

            df = pd.DataFrame({'episode_ID': [i],
                'avg_sw': [avg_score]})
            avg_sw_df = avg_sw_df.append(df, ignore_index=True)

    if i >= 10000:
        outdir = '/Users/fan/OneDrive - University of Southampton/Chandler\'s Projects/Edge-Cloud-Resource-Allocation-Using-MARL-and-Auction/scripts/MADDPG/tmp'
        outname = 'average_social_welfare.csv'
        fullname = os.path.join(outdir, outname)
        print("... saving to .csv file ...")
        avg_sw_df.to_csv(fullname, index=False)
