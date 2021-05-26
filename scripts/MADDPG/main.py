import time

import numpy as np
from scripts.MADDPG.maddpg import MADDPG
from scripts.MADDPG.buffer import MultiAgentReplayBuffer
# from scripts.MADDPG_original.maddpg import MADDPG
# from scripts.MADDPG_original.buffer import MultiAgentReplayBuffer
from make_env import make_env


def obs_list_to_state_vector(observation):
    """convert several ndarrays to one ndarray"""
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


if __name__ == '__main__':
    evaluate = False
    # evaluate = True
    # scenario = 'simple'
    scenario = 'simple_adversary'
    # print something every 500 games
    PRINT_INTERVAL = 500
    N_GAMES = 30000
    # the game do not have a terminal state so we set a max steps for each episode
    MAX_STEPS = 25
    total_steps_cntr = 0
    score_history = []
    best_score = -10  # save the model if the score > -10
    env = make_env(scenario)
    n_agents = env.n
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
        # print(env.observation_space[i])
        # exit()
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = env.action_space[0].n
    # print(env.action_space[0])
    # exit()

    # print(env.action_space[0].sample())
    # exit()
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
        fc1=64, fc2=64, alpha=0.01, beta=0.01, scenario=scenario,
        chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(int(1e6), critic_dims, actor_dims,
        n_actions, n_agents, batch_size=1024)

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
    # for i in range(1):
        obs = env.reset()
        # print(f"next observation = {obs}")
        # exit()
        score = 0
        done = [False] * n_agents
        episode_step_cntr = 0
        while not any(done):
            if evaluate:
                env.render()
                time.sleep(0.1)  # to slow down the action for the video
            actions = maddpg_agents.choose_action(obs)
            # print("actions:")
            # print(actions)
            # exit()
            obs_, reward, done, info = env.step(actions)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step_cntr >= MAX_STEPS:
                done = [True] * n_agents

            memory.store_transition(obs, state, actions, reward, obs_, state_,
                done)

            # print(f"store transition:")
            # print(f"current observation = {obs}")
            # print(f"next observation = {obs_}")
            # print(f"current state = {state}")
            # print(f"next state = {state_}")
            # print(f"actions = {actions}")
            # print(f"reward = {reward}")

            # do not learn when evaluate, learn every 100 steps
            if total_steps_cntr % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            # set the current state to new state
            obs = obs_

            score += sum(reward)
            total_steps_cntr += 1
            episode_step_cntr += 1

        score_history.append(score)
        # average score of previous 100 games
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
