import torch as T
import torch.nn.functional as F
from scripts.MADDPG.agent import Agent


class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
            scenario='simple', alpha=0.01, beta=0.01, fc1=64,
            fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        """

        Args:
            actor_dims:
            critic_dims:
            n_agents:
            n_actions:
            scenario: default to the 'simple' scenario
            alpha:
            beta:
            fc1:
            fc2:
            gamma:
            tau:
            chkpt_dir:
        """
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        # append the scenario to the checkpoint directory so that models trained with different scenarios are saved to different directories.
        chkpt_dir += scenario
        # create a list of agents
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                chkpt_dir=chkpt_dir))

    def save_checkpoint(self):
        """save models for each agent"""
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        """because the env. expects a list of actions"""
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def learn(self, memory, verbose=False):
        """

        Args:
            memory: the memory buffer

        Returns:

        """
        # we don't want to learn if the buffer size is too small
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
            actor_new_states, states_, dones = memory.sample_buffer()

        # we have to send to a device
        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx],
                dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx],
                dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        # make it more suitable for our NNs,
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        # actions according to the current actor?
        mu_actions = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        # the actually actions that we took
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        if verbose:
            print(f"old actions = {old_actions}")
            print(f"mu actions = {mu_actions}")
            print(f"new actions = {new_actions}")
            print(f"reward = {rewards}")
            exit()

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_,
                new_actions).flatten()
            critic_value_[dones[:, 0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            # y in the paper
            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu_actions).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()
