import torch as T
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx,
            chkpt_dir,
            alpha=0.01, beta=0.01, fc1=64,
            fc2=64, gamma=0.95, tau=0.01):
        """

        Args:
            actor_dims:
            critic_dims:
            n_actions: number of actions
            n_agents:
            agent_idx: agent index
            chkpt_dir: checkpoint directory
            alpha: learning rate
            beta: learning rate
            fc1:
            fc2:
            gamma: discount factor
            tau: soft update parameter
        """
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        # e.g., name = agent_1_actor
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
            chkpt_dir=chkpt_dir, name=self.agent_name + '_actor')
        self.critic = CriticNetwork(beta, critic_dims,
            fc1, fc2, n_agents, n_actions,
            chkpt_dir=chkpt_dir, name=self.agent_name + '_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
            chkpt_dir=chkpt_dir,
            name=self.agent_name + '_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims,
            fc1, fc2, n_agents, n_actions,
            chkpt_dir=chkpt_dir,
            name=self.agent_name + '_target_critic')

        # initially target networks and networks have the same parameters
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        """

        Args:
            observation:

        Returns: action w.r.t. the current policy and exploration

        """
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        # action of current policy
        actions = self.actor.forward(state)
        # exploration (0.1 is the parameter of the exploration)
        noise = 0.1 * T.rand(self.n_actions).to(self.actor.device)
        # print(f"action={action}, noise={noise}")
        action = actions + noise
        # action = actions

        return action.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        # use default tau if nothing is input
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        # soft update of target networks
        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[
                                         name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_state_dict[
                                          name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
