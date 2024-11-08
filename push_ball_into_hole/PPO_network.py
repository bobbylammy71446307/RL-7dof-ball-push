import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions
import numpy as np

class PushNetwork(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim):
        super(PushNetwork, self).__init__()

        # Observation processing network
        self.obs_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Goal processing network
        self.goal_net = nn.Sequential(
            nn.Linear(goal_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Combine features
        combined_dim = 128 + 128

        # Policy network (Actor)
        self.policy_net = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.action_mean = nn.Linear(256, action_dim)
        # Use a learnable parameter for log std
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))

        # Value network (Critic)
        self.value_net = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, observation, achieved_goal, desired_goal):
        # Process observation
        obs_features = self.obs_net(observation)

        # Process goals
        goal_input = torch.cat([achieved_goal, desired_goal], dim=-1)
        goal_features = self.goal_net(goal_input)

        # Combine features
        combined_features = torch.cat([obs_features, goal_features], dim=-1)

        # Policy network
        policy_features = self.policy_net(combined_features)
        action_mean = self.action_mean(policy_features)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        # Value network
        value = self.value_net(combined_features)

        return action_mean, action_std, value

class PPOAgent:
    def __init__(self, network, learning_rate=3e-4):
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)

    def select_action(self, observation, achieved_goal, desired_goal):
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        achieved_goal = torch.tensor(achieved_goal, dtype=torch.float32).unsqueeze(0).to(self.device)
        desired_goal = torch.tensor(desired_goal, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_mean, action_std, value = self.network(observation, achieved_goal, desired_goal)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            action_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            value = value.squeeze(-1)

        action = action.cpu().numpy()[0]
        action_log_prob = action_log_prob.cpu().numpy()[0]
        value = value.cpu().numpy()[0]

        return action, action_log_prob, value

    def evaluate_actions(self, observations, achieved_goals, desired_goals, actions):
        observations = torch.tensor(observations, dtype=torch.float32).to(self.device)
        achieved_goals = torch.tensor(achieved_goals, dtype=torch.float32).to(self.device)
        desired_goals = torch.tensor(desired_goals, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)

        action_mean, action_std, values = self.network(observations, achieved_goals, desired_goals)
        dist = torch.distributions.Normal(action_mean, action_std)
        action_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1)

        values = values.squeeze(-1)

        return action_log_probs, values, dist_entropy

    def compute_loss(self, batch_data, gamma, gae_lambda, clip_epsilon, value_loss_coef, entropy_coef):
        observations = batch_data['observations']
        achieved_goals = batch_data['achieved_goals']
        desired_goals = batch_data['desired_goals']
        actions = batch_data['actions']
        old_log_probs = batch_data['old_log_probs']
        returns = batch_data['returns']
        advantages = batch_data['advantages']

        action_log_probs, values, dist_entropy = self.evaluate_actions(
            observations, achieved_goals, desired_goals, actions
        )

        ratios = torch.exp(action_log_probs - torch.tensor(old_log_probs, dtype=torch.float32).to(self.device))

        advantages= torch.tensor(advantages, dtype=torch.float32).to(self.device)

        surr1 = ratios * advantages.unsqueeze(-1)
        surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages.unsqueeze(-1)
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.functional.mse_loss(values, torch.tensor(returns, dtype=torch.float32).to(self.device))

        entropy_loss = -dist_entropy.mean() * entropy_coef

        total_loss = policy_loss + value_loss_coef * value_loss + entropy_loss

        return total_loss, policy_loss, value_loss

    def update(self, total_loss, max_grad_norm):
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
        self.optimizer.step()

